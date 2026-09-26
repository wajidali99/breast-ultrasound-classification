# =============================================================
# STEP 6 — Final model + ONE-TIME evaluation on the locked test set
# Run:  python scripts/06_test_evaluation.py
# Order is deliberate:
#   1) train the 5 baseline DenseNet121 fold models (trainval only)
#   2) choose the threshold from their out-of-fold predictions (trainval only)
#   3) ONLY THEN predict the test set, once, with the threshold already frozen
# =============================================================
import sys, json, gc, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.utils import get_paths, seed_everything, load_config, get_device
from src.train import train_fold, run_epoch
from src.model import build_model
from src.data import CLASS_MAPS, BUSIDataset, build_transforms, test_split
from src.metrics import binary_metrics, threshold_for_sensitivity
from src.evaluate import group_bootstrap

ARCH, TARGET_SENS = "densenet121", 0.90
STEP5_FOLD_AUC = [0.962, 0.939, 0.949, 0.913, 0.903]      # Step 5 baseline, for reproducibility check
P, device = get_paths(), get_device()
cfg = load_config(str(ROOT / "configs" / "base.yaml"), {"model.arch": ARCH})
df = pd.read_csv(P["out_dir"] / cfg["data"]["splits_csv"])
out = P["out_dir"] / "step6"
(out / "checkpoints").mkdir(parents=True, exist_ok=True)
print(f"Device: {device} | arch={ARCH} | recipe = Step 5 baseline (224px, augmentation, class weights)")

# ---------- 1) Train the 5 fold models (or reuse checkpoints from this session) ----------
oof_parts = []
for fold in range(5):
    ck, pr = out / "checkpoints" / f"{ARCH}_fold{fold}.pt", out / f"oof_fold{fold}.csv"
    if ck.exists() and pr.exists():
        print(f"[skip] fold {fold} already trained"); oof_parts.append(pd.read_csv(pr)); continue
    seed_everything(cfg["seed"])
    state, hist, final, val_pred = train_fold(cfg, df, P["data_root"], fold, device, log=lambda s: None)
    torch.save({k: v.cpu() for k, v in state.items()}, ck)
    vp = val_pred[["fname", "label", "group", "y", "prob_malignant"]].assign(fold=fold)
    vp.to_csv(pr, index=False); oof_parts.append(vp)
    flag = "✓ matches Step 5" if abs(final["auc"] - STEP5_FOLD_AUC[fold]) < 0.005 else "⚠ differs from Step 5"
    print(f"fold {fold}: best ep {final['best_epoch']}/{final['epochs_run']} | val AUC {final['auc']:.3f} {flag}")
    del state, hist, val_pred; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
oof = pd.concat(oof_parts, ignore_index=True)
oof.to_csv(out / "oof_predictions.csv", index=False)

# ---------- 2) Freeze the threshold using trainval (OOF) predictions only ----------
thr = threshold_for_sensitivity(oof["y"], oof["prob_malignant"], TARGET_SENS)
oof_m = binary_metrics(oof["y"], oof["prob_malignant"], thr)
print(f"\nThreshold frozen from OOF (sens ≥ {TARGET_SENS}): {thr:.3f}  "
      f"[OOF AUC {binary_metrics(oof['y'], oof['prob_malignant'])['auc']:.3f}, "
      f"sens {oof_m['sensitivity']:.3f}, spec {oof_m['specificity']:.3f}]")
with open(out / "frozen_threshold.json", "w") as f:
    json.dump({"threshold": thr, "rule": f"sensitivity >= {TARGET_SENS} on OOF", "arch": ARCH}, f, indent=2)

# ---------- 3) Open the test set — once ----------
test_df = test_split(df, cfg["task"])
tv_groups = set(df.loc[df.split == "trainval", "group"])
assert set(test_df["group"]).isdisjoint(tv_groups), "Leakage: test group in trainval!"
assert set(test_df["fname"]).isdisjoint(set(oof["fname"])), "Leakage: test image in OOF!"
print(f"Test set: {len(test_df)} images {test_df['label'].value_counts().to_dict()}")
test_ds = BUSIDataset(test_df, P["data_root"], cfg["data"]["img_size"], build_transforms(False))
test_dl = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                     num_workers=cfg["data"]["num_workers"])
probs, rows = [], []
for fold in range(5):
    model = build_model(ARCH, pretrained=False, n_classes=2, dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(torch.load(out / "checkpoints" / f"{ARCH}_fold{fold}.pt", map_location=device))
    with torch.no_grad():
        _, y_t, p_t = run_epoch(model, test_dl, nn.CrossEntropyLoss(), device)
    assert (y_t == test_df["y"].to_numpy()).all(), "Test label order mismatch"
    probs.append(p_t)
    m = binary_metrics(y_t, p_t, thr)
    rows.append({"model": f"fold{fold}", **m})
    del model; gc.collect()
probs = np.vstack(probs); y = test_df["y"].to_numpy()
per_model = pd.DataFrame(rows); per_model.to_csv(out / "test_per_model.csv", index=False)
ens = probs.mean(0)
ens_m = binary_metrics(y, ens, thr)
ci = group_bootstrap(y, probs, test_df["group"].to_numpy(), thr, n_boot=2000, seed=cfg["seed"])

pred_df = test_df[["fname", "label", "group", "y"]].copy()
for k in range(5):
    pred_df[f"prob_fold{k}"] = probs[k]
pred_df["prob_ensemble"] = ens
pred_df["n_models_malignant"] = (probs >= thr).sum(0)
pred_df["n_models_wrong"] = ((probs >= thr).astype(int) != y[None, :]).sum(0)
pred_df.to_csv(out / "test_predictions.csv", index=False)

summary = {
    "arch": ARCH, "threshold": thr, "n_test": int(len(y)), "n_malignant": int(y.sum()),
    "primary_single_models_at_threshold": {
        "auc_mean": per_model["auc"].mean(), "auc_sd": per_model["auc"].std(ddof=1),
        "sensitivity_mean": per_model["sensitivity"].mean(), "sensitivity_sd": per_model["sensitivity"].std(ddof=1),
        "sensitivity_95ci": ci["mean_model_sens_ci"],
        "specificity_mean": per_model["specificity"].mean(), "specificity_sd": per_model["specificity"].std(ddof=1),
        "specificity_95ci": ci["mean_model_spec_ci"]},
    "secondary_ensemble": {"auc": ens_m["auc"], "auc_95ci": ci["ensemble_auc_ci"],
                           "sens_at_threshold": ens_m["sensitivity"], "spec_at_threshold": ens_m["specificity"]},
    "bootstrap": {"n_boot_used": ci["n_boot_used"], "unit": "duplicate group"},
}
with open(out / "test_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---------- 4) Report ----------
print("\n================ LOCKED TEST SET — single-model results (primary) ================")
for _, r in per_model.iterrows():
    print(f"{r['model']}: AUC {r['auc']:.3f}  sens {r['sensitivity']:.3f} ({r['tp']}/{r['tp']+r['fn']})  "
          f"spec {r['specificity']:.3f} ({r['tn']}/{r['tn']+r['fp']})  missed={r['fn']}  false alarms={r['fp']}")
s = summary["primary_single_models_at_threshold"]
print(f"\nMEAN of 5 models @ thr {thr:.3f}:")
print(f"  AUC          {s['auc_mean']:.3f} ± {s['auc_sd']:.3f}")
print(f"  Sensitivity  {s['sensitivity_mean']:.3f} ± {s['sensitivity_sd']:.3f}   95% CI {s['sensitivity_95ci'][0]:.3f}–{s['sensitivity_95ci'][1]:.3f}")
print(f"  Specificity  {s['specificity_mean']:.3f} ± {s['specificity_sd']:.3f}   95% CI {s['specificity_95ci'][0]:.3f}–{s['specificity_95ci'][1]:.3f}")
print(f"\nENSEMBLE (secondary): AUC {ens_m['auc']:.3f}  95% CI {ci['ensemble_auc_ci'][0]:.3f}–{ci['ensemble_auc_ci'][1]:.3f}")
print(f"  (at the same threshold: sens {ens_m['sensitivity']:.3f}, spec {ens_m['specificity']:.3f} — "
      f"threshold was tuned for single models, so treat as indicative)")
hard = pred_df[pred_df["n_models_wrong"] >= 3].sort_values("n_models_wrong", ascending=False)
print(f"\nImages misclassified by ≥3 of 5 models: {len(hard)} "
      f"({(hard.y == 1).sum()} malignant missed, {(hard.y == 0).sum()} benign false alarms)")
print(hard[["fname", "label", "n_models_wrong", "prob_ensemble"]].head(15).to_string(index=False))
if len(hard) > 15:
    print(f"... ({len(hard) - 15} more in test_predictions.csv)")

# ---------- 5) Figure ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
f_o, t_o, _ = roc_curve(oof["y"], oof["prob_malignant"]); f_t, t_t, _ = roc_curve(y, ens)
ax[0].plot(f_o, t_o, lw=1.5, ls="--", label=f"validation OOF (AUC {binary_metrics(oof['y'], oof['prob_malignant'])['auc']:.3f})")
ax[0].plot(f_t, t_t, lw=2.2, label=f"TEST ensemble (AUC {ens_m['auc']:.3f}, 95% CI {ci['ensemble_auc_ci'][0]:.2f}–{ci['ensemble_auc_ci'][1]:.2f})")
ax[0].plot([0, 1], [0, 1], ls=":", c="gray")
ax[0].set(title="ROC: validation vs locked test", xlabel="1 - specificity", ylabel="sensitivity")
ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3)
x = np.arange(5); w = 0.38
ax[1].bar(x - w/2, per_model["sensitivity"], w, label="sensitivity")
ax[1].bar(x + w/2, per_model["specificity"], w, label="specificity")
ax[1].axhline(TARGET_SENS, ls="--", c="gray", lw=1, label=f"target sens {TARGET_SENS}")
ax[1].set_xticks(x, per_model["model"]); ax[1].set_ylim(0, 1.22)
ax[1].set_title(f"Each fold model on TEST at thr {thr:.2f}"); ax[1].legend(fontsize=8, loc="upper center", ncol=3)
ax[1].grid(axis="y", alpha=0.3)
for cls, name in [(0, "benign"), (1, "malignant")]:
    ax[2].hist(ens[y == cls], bins=20, range=(0, 1), alpha=0.6, label=f"{name} (n={int((y == cls).sum())})")
ax[2].axvline(thr, ls="--", c="k", lw=1, label=f"threshold {thr:.2f}")
ax[2].set(title="TEST ensemble probability by true class", xlabel="P(malignant)", ylabel="images")
ax[2].legend(fontsize=8)
fig.suptitle("Step 6: DenseNet121 on the locked test set (105 images, evaluated once)", fontweight="bold")
plt.savefig(out / "test_results.png", dpi=100, bbox_inches="tight")
print(f"\nSaved to: {out}")

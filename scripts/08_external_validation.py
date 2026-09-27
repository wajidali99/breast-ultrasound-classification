# =============================================================
# STEP 8 — External validation on BUS-BRA (Brazil; different hospital, devices, patients)
# Run:  python scripts/08_external_validation.py
# The BUSI-trained fold models and the BUSI-frozen threshold (0.225) are applied AS IS.
# Nothing is retrained or re-tuned on BUS-BRA.
# =============================================================
import sys, json, gc
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

from src.utils import get_paths, load_config, get_device, seed_everything
from src.data import PathDataset, build_transforms
from src.model import build_model
from src.train import run_epoch
from src.metrics import binary_metrics, threshold_for_sensitivity
from src.evaluate import group_bootstrap
from src.external import find_external_root, index_external
from src.final_models import ensure_fold_models, ARCH

P, device = get_paths(), get_device()
cfg = load_config(str(ROOT / "configs" / "base.yaml"), {"model.arch": ARCH})
seed_everything(cfg["seed"])
df_busi = pd.read_csv(P["out_dir"] / cfg["data"]["splits_csv"])
out = P["out_dir"] / "step8"; out.mkdir(parents=True, exist_ok=True)

# ---------- 1) Index BUS-BRA ----------
ext_root = find_external_root()
ext, info = index_external(ext_root)
print(f"External root: {ext_root}")
print(f"Label CSV: {info['csv']}  (id column '{info['id_column']}', pathology column '{info['pathology_column']}')")
print(f"CSV columns: {info['columns']}")
print(f"Images matched: {info['matched']} of {info['csv_rows']} rows  (missing: {info['missing']} {info['missing_examples']})")
assert info["matched"] > 0, "No external images matched the CSV"
assert ext["y"].nunique() == 2, "External set must contain both benign and malignant"
print(f"External set: {len(ext)} images, {ext['patient'].nunique()} patients, "
      f"{ext['label'].value_counts().to_dict()}")
ext.to_csv(out / "external_index.csv", index=False)

# ---------- 2) Models + frozen threshold (from BUSI only) ----------
ckpts, thr = ensure_fold_models(cfg, df_busi, P["data_root"], P["out_dir"] / "step6", device)
print(f"Using {len(ckpts)} BUSI fold models, frozen BUSI threshold {thr:.3f}")

# ---------- 3) Predict ----------
ds = PathDataset(ext, ext_root, cfg["data"]["img_size"], build_transforms(False))
dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])
probs, rows = [], []
y = ext["y"].to_numpy()
for k, ck in enumerate(ckpts):
    model = build_model(ARCH, pretrained=False, n_classes=2, dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(torch.load(ck, map_location=device))
    with torch.no_grad():
        _, y_k, p_k = run_epoch(model, dl, nn.CrossEntropyLoss(), device)
    assert (y_k == y).all(), "Label order mismatch"
    probs.append(p_k); rows.append({"model": f"fold{k}", **binary_metrics(y, p_k, thr)})
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
probs = np.vstack(probs); ens = probs.mean(0)
per_model = pd.DataFrame(rows); per_model.to_csv(out / "external_per_model.csv", index=False)
ci = group_bootstrap(y, probs, ext["patient"].to_numpy(), thr, n_boot=2000, seed=cfg["seed"])
ens_m = binary_metrics(y, ens, thr)
pred = ext[["image_id", "patient", "label", "y"]].copy()
for k in range(len(ckpts)):
    pred[f"prob_fold{k}"] = probs[k]
pred["prob_ensemble"] = ens
pred.to_csv(out / "external_predictions.csv", index=False)

# ---------- 4) Compare with the BUSI test set (Step 6) ----------
s6 = P["out_dir"] / "step6" / "test_summary.json"
busi = json.load(open(s6)) if s6.exists() else None
s = {"auc_mean": per_model["auc"].mean(), "auc_sd": per_model["auc"].std(ddof=1),
     "sens_mean": per_model["sensitivity"].mean(), "sens_sd": per_model["sensitivity"].std(ddof=1),
     "spec_mean": per_model["specificity"].mean(), "spec_sd": per_model["specificity"].std(ddof=1)}
print("\n================ EXTERNAL (BUS-BRA) — each BUSI model at the frozen threshold ================")
for _, r in per_model.iterrows():
    print(f"{r['model']}: AUC {r['auc']:.3f}  sens {r['sensitivity']:.3f} ({r['tp']}/{r['tp']+r['fn']})  "
          f"spec {r['specificity']:.3f} ({r['tn']}/{r['tn']+r['fp']})")
print(f"\n{'':<22}{'BUSI test (Step 6)':>22}{'BUS-BRA external':>24}")
b = busi["primary_single_models_at_threshold"] if busi else None
fmt = lambda m, sd: f"{m:.3f} ± {sd:.3f}"
print(f"{'AUC (mean of 5)':<22}{fmt(b['auc_mean'], b['auc_sd']) if b else 'n/a':>22}{fmt(s['auc_mean'], s['auc_sd']):>24}")
print(f"{'Sensitivity @ thr':<22}{fmt(b['sensitivity_mean'], b['sensitivity_sd']) if b else 'n/a':>22}{fmt(s['sens_mean'], s['sens_sd']):>24}")
print(f"{'Specificity @ thr':<22}{fmt(b['specificity_mean'], b['specificity_sd']) if b else 'n/a':>22}{fmt(s['spec_mean'], s['spec_sd']):>24}")
bens = busi["secondary_ensemble"]["auc"] if busi else float("nan")
print(f"{'Ensemble AUC':<22}{bens:>22.3f}{ens_m['auc']:>24.3f}")
print(f"\nBUS-BRA 95% CIs (patient-level bootstrap): ensemble AUC {ci['ensemble_auc_ci'][0]:.3f}–{ci['ensemble_auc_ci'][1]:.3f} | "
      f"sens {ci['mean_model_sens_ci'][0]:.3f}–{ci['mean_model_sens_ci'][1]:.3f} | "
      f"spec {ci['mean_model_spec_ci'][0]:.3f}–{ci['mean_model_spec_ci'][1]:.3f}")

# ---------- 5) Post-hoc (NOT a claim): where would a 90%-sensitivity threshold sit on BUS-BRA? ----------
post_thr = threshold_for_sensitivity(y, ens, 0.90)
post = binary_metrics(y, ens, post_thr)
print(f"\nPost-hoc only (tuned ON BUS-BRA, shows calibration shift): ensemble threshold for sens≥0.90 = "
      f"{post_thr:.3f} (BUSI: {thr:.3f}) → spec {post['specificity']:.3f}")

# Optional breakdown by device, if the CSV has such a column
dev_col = next((c for c in ext.columns if c.lower() == "meta_device"), None)
dev_rows = []
if dev_col:
    for dv, g in pred.join(ext[[dev_col]]).groupby(dev_col):
        if g["y"].nunique() == 2 and len(g) >= 30:
            dev_rows.append({"device": dv, "n": len(g), "n_malignant": int(g["y"].sum()),
                             "ensemble_auc": binary_metrics(g["y"], g["prob_ensemble"])["auc"]})
    if dev_rows:
        print("\nEnsemble AUC by ultrasound device (groups with ≥30 images and both classes):")
        for r in dev_rows:
            print(f"  {str(r['device']):<25} n={r['n']:<5} malignant={r['n_malignant']:<4} AUC {r['ensemble_auc']:.3f}")
pd.DataFrame(dev_rows).to_csv(out / "external_by_device.csv", index=False)

json.dump({"dataset": "BUS-BRA", "n_images": int(len(ext)), "n_patients": int(ext["patient"].nunique()),
           "n_malignant_images": int(y.sum()), "threshold_from_busi": thr,
           "primary_single_models_at_threshold": {**s, "sens_95ci": ci["mean_model_sens_ci"], "spec_95ci": ci["mean_model_spec_ci"]},
           "secondary_ensemble": {"auc": ens_m["auc"], "auc_95ci": ci["ensemble_auc_ci"]},
           "posthoc_threshold_for_sens90_on_external": post_thr, "posthoc_spec_at_that_threshold": post["specificity"],
           "bootstrap_unit": "patient", "by_device": dev_rows, "index_info": info},
          open(out / "external_summary.json", "w"), indent=2, default=str)

# ---------- 6) Figures ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
bt = P["out_dir"] / "step6" / "test_predictions.csv"
if bt.exists():
    t6 = pd.read_csv(bt); f6, r6, _ = roc_curve(t6["y"], t6["prob_ensemble"])
    ax[0].plot(f6, r6, ls="--", lw=1.6, label=f"BUSI test (AUC {binary_metrics(t6['y'], t6['prob_ensemble'])['auc']:.3f})")
fe, te, _ = roc_curve(y, ens)
ax[0].plot(fe, te, lw=2.2, label=f"BUS-BRA external (AUC {ens_m['auc']:.3f}, 95% CI {ci['ensemble_auc_ci'][0]:.2f}–{ci['ensemble_auc_ci'][1]:.2f})")
ax[0].plot([0, 1], [0, 1], ls=":", c="gray")
ax[0].set(title="ROC: internal test vs external (5-model ensemble)", xlabel="1 - specificity", ylabel="sensitivity")
ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3)
x = np.arange(len(per_model)); w = 0.38
ax[1].bar(x - w/2, per_model["sensitivity"], w, label="sensitivity")
ax[1].bar(x + w/2, per_model["specificity"], w, label="specificity")
ax[1].axhline(0.90, ls="--", c="gray", lw=1, label="target sens 0.90")
ax[1].set_xticks(x, per_model["model"]); ax[1].set_ylim(0, 1.22)
ax[1].set_title(f"Each BUSI model on BUS-BRA at thr {thr:.2f}"); ax[1].legend(fontsize=8, loc="upper center", ncol=3)
ax[1].grid(axis="y", alpha=0.3)
for cls, name in [(0, "benign"), (1, "malignant")]:
    ax[2].hist(ens[y == cls], bins=25, range=(0, 1), alpha=0.6, label=f"{name} (n={int((y == cls).sum())})")
ax[2].axvline(thr, ls="--", c="k", lw=1, label=f"BUSI threshold {thr:.2f}")
ax[2].set(title="BUS-BRA ensemble probability by true class", xlabel="P(malignant)", ylabel="images"); ax[2].legend(fontsize=8)
fig.suptitle(f"Step 8: external validation on BUS-BRA ({len(ext)} images, {ext['patient'].nunique()} patients) — no retraining, no re-tuning",
             fontweight="bold")
plt.savefig(out / "external_results.png", dpi=100, bbox_inches="tight"); plt.close(fig)

# Side-by-side look at the two domains
from PIL import Image
busi_rows = df_busi[(df_busi.split == "test") & (df_busi.label != "normal")].sample(4, random_state=0)
ext_rows = pd.concat([ext[ext.label == lab].sample(2, random_state=0) for lab in ["benign", "malignant"]])
fig, axs = plt.subplots(2, 4, figsize=(13, 6.5), layout="constrained")
for a, (_, r) in zip(axs[0], busi_rows.iterrows()):
    a.imshow(Image.open(P["data_root"] / r["label"] / r["fname"]).convert("L"), cmap="gray"); a.set_title(f"BUSI · {r['label']}", fontsize=9); a.axis("off")
for a, (_, r) in zip(axs[1], ext_rows.iterrows()):
    a.imshow(Image.open(r["path"]).convert("L"), cmap="gray"); a.set_title(f"BUS-BRA · {r['label']}", fontsize=9); a.axis("off")
fig.suptitle("Domain shift: BUSI (Egypt) vs BUS-BRA (Brazil)", fontweight="bold")
plt.savefig(out / "domain_examples.png", dpi=100, bbox_inches="tight")
print(f"\nSaved to: {out}")

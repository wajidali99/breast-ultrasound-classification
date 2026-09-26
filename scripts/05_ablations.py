# =============================================================
# STEP 5 — Ablations on DenseNet121 (5-fold CV) + threshold selection
# Run:  python scripts/05_ablations.py
# Optional:  --only baseline no_aug        (sirf kuch experiments)
# Resume-safe: jo experiment/fold ho chuka hai wo skip hota hai (same session mein).
# =============================================================
import sys, json, gc, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.utils import get_paths, seed_everything, load_config, get_device
from src.train import train_fold
from src.metrics import binary_metrics, threshold_for_sensitivity, youden_threshold

ARCH = "densenet121"   # Step 4 winner
# Har experiment baseline se sirf EK cheez badalta hai
EXPERIMENTS = {
    "baseline":         {},
    "no_aug":           {"data.augment": False},
    "no_class_weights": {"train.class_weights": False},
    "img320":           {"data.img_size": 320},
    "add_conflicting":  {"data.add_conflicting_to_train": True},
}
DESCRIPTION = {
    "baseline":         "DenseNet121, 224px, augmentation, class weights",
    "no_aug":           "augmentation OFF",
    "no_class_weights": "class weights OFF",
    "img320":           "input 320x320 instead of 224",
    "add_conflicting":  "18 conflicting-label images added to training",
}
ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="+", default=list(EXPERIMENTS))
ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
args = ap.parse_args()
assert "baseline" in args.only, "Baseline zaroori hai — baaqi sab us se compare hote hain"

P, device = get_paths(), get_device()
base_path = str(ROOT / "configs" / "base.yaml")
df = pd.read_csv(P["out_dir"] / load_config(base_path)["data"]["splits_csv"])
out = P["out_dir"] / "step5"
(out / "runs").mkdir(parents=True, exist_ok=True)
print(f"Device: {device} | arch={ARCH} | experiments={args.only} | folds={args.folds}")

# ---------- 1) Train ----------
for name in args.only:
    for fold in args.folds:
        tag = f"{name}_fold{fold}"
        if (out / "runs" / f"{tag}_summary.json").exists():
            print(f"[skip] {tag}"); continue
        cfg = load_config(base_path, {"model.arch": ARCH, **EXPERIMENTS[name]})
        seed_everything(cfg["seed"])
        print(f"\n===== {tag}  ({DESCRIPTION[name]}) =====", flush=True)
        t0 = time.time()
        state, hist, final, val_pred = train_fold(cfg, df, P["data_root"], fold, device, log=lambda s: None)
        final.update({"experiment": name, "minutes": round((time.time() - t0) / 60, 2)})
        hist.to_csv(out / "runs" / f"{tag}_history.csv", index=False)
        val_pred[["fname", "label", "group", "y", "prob_malignant"]].to_csv(
            out / "runs" / f"{tag}_val_predictions.csv", index=False)
        with open(out / "runs" / f"{tag}_summary.json", "w") as f:
            json.dump(final, f, indent=2)
        print(f"best ep {final['best_epoch']}/{final['epochs_run']} | AUC {final['auc']:.3f} "
              f"sens {final['sensitivity']:.3f} spec {final['specificity']:.3f} | {final['minutes']} min", flush=True)
        del state, hist, val_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------- 2) Ablation table (paired against baseline, fold by fold) ----------
res = pd.DataFrame([json.load(open(p)) for p in sorted((out / "runs").glob("*_summary.json"))])
res.to_csv(out / "ablation_per_fold.csv", index=False)
base = res[res.experiment == "baseline"].set_index("val_fold")
rows = []
for name in [n for n in EXPERIMENTS if n in set(res.experiment)]:
    g = res[res.experiment == name].set_index("val_fold")
    common = sorted(set(g.index) & set(base.index))
    delta = g.loc[common, "auc"] - base.loc[common, "auc"]
    preds = pd.concat([pd.read_csv(out / "runs" / f"{name}_fold{f}_val_predictions.csv") for f in g.index])
    assert not preds["fname"].duplicated().any(), "Same image predicted twice"
    rows.append({
        "experiment": name, "change": DESCRIPTION[name], "n_folds": len(g),
        "auc_mean": g["auc"].mean(), "auc_std": g["auc"].std(ddof=1),
        "sens_mean": g["sensitivity"].mean(), "spec_mean": g["specificity"].mean(),
        "pooled_oof_auc": binary_metrics(preds["y"], preds["prob_malignant"])["auc"],
        "delta_auc_vs_baseline": delta.mean() if name != "baseline" else 0.0,
        "folds_better_than_baseline": int((delta > 0).sum()) if name != "baseline" else None,
    })
abl = pd.DataFrame(rows)
abl.to_csv(out / "ablation_summary.csv", index=False)

print("\n=================== ABLATIONS (DenseNet121, 5-fold CV, validation) ===================")
for _, r in abl.iterrows():
    extra = "" if r.experiment == "baseline" else \
        f"  ΔAUC {r.delta_auc_vs_baseline:+.3f} (better on {int(r.folds_better_than_baseline)}/{int(r.n_folds)} folds)"
    print(f"{r.experiment:<17} AUC {r.auc_mean:.3f} ± {r.auc_std:.3f}  pooled {r.pooled_oof_auc:.3f}  "
          f"sens {r.sens_mean:.3f}  spec {r.spec_mean:.3f}{extra}")

# ---------- 3) Threshold selection on baseline out-of-fold predictions ----------
oof = pd.concat([pd.read_csv(out / "runs" / f"baseline_fold{f}_val_predictions.csv") for f in base.index])
y, p = oof["y"].to_numpy(), oof["prob_malignant"].to_numpy()
thresholds = {
    "default_0.5": 0.5,
    "youden": youden_threshold(y, p),
    "sens_0.90": threshold_for_sensitivity(y, p, 0.90),
    "sens_0.95": threshold_for_sensitivity(y, p, 0.95),
}
print("\n=========== THRESHOLDS (chosen on baseline OOF predictions, trainval only) ===========")
thr_rows = []
for k, t in thresholds.items():
    m = binary_metrics(y, p, t)
    thr_rows.append({"rule": k, "threshold": t, **{x: m[x] for x in ["sensitivity", "specificity", "balanced_acc", "tp", "fn", "tn", "fp"]}})
    print(f"{k:<12} thr={t:.3f}  sens {m['sensitivity']:.3f} ({m['tp']}/{m['tp']+m['fn']})  "
          f"spec {m['specificity']:.3f} ({m['tn']}/{m['tn']+m['fp']})  missed cancers={m['fn']}  false alarms={m['fp']}")
pd.DataFrame(thr_rows).to_csv(out / "threshold_analysis.csv", index=False)
with open(out / "chosen_thresholds.json", "w") as f:
    json.dump({"arch": ARCH, "source": "baseline pooled OOF (trainval only)", **thresholds}, f, indent=2)

# ---------- 4) Figures ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
order = abl["experiment"].tolist()
data = [res.loc[res.experiment == n, "auc"].values for n in order]
ax[0].boxplot(data, showmeans=True)
ax[0].set_xticks(range(1, len(order) + 1), order)
rng = np.random.default_rng(0)
for i, d in enumerate(data, start=1):
    ax[0].scatter(np.full(len(d), i) + rng.uniform(-0.08, 0.08, len(d)), d, s=18, alpha=0.7, zorder=3)
ax[0].axhline(abl.loc[abl.experiment == "baseline", "auc_mean"].item(), ls="--", c="gray", lw=1)
ax[0].set_title("Validation AUC per fold (dashed = baseline mean)"); ax[0].tick_params(axis="x", rotation=15)
ax[0].grid(axis="y", alpha=0.3)

fpr, tpr, _ = roc_curve(y, p)
ax[1].plot(fpr, tpr, lw=2, label=f"baseline OOF (AUC {binary_metrics(y, p)['auc']:.3f})")
ax[1].plot([0, 1], [0, 1], ls=":", c="gray")
for (k, t), mk in zip(thresholds.items(), ["o", "s", "^", "D"]):
    m = binary_metrics(y, p, t)
    ax[1].scatter(1 - m["specificity"], m["sensitivity"], s=70, marker=mk, zorder=3,
                  label=f"{k} (thr {t:.2f}): sens {m['sensitivity']:.2f}, spec {m['specificity']:.2f}")
ax[1].set(title="ROC of out-of-fold predictions with candidate thresholds",
          xlabel="1 - specificity (false alarm rate)", ylabel="sensitivity")
ax[1].legend(fontsize=8, loc="lower right"); ax[1].grid(alpha=0.3)
fig.suptitle("Step 5: DenseNet121 ablations and operating threshold", fontweight="bold")
plt.savefig(out / "ablations_thresholds.png", dpi=100, bbox_inches="tight")
print(f"\nSaved to: {out}")

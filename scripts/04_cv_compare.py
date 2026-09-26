# =============================================================
# STEP 4 — 5-fold cross-validation + architecture comparison
# Run:  python scripts/04_cv_compare.py
# Optional:  --archs resnet50 efficientnet_b0   --folds 0 1 2
# Resume-safe: agar kisi arch/fold ka result pehle se saved hai, to wo skip hota hai.
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

from src.utils import get_paths, seed_everything, load_config, get_device
from src.train import train_fold
from src.metrics import binary_metrics

DEFAULT_ARCHS = ["resnet50", "efficientnet_b0", "densenet121", "convnext_tiny"]
ap = argparse.ArgumentParser()
ap.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS)
ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
args = ap.parse_args()

P, device = get_paths(), get_device()
base_cfg = load_config(str(ROOT / "configs" / "base.yaml"))
splits_csv = P["out_dir"] / base_cfg["data"]["splits_csv"]
assert splits_csv.exists(), f"{splits_csv} nahi mili — pehle Step 1 script chalayein."
df = pd.read_csv(splits_csv)
out = P["out_dir"] / "step4"
(out / "checkpoints").mkdir(parents=True, exist_ok=True)
(out / "runs").mkdir(exist_ok=True)
print(f"Device: {device} | archs={args.archs} | folds={args.folds}")

# ---------- 1) Train every arch on every fold ----------
for arch in args.archs:
    for fold in args.folds:
        tag = f"{arch}_fold{fold}"
        if (out / "runs" / f"{tag}_summary.json").exists():
            print(f"[skip] {tag} already done"); continue
        cfg = load_config(str(ROOT / "configs" / "base.yaml"), {"model.arch": arch})
        seed_everything(cfg["seed"])                      # same seed -> fair comparison
        print(f"\n===== {tag} =====", flush=True)
        t0 = time.time()
        state, hist, final, val_pred = train_fold(cfg, df, P["data_root"], fold, device,
                                                  log=lambda s: None)  # quiet per-epoch log
        final["minutes"] = round((time.time() - t0) / 60, 2)
        torch.save({k: v.cpu() for k, v in state.items()}, out / "checkpoints" / f"{tag}.pt")
        hist.to_csv(out / "runs" / f"{tag}_history.csv", index=False)
        val_pred[["fname", "label", "group", "y", "prob_malignant"]].to_csv(
            out / "runs" / f"{tag}_val_predictions.csv", index=False)
        with open(out / "runs" / f"{tag}_summary.json", "w") as f:
            json.dump(final, f, indent=2)
        print(f"best ep {final['best_epoch']}/{final['epochs_run']} | AUC {final['auc']:.3f} "
              f"sens {final['sensitivity']:.3f} spec {final['specificity']:.3f} | {final['minutes']} min",
              flush=True)
        del state, hist, val_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------- 2) Collect all saved runs ----------
rows = [json.load(open(p)) for p in sorted((out / "runs").glob("*_summary.json"))]
assert len(rows), "Koi run nahi mila"
res = pd.DataFrame(rows).sort_values(["arch", "val_fold"])
res.to_csv(out / "cv_per_fold.csv", index=False)

metrics = ["auc", "sensitivity", "specificity", "balanced_acc"]
summary = []
for arch, g in res.groupby("arch"):
    row = {"arch": arch, "n_folds": len(g)}
    for m in metrics:
        row[f"{m}_mean"] = g[m].mean()
        row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
    # Pooled out-of-fold AUC: har trainval image ek dafa predict hui, us model se jis ne use kabhi train mein nahi dekha
    preds = pd.concat([pd.read_csv(out / "runs" / f"{arch}_fold{f}_val_predictions.csv") for f in g["val_fold"]])
    assert not preds["fname"].duplicated().any(), "Same image predicted twice across folds"
    row["pooled_oof_auc"] = binary_metrics(preds["y"], preds["prob_malignant"])["auc"]
    row["mean_best_epoch"] = g["best_epoch"].mean()
    row["minutes_total"] = g["minutes"].sum() if "minutes" in g else float("nan")
    summary.append(row)
summ = pd.DataFrame(summary).sort_values("auc_mean", ascending=False).reset_index(drop=True)
summ.to_csv(out / "cv_summary.csv", index=False)

print("\n================ 5-FOLD CV SUMMARY (validation, threshold 0.5) ================")
for _, r in summ.iterrows():
    print(f"{r['arch']:<16} folds={r['n_folds']}  AUC {r['auc_mean']:.3f} ± {r['auc_std']:.3f}  "
          f"sens {r['sensitivity_mean']:.3f} ± {r['sensitivity_std']:.3f}  "
          f"spec {r['specificity_mean']:.3f} ± {r['specificity_std']:.3f}  "
          f"pooled-OOF AUC {r['pooled_oof_auc']:.3f}  ({r['minutes_total']:.1f} min)")

# ---------- 3) Figure: fold-level AUC and sensitivity per arch ----------
order = summ["arch"].tolist()
rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
for a, m, title in [(ax[0], "auc", "Validation AUC per fold"),
                    (ax[1], "sensitivity", "Sensitivity per fold (threshold 0.5)")]:
    data = [res.loc[res.arch == arch, m].values for arch in order]
    a.boxplot(data, showmeans=True)
    a.set_xticks(range(1, len(order) + 1), order)
    for i, d in enumerate(data, start=1):
        a.scatter(np.full(len(d), i) + rng.uniform(-0.08, 0.08, len(d)), d, s=18, alpha=0.7, zorder=3)
    a.set_title(title); a.tick_params(axis="x", rotation=15); a.grid(axis="y", alpha=0.3)
fig.suptitle("5-fold CV: architecture comparison (benign vs malignant)", fontweight="bold")
plt.savefig(out / "cv_comparison.png", dpi=100, bbox_inches="tight")
print(f"\nSaved to: {out}")

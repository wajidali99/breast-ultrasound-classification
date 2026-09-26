# =============================================================
# STEP 3 — Baseline CNN on ONE fold
# Run:  python scripts/03_train_baseline.py --fold 0
# Optional overrides: --arch resnet50 --epochs 30 --lr 1e-4
# =============================================================
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import get_paths, seed_everything, load_config, get_device
from src.train import train_fold

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--arch", type=str, default=None)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--lr", type=float, default=None)
args = ap.parse_args()

overrides = {k: v for k, v in {"model.arch": args.arch, "train.epochs": args.epochs,
                               "train.lr": args.lr}.items() if v is not None}
cfg = load_config(str(ROOT / "configs" / "base.yaml"), overrides)
seed_everything(cfg["seed"])
P, device = get_paths(), get_device()
print(f"Device: {device} | config: arch={cfg['model']['arch']} lr={cfg['train']['lr']} "
      f"epochs={cfg['train']['epochs']} batch={cfg['train']['batch_size']}")

splits_csv = P["out_dir"] / cfg["data"]["splits_csv"]
assert splits_csv.exists(), f"{splits_csv} nahi mili — pehle Step 1 script chalayein."
df = pd.read_csv(splits_csv)

state, hist, final, val_pred = train_fold(cfg, df, P["data_root"], args.fold, device)

# ---------- Save everything ----------
tag = f"{cfg['model']['arch']}_fold{args.fold}"
out = P["out_dir"] / "step3"
(out / "checkpoints").mkdir(parents=True, exist_ok=True)
torch.save(state, out / "checkpoints" / f"{tag}.pt")
hist.to_csv(out / f"{tag}_history.csv", index=False)
val_pred[["fname", "label", "group", "y", "prob_malignant"]].to_csv(out / f"{tag}_val_predictions.csv", index=False)
with open(out / f"{tag}_summary.json", "w") as f:
    json.dump(final, f, indent=2)

fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
ax[0].plot(hist.epoch, hist.train_loss, "o-", label="train"); ax[0].plot(hist.epoch, hist.val_loss, "o-", label="val")
ax[0].axvline(final["best_epoch"], ls="--", c="gray"); ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend()
ax[1].plot(hist.epoch, hist.train_auc, "o-", label="train"); ax[1].plot(hist.epoch, hist.val_auc, "o-", label="val")
ax[1].axvline(final["best_epoch"], ls="--", c="gray", label=f"best (ep {final['best_epoch']})")
ax[1].set(title="AUC", xlabel="epoch", ylim=(0.4, 1.01)); ax[1].legend()
fig.suptitle(f"{cfg['model']['arch']} — fold {args.fold}", fontweight="bold")
plt.savefig(out / f"{tag}_curves.png", dpi=100, bbox_inches="tight")

print("\n===== BEST EPOCH (validation, threshold 0.5) =====")
for k in ["best_epoch", "epochs_run", "auc", "sensitivity", "specificity", "balanced_acc", "accuracy"]:
    v = final[k]; print(f"{k:>13}: {v:.3f}" if isinstance(v, float) else f"{k:>13}: {v}")
print(f"Confusion: TP={final['tp']} FN={final['fn']} TN={final['tn']} FP={final['fp']}")
print(f"Saved to: {out}")

# =============================================================
# STEP 2 — Data pipeline sanity check
# Run from repo root:  python scripts/02_data_check.py
# Needs Step 1 output: busi_splits.csv (in out_dir)
# =============================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import get_paths, seed_everything, load_config
from src.data import (CLASS_MAPS, IMAGENET_MEAN, IMAGENET_STD, select_rows, fold_split,
                      test_split, class_weights, make_loaders)

cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"))
seed_everything(cfg["seed"])
P = get_paths()
TASK, SIZE, BS = cfg["task"], cfg["data"]["img_size"], cfg["train"]["batch_size"]

splits_csv = P["out_dir"] / cfg["data"]["splits_csv"]
assert splits_csv.exists(), f"{splits_csv} nahi mili — pehle Step 1 script chalayein."
df = pd.read_csv(splits_csv)

# ---------- 1) How many images per split for this task ----------
names = {v: k for k, v in CLASS_MAPS[TASK].items()}
print(f"Task: {TASK}   labels: {CLASS_MAPS[TASK]}")
te = test_split(df, TASK)
print(f"Test set (locked, not used here): {len(te)} images  {te['label'].value_counts().to_dict()}")
for k in range(5):
    tr, va = fold_split(df, TASK, k)
    print(f"Fold {k}:  train={len(tr):3d} {tr['label'].value_counts().to_dict()}   "
          f"val={len(va):3d} {va['label'].value_counts().to_dict()}")
    assert set(tr["group"]).isdisjoint(te["group"]) and set(va["group"]).isdisjoint(te["group"]), \
        "Leakage: test group in train/val!"

# ---------- 2) Build loaders for fold 0 ----------
train_dl, val_dl, train_df, val_df = make_loaders(df, P["data_root"], TASK, val_fold=0,
                                                  img_size=SIZE, batch_size=BS,
                                                  num_workers=cfg["data"]["num_workers"],
                                                  seed=cfg["seed"])
w = class_weights(train_df, len(CLASS_MAPS[TASK]))
print(f"\nClass weights (fold 0): { {names[i]: round(float(v), 3) for i, v in enumerate(w)} }")

x, y = next(iter(train_dl))
print(f"Batch images: {tuple(x.shape)}  dtype={x.dtype}  min={x.min():.2f}  max={x.max():.2f}")
print(f"Batch labels: {tuple(y.shape)}  values={sorted(set(y.tolist()))}")

# ---------- 3) Sanity checks ----------
n_cls = len(CLASS_MAPS[TASK])
assert x.shape[1:] == (3, SIZE, SIZE), f"Wrong image shape {x.shape}"
assert set(y.tolist()) <= set(range(n_cls)), "Unknown label in batch"
assert torch.isfinite(x).all(), "NaN/Inf in images"
if TASK == "binary":
    assert "normal" not in set(train_df["label"]) | set(val_df["label"]), "Normal images in binary task!"
xv, _ = next(iter(val_dl))
assert xv.shape[1:] == (3, SIZE, SIZE)
print("All data checks passed ✅")

# ---------- 4) Save a grid of augmented training images ----------
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
n = min(12, len(x))
fig, ax = plt.subplots(2, 6, figsize=(14, 5), layout="constrained")
for i, a in enumerate(ax.flat):
    if i >= n:
        a.axis("off"); continue
    img = (x[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    a.imshow(img); a.set_title(names[int(y[i])], fontsize=9); a.axis("off")
fig.suptitle(f"Augmented training batch (fold 0, {SIZE}x{SIZE}, letterboxed)", fontweight="bold")
out = P["out_dir"] / "busi_augmented_batch.png"
plt.savefig(out, dpi=100, bbox_inches="tight")
print("Saved:", out)

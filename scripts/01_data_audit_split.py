# =============================================================
# STEP 1 — BUSI data audit + near-duplicate detection + leak-free split
# Run from repo root:  python scripts/01_data_audit_split.py
# Paths auto-detected via src.utils.get_paths() (Kaggle / local / SageMaker)
# =============================================================
import os, glob, hashlib, itertools, json
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from sklearn.model_selection import StratifiedGroupKFold

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import get_paths, seed_everything

PATHS = get_paths()
ROOT = str(PATHS["data_root"])
OUT = str(PATHS["out_dir"])
seed_everything(42)
CLASSES = ["benign", "malignant", "normal"]
PHASH_THRESH = 4   # Hamming distance <= 4  => near-duplicate (visually verify karein)
SEED = 42

# ---------- 1) Index images (mask files exclude) ----------
rows = []
for label in CLASSES:
    for p in sorted(glob.glob(os.path.join(ROOT, label, "*.png"))):
        name = os.path.basename(p)
        if "_mask" in name:
            continue
        img = Image.open(p)
        with open(p, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        masks = sorted(glob.glob(p[:-4] + "_mask*.png"))
        rows.append(dict(
            path=p, fname=name, label=label,
            w=img.size[0], h=img.size[1], img_mode=img.mode,
            n_masks=len(masks), mask_paths="|".join(masks),
            md5=md5, phash=str(imagehash.phash(img.convert("L"))),
        ))
df = pd.DataFrame(rows)

print("Class counts:\n", df["label"].value_counts(), "\nTotal:", len(df))
print("Image modes:", df["img_mode"].value_counts().to_dict())
print("Images with >1 mask (multi-lesion):", (df["n_masks"] > 1).sum())
print("Images with 0 masks:", (df["n_masks"] == 0).sum())
print("Size range  W:", df.w.min(), "-", df.w.max(), " H:", df.h.min(), "-", df.h.max())

# ---------- 2) Exact + near-duplicate detection (union-find grouping) ----------
hashes = [imagehash.hex_to_hash(h) for h in df["phash"]]
parent = list(range(len(df)))

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

names, labels = df["fname"].tolist(), df["label"].tolist()
pairs = []
for i, j in itertools.combinations(range(len(df)), 2):
    d = hashes[i] - hashes[j]
    if d <= PHASH_THRESH:
        pairs.append((names[i], names[j], labels[i], labels[j], d))
        parent[find(i)] = find(j)

df["group"] = [find(i) for i in range(len(df))]
dups = pd.DataFrame(pairs, columns=["img_a", "img_b", "label_a", "label_b", "dist"])
print("\nExact duplicates (md5):", df["md5"].duplicated().sum())
print("Near-duplicate pairs:", len(dups))
print("Cross-label near-dup pairs (!!):", (dups.label_a != dups.label_b).sum())
print("Unique groups:", df["group"].nunique(), "of", len(df), "images")
dups.to_csv(f"{OUT}/busi_near_duplicates.csv", index=False)

# ---------- 3) Conflicting-label groups (same image, different labels) ----------
# Agar ek group mein 1 se zyada label hain, to us image ka sahi label maloom nahi.
# Aise poore group ko experiments se nikaal dete hain (DROP_CONFLICTING = False se ablation).
DROP_CONFLICTING = True
df["conflict"] = df.groupby("group")["label"].transform("nunique") > 1
conf = df[df["conflict"]].sort_values(["group", "label", "fname"])
print(f"\nConflicting-label groups: {conf['group'].nunique()}  ({len(conf)} images)")
if len(conf):
    print(conf[["group", "fname", "label"]].to_string(index=False))
conf[["group", "fname", "label", "path"]].to_csv(f"{OUT}/busi_conflicting_labels.csv", index=False)

# ---------- 4) Group-aware stratified split: ~1/6 test + 5 CV folds ----------
df["split"] = "excluded"
df["fold"] = -1
use = df[~df["conflict"]] if DROP_CONFLICTING else df

sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
test_pos = next(sgkf.split(use, use["label"], use["group"]))[1]
df.loc[use.index, "split"] = "trainval"
df.loc[use.index[test_pos], "split"] = "test"

tv = df[df["split"] == "trainval"]
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for k, (_, va) in enumerate(cv.split(tv, tv["label"], tv["group"])):
    df.loc[tv.index[va], "fold"] = k

# ---------- 5) Sanity checks ----------
test_g = set(df.loc[df.split == "test", "group"])
tv_g = set(df.loc[df.split == "trainval", "group"])
assert test_g.isdisjoint(tv_g), "Leakage: group test aur trainval dono mein!"
assert (df.groupby("group")["fold"].nunique() == 1).all(), "Leakage: group 2 folds mein bata!"
assert (df.loc[df.split == "trainval", "fold"] >= 0).all(), "Kuch trainval images ko fold nahi mila"
if DROP_CONFLICTING:
    assert not df.loc[df.split != "excluded", "conflict"].any(), "Conflicting image split mein reh gayi!"

print("\n", pd.crosstab(df["split"], df["label"]))
print("\n", pd.crosstab(df.loc[df.split == "trainval", "fold"], df.loc[df.split == "trainval", "label"]))

# ---------- 6) Save ----------
df.to_csv(f"{OUT}/busi_splits.csv", index=False)
summary = {
    "total_images": int(len(df)),
    "class_counts": df["label"].value_counts().to_dict(),
    "multi_lesion_images": int((df["n_masks"] > 1).sum()),
    "exact_duplicates_md5": int(df["md5"].duplicated().sum()),
    "phash_threshold": PHASH_THRESH,
    "near_duplicate_pairs": int(len(dups)),
    "cross_label_pairs": int((dups.label_a != dups.label_b).sum()),
    "unique_groups": int(df["group"].nunique()),
    "group_size_counts": {int(k): int(v) for k, v in df.groupby("group").size().value_counts().sort_index().items()},
    "conflicting_groups": int(conf["group"].nunique()),
    "conflicting_images_excluded": int(len(conf)) if DROP_CONFLICTING else 0,
    "split_counts": {s: df.loc[df.split == s, "label"].value_counts().to_dict() for s in ["trainval", "test", "excluded"]},
}
with open(f"{OUT}/busi_audit_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved: busi_splits.csv, busi_near_duplicates.csv, busi_conflicting_labels.csv, busi_audit_summary.json")

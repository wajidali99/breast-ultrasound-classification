# =============================================================
# STEP 1 — BUSI data audit + near-duplicate detection + leak-free split
# Kaggle: Add Data -> "breast-ultrasound-images-dataset" (aryashah2k)
# Internet ON karein, phir pehle cell mein:  !pip install -q imagehash
# =============================================================
import os, glob, hashlib, itertools
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from sklearn.model_selection import StratifiedGroupKFold

ROOT = "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"
OUT = "/kaggle/working"
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

# ---------- 3) Group-aware stratified split: ~1/6 test + 5 CV folds ----------
sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
test_idx = next(sgkf.split(df, df["label"], df["group"]))[1]
df["split"] = "trainval"
df.loc[test_idx, "split"] = "test"

df["fold"] = -1
tv = df[df["split"] == "trainval"]
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for k, (_, va) in enumerate(cv.split(tv, tv["label"], tv["group"])):
    df.loc[tv.index[va], "fold"] = k

# Leakage sanity check: koi group test aur trainval dono mein na ho
assert set(df[df.split == "test"].group).isdisjoint(set(df[df.split != "test"].group))

print("\n", pd.crosstab(df["split"], df["label"]))
print("\n", pd.crosstab(df["fold"], df["label"]))
df.to_csv(f"{OUT}/busi_splits.csv", index=False)
print("\nSaved: busi_splits.csv, busi_near_duplicates.csv")

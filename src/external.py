"""Step 8 — locate and index an external breast-ultrasound dataset (BUS-BRA) robustly."""
import os
import re
from pathlib import Path

import pandas as pd

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_label_csv(root: Path):
    """Return (csv_path, dataframe) for the first CSV that has a pathology/label column."""
    for csv in sorted(root.rglob("*.csv")):
        try:
            d = pd.read_csv(csv)
        except Exception:
            continue
        if any(re.search(r"patholog", c, re.I) for c in d.columns):
            return csv, d
    raise FileNotFoundError(f"No CSV with a 'Pathology' column under {root}")


def index_external(root: Path) -> pd.DataFrame:
    """Build a table: path, y (1 = malignant), label, patient, plus any metadata columns."""
    csv, meta = find_label_csv(root)
    pcol = next(c for c in meta.columns if re.search(r"patholog", c, re.I))
    idcol = next((c for c in meta.columns if re.fullmatch(r"(image_?)?id", c, re.I)), meta.columns[0])
    # all non-mask images under root, keyed by file stem
    imgs = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and "mask" not in p.name.lower() and "mask" not in p.parent.name.lower():
            imgs[p.stem] = p
    rows, missing = [], []
    for _, r in meta.iterrows():
        key = Path(str(r[idcol])).stem
        p = imgs.get(key)
        if p is None:
            missing.append(key); continue
        lab = str(r[pcol]).strip().lower()
        if lab not in {"benign", "malignant"}:
            continue
        num = re.search(r"(\d+)", key)
        rows.append({"path": str(p), "fname": p.name, "image_id": key, "label": lab,
                     "y": int(lab == "malignant"), "patient": num.group(1) if num else key,
                     **{f"meta_{c}": r[c] for c in meta.columns if c not in (idcol, pcol)}})
    df = pd.DataFrame(rows)
    info = {"csv": str(csv), "id_column": idcol, "pathology_column": pcol,
            "csv_rows": len(meta), "matched": len(df), "missing": len(missing),
            "missing_examples": missing[:5], "columns": list(meta.columns)}
    return df, info


def find_external_root(base: str = "/kaggle/input") -> Path:
    env = os.environ.get("EXT_ROOT")
    if env:
        return Path(env)
    base = Path(base)
    for csv in sorted(base.rglob("*.csv")):
        try:
            cols = pd.read_csv(csv, nrows=1).columns
        except Exception:
            continue
        if any(re.search(r"patholog", c, re.I) for c in cols) and "busi" not in str(csv).lower():
            # dataset root = first folder below the csv that also contains images
            for parent in [csv.parent, *csv.parents]:
                if parent == base:
                    break
                if any(parent.rglob("*.png")):
                    return parent
    raise FileNotFoundError("External dataset not found under /kaggle/input — add BUS-BRA via 'Add Input'.")

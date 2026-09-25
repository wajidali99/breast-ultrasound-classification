"""Step 2 — BUSI dataset, preprocessing and data loaders.

Flow:  busi_splits.csv  ->  select rows (task/split/fold)  ->  BUSIDataset  ->  DataLoader
Image path is rebuilt as  data_root / label / fname  (never taken from the CSV),
so the same code works on Kaggle, a laptop, or AWS.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils import seed_worker

# Label -> number. Binary task drops 'normal' (no lesion to classify).
CLASS_MAPS = {
    "binary": {"benign": 0, "malignant": 1},
    "multiclass": {"normal": 0, "benign": 1, "malignant": 2},
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- row selection
def select_rows(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Keep usable rows for this task and add the numeric label column 'y'."""
    cmap = CLASS_MAPS[task]
    out = df[(df["split"] != "excluded") & (df["label"].isin(cmap))].copy()
    out["y"] = out["label"].map(cmap).astype(int)
    return out.reset_index(drop=True)


def fold_split(df: pd.DataFrame, task: str, val_fold: int):
    """Train = the other 4 folds, Val = val_fold. Test set is never touched here."""
    tv = select_rows(df[df["split"] == "trainval"], task)
    train = tv[tv["fold"] != val_fold].reset_index(drop=True)
    val = tv[tv["fold"] == val_fold].reset_index(drop=True)
    assert len(train) and len(val), f"Empty train/val for fold {val_fold}"
    assert set(train["group"]).isdisjoint(val["group"]), "Leakage: group in both train and val!"
    return train, val


def test_split(df: pd.DataFrame, task: str) -> pd.DataFrame:
    return select_rows(df[df["split"] == "test"], task)


def class_weights(frame: pd.DataFrame, n_classes: int) -> torch.Tensor:
    """Rare class gets a bigger weight: w_c = N / (n_classes * count_c)."""
    counts = np.bincount(frame["y"].to_numpy(), minlength=n_classes).astype(float)
    assert (counts > 0).all(), f"Some class has zero samples: {counts}"
    return torch.tensor(len(frame) / (n_classes * counts), dtype=torch.float32)


# ---------------------------------------------------------------- preprocessing
def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Resize keeping the aspect ratio, then pad with black to size x size.
    (Stretching would distort lesion shape, which matters for benign vs malignant.)"""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new(img.mode, (size, size), 0)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def build_transforms(train: bool):
    """Ultrasound-safe augmentation: horizontal flip yes, vertical flip NO
    (skin is always at the top of an ultrasound image)."""
    from torchvision import transforms as T
    tail = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose(tail)
    aug = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    return T.Compose(aug + tail)


# ---------------------------------------------------------------- dataset
class BUSIDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root, img_size: int = 224,
                 transform=None, cache: bool = True):
        self.frame = frame.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.transform = transform
        # 762 small images fit easily in RAM -> load once, faster epochs
        self._cache = [self._load(i) for i in range(len(self.frame))] if cache else None

    def path(self, i: int) -> Path:
        r = self.frame.iloc[i]
        return self.data_root / r["label"] / r["fname"]

    def _load(self, i: int) -> Image.Image:
        img = Image.open(self.path(i)).convert("L")          # grayscale
        return letterbox(img, self.img_size).convert("RGB")  # 3 same channels for ImageNet models

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i: int):
        img = self._cache[i] if self._cache is not None else self._load(i)
        x = self.transform(img) if self.transform else img
        return x, int(self.frame.at[i, "y"])


def make_loaders(df: pd.DataFrame, data_root, task: str, val_fold: int,
                 img_size: int = 224, batch_size: int = 32, num_workers: int = 2, seed: int = 42):
    train_df, val_df = fold_split(df, task, val_fold)
    train_ds = BUSIDataset(train_df, data_root, img_size, build_transforms(True))
    val_ds = BUSIDataset(val_df, data_root, img_size, build_transforms(False))
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          worker_init_fn=seed_worker, generator=g, drop_last=False, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    return train_dl, val_dl, train_df, val_df

"""v2 — mask-guided training (see docs/v2_preregistration.md).

- SegDataset: returns (image, label, lesion mask) with the SAME geometric augmentation on image and mask
- aspect_crop: random narrow/tall crop that always keeps the whole lesion
- DenseNetSeg: DenseNet121 classifier + 1x1-conv lesion head on the final 7x7 feature map
- train_fold_v2: cross-entropy + lambda * BCE(mask head, 7x7 mask)
"""
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.data import letterbox, fold_split, class_weights, CLASS_MAPS, IMAGENET_MEAN, IMAGENET_STD
from src.metrics import binary_metrics
from src.utils import seed_worker


# ------------------------------------------------------------------ masks + crop (numpy/PIL only)
def raw_union_mask(data_root, label: str, fname: str) -> Image.Image:
    stem = Path(fname).stem
    paths = sorted(Path(data_root, label).glob(f"{stem}_mask*.png"))
    assert paths, f"No mask for {label}/{fname}"
    m = None
    for p in paths:
        a = np.array(Image.open(p).convert("L")) > 127
        m = a if m is None else (m | a)
    return Image.fromarray(m.astype(np.uint8) * 255)


def aspect_crop(img: Image.Image, mask: Image.Image, rng: random.Random,
                w_frac=(0.5, 1.0), h_frac=(0.8, 1.0)):
    """Random crop to a narrower/shorter window that always contains the whole lesion."""
    W, H = img.size
    m = np.array(mask) > 127
    ys, xs = np.where(m)
    if len(xs) == 0:
        return img, mask
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cw = max(int(round(W * rng.uniform(*w_frac))), bx1 - bx0)
    ch = max(int(round(H * rng.uniform(*h_frac))), by1 - by0)
    cw, ch = min(cw, W), min(ch, H)
    x_lo, x_hi = max(0, bx1 - cw), min(bx0, W - cw)
    y_lo, y_hi = max(0, by1 - ch), min(by0, H - ch)
    x0 = rng.randint(x_lo, x_hi) if x_hi >= x_lo else 0
    y0 = rng.randint(y_lo, y_hi) if y_hi >= y_lo else 0
    box = (x0, y0, x0 + cw, y0 + ch)
    return img.crop(box), mask.crop(box)


# ------------------------------------------------------------------ dataset
class SegDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root, img_size=224, train=False,
                 augment=True, aspect_aug=False, aspect_p=0.5):
        self.frame = frame.reset_index(drop=True)
        self.size, self.train, self.augment = img_size, train, augment
        self.aspect_aug, self.aspect_p = aspect_aug, aspect_p
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2)
        self.cache = []
        for _, r in self.frame.iterrows():
            img = Image.open(Path(data_root, r["label"], r["fname"])).convert("L")
            self.cache.append((img, raw_union_mask(data_root, r["label"], r["fname"])))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        img, mask = self.cache[i]
        if self.train and self.aspect_aug:
            rng = random.Random(torch.randint(0, 2**31 - 1, (1,)).item())
            if rng.random() < self.aspect_p:
                img, mask = aspect_crop(img, mask, rng)
        img = letterbox(img, self.size).convert("RGB")
        mask = letterbox(mask, self.size, resample=Image.NEAREST)
        x = TF.to_tensor(img)
        m = torch.from_numpy((np.array(mask) > 127).astype(np.float32))[None]
        if self.train and self.augment:
            if torch.rand(1).item() < 0.5:                              # horizontal flip only
                x, m = TF.hflip(x), TF.hflip(m)
            angle, translate, scale, shear = T.RandomAffine.get_params(
                [-10.0, 10.0], [0.05, 0.05], [0.9, 1.1], None, [self.size, self.size])
            x = TF.affine(x, angle, list(translate), scale, list(shear), interpolation=InterpolationMode.BILINEAR)
            m = TF.affine(m, angle, list(translate), scale, list(shear), interpolation=InterpolationMode.NEAREST)
            x = self.jitter(x)
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        return x, int(self.frame.at[i, "y"]), m


# ------------------------------------------------------------------ model
class DenseNetSeg(nn.Module):
    """forward(x) -> class logits (so evaluation code, Grad-CAM and demos work unchanged);
    forward_both(x) -> (class logits, 7x7 lesion logits) for training."""
    def __init__(self, pretrained=True, n_classes=2, dropout=0.2):
        super().__init__()
        import timm
        self.backbone = timm.create_model("densenet121", pretrained=pretrained,
                                          num_classes=n_classes, drop_rate=dropout)
        self.seg_head = nn.Conv2d(self.backbone.num_features, 1, kernel_size=1)

    def forward_both(self, x):
        f = self.backbone.forward_features(x)
        return self.backbone.forward_head(f), self.seg_head(f)

    def forward(self, x):
        return self.forward_both(x)[0]


# ------------------------------------------------------------------ training
def _evaluate(model, loader, device):
    model.eval(); probs, ys = [], []
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
        for x, y, _ in loader:
            probs.append(torch.softmax(model(x.to(device)).float(), 1)[:, 1].cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train_fold_v2(cfg, df, data_root, val_fold, device, lam=0.5, aspect_aug=False, log=print):
    task, tc = cfg["task"], cfg["train"]
    n_cls = len(CLASS_MAPS[task])
    tr_df, va_df = fold_split(df, task, val_fold)
    size, nw = cfg["data"]["img_size"], cfg["data"]["num_workers"]
    tr_ds = SegDataset(tr_df, data_root, size, train=True, augment=True, aspect_aug=aspect_aug)
    va_ds = SegDataset(va_df, data_root, size, train=False)
    g = torch.Generator().manual_seed(cfg["seed"])
    tr_dl = DataLoader(tr_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=nw,
                       worker_init_fn=seed_worker, generator=g, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=nw, pin_memory=True)

    model = DenseNetSeg(cfg["model"]["pretrained"], n_cls, cfg["model"]["dropout"]).to(device)
    w = class_weights(tr_df, n_cls).to(device) if tc["class_weights"] else None
    ce, bce = nn.CrossEntropyLoss(weight=w), nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc, best_state, best_epoch, bad, hist = -1.0, None, -1, 0, []
    for epoch in range(1, tc["epochs"] + 1):
        model.train(); t0 = time.time(); tot = seg_tot = n = 0
        for x, y, m in tr_dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits, seg = model.forward_both(x)
                target = F.adaptive_avg_pool2d(m, seg.shape[-2:])        # soft 7x7 lesion map
                l_cls, l_seg = ce(logits, y), bce(seg.float(), target.float())
                loss = l_cls + lam * l_seg
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            assert torch.isfinite(loss), "Loss NaN/Inf"
            tot += l_cls.item() * len(y); seg_tot += l_seg.item() * len(y); n += len(y)
        sched.step()
        vy, vp = _evaluate(model, va_dl, device)
        auc = binary_metrics(vy, vp)["auc"]
        hist.append({"epoch": epoch, "train_cls_loss": tot / n, "train_seg_loss": seg_tot / n,
                     "val_auc": auc, "sec": time.time() - t0})
        if auc > best_auc:
            best_auc, best_epoch, bad = auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict()); best_val = (vy, vp)
        else:
            bad += 1
        log(f"ep {epoch:02d} cls {tot/n:.3f} seg {seg_tot/n:.3f} | val AUC {auc:.3f}{'  *' if auc == best_auc else ''}")
        if bad >= tc["early_stopping_patience"]:
            break
    final = binary_metrics(*best_val)
    final.update({"best_epoch": best_epoch, "epochs_run": len(hist), "val_fold": val_fold})
    return best_state, pd.DataFrame(hist), final, va_df.assign(prob_malignant=best_val[1])

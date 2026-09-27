"""Step 7 — Grad-CAM and lesion-localisation metrics.

Grad-CAM target = the MALIGNANT logit for every image: the heatmap shows which regions
push the model's malignancy score up, i.e. the score that is compared with the threshold.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from src.data import letterbox


# ---------------------------------------------------------------- masks (numpy only)
def load_lesion_mask(data_root, label: str, fname: str, size: int) -> np.ndarray:
    """Union of all lesion masks for an image, letterboxed exactly like the image."""
    stem = Path(fname).stem
    paths = sorted(Path(data_root, label).glob(f"{stem}_mask*.png"))
    assert paths, f"No mask for {label}/{fname}"
    union = None
    for p in paths:
        m = np.array(Image.open(p).convert("L")) > 127
        union = m if union is None else (union | m)
    lb = letterbox(Image.fromarray(union.astype(np.uint8) * 255), size, resample=Image.NEAREST)
    return np.array(lb) > 127


def content_mask(data_root, label: str, fname: str, size: int) -> np.ndarray:
    """True where the letterboxed image has real pixels (not black padding)."""
    w, h = Image.open(Path(data_root, label, fname)).size
    ones = Image.new("L", (w, h), 255)
    return np.array(letterbox(ones, size, resample=Image.NEAREST)) > 127


def localisation_metrics(cam: np.ndarray, lesion: np.ndarray, content: np.ndarray,
                         tolerance_px: int = 7) -> dict:
    """cam: (H, W) in [0, 1]. Returns
    - pointing_hit: the hottest CAM pixel lies inside the lesion (dilated by tolerance_px)
    - energy_in_lesion: share of CAM energy inside the lesion (within the real image area)
    - lesion_area_frac: lesion share of the real image area (= energy expected by chance)
    - energy_ratio: energy_in_lesion / lesion_area_frac  (>1 = more focused on lesion than chance)
    """
    assert cam.shape == lesion.shape == content.shape
    assert lesion.any(), "Empty lesion mask"
    cam = np.where(content, cam, 0.0)
    target = binary_dilation(lesion, iterations=tolerance_px) if tolerance_px else lesion
    peak = np.unravel_index(np.argmax(cam), cam.shape)
    total = cam.sum()
    e_in = float(cam[lesion & content].sum() / total) if total > 0 else 0.0
    area = float((lesion & content).sum() / content.sum())
    return {"pointing_hit": bool(target[peak]), "energy_in_lesion": e_in,
            "lesion_area_frac": area, "energy_ratio": e_in / area if area > 0 else float("nan")}


# ---------------------------------------------------------------- Grad-CAM (torch)
def target_layer(model, arch: str):
    if arch.startswith("densenet"):
        return model.features
    if arch.startswith("resnet"):
        return model.layer4
    if arch.startswith("efficientnet"):
        return model.conv_head
    if arch.startswith("convnext"):
        return model.stages[-1]
    raise ValueError(f"No Grad-CAM layer defined for {arch}")


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self._h = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx: int = 1):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(False)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            logits[:, class_idx].sum().backward()
            w = self.grads.mean(dim=(2, 3), keepdim=True)           # importance of each channel
            cam = F.relu((w * self.acts).sum(1, keepdim=True))       # (B, 1, h, w)
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            cam = cam / cam.flatten(1).max(1).values.clamp_min(1e-8)[:, None, None]
        prob = torch.softmax(logits.detach().float(), 1)[:, 1]
        return cam.detach().cpu().numpy(), prob.cpu().numpy()

    def remove(self):
        self._h.remove()

# =============================================================
# STEP 10a — Export the v2b fold models to the Hugging Face Hub
# Run on Kaggle:  python scripts/10_export_models_to_hf.py --hf_user Wajiddev99
# Needs a Kaggle secret named HF_TOKEN (Hugging Face token with WRITE access).
# If the v2b checkpoints are missing (session restarted) they are retrained (~8 min, deterministic).
# =============================================================
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import get_paths, load_config, get_device, seed_everything
from src.data import fold_split
from src.metrics import binary_metrics, threshold_for_sensitivity
from src.v2 import DenseNetSeg, SegDataset, train_fold_v2

ap = argparse.ArgumentParser()
ap.add_argument("--hf_user", required=True)
ap.add_argument("--repo", default="busi-densenet121-v2b")
args = ap.parse_args()

EXPECTED_VAL_AUC = [0.970, 0.928, 0.978, 0.932, 0.927]      # v2b folds from the v2 run
P, device = get_paths(), get_device()
cfg = load_config(str(ROOT / "configs" / "base.yaml"), {"model.arch": "densenet121"})
df = pd.read_csv(P["out_dir"] / cfg["data"]["splits_csv"])
ck_dir = P["out_dir"] / "v2" / "checkpoints"; ck_dir.mkdir(parents=True, exist_ok=True)
export = P["out_dir"] / "hf_export"; export.mkdir(exist_ok=True)

# ---------- 1) Make sure all 5 v2b checkpoints exist ----------
for fold in range(5):
    ck = ck_dir / f"v2b_fold{fold}.pt"
    if not ck.exists():
        print(f"Retraining v2b fold {fold} (checkpoint missing) ...", flush=True)
        seed_everything(cfg["seed"])
        state, _, final, _ = train_fold_v2(cfg, df, P["data_root"], fold, device, lam=0.5,
                                           aspect_aug=True, log=lambda s: None)
        torch.save({k: v.cpu() for k, v in state.items()}, ck)

# ---------- 2) Verify each checkpoint on its own validation fold + rebuild the threshold ----------
oof = []
for fold in range(5):
    model = DenseNetSeg(pretrained=False, n_classes=2, dropout=cfg["model"]["dropout"])
    model.load_state_dict(torch.load(ck_dir / f"v2b_fold{fold}.pt", map_location="cpu"))
    model.to(device).eval()
    _, va = fold_split(df, cfg["task"], fold)
    dl = DataLoader(SegDataset(va, P["data_root"], cfg["data"]["img_size"], train=False), batch_size=32)
    ps, ys = [], []
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
        for x, y, _ in dl:
            ps.append(torch.softmax(model(x.to(device)).float(), 1)[:, 1].cpu().numpy()); ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    auc = binary_metrics(y, p)["auc"]
    ok = "✓" if abs(auc - EXPECTED_VAL_AUC[fold]) < 0.005 else "⚠ differs from v2 run"
    print(f"fold {fold}: val AUC {auc:.3f} (v2 run: {EXPECTED_VAL_AUC[fold]}) {ok}")
    oof.append(pd.DataFrame({"y": y, "p": p}))
    torch.save(model.state_dict(), export / f"v2b_fold{fold}.pt")
oof = pd.concat(oof)
thr = threshold_for_sensitivity(oof["y"], oof["p"], 0.90)
print(f"Frozen threshold (sens ≥ 0.90 on OOF): {thr:.3f}   pooled OOF AUC {binary_metrics(oof['y'], oof['p'])['auc']:.3f}")

# ---------- 3) Config + model card ----------
json.dump({"architecture": "DenseNet121 + 1x1 lesion head (timm densenet121)", "n_models": 5,
           "input_size": 224, "preprocessing": "grayscale -> letterbox 224x224 (black padding) -> 3 channels -> ImageNet mean/std",
           "classes": ["benign", "malignant"], "threshold_per_model": thr,
           "decision_rule": "each fold model votes malignant if p >= threshold; label = majority of 5",
           "training": "BUSI (benign vs malignant), 5-fold CV, leakage-aware group split, mask-guided loss (lambda=0.5), aspect-ratio crop augmentation"},
          open(export / "config.json", "w"), indent=2)
card = f"""---
license: mit
tags: [medical-imaging, ultrasound, breast-cancer, image-classification, pytorch, timm]
---
# Breast ultrasound classifier — DenseNet121 v2b (research only)

> ⚠️ **Not a medical device.** Research model trained on public datasets; must not be used for diagnosis.

Five DenseNet121 fold models (v2b: mask-guided training + aspect-ratio crop augmentation) for **benign vs malignant** classification of breast ultrasound lesions.
Code, data audit and full evaluation: https://github.com/wajidali99/breast-ultrasound-classification

## Decision rule
Each model outputs P(malignant). A model votes *malignant* if P ≥ **{thr:.3f}** (threshold frozen on BUSI out-of-fold predictions for ≥90% sensitivity). The label is the majority of the 5 votes.

## Performance (ensemble AUC)
| Data | AUC |
|---|---|
| BUSI 5-fold CV (pooled out-of-fold) | 0.940 |
| BUSI locked test set (105 images) | 0.970 |
| BUS-BRA external (1,875 images, Brazil) | 0.776 (95% CI 0.749–0.804) |

## Limitations
- Large performance drop on an external hospital (AUC 0.97 → 0.78): results do not transfer reliably across sites.
- At the BUSI threshold, external sensitivity was 0.70 (target 0.90); site-specific calibration would be required.
- Trained on ~530 BUSI images from a single centre; images without a lesion (normal) are out of scope.
"""
(export / "README.md").write_text(card)

# ---------- 4) Upload ----------
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    import os; token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not found — add it in Kaggle: Add-ons → Secrets"
from huggingface_hub import HfApi
api = HfApi(token=token)
repo_id = f"{args.hf_user}/{args.repo}"
api.create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(export), repo_id=repo_id, repo_type="model",
                  commit_message="Upload v2b fold models, config and model card")
print(f"\nUploaded to https://huggingface.co/{repo_id}")

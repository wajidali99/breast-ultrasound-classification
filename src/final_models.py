"""Rebuild the Step 6 fold models if they are missing (training is fully reproducible)."""
import json
from pathlib import Path

import pandas as pd
import torch

from src.utils import seed_everything
from src.train import train_fold
from src.metrics import threshold_for_sensitivity

ARCH, TARGET_SENS = "densenet121", 0.90


def ensure_fold_models(cfg, df, data_root, step6_dir: Path, device, log=print):
    """Returns (list of checkpoint paths, frozen threshold). Uses trainval data only."""
    ck_dir = step6_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    oof_parts = []
    for fold in range(5):
        ck, pr = ck_dir / f"{ARCH}_fold{fold}.pt", step6_dir / f"oof_fold{fold}.csv"
        if ck.exists() and pr.exists():
            oof_parts.append(pd.read_csv(pr)); continue
        log(f"Rebuilding fold {fold} model (checkpoint missing) ...")
        seed_everything(cfg["seed"])
        state, _, final, val_pred = train_fold(cfg, df, data_root, fold, device, log=lambda s: None)
        torch.save({k: v.cpu() for k, v in state.items()}, ck)
        vp = val_pred[["fname", "label", "group", "y", "prob_malignant"]].assign(fold=fold)
        vp.to_csv(pr, index=False); oof_parts.append(vp)
        log(f"  fold {fold}: val AUC {final['auc']:.3f}")
    thr_file = step6_dir / "frozen_threshold.json"
    if thr_file.exists():
        thr = json.load(open(thr_file))["threshold"]
    else:
        oof = pd.concat(oof_parts, ignore_index=True)
        thr = threshold_for_sensitivity(oof["y"], oof["prob_malignant"], TARGET_SENS)
        json.dump({"threshold": thr, "rule": f"sensitivity >= {TARGET_SENS} on OOF", "arch": ARCH},
                  open(thr_file, "w"), indent=2)
    return [ck_dir / f"{ARCH}_fold{k}.pt" for k in range(5)], float(thr)

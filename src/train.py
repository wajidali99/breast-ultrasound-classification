"""Step 3 — train + evaluate one CV fold."""
import time
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import CLASS_MAPS, make_loaders, class_weights
from src.model import build_model, count_params
from src.metrics import binary_metrics


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over the loader. Trains if optimizer is given, else evaluates."""
    train = optimizer is not None
    model.train(train)
    total_loss, n, probs, labels = 0.0, 0, [], []
    use_amp = device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        assert torch.isfinite(loss), "Loss became NaN/Inf — training unstable"
        total_loss += loss.item() * len(y)
        n += len(y)
        probs.append(torch.softmax(logits.float(), 1)[:, 1].detach().cpu().numpy())
        labels.append(y.cpu().numpy())
    return total_loss / n, np.concatenate(labels), np.concatenate(probs)


def train_fold(cfg: dict, df: pd.DataFrame, data_root, val_fold: int, device, log=print):
    task = cfg["task"]
    assert task == "binary", "Step 3 baseline is binary only"
    n_cls = len(CLASS_MAPS[task])
    tc = cfg["train"]

    train_dl, val_dl, train_df, val_df = make_loaders(
        df, data_root, task, val_fold, img_size=cfg["data"]["img_size"],
        batch_size=tc["batch_size"], num_workers=cfg["data"]["num_workers"], seed=cfg["seed"],
        augment=cfg["data"].get("augment", True),
        add_conflicting_to_train=cfg["data"].get("add_conflicting_to_train", False))

    model = build_model(cfg["model"]["arch"], cfg["model"]["pretrained"], n_cls,
                        cfg["model"]["dropout"]).to(device)
    log(f"Model {cfg['model']['arch']}: {count_params(model)/1e6:.1f}M params | "
        f"train={len(train_df)} val={len(val_df)}")

    w = class_weights(train_df, n_cls).to(device) if tc["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best_auc, best_state, best_epoch, bad_epochs = [], -1.0, None, -1, 0
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_y, tr_p = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
        with torch.no_grad():
            va_loss, va_y, va_p = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        tr_m, va_m = binary_metrics(tr_y, tr_p), binary_metrics(va_y, va_p)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": tr_loss, "val_loss": va_loss,
               "train_auc": tr_m["auc"], "val_auc": va_m["auc"],
               "val_sens": va_m["sensitivity"], "val_spec": va_m["specificity"],
               "val_bal_acc": va_m["balanced_acc"], "sec": time.time() - t0}
        history.append(row)
        improved = va_m["auc"] > best_auc
        if improved:
            best_auc, best_epoch, bad_epochs = va_m["auc"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            best_val = {"y": va_y, "prob": va_p}
        else:
            bad_epochs += 1
        log(f"ep {epoch:02d} | train loss {tr_loss:.3f} auc {tr_m['auc']:.3f} | "
            f"val loss {va_loss:.3f} auc {va_m['auc']:.3f} sens {va_m['sensitivity']:.2f} "
            f"spec {va_m['specificity']:.2f} | {row['sec']:.0f}s{'  *best*' if improved else ''}")
        if bad_epochs >= tc["early_stopping_patience"]:
            log(f"Early stopping: val AUC {tc['early_stopping_patience']} epochs se behtar nahi hua.")
            break

    assert best_state is not None, "No epoch completed"
    final = binary_metrics(best_val["y"], best_val["prob"])
    final.update({"best_epoch": best_epoch, "epochs_run": len(history), "val_fold": val_fold,
                  "arch": cfg["model"]["arch"]})
    return best_state, pd.DataFrame(history), final, val_df.assign(prob_malignant=best_val["prob"])

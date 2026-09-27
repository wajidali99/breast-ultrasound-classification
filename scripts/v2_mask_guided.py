# =============================================================
# v2 — Mask-guided training (plan: docs/v2_preregistration.md, committed BEFORE running)
# Run:  python scripts/v2_mask_guided.py
# Needs BUS-BRA added as a Kaggle input (same as Step 8).
# =============================================================
import sys, json, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.utils import get_paths, load_config, get_device, seed_everything
from src.data import BUSIDataset, PathDataset, build_transforms, test_split
from src.model import build_model
from src.train import run_epoch
from src.metrics import binary_metrics, threshold_for_sensitivity
from src.evaluate import group_bootstrap, paired_auc_diff
from src.external import find_external_root, index_external
from src.final_models import ensure_fold_models, ARCH
from src.v2 import DenseNetSeg, train_fold_v2

LAMBDA = 0.5                                  # pre-registered, not tuned
VARIANTS = {"v2a": {"aspect_aug": False}, "v2b": {"aspect_aug": True}}
P, device = get_paths(), get_device()
cfg = load_config(str(ROOT / "configs" / "base.yaml"), {"model.arch": ARCH})
df = pd.read_csv(P["out_dir"] / cfg["data"]["splits_csv"])
out = P["out_dir"] / "v2"; (out / "checkpoints").mkdir(parents=True, exist_ok=True)
print(f"Device: {device} | λ={LAMBDA} | variants={list(VARIANTS)}")

# ---------- 1) 5-fold CV for each v2 variant (BUSI trainval only) ----------
oof, cv_rows = {}, []
for name, kw in VARIANTS.items():
    parts = []
    for fold in range(5):
        ck, pr = out / "checkpoints" / f"{name}_fold{fold}.pt", out / f"{name}_oof_fold{fold}.csv"
        if ck.exists() and pr.exists():
            print(f"[skip] {name} fold {fold}"); parts.append(pd.read_csv(pr))
            cv_rows.append({"variant": name, "fold": fold, **json.load(open(out / f"{name}_fold{fold}.json"))}); continue
        seed_everything(cfg["seed"])
        state, hist, final, vp = train_fold_v2(cfg, df, P["data_root"], fold, device, lam=LAMBDA, log=lambda s: None, **kw)
        torch.save({k: v.cpu() for k, v in state.items()}, ck)
        vp = vp[["fname", "label", "group", "y", "prob_malignant"]].assign(fold=fold); vp.to_csv(pr, index=False)
        hist.to_csv(out / f"{name}_fold{fold}_history.csv", index=False)
        json.dump(final, open(out / f"{name}_fold{fold}.json", "w"), indent=2)
        parts.append(vp); cv_rows.append({"variant": name, "fold": fold, **final})
        print(f"{name} fold {fold}: best ep {final['best_epoch']}/{final['epochs_run']} | val AUC {final['auc']:.3f}", flush=True)
        del state; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    oof[name] = pd.concat(parts, ignore_index=True)

# v1 (Step 6 models) — rebuilt if missing; gives v1 OOF + threshold
v1_ckpts, v1_thr = ensure_fold_models(cfg, df, P["data_root"], P["out_dir"] / "step6", device)
oof["v1"] = pd.concat([pd.read_csv(P["out_dir"] / "step6" / f"oof_fold{k}.csv") for k in range(5)], ignore_index=True)
cv = pd.DataFrame(cv_rows); cv.to_csv(out / "v2_cv_per_fold.csv", index=False)

# ---------- 2) Pre-registered selection on BUSI OOF only + frozen thresholds ----------
pooled = {k: binary_metrics(v["y"], v["prob_malignant"])["auc"] for k, v in oof.items()}
thr = {"v1": v1_thr, **{k: threshold_for_sensitivity(oof[k]["y"], oof[k]["prob_malignant"], 0.90) for k in VARIANTS}}
selected = "v2b" if pooled["v2b"] - pooled["v2a"] >= 0.005 else "v2a"
print("\n=========== BUSI 5-fold CV (validation) ===========")
print(f"v1   pooled OOF AUC {pooled['v1']:.3f}  (Step 5 baseline)")
for k in VARIANTS:
    g = cv[cv.variant == k]
    print(f"{k:<4} pooled OOF AUC {pooled[k]:.3f}  mean fold AUC {g['auc'].mean():.3f} ± {g['auc'].std(ddof=1):.3f}  threshold {thr[k]:.3f}")
print(f"→ Selected by pre-registered rule: {selected}")

# ---------- 3) Predict BUSI test + BUS-BRA with all three model sets ----------
def load_models(kind):
    ms = []
    for k in range(5):
        if kind == "v1":
            m = build_model(ARCH, pretrained=False, n_classes=2, dropout=cfg["model"]["dropout"])
            m.load_state_dict(torch.load(v1_ckpts[k], map_location="cpu"))
        else:
            m = DenseNetSeg(pretrained=False, n_classes=2, dropout=cfg["model"]["dropout"])
            m.load_state_dict(torch.load(out / "checkpoints" / f"{kind}_fold{k}.pt", map_location="cpu"))
        ms.append(m.to(device).eval())
    return ms

def predict(models, loader):
    ps = []
    for m in models:
        with torch.no_grad():
            _, y_, p_ = run_epoch(m, loader, nn.CrossEntropyLoss(), device)
        ps.append(p_)
    return y_, np.vstack(ps)

test_df = test_split(df, cfg["task"])
test_dl = DataLoader(BUSIDataset(test_df, P["data_root"], cfg["data"]["img_size"], build_transforms(False)),
                     batch_size=32, shuffle=False, num_workers=cfg["data"]["num_workers"])
ext_root = find_external_root(); ext, info = index_external(ext_root)
ext_dl = DataLoader(PathDataset(ext, ext_root, cfg["data"]["img_size"], build_transforms(False)),
                    batch_size=32, shuffle=False, num_workers=cfg["data"]["num_workers"])
print(f"\nBUS-BRA: {len(ext)} images, {ext['patient'].nunique()} patients")

res, probs_store = [], {}
for kind in ["v1", "v2a", "v2b"]:
    models = load_models(kind)
    for dname, dl, groups in [("BUSI test", test_dl, test_df["group"].to_numpy()),
                              ("BUS-BRA", ext_dl, ext["patient"].to_numpy())]:
        y, pr = predict(models, dl); probs_store[(kind, dname)] = (y, pr)
        per = [binary_metrics(y, p, thr[kind]) for p in pr]
        ci = group_bootstrap(y, pr, groups, thr[kind], n_boot=2000, seed=cfg["seed"])
        res.append({"model": kind, "dataset": dname, "threshold": thr[kind],
                    "auc_single_mean": np.mean([m["auc"] for m in per]), "auc_single_sd": np.std([m["auc"] for m in per], ddof=1),
                    "ensemble_auc": binary_metrics(y, pr.mean(0))["auc"],
                    "ensemble_auc_lo": ci["ensemble_auc_ci"][0], "ensemble_auc_hi": ci["ensemble_auc_ci"][1],
                    "sens_mean": np.mean([m["sensitivity"] for m in per]), "spec_mean": np.mean([m["specificity"] for m in per]),
                    "spec_min": np.min([m["specificity"] for m in per]), "spec_max": np.max([m["specificity"] for m in per])})
    del models; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
res = pd.DataFrame(res); res.to_csv(out / "v1_vs_v2_results.csv", index=False)

print("\n=========== v1 vs v2 (each fold model at its own frozen threshold; ensemble AUC with 95% CI) ===========")
for dname in ["BUSI test", "BUS-BRA"]:
    print(f"\n{dname}")
    for _, r in res[res.dataset == dname].iterrows():
        tag = "  ← selected v2" if r["model"] == selected else ""
        print(f"  {r['model']:<4} ens AUC {r['ensemble_auc']:.3f} ({r['ensemble_auc_lo']:.3f}–{r['ensemble_auc_hi']:.3f})  "
              f"single AUC {r['auc_single_mean']:.3f} ± {r['auc_single_sd']:.3f}  "
              f"sens {r['sens_mean']:.3f}  spec {r['spec_mean']:.3f} (range {r['spec_min']:.2f}–{r['spec_max']:.2f}){tag}")

# ---------- 4) Primary pre-registered test: selected v2 vs v1 on BUS-BRA (paired, by patient) ----------
cmp_rows = []
for dname, groups in [("BUS-BRA", ext["patient"].to_numpy()), ("BUSI test", test_df["group"].to_numpy())]:
    for kind in ["v2a", "v2b"]:
        y, p1 = probs_store[("v1", dname)]; _, p2 = probs_store[(kind, dname)]
        d = paired_auc_diff(y, p1.mean(0), p2.mean(0), groups, n_boot=2000, seed=cfg["seed"])
        cmp_rows.append({"dataset": dname, "comparison": f"{kind} - v1", **d, "primary": dname == "BUS-BRA" and kind == selected})
cmp = pd.DataFrame(cmp_rows); cmp.to_csv(out / "v2_paired_comparison.csv", index=False)
print("\n=========== Paired ΔAUC (ensemble), bootstrap 95% CI ===========")
for _, r in cmp.iterrows():
    verdict = "improvement" if r["ci"][0] > 0 else ("worse" if r["ci"][1] < 0 else "no clear difference")
    star = "  ← PRIMARY (pre-registered)" if r["primary"] else ""
    print(f"  {r['dataset']:<9} {r['comparison']:<9} ΔAUC {r['delta_auc']:+.3f}  CI {r['ci'][0]:+.3f} to {r['ci'][1]:+.3f}  → {verdict}{star}")

json.dump({"lambda": LAMBDA, "pooled_oof_auc": pooled, "thresholds": thr, "selected": selected,
           "results": res.to_dict("records"), "paired": cmp.to_dict("records")},
          open(out / "v2_summary.json", "w"), indent=2, default=str)

# ---------- 5) Figure ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
for a, dname in zip(ax, ["BUSI test", "BUS-BRA"]):
    for kind, ls in [("v1", "--"), ("v2a", "-"), ("v2b", "-")]:
        y, pr = probs_store[(kind, dname)]; f, t, _ = roc_curve(y, pr.mean(0))
        a.plot(f, t, ls=ls, lw=2 if kind == selected else 1.4,
               label=f"{kind} (AUC {binary_metrics(y, pr.mean(0))['auc']:.3f}){' *' if kind == selected else ''}")
    a.plot([0, 1], [0, 1], ls=":", c="gray"); a.grid(alpha=0.3); a.legend(fontsize=9, loc="lower right")
    a.set(title=f"{dname} — ensemble ROC", xlabel="1 - specificity", ylabel="sensitivity")
fig.suptitle("v1 vs v2 (mask-guided) — * = v2 variant selected on BUSI CV", fontweight="bold")
plt.savefig(out / "v1_vs_v2.png", dpi=100, bbox_inches="tight")
print(f"\nSaved to: {out}")

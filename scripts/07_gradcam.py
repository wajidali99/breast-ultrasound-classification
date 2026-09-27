# =============================================================
# STEP 7 — Explainability: where does the model look?
# Grad-CAM (malignant logit, averaged over the 5 fold models) on the TEST set,
# compared with the radiologists' lesion masks.
# Run:  python scripts/07_gradcam.py
# Needs Step 6 checkpoints; if the session restarted they are rebuilt automatically (~6 min).
# =============================================================
import sys, json, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from src.utils import get_paths, load_config, get_device, seed_everything
from src.data import BUSIDataset, build_transforms, test_split
from src.model import build_model
from src.explain import GradCAM, target_layer, load_lesion_mask, content_mask, localisation_metrics
from src.final_models import ensure_fold_models, ARCH

P, device = get_paths(), get_device()
cfg = load_config(str(ROOT / "configs" / "base.yaml"), {"model.arch": ARCH})
seed_everything(cfg["seed"])
df = pd.read_csv(P["out_dir"] / cfg["data"]["splits_csv"])
SIZE = cfg["data"]["img_size"]
out = P["out_dir"] / "step7"; out.mkdir(parents=True, exist_ok=True)

ckpts, thr = ensure_fold_models(cfg, df, P["data_root"], P["out_dir"] / "step6", device)
print(f"Device: {device} | {len(ckpts)} fold models | frozen threshold {thr:.3f}")

# ---------- 1) Grad-CAM on the test set, averaged over the 5 models ----------
test_df = test_split(df, cfg["task"])
ds = BUSIDataset(test_df, P["data_root"], SIZE, build_transforms(False))
dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
cam_sum = np.zeros((len(ds), SIZE, SIZE), dtype=np.float32)
probs = np.zeros((len(ckpts), len(ds)), dtype=np.float32)
for k, ck in enumerate(ckpts):
    model = build_model(ARCH, pretrained=False, n_classes=2, dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(torch.load(ck, map_location=device))
    gc_ = GradCAM(model, target_layer(model, ARCH))
    i0 = 0
    for x, _ in dl:
        cam, p = gc_(x.to(device), class_idx=1)
        cam_sum[i0:i0 + len(x)] += cam
        probs[k, i0:i0 + len(x)] = p
        i0 += len(x)
    gc_.remove(); del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"model {k}: Grad-CAM done")
cams = cam_sum / cam_sum.reshape(len(ds), -1).max(1).clip(1e-8)[:, None, None]
assert np.isfinite(cams).all() and cams.max() <= 1.0 + 1e-6

# ---------- 2) Compare with lesion masks ----------
y = test_df["y"].to_numpy()
rows = []
for i, r in test_df.iterrows():
    les = load_lesion_mask(P["data_root"], r["label"], r["fname"], SIZE)
    con = content_mask(P["data_root"], r["label"], r["fname"], SIZE)
    m = localisation_metrics(cams[i], les, con, tolerance_px=7)
    tol_area = float((binary_dilation(les, iterations=7) & con).sum() / con.sum())
    wrong = int(((probs[:, i] >= thr).astype(int) != y[i]).sum())
    rows.append({"fname": r["fname"], "label": r["label"], "y": int(y[i]),
                 "prob_ensemble": float(probs[:, i].mean()), "n_models_wrong": wrong,
                 "hard_case": wrong >= 3, "chance_hit": tol_area, **m})
res = pd.DataFrame(rows)
res.to_csv(out / "gradcam_localisation.csv", index=False)
np.save(out / "test_cams.npy", cams.astype(np.float16))

def summarise(g):
    return {"n": int(len(g)), "pointing_hit_rate": float(g["pointing_hit"].mean()),
            "chance_hit_rate": float(g["chance_hit"].mean()),
            "median_energy_in_lesion": float(g["energy_in_lesion"].median()),
            "median_lesion_area_frac": float(g["lesion_area_frac"].median()),
            "median_energy_ratio": float(g["energy_ratio"].median())}
groups = {"all": res, "benign": res[res.y == 0], "malignant": res[res.y == 1],
          "correct (<3 models wrong)": res[~res.hard_case], "hard (≥3 models wrong)": res[res.hard_case]}
summ = {k: summarise(g) for k, g in groups.items() if len(g)}
json.dump({"threshold": thr, "tolerance_px": 7, "cam_target": "malignant logit, mean of 5 fold models",
           **summ}, open(out / "gradcam_summary.json", "w"), indent=2)

print("\n============ Grad-CAM vs lesion masks (TEST set) ============")
print(f"{'group':<26}{'n':>4}  {'pointing hit':>12}  {'chance':>7}  {'energy in lesion':>16}  {'lesion area':>11}  {'ratio':>6}")
for k, s in summ.items():
    print(f"{k:<26}{s['n']:>4}  {s['pointing_hit_rate']:>12.2f}  {s['chance_hit_rate']:>7.2f}  "
          f"{s['median_energy_in_lesion']:>16.2f}  {s['median_lesion_area_frac']:>11.2f}  {s['median_energy_ratio']:>6.2f}")
print("(pointing hit = hottest pixel inside lesion ±7px; chance = hit rate of a random point; "
      "ratio > 1 = heat concentrated on the lesion more than chance)")

# ---------- 3) Galleries ----------
def overlay(ax, i, title):
    img = np.array(ds._cache[i].convert("L"))
    les = load_lesion_mask(P["data_root"], test_df.at[i, "label"], test_df.at[i, "fname"], SIZE)
    con = content_mask(P["data_root"], test_df.at[i, "label"], test_df.at[i, "fname"], SIZE)
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(~con, cams[i]), cmap="jet", alpha=0.40, vmin=0, vmax=1)
    ax.contour(les, levels=[0.5], colors="lime", linewidths=1.2)
    py, px = np.unravel_index(np.argmax(np.where(con, cams[i], 0)), cams[i].shape)
    ax.scatter([px], [py], marker="x", c="white", s=50, linewidths=2)
    ax.set_title(title, fontsize=8); ax.axis("off")

def gallery(idx, fname, heading):
    if not len(idx):
        return
    n = len(idx); cols = 4; rws = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(3.2 * cols, 3.4 * rws), squeeze=False, layout="constrained")
    for a in axs.flat:
        a.axis("off")
    for a, i in zip(axs.flat, idx):
        r = res.loc[i]
        overlay(a, i, f"{r.fname}\ntrue {r.label} | p={r.prob_ensemble:.2f} | "
                      f"{'HIT' if r.pointing_hit else 'miss'}")
    fig.suptitle(heading + "\n(green = radiologist lesion mask, colour = Grad-CAM, × = hottest point)",
                 fontweight="bold", fontsize=10)
    plt.savefig(out / fname, dpi=100, bbox_inches="tight"); plt.close(fig)

ok = res[~res.hard_case]
good = list(ok[ok.y == 0].nsmallest(4, "prob_ensemble").index) + list(ok[ok.y == 1].nlargest(4, "prob_ensemble").index)
gallery(good, "gradcam_correct.png", "Confident correct predictions (test set)")
gallery(list(res[res.hard_case].sort_values("n_models_wrong", ascending=False).index[:16]),
        "gradcam_hard_cases.png", "Hard cases: misclassified by ≥3 of 5 models (test set)")
print(f"\nSaved to: {out}")

"""Step 6 — held-out test evaluation helpers (pure numpy, no torch)."""
import numpy as np
from sklearn.metrics import roc_auc_score


def _sens_spec(y, pred):
    tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == 0) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0)); fp = np.sum((pred == 1) & (y == 0))
    return tp / (tp + fn), tn / (tn + fp)


def group_bootstrap(y, probs, groups, thr, n_boot=2000, seed=42):
    """95% CIs by resampling whole duplicate-groups (not single images), so copies of
    the same scan are always drawn together.
    probs: array (n_models, n_images). Returns CIs for
      - ensemble AUC (mean probability of the models)
      - mean single-model sensitivity / specificity at `thr`
    """
    y = np.asarray(y).astype(int); probs = np.atleast_2d(np.asarray(probs, dtype=float))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    ens = probs.mean(0)
    aucs, sens, specs = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue                      # need both classes
        aucs.append(roc_auc_score(yb, ens[idx]))
        ss = np.array([_sens_spec(yb, (p[idx] >= thr).astype(int)) for p in probs])
        sens.append(ss[:, 0].mean()); specs.append(ss[:, 1].mean())
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"ensemble_auc_ci": ci(aucs), "mean_model_sens_ci": ci(sens),
            "mean_model_spec_ci": ci(specs), "n_boot_used": len(aucs)}

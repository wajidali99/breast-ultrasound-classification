"""Step 3 — evaluation metrics for binary benign (0) vs malignant (1)."""
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


def binary_metrics(y_true, prob_malignant, threshold: float = 0.5) -> dict:
    """AUC uses probabilities (threshold-free). The rest use prob >= threshold -> malignant."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob_malignant, dtype=float)
    assert y_true.shape == prob.shape, "y_true and prob length mismatch"
    assert set(np.unique(y_true)) <= {0, 1}, "Labels must be 0/1"
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # malignant pakre gaye
    spec = tn / (tn + fp) if (tn + fp) else float("nan")   # benign sahi pehchane
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "auc": float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_acc": float((sens + spec) / 2),
        "accuracy": float((tp + tn) / len(y_true)),
        "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp),
    }


def threshold_for_sensitivity(y_true, prob, target: float = 0.90) -> float:
    """Highest threshold whose sensitivity is >= target (so specificity is as high as possible)."""
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(prob, dtype=float))
    ok = np.where((tpr >= target) & np.isfinite(thr))[0]
    assert len(ok), f"No threshold reaches sensitivity {target}"
    return float(thr[ok[0]])


def youden_threshold(y_true, prob) -> float:
    """Threshold maximising sensitivity + specificity - 1 (Youden's J)."""
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(prob, dtype=float))
    j = np.where(np.isfinite(thr), tpr - fpr, -np.inf)
    return float(thr[int(np.argmax(j))])

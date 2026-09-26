"""Step 3 — evaluation metrics for binary benign (0) vs malignant (1)."""
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


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

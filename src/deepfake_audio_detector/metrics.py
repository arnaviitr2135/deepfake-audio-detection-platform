from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_curve


def equal_error_rate(y_true: np.ndarray, fake_scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, fake_scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def classification_report_dict(y_true: np.ndarray, fake_scores: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (fake_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    eer, eer_threshold = equal_error_rate(y_true, fake_scores)
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": {
            "real": float(per_class_acc[0]),
            "fake": float(per_class_acc[1]),
        },
        "precision": {"real": float(precision[0]), "fake": float(precision[1])},
        "recall": {"real": float(recall[0]), "fake": float(recall[1])},
        "f1": {"real": float(f1[0]), "fake": float(f1[1])},
    }

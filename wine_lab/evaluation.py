"""
Computes classification metrics and saves diagnostic plots for a candidate model.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

DECISION_THRESHOLD = 0.5


def compute_metrics(y_true, y_prob, threshold: float = DECISION_THRESHOLD) -> dict:
    """
    Compute ranking and threshold-based metrics for binary predictions.

    Parameters
    ----------
    y_true : array-like of int
        True labels (0 or 1).
    y_prob : array-like of float
        Predicted probability of the positive class.
    threshold : float
        Probability at or above which a prediction counts as positive.

    Returns
    -------
    metric_dict : dict of str to float
        auc, average_precision, log_loss, accuracy, precision, recall, and f1.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_roc_curve(y_true, y_prob, output_path) -> None:
    """Save a ROC curve plot."""
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC curve (validation)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true, y_prob, output_path,
                          threshold: float = DECISION_THRESHOLD) -> None:
    """Save a confusion matrix plot at the given threshold."""
    fig, ax = plt.subplots(figsize=(5, 5))
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["standard", "high quality"], ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix (threshold {threshold})")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(importance_list, feature_list, output_path) -> None:
    """
    Save a horizontal bar chart of feature importances, largest first.

    Parameters
    ----------
    importance_list : array-like of float
        Importance per feature, for example model.feature_importances_.
    feature_list : list of str
        Feature names in the same order.
    output_path : str or pathlib.Path
        PNG file to write.
    """
    order = np.argsort(importance_list)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.asarray(feature_list)[order], np.asarray(importance_list)[order])
    ax.set_title("Feature importance")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

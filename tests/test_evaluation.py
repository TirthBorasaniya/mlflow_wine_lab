"""
Tests for metric computation and diagnostic plots.
"""

import pytest

from wine_lab.evaluation import (
    compute_metrics,
    save_confusion_matrix,
    save_feature_importance,
    save_roc_curve,
)

Y_TRUE = [0, 0, 1, 1, 0, 1]
Y_PROB = [0.1, 0.4, 0.8, 0.6, 0.7, 0.3]


def test_compute_metrics_values():
    metric_dict = compute_metrics(Y_TRUE, Y_PROB)
    assert set(metric_dict) == {
        "auc", "average_precision", "log_loss", "accuracy", "precision", "recall", "f1"
    }
    # 6 of the 9 positive-negative pairs are ranked correctly
    assert metric_dict["auc"] == pytest.approx(6 / 9)
    assert metric_dict["accuracy"] == pytest.approx(4 / 6)
    assert metric_dict["precision"] == pytest.approx(2 / 3)
    assert metric_dict["recall"] == pytest.approx(2 / 3)


def test_perfect_predictions():
    metric_dict = compute_metrics([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0])
    assert metric_dict["auc"] == 1.0
    assert metric_dict["f1"] == 1.0
    assert metric_dict["log_loss"] < 1e-5


def test_threshold_changes_labels_only():
    low = compute_metrics(Y_TRUE, Y_PROB, threshold=0.2)
    high = compute_metrics(Y_TRUE, Y_PROB, threshold=0.9)
    assert low["auc"] == high["auc"]
    assert low["recall"] == 1.0
    assert high["recall"] == 0.0


def test_plots_are_written(tmp_path):
    save_roc_curve(Y_TRUE, Y_PROB, tmp_path / "roc.png")
    save_confusion_matrix(Y_TRUE, Y_PROB, tmp_path / "cm.png")
    save_feature_importance([0.2, 0.5, 0.3], ["a", "b", "c"], tmp_path / "fi.png")
    for name in ("roc.png", "cm.png", "fi.png"):
        assert (tmp_path / name).stat().st_size > 0

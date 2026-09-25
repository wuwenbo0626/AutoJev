import numpy as np

from laya_autofinetune.metrics import classification_metrics


def test_metrics_perfect_predictions():
    result = classification_metrics(
        [np.array([0.9, 0.1]), np.array([0.2, 0.8])],
        [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
    )
    assert result["accuracy"] == 1.0
    assert result["brier"] < 0.1
    assert result["finite"] is True


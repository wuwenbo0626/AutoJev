from __future__ import annotations

import math
from typing import Any

import numpy as np


def classification_metrics(probabilities: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, Any]:
    if not probabilities:
        return {"count": 0}
    correct, confidence, nll, brier = [], [], [], []
    for probs, target in zip(probabilities, targets):
        pred = int(np.argmax(probs))
        gold = int(np.argmax(target))
        correct.append(float(pred == gold))
        confidence.append(float(np.max(probs)))
        nll.append(-float(np.sum(target * np.log(np.clip(probs, 1e-12, 1.0)))))
        brier.append(float(np.sum((probs - target) ** 2)))
    edges = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    conf = np.asarray(confidence)
    corr = np.asarray(correct)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (conf >= low if index == 0 else conf > low) & (conf <= high)
        if selected.any():
            ece += float(selected.mean() * abs(conf[selected].mean() - corr[selected].mean()))
    return {
        "count": len(correct),
        "accuracy": float(np.mean(correct)),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "ece_15": ece,
        "mean_confidence": float(np.mean(confidence)),
        "finite": all(math.isfinite(value) for value in nll + brier),
    }


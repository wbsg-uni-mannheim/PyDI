from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple


def compute_binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float | int]:
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def tune_threshold_for_f1(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    step: float = 0.01,
) -> Tuple[float, Dict[str, float | int]]:
    """
    Find threshold in [0,1] that maximizes F1 on validation data.
    """
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have same length")
    if len(y_true) == 0:
        return 0.5, compute_binary_metrics([], [])

    best_th = 0.5
    best_metrics = None
    best_f1 = -1.0

    # Include both endpoints and use deterministic tie-break preferring lower threshold
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        th = i * step
        pred = [1 if p > th else 0 for p in y_prob]
        m = compute_binary_metrics(y_true, pred)
        f1 = float(m["f1"])
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_metrics = m

    assert best_metrics is not None
    return best_th, best_metrics

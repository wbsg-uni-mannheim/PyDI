"""Per-attribute scoring for the Normalization committee.

Closes the Pending #5 closeness contract over the normalization stage:
for each fusion-protected (entity, canonical-attribute) cell that the
SM mapping resolves to a source column, score the normalizer's output
against the fusion val/test reference value via
:func:`protection.is_close_enough`.

Metric: per-attribute precision / recall / F1 with the standard EM/SM
convention.

- ``correct``: normalizer output is close-enough to canonical
- ``wrong_output``: normalizer output is non-null and *not* close-enough
- ``abstained``: normalizer returned ``None``
- precision = correct / max(correct + wrong_output, 1)
- recall    = correct / max(correct + wrong_output + abstained, 1)
- F1 = 2 * P * R / (P + R)

All counters are over the per-source axis: an entity that is carried
by three sources contributes three (source, cell) observations on each
applicable attribute (one per source-column the SM mapping resolves
for that attribute).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .protection import (
    ToleranceSpec,
    is_close_enough,
)


@dataclass
class CellOutcome:
    """Per-(member, source, entity, attribute) outcome counter."""

    correct: int = 0
    wrong: int = 0
    abstained: int = 0
    total: int = 0


@dataclass
class AttributeScore:
    """Per-attribute aggregated counters + derived F1."""

    correct: int = 0
    wrong: int = 0
    abstained: int = 0
    total: int = 0

    @property
    def precision(self) -> float:
        denom = self.correct + self.wrong
        return self.correct / denom if denom else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_cell(
    output: str | None,
    target_values: list[str],
    tolerance: ToleranceSpec,
) -> tuple[bool, bool]:
    """Score a single normalizer output cell.

    Returns ``(correct, abstained)``. ``wrong`` is ``not correct and
    not abstained``.
    """
    if output is None:
        return False, True
    if not target_values:
        # No ground truth → cannot score.
        return False, True
    for target in target_values:
        if is_close_enough(str(output), str(target), tolerance):
            return True, False
    return False, False


@dataclass
class MemberPerAttributeScores:
    """Per-attribute counters for one committee member."""

    member: str
    by_attribute: dict[str, AttributeScore] = field(default_factory=dict)

    def record(
        self,
        attribute: str,
        output: str | None,
        target_values: list[str],
        tolerance: ToleranceSpec,
    ) -> None:
        score = self.by_attribute.setdefault(attribute, AttributeScore())
        correct, abstained = score_cell(output, target_values, tolerance)
        score.total += 1
        if correct:
            score.correct += 1
        elif abstained:
            score.abstained += 1
        else:
            score.wrong += 1

    def macro_metrics(self) -> dict[str, float]:
        """Return macro-averaged F1 / precision / recall across attributes."""
        if not self.by_attribute:
            return {
                "macro_f1": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "min_f1": 0.0,
                "max_f1": 0.0,
                "n_attributes": 0,
                "n_cells": 0,
            }
        f1s = [s.f1 for s in self.by_attribute.values()]
        ps = [s.precision for s in self.by_attribute.values()]
        rs = [s.recall for s in self.by_attribute.values()]
        n_cells = sum(s.total for s in self.by_attribute.values())
        return {
            "macro_f1": sum(f1s) / len(f1s),
            "macro_precision": sum(ps) / len(ps),
            "macro_recall": sum(rs) / len(rs),
            "min_f1": min(f1s),
            "max_f1": max(f1s),
            "n_attributes": float(len(self.by_attribute)),
            "n_cells": float(n_cells),
        }


__all__ = [
    "AttributeScore",
    "CellOutcome",
    "MemberPerAttributeScores",
    "score_cell",
]

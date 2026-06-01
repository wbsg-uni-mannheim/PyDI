"""Committee runner framework for validation.

An abstract base that every stage-specific committee (SM, EM, fusion)
inherits from. The base carries no stage-specific logic — it defines
the interface, the result dataclasses, and the roster/config plumbing.
Concrete subclasses live in ``committee_sm.py``, ``committee_em.py``,
and ``committee_fusion.py`` (created in M2/M3/M4).

The contract is narrow on purpose: every runner takes a
:class:`VariantBundle` (the output of
:func:`usecases_synthetic.lib.variant_loader.load_variant`) and returns
a :class:`CommitteeResult` with a flat ``aggregated`` metric dict so
:func:`usecases_synthetic.lib.validation_metrics.delta` can subtract
baselines without schema knowledge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - import kept for type hints only
    from .variant_loader import VariantBundle


Stage = Literal["sm", "norm", "em", "em_blocking", "em_matching", "fusion"]


@dataclass
class MemberResult:
    """Result of a single committee member.

    Parameters
    ----------
    name : str
        Member identifier (e.g. ``"label_based"``, ``"magellan_rf"``).
    predictions : Any
        Stage-specific predictions. SM members return a mapping
        DataFrame, EM members return correspondence DataFrames, fusion
        members return the fused DataFrame. The base class does not
        inspect this field.
    metrics : dict[str, float]
        Flat per-member metric dict (``"precision"``, ``"recall"``,
        ``"f1"``, ...). Aggregation happens in
        :class:`CommitteeResult`.
    runtime_s : float
        Wall-clock runtime for this member.
    notes : dict[str, Any]
        Optional member-specific notes (e.g. member config hash,
        deterministic seed). Not used for metrics.
    """

    name: str
    predictions: Any
    metrics: dict[str, float]
    runtime_s: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommitteeResult:
    """Aggregated outcome of a committee run on a single variant.

    Parameters
    ----------
    stage : {"sm", "em", "fusion"}
        Pipeline stage this result belongs to.
    domain : str
        Domain name (e.g. ``"companies"``).
    level : str
        Variant level (e.g. ``"baseline"``, ``"easy"``, ``"medium"``,
        ``"hard"``).
    per_member : dict[str, MemberResult]
        Per-member results keyed by member name.
    aggregated : dict[str, float]
        Flat cross-member metric dict. What gets diffed against
        baselines by
        :func:`usecases_synthetic.lib.validation_metrics.delta`.
    per_attribute : dict[str, dict[str, float]]
        Per-attribute metrics (fusion: per column; SM: per source
        column; EM: unused unless a runner chooses to populate it).
    per_partition : dict[str, dict[str, float]]
        Per-partition metrics (EM: per source pair; fusion: per
        attribute class; SM: per source). Partitions are named with
        strings so they survive JSON round-trip.
    per_blocker : dict[str, MemberResult]
        EM-specific: blocking-committee results (one entry per blocker).
        Empty for SM/fusion committees. Populated by ``EMCommitteeRunner``
        under the split-roster architecture where blockers and matchers
        run in two separate phases; ``per_member`` carries matchers only
        so ``aggregated`` macro F1 is not polluted by blocker pair-recall
        numbers.
    runtime_s : float
        Total committee wall-clock runtime.
    roster : list[str]
        Member names in the roster, in declaration order.
    """

    stage: Stage
    domain: str
    level: str
    per_member: dict[str, MemberResult]
    aggregated: dict[str, float]
    per_attribute: dict[str, dict[str, float]] = field(default_factory=dict)
    per_partition: dict[str, dict[str, float]] = field(default_factory=dict)
    per_blocker: dict[str, MemberResult] = field(default_factory=dict)
    runtime_s: float = 0.0
    roster: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of this committee result.

        Predictions are dropped from ``per_member`` because they are not
        metrics and may not be serialisable. The metric payload is what
        :func:`usecases_synthetic.lib.report.write_metrics_json` persists.

        Returns
        -------
        dict[str, Any]
            JSON-serialisable dict.
        """
        return {
            "stage": self.stage,
            "domain": self.domain,
            "level": self.level,
            "runtime_s": self.runtime_s,
            "roster": list(self.roster),
            "aggregated": dict(self.aggregated),
            "per_attribute": {
                attr: dict(metrics) for attr, metrics in self.per_attribute.items()
            },
            "per_partition": {
                part: dict(metrics) for part, metrics in self.per_partition.items()
            },
            "per_member": {
                name: {
                    "metrics": dict(member.metrics),
                    "runtime_s": member.runtime_s,
                    "notes": dict(member.notes),
                }
                for name, member in self.per_member.items()
            },
            "per_blocker": {
                name: {
                    "metrics": dict(member.metrics),
                    "runtime_s": member.runtime_s,
                    "notes": dict(member.notes),
                }
                for name, member in self.per_blocker.items()
            },
        }


class CommitteeRunner(ABC):
    """Abstract base class for committee runners.

    Parameters
    ----------
    roster : list[Any]
        Ordered roster of committee members. The concrete type is
        stage-specific — SM runners expect matcher objects, EM runners
        expect matcher objects or callables, fusion runners expect
        strategy specs. The base only stores the list.
    config : dict[str, Any]
        Runner configuration (e.g. metric thresholds, partition
        settings). Opaque to the base class.
    """

    stage: Stage = "sm"  # Overridden by subclasses.

    def __init__(
        self,
        roster: list[Any],
        config: dict[str, Any] | None = None,
    ) -> None:
        self.roster = roster
        self.config = dict(config or {})

    @property
    def roster_names(self) -> list[str]:
        """Return the member names in declaration order.

        Subclasses whose roster elements are not plain objects with a
        ``name`` attribute should override this property.

        Returns
        -------
        list[str]
            Ordered member names.
        """
        names: list[str] = []
        for index, member in enumerate(self.roster):
            name = getattr(member, "name", None)
            if not isinstance(name, str):
                name = f"{type(member).__name__}_{index}"
            names.append(name)
        return names

    @abstractmethod
    def run(self, bundle: "VariantBundle") -> CommitteeResult:
        """Run every roster member against ``bundle`` and aggregate.

        Parameters
        ----------
        bundle : VariantBundle
            Loaded variant (baseline or augmented).

        Returns
        -------
        CommitteeResult
            Aggregated result for this stage.
        """

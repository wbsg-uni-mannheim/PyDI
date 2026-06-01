"""Baseline metric reader.

Reads ``usecases_synthetic/baselines/<domain>/baseline_metrics.json``
into a :class:`BaselineMetrics` dataclass. The on-disk schema is the
same shape that :func:`usecases_synthetic.lib.report.write_metrics_json`
produces, so M5 (the baseline measurement script) can write via
``write_metrics_json`` and M7/M8 can read via this loader without a
format adapter in between.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .domain_config import SYNTHETIC_DIR

BASELINES_DIR: Path = SYNTHETIC_DIR / "baselines"


@dataclass
class BaselineMetrics:
    """Baseline metric payload for a single domain.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    per_stage : dict[str, dict[str, Any]]
        Mapping from stage (``"sm"``, ``"em"``, ``"fusion"``) to its
        committee metric dict as produced by
        :meth:`usecases_synthetic.lib.committee.CommitteeResult.as_dict`.
    meta : dict[str, Any]
        Arbitrary metadata (measurement timestamp, git SHA, runtime).
    """

    domain: str
    per_stage: dict[str, dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)

    def aggregated(self, stage: str) -> dict[str, float]:
        """Return the flat ``aggregated`` metric dict for ``stage``.

        Parameters
        ----------
        stage : str
            ``"sm"``, ``"em"``, or ``"fusion"``.

        Returns
        -------
        dict[str, float]
            The aggregated dict. Empty if the stage is not present.
        """
        block = self.per_stage.get(stage, {})
        agg = block.get("aggregated", {})
        # The em_blocking + em_matching runners surface ``best_member_name``
        # (a string) alongside numeric macros. Filter non-numeric values
        # so the float cast stays safe; callers that want the string-valued
        # keys should read ``per_stage[stage]["aggregated"]`` directly.
        out: dict[str, float] = {}
        for k, v in agg.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def per_attribute(self, stage: str) -> dict[str, dict[str, float]]:
        """Return the per-attribute metric dict for ``stage``.

        Parameters
        ----------
        stage : str
            Stage name.

        Returns
        -------
        dict[str, dict[str, float]]
            Attribute -> metric -> value. Empty when absent.
        """
        block = self.per_stage.get(stage, {})
        per_attr = block.get("per_attribute", {})
        return {
            attr: {k: float(v) for k, v in metrics.items()}
            for attr, metrics in per_attr.items()
        }

    def per_partition(self, stage: str) -> dict[str, dict[str, float]]:
        """Return the per-partition metric dict for ``stage``.

        Parameters
        ----------
        stage : str
            Stage name.

        Returns
        -------
        dict[str, dict[str, float]]
            Partition -> metric -> value. Empty when absent.
        """
        block = self.per_stage.get(stage, {})
        per_part = block.get("per_partition", {})
        return {
            part: {k: float(v) for k, v in metrics.items()}
            for part, metrics in per_part.items()
        }


def baseline_path(domain: str) -> Path:
    """Return the canonical baseline metrics path for a domain.

    Parameters
    ----------
    domain : str
        Domain name.

    Returns
    -------
    Path
        ``usecases_synthetic/baselines/<domain>/baseline_metrics.json``.
    """
    return BASELINES_DIR / domain / "baseline_metrics.json"


def load_baseline(
    domain: str,
    *,
    path_override: Path | None = None,
) -> BaselineMetrics:
    """Load baseline metrics for ``domain``.

    Parameters
    ----------
    domain : str
        Domain name.
    path_override : Path, optional
        Read from this path instead of the canonical location. Used by
        tests.

    Returns
    -------
    BaselineMetrics
        Parsed baseline.

    Raises
    ------
    FileNotFoundError
        If the baseline file does not exist.
    """
    path = path_override if path_override is not None else baseline_path(domain)
    if not path.exists():
        raise FileNotFoundError(f"Baseline metrics not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    per_stage: dict[str, dict[str, Any]] = raw.get("per_stage", {})
    meta: dict[str, Any] = raw.get("meta", {})
    return BaselineMetrics(domain=domain, per_stage=per_stage, meta=meta)

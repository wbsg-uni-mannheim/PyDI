"""Load per-stage per-member val-best hyperparameters from tuning caches.

The synthetic-pipeline's tuning harnesses (``_tune_*_committee.py``)
emit per-domain sweep caches at ``cache/<stage>_tuning/sweep.json``.
Each cache has a stage-specific schema; this module folds them into a
uniform shape so the best-of-breed stage runners can override their
YAML-locked defaults with the val-best HP per member for a given
domain.

The uniform shape returned by :func:`load_val_best_hp_for_domain` is::

    {
        "sm": {
            "<member_name>": {
                "init": {<init kwarg>: <value>, ...},
                "match_kwargs": {<match kwarg>: <value>, ...},
                "val_score": float,
                "metric_key": "mean_f1",
            },
            ...
        },
        "norm": {
            "<member_name>": {
                "init": {...},
                "val_score": float,
                "metric_key": "macro_f1",
            },
            ...
        },
        "em_blocking": {
            "<member_name>": {
                "params": {...},
                "val_score": float,
                "metric_key": "reduction_ratio",
            },
            ...
        },
        "em_matching": {
            "<member_name>": {
                "params": {...},
                "val_score": float,
                "metric_key": "mean_f1",
            },
            ...
        },
        # fusion sweeps are sub-sweep-keyed (trust / tolerance / trim / ...)
        # rather than member-keyed, so we expose them as a separate flat
        # dict to be consumed by the fusion-specific overrides
        # mechanism (see :func:`load_fusion_tuned_overrides`).
    }
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def _is_nan(value: Any) -> bool:
    """Return True for NaN; never for non-floats."""
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Per-stage loaders
# ---------------------------------------------------------------------------


def _load_sm_val_best(cache_path: Path, domain: str) -> dict[str, dict[str, Any]]:
    """SM sweep schema: {member: [{init.<k>: ..., match.<k>: ..., per_domain: {<d>: {f1}}}]}."""
    if not cache_path.exists():
        return {}
    raw = json.loads(cache_path.read_text())
    out: dict[str, dict[str, Any]] = {}
    for member, rows in raw.items():
        if not isinstance(rows, list):
            continue
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            per_domain = row.get("per_domain") or {}
            if domain not in per_domain:
                continue
            f1 = float(per_domain[domain].get("f1", 0.0))
            if _is_nan(f1):
                continue
            scored.append((f1, row))
        if not scored:
            continue
        # Tie-break: higher F1 first; then a stable hash on the row body.
        scored.sort(
            key=lambda x: (-x[0], json.dumps(x[1], sort_keys=True, default=str))
        )
        best_f1, best_row = scored[0]
        init = {
            k[len("init.") :]: v for k, v in best_row.items() if k.startswith("init.")
        }
        match_kwargs = {
            k[len("match.") :]: v for k, v in best_row.items() if k.startswith("match.")
        }
        out[member] = {
            "init": init,
            "match_kwargs": match_kwargs,
            "val_score": best_f1,
            "metric_key": "f1",
        }
    return out


def _load_norm_val_best(cache_path: Path, domain: str) -> dict[str, dict[str, Any]]:
    """Norm sweep schema: {member: [{init: {...}, per_domain: {<d>: {macro_f1}}}]}."""
    if not cache_path.exists():
        return {}
    raw = json.loads(cache_path.read_text())
    out: dict[str, dict[str, Any]] = {}
    for member, rows in raw.items():
        if not isinstance(rows, list):
            continue
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            per_domain = row.get("per_domain") or {}
            if domain not in per_domain:
                continue
            f1 = float(per_domain[domain].get("macro_f1", 0.0))
            if _is_nan(f1):
                continue
            scored.append((f1, row))
        if not scored:
            continue
        scored.sort(
            key=lambda x: (
                -x[0],
                json.dumps(x[1].get("init", {}), sort_keys=True, default=str),
            )
        )
        best_f1, best_row = scored[0]
        out[member] = {
            "init": dict(best_row.get("init") or {}),
            "val_score": best_f1,
            "metric_key": "macro_f1",
        }
    return out


def _load_em_blocking_val_best(
    cache_path: Path, domain: str
) -> dict[str, dict[str, Any]]:
    """EM blocking sweep schema: {sub_sweep: [...]}.

    Three sub-sweeps:
    - ``embedding``: per-(model, top_k, threshold), per_domain.<d>.{mean_pair_recall, mean_reduction_ratio}
    - ``standard``: per-(domain, on_cols, key_spec), mean_recall + mean_rr
    - ``sn``: per-(domain, key_col, window), mean_recall + mean_rr

    For each member we want the per-domain val-best config. Members
    correspond to the sub-sweep that targets them:
    - ``embedding_blocker`` → embedding sweep
    - ``standard_blocker`` → standard sweep
    - ``sorted_neighbourhood_blocker`` → sn sweep
    - ``token_blocker``, ``bm25_blocker``, ``sc_block`` → not swept; YAML defaults stand

    Selection rule per the tune script: highest reduction_ratio among
    configs clearing the recall_floor (0.97 default), else
    highest-recall.
    """
    if not cache_path.exists():
        return {}
    raw = json.loads(cache_path.read_text())
    out: dict[str, dict[str, Any]] = {}

    embedding_rows = raw.get("embedding") or []
    standard_rows = raw.get("standard") or []
    sn_rows = raw.get("sn") or []

    recall_floor = 0.97

    def _pick(
        rows: list[dict[str, Any]],
        *,
        recall_key: str,
        rr_key: str,
        param_extract: Any,
    ) -> tuple[float, dict[str, Any]] | None:
        clearing: list[tuple[float, float, dict[str, Any]]] = []
        for row in rows:
            recall = float(row.get(recall_key, 0.0) or 0.0)
            rr = float(row.get(rr_key, 0.0) or 0.0)
            if _is_nan(recall) or _is_nan(rr):
                continue
            clearing.append((recall, rr, row))
        if not clearing:
            return None
        # If every candidate has 0 recall, the sweep was run against the
        # wrong column set for this domain (e.g. embedding sub-sweep with
        # text_cols=['name'] on products which keys on 'title'). Return
        # None so the stage runner falls back to the YAML default.
        if max(c[0] for c in clearing) <= 0.0:
            return None
        above = [c for c in clearing if c[0] >= recall_floor]
        if above:
            above.sort(key=lambda x: (-x[1], -x[0]))
            picked = above[0]
        else:
            clearing.sort(key=lambda x: (-x[0], -x[1]))
            picked = clearing[0]
        return picked[1], param_extract(picked[2])

    # embedding_blocker
    emb_for_domain = [
        r for r in embedding_rows if (r.get("per_domain") or {}).get(domain)
    ]
    for r in emb_for_domain:
        per_dom = r["per_domain"][domain]
        r["_recall_in_domain"] = float(per_dom.get("mean_pair_recall", 0.0))
        r["_rr_in_domain"] = float(per_dom.get("mean_reduction_ratio", 0.0))
    picked = _pick(
        emb_for_domain,
        recall_key="_recall_in_domain",
        rr_key="_rr_in_domain",
        param_extract=lambda r: dict(r.get("params") or {}),
    )
    if picked:
        score, params = picked
        out["embedding_blocker"] = {
            "params": params,
            "val_score": score,
            "metric_key": "reduction_ratio",
        }

    # standard_blocker
    standard_for_domain = [r for r in standard_rows if r.get("domain") == domain]
    picked = _pick(
        standard_for_domain,
        recall_key="mean_recall",
        rr_key="mean_rr",
        param_extract=lambda r: {
            "on": list(r.get("on_cols") or []),
            "key_spec": list(r.get("key_spec") or []),
        },
    )
    if picked:
        score, params = picked
        out["standard_blocker"] = {
            "params": params,
            "val_score": score,
            "metric_key": "reduction_ratio",
        }

    # sorted_neighbourhood_blocker
    sn_for_domain = [r for r in sn_rows if r.get("domain") == domain]
    picked = _pick(
        sn_for_domain,
        recall_key="mean_recall",
        rr_key="mean_rr",
        param_extract=lambda r: {
            "key": r.get("key_col"),
            "key_spec": dict(r.get("key_spec") or {}),
            "window": r.get("window"),
        },
    )
    if picked:
        score, params = picked
        out["sorted_neighbourhood_blocker"] = {
            "params": params,
            "val_score": score,
            "metric_key": "reduction_ratio",
        }

    return out


def _load_em_matching_val_best(
    cache_path: Path, domain: str
) -> dict[str, dict[str, Any]]:
    """EM matching sweep schema: ``{"rows": [...], "winners": {<member>::<domain>: row}}``."""
    if not cache_path.exists():
        return {}
    raw = json.loads(cache_path.read_text())
    winners = raw.get("winners") or {}
    out: dict[str, dict[str, Any]] = {}
    for key, row in winners.items():
        if "::" not in key:
            continue
        member, dom = key.split("::", 1)
        if dom != domain:
            continue
        params = {
            k: v
            for k, v in row.items()
            if k
            not in {"member", "domain", "mean_f1", "min_f1", "config_id", "per_pair"}
        }
        out[member] = {
            "params": params,
            "val_score": float(row.get("mean_f1", 0.0)),
            "metric_key": "f1",
        }
    return out


def _load_fusion_val_best(cache_path: Path, domain: str) -> dict[str, Any]:
    """Fusion sweep schema: per-sub-sweep tunings of single params.

    The fusion tune layout is different — sweeps target single
    hyperparameters (trust, tolerance, trim, ...) rather than per-member
    grids. We expose a flat dict of sub-sweep → winner-row for the
    given domain, leaving interpretation to the fusion stage runner.
    """
    if not cache_path.exists():
        return {}
    raw = json.loads(cache_path.read_text())
    out: dict[str, Any] = {}
    for sub_name, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        # Each fusion sub-sweep emits {<domain>: [...rows...]} typically.
        dom_rows = payload.get(domain)
        if dom_rows:
            out[sub_name] = dom_rows
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_val_best_hp_for_domain(
    domain: str,
    *,
    cache_root: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load val-best HP per stage member for ``domain``.

    Parameters
    ----------
    domain : str
        Domain name (``"products"``, ``"music"``, ...).
    cache_root : Path
        Repo-level ``cache/`` directory (i.e. the parent of
        ``cache/sm_tuning/sweep.json``).

    Returns
    -------
    dict
        ``{stage: {member: {<HP-dict>, "val_score": float, "metric_key": str}}}``.
        Missing stages or members fall back to YAML defaults in the
        stage runner.
    """
    return {
        "sm": _load_sm_val_best(cache_root / "sm_tuning" / "sweep.json", domain),
        "norm": _load_norm_val_best(cache_root / "norm_tuning" / "sweep.json", domain),
        "em_blocking": _load_em_blocking_val_best(
            cache_root / "em_blocking_tuning" / "sweep.json", domain
        ),
        "em_matching": _load_em_matching_val_best(
            cache_root / "em_matching_tuning" / "sweep.json", domain
        ),
        "fusion": _load_fusion_val_best(
            cache_root / "fusion_tuning" / "sweep.json", domain
        ),
    }


def summarize_tuned_hps(tuned: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    """Pretty one-line-per-member summary for the report."""
    lines: list[str] = []
    for stage, members in tuned.items():
        if not members:
            lines.append(f"{stage}: no tuned HPs in cache (using YAML defaults)")
            continue
        lines.append(f"{stage}:")
        for member, info in members.items():
            if not isinstance(info, dict) or "val_score" not in info:
                continue
            score = info.get("val_score", float("nan"))
            metric = info.get("metric_key", "?")
            lines.append(f"  - {member}: val_{metric}={score:.4f}")
    return "\n".join(lines)


__all__ = ["load_val_best_hp_for_domain", "summarize_tuned_hps"]

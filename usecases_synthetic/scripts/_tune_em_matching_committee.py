"""Tune EM matching committee hyperparameters per member.

One-off sweep harness for the R5 EM matching tuning pass. Scope per
R5 sign-off (plans/plan_s1_scale.md):

- **ditto_plm**: no sweep (R2 LR×class-balance sweep is the source of
  truth; checkpoints at ``cache/ditto_checkpoints/<d>/best/`` already
  picked. Threshold locked at 0.5.).
- **magellan**: classifier sweep only (auto-feature-gen handles the
  similarity-function dimension). Grid: ``n_estimators × max_depth ×
  class_weight``. Threshold locked at 0.5.
- **matchgpt**: no sweep (locked at zero-shot, ``gpt-5.4-mini``,
  threshold 0.5).
- **comem**: no sweep (locked at paper defaults — ``stage1_set_size=10``,
  ``stage2_model=None`` reuses ``gpt-5.4-mini``, ``threshold=0.5``).

Per-domain results land at ``cache/em_matching_tuning/sweep.json``.
Winner per domain: highest mean F1 across the domain's source pairs,
ties broken by min F1 (robustness) then alphabetically.

Usage::

    python usecases_synthetic/scripts/_tune_em_matching_committee.py \\
        --domains companies,games,music \\
        --members magellan
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from usecases_synthetic.lib.column_mapping import apply_column_mapping  # noqa: E402
from usecases_synthetic.lib.committee_em_scoring import (  # noqa: E402
    score_em_correspondences_closed_set,
)
from usecases_synthetic.lib.variant_loader import load_variant  # noqa: E402

logger = logging.getLogger("tune_em_matching")


CACHE_DIR = REPO_ROOT / "cache" / "em_matching_tuning"


# ---------------------------------------------------------------------------
# Domain context
# ---------------------------------------------------------------------------


def _committee_yaml_path(domain: str, base: str) -> Path:
    """Resolve ``config/committees/<base>_<domain>.yaml`` (or unsuffixed for companies)."""
    committee_dir = REPO_ROOT / "usecases_synthetic" / "config" / "committees"
    suffix = "" if domain == "companies" else f"_{domain}"
    return committee_dir / f"{base}{suffix}.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _domain_context(domain: str) -> dict[str, Any]:
    """Load + column-map baseline sources for ``domain``."""
    bundle = load_variant(domain, level="baseline")
    em_yaml = _committee_yaml_path(domain, "em_matching_committee")
    blocking_yaml = _committee_yaml_path(domain, "em_blocking_committee")
    em_cfg = _load_yaml(em_yaml)
    blocking_cfg = _load_yaml(blocking_yaml)
    column_mapping = bundle.resolve_column_mapping(em_cfg.get("column_mapping", {}))
    sources_mapped: dict[str, pd.DataFrame] = {}
    for src_name, df in bundle.sources.items():
        rename_map = column_mapping.get(src_name, {})
        sources_mapped[src_name] = (
            apply_column_mapping(df, rename_map) if rename_map else df.copy()
        )
    em_dir = bundle.variant_root / "input" / "entitymatching"
    return {
        "domain": domain,
        "bundle": bundle,
        "sources": sources_mapped,
        "em_gold": bundle.em_gold,
        "em_gold_regenerated": bundle.em_gold_regenerated,
        "source_pairs": bundle.source_pairs,
        "column_mapping": column_mapping,
        "em_dir": em_dir,
        "em_matching_cfg": em_cfg,
        "em_blocking_cfg": blocking_cfg,
    }


def _resolve_train_path(em_dir: Path, pair: tuple[str, str]) -> Path | None:
    """Mirror committee_em._resolve_pair_train_path."""
    src1, src2 = pair
    candidates = [
        em_dir / f"{src1}_2_{src2}_train.csv",
        em_dir / f"{src2}_2_{src1}_train.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Candidate pre-blocking
# ---------------------------------------------------------------------------


def _import_class(module: str, cls: str):
    mod = importlib.import_module(module)
    return getattr(mod, cls)


def _derive_blocking_keys(df: pd.DataFrame, blocking_cfg: dict[str, Any]) -> None:
    """Add derived blocking-key columns to *df* in place (mirrors runner)."""
    # Locate any standard_blocker `on` keys + sorted_neighbourhood `key`.
    from usecases_synthetic.lib.committee_em import _generate_blocking_keys

    keys: set[str] = set()
    name_col = "name"
    for member in blocking_cfg.get("members", []):
        if not member.get("enabled_by_default", False):
            continue
        blocker = member.get("blocker", {})
        cls = blocker.get("class")
        params = blocker.get("params", {}) or {}
        if cls == "StandardBlocker":
            on = params.get("on", [])
            if isinstance(on, str):
                on = [on]
            keys.update(str(k) for k in on)
        elif cls == "SortedNeighbourhoodBlocker":
            k = params.get("key")
            if k:
                keys.add(str(k))
    if not keys:
        return
    _generate_blocking_keys(df, column=name_col, required_keys=sorted(keys))


def _pre_block_pair(
    ctx: dict[str, Any],
    pair: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the locked blocker winner once per pair; return (df_l, df_r, candidates).

    Picks the highest-priority blocker that clears the recall floor on
    this pair against EM gold. For the sweep we use a fast deterministic
    blocker — ``StandardBlocker`` on the per-domain key (which already
    won at R5 EM blocking sub-B). Embedding-based blockers are skipped
    here for sweep speed; the classifier scoring is stable across
    candidate-set sources.
    """
    from usecases_synthetic.lib.committee_em import _generate_blocking_keys

    src1, src2 = pair
    df_left = ctx["sources"][src1].copy()
    df_right = ctx["sources"][src2].copy()
    _derive_blocking_keys(df_left, ctx["em_blocking_cfg"])
    _derive_blocking_keys(df_right, ctx["em_blocking_cfg"])

    # Find the standard_blocker member spec to reuse its `on` key.
    blocking_cfg = ctx["em_blocking_cfg"]
    std_member = next(
        (
            m
            for m in blocking_cfg.get("members", [])
            if m.get("blocker", {}).get("class") == "StandardBlocker"
            and m.get("enabled_by_default", False)
        ),
        None,
    )
    if std_member is None:
        raise RuntimeError(
            f"No enabled StandardBlocker in {ctx['domain']!r} em_blocking_committee"
        )
    cls = _import_class(std_member["blocker"]["module"], std_member["blocker"]["class"])
    params = dict(std_member["blocker"].get("params", {}))
    blocker = cls(df_left, df_right, id_column="id", **params)
    candidates = blocker.materialize()
    return df_left, df_right, candidates


# ---------------------------------------------------------------------------
# Magellan sweep
# ---------------------------------------------------------------------------


def _magellan_grid() -> list[dict[str, Any]]:
    """Classifier-only sweep grid; auto-feature-gen handles the rest."""
    grid: list[dict[str, Any]] = []
    for n_est, max_depth, class_weight in itertools.product(
        [100, 300],
        [10, 20, None],
        [None, "balanced"],
    ):
        grid.append(
            {
                "n_estimators": n_est,
                "max_depth": max_depth,
                "class_weight": class_weight,
            }
        )
    return grid


def _config_id(cfg: dict[str, Any]) -> str:
    """Stable string id for a sweep cell."""
    return "_".join(f"{k}={cfg[k]}" for k in sorted(cfg))


def _score_magellan_cell(
    ctx: dict[str, Any],
    pair: tuple[str, str],
    classifier_params: dict[str, Any],
    member_yaml: dict[str, Any],
) -> dict[str, float]:
    """Train + score Magellan on a single (pair, classifier_combo) cell.

    Sweep-fast path: score the matcher on **EM gold pairs only** (closed-
    set semantic) instead of running it across the full blocker
    candidate set. The relative ranking of classifier configs is
    preserved under closed-set scoring; the full-candidate open-set
    measurement runs at R6.1 baseline (``measure_baseline.py``), not in
    the sweep. This is ~100× faster on large domains (games / music)
    because feature extraction only runs on a few hundred gold pairs
    instead of the ~50-100k blocker output.
    """
    from usecases_synthetic.lib.magellan_em_matcher import MagellanMatcher

    train_path = _resolve_train_path(ctx["em_dir"], pair)
    gold = ctx["em_gold"].get(pair)
    if train_path is None or gold is None or gold.empty:
        return {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}
    df_left = ctx["sources"][pair[0]]
    df_right = ctx["sources"][pair[1]]
    member_params = dict(member_yaml.get("params", {}) or {})
    matcher = MagellanMatcher(
        training_gold_path=str(train_path),
        numeric_attributes=member_params.get("numeric_attributes", []),
        date_attributes=member_params.get("date_attributes", []),
        classifier_class=member_params.get(
            "classifier_class", "sklearn.ensemble.RandomForestClassifier"
        ),
        classifier_params={
            **(member_params.get("classifier_params") or {}),
            **classifier_params,
            "random_state": 42,
            "n_jobs": 1,
        },
        preprocess=member_params.get("preprocess"),
        seed=42,
    )
    # Score on the gold pair set directly (id1, id2 only — labels stay
    # in the gold frame for scoring). Closed-set semantic: predictions
    # outside the gold universe are out-of-scope FPs by construction.
    gold_pairs = gold[["id1", "id2"]].copy()
    preds = matcher.match(
        df_left,
        df_right,
        gold_pairs,
        id_column="id",
        threshold=member_yaml.get("threshold", 0.5),
    )
    return score_em_correspondences_closed_set(preds, gold)


def _run_magellan_sweep(contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the Magellan classifier sweep across all contexts."""
    grid = _magellan_grid()
    rows: list[dict[str, Any]] = []
    for domain, ctx in contexts.items():
        # Locate the magellan member spec in the per-domain YAML.
        magellan_member = next(
            (m for m in ctx["em_matching_cfg"]["members"] if m["name"] == "magellan"),
            None,
        )
        if magellan_member is None:
            logger.warning("No magellan member in %s em_matching cfg", domain)
            continue
        for cfg in grid:
            t0 = time.monotonic()
            per_pair: dict[str, dict[str, float]] = {}
            for pair in ctx["source_pairs"]:
                metrics = _score_magellan_cell(ctx, pair, cfg, magellan_member)
                per_pair[f"{pair[0]}_{pair[1]}"] = metrics
            elapsed = time.monotonic() - t0
            valid_f1s = [m["f1"] for m in per_pair.values() if not _is_nan(m["f1"])]
            row = {
                "member": "magellan",
                "domain": domain,
                "config": cfg,
                "config_id": _config_id(cfg),
                "per_pair": per_pair,
                "mean_f1": (
                    float(sum(valid_f1s) / len(valid_f1s))
                    if valid_f1s
                    else float("nan")
                ),
                "min_f1": float(min(valid_f1s)) if valid_f1s else float("nan"),
                "n_pairs": len(valid_f1s),
                "runtime_s": elapsed,
            }
            rows.append(row)
            logger.info(
                "magellan | %s | %s | mean_f1=%.3f min_f1=%.3f n=%d (%.1fs)",
                domain,
                row["config_id"],
                row["mean_f1"],
                row["min_f1"],
                row["n_pairs"],
                elapsed,
            )
    return rows


def _is_nan(x: Any) -> bool:
    try:
        return x != x  # NaN trick
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------


def _pick_winner(
    rows: list[dict[str, Any]], member: str, domain: str
) -> dict[str, Any]:
    """Highest mean_f1 across the domain; tie-break by min_f1, then config_id."""
    domain_rows = [r for r in rows if r["member"] == member and r["domain"] == domain]
    if not domain_rows:
        return {}
    domain_rows.sort(
        key=lambda r: (
            -r["mean_f1"] if not _is_nan(r["mean_f1"]) else 0.0,
            -r["min_f1"] if not _is_nan(r["min_f1"]) else 0.0,
            r["config_id"],
        )
    )
    return domain_rows[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        default="companies,games,music",
        help="Comma-separated domain list (default: companies,games,music).",
    )
    parser.add_argument(
        "--members",
        default="magellan",
        help=(
            "Comma-separated member list. "
            "Only `magellan` is swept (the other 3 are locked per R5 sign-off)."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(CACHE_DIR / "sweep.json"),
        help="Output path for sweep results.",
    )
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    members = {m.strip() for m in args.members.split(",") if m.strip()}

    contexts: dict[str, dict[str, Any]] = {}
    for d in domains:
        logger.info("Loading context for %s", d)
        contexts[d] = _domain_context(d)

    all_rows: list[dict[str, Any]] = []
    if "magellan" in members:
        all_rows.extend(_run_magellan_sweep(contexts))
    skipped_members = members - {"magellan"}
    if skipped_members:
        logger.warning(
            "Skipping non-swept members per R5 lock: %s "
            "(ditto/matchgpt/comem are locked, see R5 EM matching sign-off)",
            sorted(skipped_members),
        )

    # Winner per (member, domain)
    winners: dict[str, dict[str, Any]] = {}
    for member in members:
        for d in domains:
            w = _pick_winner(all_rows, member, d)
            if w:
                winners[f"{member}::{d}"] = w

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": all_rows,
        "winners": winners,
        "schema_version": "v1",
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(
        "Wrote %d rows + %d winners to %s", len(all_rows), len(winners), out_path
    )


if __name__ == "__main__":
    main()

"""Schema-matching committee runner.

Instantiates each matcher in the SM roster YAML, runs it against every
source in a :class:`VariantBundle`, scores each matcher's output against
the SM ground-truth mapping, and returns a :class:`CommitteeResult`.

The runner calls ``BaseSchemaMatcher.match(source_df, target_df, ...)``
once per source per matcher.  The target DataFrame is synthesised from
the variant's ``target_schema`` JSON Schema (one column per property,
zero rows — the matchers only need the column names and optionally the
values for instance-based approaches).

For the ``duplicate_based`` matcher the runner passes the first
available EM correspondence DataFrame as the ``correspondences`` kwarg.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from PyDI.schemamatching.base import BaseSchemaMatcher, SchemaMapping
from PyDI.schemamatching.evaluation import SchemaMappingEvaluator

from .committee import CommitteeResult, CommitteeRunner, MemberResult
from .validation_metrics import precision_recall_f1
from .variant_loader import VariantBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Roster member spec
# ---------------------------------------------------------------------------


@dataclass
class _SMRosterMember:
    """Parsed representation of a single SM roster entry."""

    name: str
    module: str
    cls_name: str
    signal_type: str
    enabled_by_default: bool
    params: dict[str, Any]
    match_kwargs: dict[str, Any]


def _parse_roster(
    raw_members: list[dict[str, Any]],
    *,
    with_llm: bool = False,
) -> list[_SMRosterMember]:
    """Parse and filter the YAML roster.

    Parameters
    ----------
    raw_members : list of dict
        Raw member dicts from the YAML.
    with_llm : bool
        When ``False`` (default), members with ``signal_type == "llm"``
        are excluded even if ``enabled_by_default`` is ``True``.

    Returns
    -------
    list of _SMRosterMember
        Enabled roster members.
    """
    out: list[_SMRosterMember] = []
    for entry in raw_members:
        enabled = entry.get("enabled_by_default", True)
        is_llm = entry.get("signal_type") == "llm"

        # LLM members respect enabled_by_default like any other member.
        # The with_llm flag force-enables disabled LLM members.
        if is_llm and not enabled and not with_llm:
            continue
        if not is_llm and not enabled:
            continue

        out.append(
            _SMRosterMember(
                name=entry["name"],
                module=entry["module"],
                cls_name=entry["class"],
                signal_type=entry["signal_type"],
                enabled_by_default=enabled,
                params=dict(entry.get("params", {}) or {}),
                match_kwargs=dict(entry.get("match_kwargs", {}) or {}),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Matcher instantiation
# ---------------------------------------------------------------------------


def _instantiate_matcher(
    spec: _SMRosterMember,
) -> BaseSchemaMatcher:
    """Dynamically import and instantiate a schema matcher.

    For LLM-based matchers (``signal_type == "llm"``), the ``model_name``
    param is popped and used to construct a LangChain ``ChatOpenAI``
    instance which is passed as the ``chat_model`` argument.

    Parameters
    ----------
    spec : _SMRosterMember
        Roster entry describing which class to import and init params.

    Returns
    -------
    BaseSchemaMatcher
        Instantiated matcher.
    """
    mod = importlib.import_module(spec.module)
    cls = getattr(mod, spec.cls_name)
    params = dict(spec.params)

    if spec.signal_type == "llm":
        model_name = params.pop("model_name", "gpt-5.4-mini")
        from langchain_openai import ChatOpenAI

        chat_model = ChatOpenAI(
            model=model_name,
            temperature=params.pop("temperature", 0.0),
        )
        params["chat_model"] = chat_model

    return cls(**params)


# ---------------------------------------------------------------------------
# Target DataFrame construction
# ---------------------------------------------------------------------------


def _target_df_from_schema(
    target_schema: dict[str, Any],
    sources: dict[str, pd.DataFrame],
    *,
    target_name: str | None = None,
    fusion_frames: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build a target reference DataFrame from a JSON Schema.

    The resulting DataFrame has one column per ``properties`` key.
    When *fusion_frames* are provided, each schema property whose name
    appears as a column in any fusion frame is populated with that
    column's non-null values (concatenated across frames). Properties
    not present in any fusion frame remain empty. ``attrs[
    "dataset_name"]`` is set to *target_name* if provided, otherwise to
    the schema title (lower-cased) or ``"target"``.

    Populating from fusion val/test (rather than source DataFrames)
    avoids a leak that would let instance-based matchers trivially
    pair source columns with target columns containing the same source
    values. Fusion val/test ship the canonical reference values for the
    K10-eligible attributes; properties outside that set keep empty
    target columns and get no signal from instance-based matchers
    (only label-/embedding-/correspondence-based members vote on them).

    Parameters
    ----------
    target_schema : dict
        Parsed ``target_schema.json`` (JSON Schema draft 2020-12).
    sources : dict of str to DataFrame
        Source DataFrames. Currently only used to derive a sensible
        column ordering when needed; values are not copied into the
        target (see leak avoidance above).
    target_name : str, optional
        Explicit dataset name for the target DataFrame. When ``None``
        the schema title is used.
    fusion_frames : list of DataFrame, optional
        Fusion val + test reference frames (each carries one column per
        attribute alongside ``*_provenance`` columns; the latter are
        ignored). When ``None`` the target keeps zero rows.

    Returns
    -------
    DataFrame
        Reference target DataFrame.
    """
    properties = target_schema.get("properties", {})
    columns = list(properties.keys())

    column_values: dict[str, list[Any]] = {col: [] for col in columns}

    if fusion_frames:
        for frame in fusion_frames:
            if frame is None or frame.empty:
                continue
            for col in columns:
                if col not in frame.columns:
                    continue
                series = frame[col].dropna()
                if series.empty:
                    continue
                column_values[col].extend(series.astype(str).tolist())

    max_rows = max((len(vs) for vs in column_values.values()), default=0)
    if max_rows == 0:
        target = pd.DataFrame(columns=columns)
    else:
        padded = {
            col: vs + [None] * (max_rows - len(vs)) for col, vs in column_values.items()
        }
        target = pd.DataFrame(padded, columns=columns)

    if target_name is not None:
        target.attrs["dataset_name"] = target_name
    else:
        title = target_schema.get("title", "target")
        target.attrs["dataset_name"] = title.lower().replace(" ", "_")

    return target


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _mapping_tuples(
    mapping: pd.DataFrame,
) -> set[tuple[str, str, str, str]]:
    """Extract ``(source_dataset, source_column, target_dataset, target_column)``
    tuples from a mapping DataFrame.

    Parameters
    ----------
    mapping : DataFrame
        Mapping with columns ``source_dataset``, ``source_column``,
        ``target_dataset``, ``target_column``.

    Returns
    -------
    set of tuple
        Unique mapping tuples.
    """
    return set(
        zip(
            mapping["source_dataset"],
            mapping["source_column"],
            mapping["target_dataset"],
            mapping["target_column"],
            strict=True,
        )
    )


def score_sm_mapping(
    predicted: pd.DataFrame,
    gold: pd.DataFrame,
) -> dict[str, float]:
    """Score a predicted SM mapping against the gold standard.

    Uses set-based precision / recall / F1 over
    ``(source_dataset, source_column, target_dataset, target_column)``
    tuples.  Both directions are considered equivalent (symmetric).

    Parameters
    ----------
    predicted : DataFrame
        Matcher output — a mapping DataFrame.
    gold : DataFrame
        Gold-standard mapping.

    Returns
    -------
    dict[str, float]
        Keys ``"precision"``, ``"recall"``, ``"f1"``, ``"tp"``,
        ``"fp"``, ``"fn"``.
    """
    if predicted.empty:
        gold_size = len(_mapping_tuples(gold)) if not gold.empty else 0
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": float(gold_size),
        }

    pred_tuples = _mapping_tuples(predicted)
    gold_tuples = _mapping_tuples(gold)

    return precision_recall_f1(pred_tuples, gold_tuples)


def score_sm_per_attribute(
    predicted: pd.DataFrame,
    gold: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Score SM mapping per source column.

    For each ``(source_dataset, source_column)`` in the gold, check
    whether the predicted mapping maps it to the correct target column.

    Parameters
    ----------
    predicted : DataFrame
        Matcher output mapping.
    gold : DataFrame
        Gold-standard mapping.

    Returns
    -------
    dict[str, dict[str, float]]
        Outer key: ``"<source_dataset>.<source_column>"``.
        Inner dict: ``{"correct": 0|1, "predicted_target": str|None,
        "gold_target": str}``.
    """
    result: dict[str, dict[str, float]] = {}

    # Index predicted by (source_dataset, source_column) → target_column
    pred_index: dict[tuple[str, str], str] = {}
    if not predicted.empty:
        for _, row in predicted.iterrows():
            key = (str(row["source_dataset"]), str(row["source_column"]))
            pred_index[key] = str(row["target_column"])

    for _, row in gold.iterrows():
        src_ds = str(row["source_dataset"])
        src_col = str(row["source_column"])
        tgt_col = str(row["target_column"])
        key = (src_ds, src_col)
        attr_key = f"{src_ds}.{src_col}"

        predicted_target = pred_index.get(key)
        correct = 1.0 if predicted_target == tgt_col else 0.0

        result[attr_key] = {"correct": correct, "f1": correct}

    return result


# ---------------------------------------------------------------------------
# SM Committee Runner
# ---------------------------------------------------------------------------


class SMCommitteeRunner(CommitteeRunner):
    """Schema-matching committee runner.

    Loads the roster from a YAML file, instantiates each enabled
    matcher, runs it against every source in a ``VariantBundle``, and
    scores each matcher's mapping output against the SM ground truth.

    Parameters
    ----------
    roster_path : Path
        Path to ``sm_committee.yaml``.
    with_llm : bool
        Enable LLM-based matcher(s) from the roster. Default ``False``.
    """

    stage: Literal["sm"] = "sm"

    def __init__(
        self,
        roster_path: Path,
        *,
        with_llm: bool = False,
    ) -> None:
        raw = _load_roster_yaml(roster_path)
        members = _parse_roster(raw["members"], with_llm=with_llm)
        self._specs = members
        self._seed = raw.get("seed", 42)

        # Instantiate matchers eagerly so import errors surface early.
        self._matchers: list[BaseSchemaMatcher] = [
            _instantiate_matcher(spec) for spec in members
        ]

        # Pass specs as roster to the base; override roster_names.
        super().__init__(
            roster=list(self._matchers),
            config={"seed": self._seed, "with_llm": with_llm},
        )

    @property
    def roster_names(self) -> list[str]:
        """Return member names in declaration order."""
        return [spec.name for spec in self._specs]

    def run(self, bundle: VariantBundle) -> CommitteeResult:
        """Run every SM roster member and aggregate.

        Parameters
        ----------
        bundle : VariantBundle
            Loaded variant (baseline or augmented).

        Returns
        -------
        CommitteeResult
            Per-member and aggregated SM metrics.
        """
        gold = bundle.sm_mapping
        if gold is None:
            raise ValueError(
                f"No SM gold mapping for {bundle.domain}/{bundle.level}. "
                "Ensure sm_mapping_gold.csv (baseline) or sm_mapping.csv "
                "(variant) exists."
            )

        # Extract the target dataset name from the gold mapping so the
        # matchers produce predictions with the same target_dataset value.
        gold_target_name: str | None = None
        if "target_dataset" in gold.columns and not gold.empty:
            gold_target_name = str(gold["target_dataset"].iloc[0])

        fusion_frames: list[pd.DataFrame] = []
        if bundle.fusion_validation is not None:
            fusion_frames.append(bundle.fusion_validation)
        if bundle.fusion_gold is not None:
            fusion_frames.append(bundle.fusion_gold)

        target_df = _target_df_from_schema(
            bundle.target_schema,
            bundle.sources,
            target_name=gold_target_name,
            fusion_frames=fusion_frames or None,
        )

        per_member: dict[str, MemberResult] = {}
        all_per_attr: dict[str, dict[str, float]] = {}
        t0_total = time.monotonic()

        for spec, matcher in zip(self._specs, self._matchers):
            t0 = time.monotonic()

            # Duplicate-based matchers take a different call shape: they
            # need (source1_df, source2_df) with EM correspondences linking
            # them, not (source_df, target_reference_df). Special-case the
            # signal_type and dispatch per source-pair (Option A from
            # plan_s1_scale.md §"R5 SM duplicate-matcher fix"). The
            # cross-source predictions are then translated to canonical
            # source→target mappings via the SM gold lookup so the standard
            # scoring helper applies.
            if spec.signal_type == "duplicate":
                combined = self._run_duplicate_per_pair(matcher, spec, bundle, gold)
            else:
                # Run matcher against every source, collect all mappings.
                all_mappings: list[pd.DataFrame] = []
                for source_name, source_df in bundle.sources.items():
                    match_kwargs = dict(spec.match_kwargs)

                    try:
                        mapping = matcher.match(source_df, target_df, **match_kwargs)
                        all_mappings.append(mapping)
                    except Exception as exc:
                        logger.exception(
                            "Matcher %s failed on source %s",
                            spec.name,
                            source_name,
                        )
                        # Silent fallback historically masked silent
                        # zero-scores in committee aggregation. Re-raise
                        # so any matcher fault is loud; explicitly
                        # disable the matcher in sm_committee.yaml
                        # (or its per-domain fork) when it should not
                        # contribute on a given domain.
                        raise RuntimeError(
                            f"SM matcher {spec.name!r} failed on source "
                            f"{source_name!r}; disable it explicitly in "
                            f"sm_committee.yaml if it should not run."
                        ) from exc

                if all_mappings:
                    combined = pd.concat(all_mappings, ignore_index=True)
                else:
                    combined = pd.DataFrame(
                        columns=[
                            "source_dataset",
                            "source_column",
                            "target_dataset",
                            "target_column",
                            "score",
                        ]
                    )

            elapsed = time.monotonic() - t0

            # Score against gold.
            metrics = score_sm_mapping(combined, gold)
            per_attr = score_sm_per_attribute(combined, gold)

            per_member[spec.name] = MemberResult(
                name=spec.name,
                predictions=combined,
                metrics=metrics,
                runtime_s=elapsed,
                notes={"signal_type": spec.signal_type},
            )

            # Merge per-attribute results (track per-member).
            for attr_key, attr_metrics in per_attr.items():
                if attr_key not in all_per_attr:
                    all_per_attr[attr_key] = {}
                all_per_attr[attr_key][spec.name] = attr_metrics.get("correct", 0.0)

        total_runtime = time.monotonic() - t0_total

        # Aggregated metrics.
        f1_values = [m.metrics["f1"] for m in per_member.values()]
        aggregated = _compute_aggregated(f1_values, per_member)

        # Per-attribute: add "any_correct" — was this column mapped
        # correctly by at least one member?
        per_attribute_out: dict[str, dict[str, float]] = {}
        for attr_key, member_scores in all_per_attr.items():
            entry: dict[str, float] = dict(member_scores)
            entry["any_correct"] = (
                1.0 if any(v >= 1.0 for v in member_scores.values()) else 0.0
            )
            per_attribute_out[attr_key] = entry

        # Per-partition: per-source rollup.
        per_partition = _per_source_rollup(per_member, gold)

        return CommitteeResult(
            stage="sm",
            domain=bundle.domain,
            level=bundle.level,
            per_member=per_member,
            aggregated=aggregated,
            per_attribute=per_attribute_out,
            per_partition=per_partition,
            runtime_s=total_runtime,
            roster=self.roster_names,
        )

    def _run_duplicate_per_pair(
        self,
        matcher: BaseSchemaMatcher,
        spec: _SMRosterMember,
        bundle: VariantBundle,
        gold: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run a duplicate-based matcher per source-pair (Option A).

        For each ordered ``(srcA, srcB)`` pair in ``bundle.source_pairs``,
        filter the EM gold to label-positive correspondences and call
        ``matcher.match(srcA_df, srcB_df, correspondences=positives)``.
        The matcher emits cross-source predictions
        ``(srcA, colX, srcB, colY, score)``; we translate each side to a
        canonical source→target tuple via the SM gold lookup so the
        standard ``score_sm_mapping`` helper applies.

        Translation note: looking up the predicted columns in the SM
        gold means the matcher's reported precision is structurally
        bounded near 1.0 (every emitted target column is gold-derived).
        Recall is the meaningful signal — fraction of in-gold source
        columns the matcher confirmed via at least one cross-source
        equivalence. Documented in plan_s1_scale.md §"R5 SM
        duplicate-matcher fix".
        """
        # Build gold lookup keyed by (source_dataset, source_column).
        gold_lookup: dict[tuple[str, str], tuple[str, str]] = {}
        for _, row in gold.iterrows():
            key = (str(row["source_dataset"]), str(row["source_column"]))
            gold_lookup[key] = (
                str(row["target_dataset"]),
                str(row["target_column"]),
            )

        all_mappings: list[pd.DataFrame] = []
        for src1_name, src2_name in bundle.source_pairs:
            em_pair = bundle.em_gold.get((src1_name, src2_name))
            if em_pair is None or em_pair.empty:
                continue

            # Filter to label-positive correspondences only.
            label_col = "label" if "label" in em_pair.columns else None
            if label_col is not None:
                truthy = em_pair[label_col].astype(str).str.lower()
                em_pos = em_pair[truthy.isin(("true", "1", "yes"))]
            else:
                em_pos = em_pair
            if em_pos.empty:
                continue

            src1_df = bundle.sources.get(src1_name)
            src2_df = bundle.sources.get(src2_name)
            if src1_df is None or src2_df is None:
                continue

            match_kwargs = dict(spec.match_kwargs)
            match_kwargs["correspondences"] = em_pos

            try:
                cross = matcher.match(src1_df, src2_df, **match_kwargs)
            except Exception as exc:
                logger.exception(
                    "Duplicate matcher %s failed on %s ↔ %s",
                    spec.name,
                    src1_name,
                    src2_name,
                )
                # Re-raise instead of silently skipping the pair
                # (historical behaviour masked silent zero-scores in
                # committee aggregation). Disable the matcher in
                # sm_committee.yaml if it should not run on this domain.
                raise RuntimeError(
                    f"SM duplicate matcher {spec.name!r} failed on pair "
                    f"{src1_name!r} <-> {src2_name!r}; disable it "
                    f"explicitly in sm_committee.yaml if it should not run."
                ) from exc

            if cross is None or cross.empty:
                continue

            translated = _translate_cross_source_to_target(cross, gold_lookup)
            if not translated.empty:
                all_mappings.append(translated)

        if all_mappings:
            return pd.concat(all_mappings, ignore_index=True).drop_duplicates(
                subset=[
                    "source_dataset",
                    "source_column",
                    "target_dataset",
                    "target_column",
                ]
            )
        return pd.DataFrame(
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
            ]
        )


def _translate_cross_source_to_target(
    cross: pd.DataFrame,
    gold_lookup: dict[tuple[str, str], tuple[str, str]],
) -> pd.DataFrame:
    """Translate cross-source predictions to canonical source→target tuples.

    For each predicted ``(srcA, colX, srcB, colY)``, emit
    ``(srcA, colX, gold_target_dataset, gold_target_column)`` and
    ``(srcB, colY, gold_target_dataset, gold_target_column)`` whenever
    each side has a gold entry. Scores are propagated; predictions
    without a gold lookup are dropped.
    """
    rows: list[dict[str, Any]] = []
    for _, r in cross.iterrows():
        srcA = str(r["source_dataset"])
        colX = str(r["source_column"])
        srcB = str(r["target_dataset"])
        colY = str(r["target_column"])
        score = float(r.get("score", 0.0))
        notes = str(r.get("notes", "")) if "notes" in cross.columns else ""

        for src, col in ((srcA, colX), (srcB, colY)):
            target = gold_lookup.get((src, col))
            if target is None:
                continue
            rows.append(
                {
                    "source_dataset": src,
                    "source_column": col,
                    "target_dataset": target[0],
                    "target_column": target[1],
                    "score": score,
                    "notes": (
                        f"{notes};translated_from_cross_source"
                        if notes
                        else "translated_from_cross_source"
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
                "notes",
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_roster_yaml(path: Path) -> dict[str, Any]:
    """Load and return the roster YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _first_em_correspondences(
    bundle: VariantBundle,
) -> pd.DataFrame | None:
    """Return the first available EM correspondence set, or ``None``.

    The duplicate-based matcher needs known entity correspondences to
    vote on column alignment.  We pass the first source pair's gold.

    Parameters
    ----------
    bundle : VariantBundle
        The loaded variant.

    Returns
    -------
    DataFrame or None
        Correspondence DataFrame with ``id1``, ``id2`` columns.
    """
    for pair, gold_df in bundle.em_gold.items():
        if not gold_df.empty:
            return gold_df
    return None


def _compute_aggregated(
    f1_values: list[float],
    per_member: dict[str, MemberResult],
) -> dict[str, float | str]:
    """Compute aggregated committee metrics.

    Parameters
    ----------
    f1_values : list of float
        Per-member F1 values.
    per_member : dict
        Per-member results.

    Returns
    -------
    dict[str, float | str]
        Aggregated metric dict with ``macro_f1``, ``min_f1``,
        ``max_f1``, ``macro_precision``, ``macro_recall``,
        ``best_member_name``, ``best_member_f1``.
    """
    n = len(f1_values)
    if n == 0:
        return {
            "macro_f1": 0.0,
            "min_f1": 0.0,
            "max_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "best_member_name": "",
            "best_member_f1": 0.0,
        }

    precision_values = [m.metrics.get("precision", 0.0) for m in per_member.values()]
    recall_values = [m.metrics.get("recall", 0.0) for m in per_member.values()]

    # Promote the single best member (highest f1) so SM reports a best
    # member alongside the committee macro, consistent with the norm /
    # em_blocking / em_matching stages (committee-reporting convention).
    # per_member already carries the full breakdown.
    best_member_name, best_member_f1 = max(
        ((name, m.metrics.get("f1", 0.0)) for name, m in per_member.items()),
        key=lambda kv: kv[1],
        default=("", 0.0),
    )

    return {
        "macro_f1": sum(f1_values) / n,
        "min_f1": min(f1_values),
        "max_f1": max(f1_values),
        "macro_precision": sum(precision_values) / n,
        "macro_recall": sum(recall_values) / n,
        "best_member_name": best_member_name,
        "best_member_f1": best_member_f1,
    }


def _per_source_rollup(
    per_member: dict[str, MemberResult],
    gold: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute per-source macro F1 across members.

    For each source in the gold, filter each member's predictions to
    that source and compute F1.  Returns the macro average across
    members per source.

    Parameters
    ----------
    per_member : dict
        Per-member results with ``predictions`` as mapping DataFrames.
    gold : DataFrame
        Gold-standard mapping.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{source_name: {"macro_f1": float, "n_columns": int}}``.
    """
    sources = gold["source_dataset"].unique().tolist()
    result: dict[str, dict[str, float]] = {}

    for source in sources:
        gold_src = gold[gold["source_dataset"] == source]
        gold_tuples = _mapping_tuples(gold_src)
        n_cols = len(gold_tuples)

        member_f1s: list[float] = []
        for member in per_member.values():
            pred = member.predictions
            if pred is not None and not pred.empty:
                pred_src = pred[pred["source_dataset"] == source]
            else:
                pred_src = pd.DataFrame(
                    columns=[
                        "source_dataset",
                        "source_column",
                        "target_dataset",
                        "target_column",
                        "score",
                    ]
                )
            pred_tuples = _mapping_tuples(pred_src) if not pred_src.empty else set()
            metrics = precision_recall_f1(pred_tuples, gold_tuples)
            member_f1s.append(metrics["f1"])

        macro = sum(member_f1s) / len(member_f1s) if member_f1s else 0.0
        result[source] = {"macro_f1": macro, "n_columns": float(n_cols)}

    return result

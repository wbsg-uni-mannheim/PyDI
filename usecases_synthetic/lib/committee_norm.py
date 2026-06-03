"""Normalization committee runner.

Sits between SM and EM in the synthetic-variant pipeline. Each member
maps a per-source-cell raw value to a canonical form; the runner scores
every member's output for each fusion-protected (entity, canonical
attribute) cell against the fusion val/test reference value via the
Pending #5 closeness contract (:func:`protection.is_close_enough`).

Per-(member, attribute) F1 is the headline metric. Macro F1 across
attributes per member is reported via
:meth:`MemberPerAttributeScores.macro_metrics`; the committee's
``aggregated`` block carries cross-member macro F1 and the best-member
F1 (Pending #8 ceiling).

Roster YAML shape mirrors the SM committee — see
``config/committees/normalization_committee_<domain>.yaml`` for the
reference layout. Per-domain forks rather than a shared file so each
domain authors its own per-attribute strategy block (per the 2026-05-10
sign-off).
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd
import yaml

from .committee import CommitteeResult, CommitteeRunner, MemberResult
from .committee_norm_scoring import (
    AttributeScore,
    MemberPerAttributeScores,
)
from .protection import (
    ToleranceSpec,
    fusion_cell_tolerance,
    kind_map_for_domain,
    load_fusion_target_values,
)
from .variant_loader import VariantBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Roster member spec
# ---------------------------------------------------------------------------


@runtime_checkable
class _NormalizerLike(Protocol):
    name: str

    def normalize(
        self,
        value: Any,
        *,
        attribute: str,
        kind: str,
        domain: str,
    ) -> str | None: ...


@dataclass
class _NormRosterMember:
    """Parsed representation of a single normalization roster entry."""

    name: str
    module: str
    cls_name: str
    signal_type: str
    enabled_by_default: bool
    params: dict[str, Any]
    applies_to: list[str] | None  # None = all attributes


def _parse_roster(
    raw_members: list[dict[str, Any]],
    *,
    with_llm: bool = False,
) -> list[_NormRosterMember]:
    """Parse + filter the YAML roster.

    LLM members (``signal_type == "llm"``) are excluded by default
    unless *with_llm* is True (matches SM-runner conventions).
    """
    out: list[_NormRosterMember] = []
    for entry in raw_members:
        enabled = entry.get("enabled_by_default", True)
        is_llm = entry.get("signal_type") == "llm"
        if is_llm and not enabled and not with_llm:
            continue
        if not is_llm and not enabled:
            continue
        applies_to = entry.get("applies_to")
        if applies_to is not None and not isinstance(applies_to, list):
            raise ValueError(
                f"Roster entry {entry.get('name')!r}: applies_to must be a list."
            )
        out.append(
            _NormRosterMember(
                name=entry["name"],
                module=entry["module"],
                cls_name=entry["class"],
                signal_type=entry["signal_type"],
                enabled_by_default=enabled,
                params=dict(entry.get("params", {}) or {}),
                applies_to=list(applies_to) if applies_to else None,
            )
        )
    return out


def _instantiate_member(spec: _NormRosterMember) -> _NormalizerLike:
    """Dynamically import + instantiate a normalization member."""
    mod = importlib.import_module(spec.module)
    cls = getattr(mod, spec.cls_name)
    params = dict(spec.params)
    member = cls(name=spec.name, **params)
    if not isinstance(member, _NormalizerLike):
        raise TypeError(
            f"Member {spec.name} ({spec.cls_name}) does not implement the "
            "BaseNormalizer protocol (missing .normalize)."
        )
    return member


# ---------------------------------------------------------------------------
# SM mapping resolution
# ---------------------------------------------------------------------------


def _build_entity_linkage(
    bundle: VariantBundle,
) -> dict[str, dict[str, str]]:
    """Build ``{fusion_entity_id: {source_name: source_record_id}}`` from EM gold.

    Fusion target IDs use one source's ID convention (e.g. forbes URLs
    for companies, musicbrainz IDs for music). To score values from
    *other* sources at the same conceptual entity, we walk the EM gold
    label-positive correspondences and group by the fusion-target-style
    side (``id1``).

    Source resolution: ``id_prefix`` from the per-source ``DataFrame.attrs``
    is used to figure out which prefix maps to which source. When the
    bundle's source DFs don't carry attrs (lightweight test fixtures),
    we fall back to scanning each source's ``id`` column to detect the
    matching prefix.
    """
    linkage: dict[str, dict[str, str]] = {}
    if not bundle.em_gold:
        return linkage

    # Build source id-prefix → source-name map. Prefer DataFrame.attrs;
    # fall back to per-source id-prefix detection.
    prefix_to_source: dict[str, str] = {}
    for src_name, src_df in bundle.sources.items():
        prefix = src_df.attrs.get("id_prefix") if hasattr(src_df, "attrs") else None
        if not prefix and "id" in src_df.columns and not src_df.empty:
            sample_id = str(src_df["id"].iloc[0])
            # Heuristic: take the longest common prefix across the first
            # 5 ids as the source prefix.
            sample = [str(v) for v in src_df["id"].head(5).tolist() if v]
            if sample:
                prefix = sample[0]
                for s in sample[1:]:
                    while prefix and not s.startswith(prefix):
                        prefix = prefix[:-1]
        if prefix:
            prefix_to_source[prefix] = src_name

    def _resolve_source(record_id: str) -> str | None:
        # Pick the longest matching prefix to disambiguate overlapping
        # prefixes (e.g. ``http://`` shared by multiple sources).
        best: tuple[int, str | None] = (0, None)
        for prefix, source in prefix_to_source.items():
            if record_id.startswith(prefix) and len(prefix) > best[0]:
                best = (len(prefix), source)
        return best[1]

    for (_src_left, _src_right), gold_df in bundle.em_gold.items():
        if gold_df is None or gold_df.empty:
            continue
        if "label" in gold_df.columns:
            truthy = gold_df["label"].astype(str).str.lower()
            positives = gold_df[truthy.isin(("true", "1", "yes"))]
        else:
            positives = gold_df
        for _, row in positives.iterrows():
            id1 = str(row["id1"])
            id2 = str(row["id2"])
            src1 = _resolve_source(id1)
            src2 = _resolve_source(id2)
            entry = linkage.setdefault(id1, {})
            if src1:
                entry[src1] = id1
            if src2:
                entry[src2] = id2
            # Also index by the right-hand side so callers can resolve
            # symmetric lookups when the fusion target IDs happen to
            # match the right-side convention on a particular source pair.
            entry2 = linkage.setdefault(id2, {})
            if src1:
                entry2[src1] = id1
            if src2:
                entry2[src2] = id2
    return linkage


def _build_source_attribute_index(
    sm_mapping: pd.DataFrame,
    knob_08_renames: dict[str, dict[str, str]] | None,
) -> dict[tuple[str, str], list[str]]:
    """Build ``{(source, canonical_attribute): [post_k8_col, ...]}`` from SM gold.

    A single canonical attribute can be backed by multiple source
    columns (rare but allowed); the runner emits one observation per
    (source, canonical_attribute, source_column).

    K8 renames the original column names; the SM gold is authored
    against post-K8 names so the lookup is identity in normal use, but
    the helper accepts the rename map for safety on baseline-vs-variant
    drift.
    """
    index: dict[tuple[str, str], list[str]] = {}
    if sm_mapping is None or sm_mapping.empty:
        return index
    renames = knob_08_renames or {}
    for _, row in sm_mapping.iterrows():
        source = str(row["source_dataset"])
        source_col = str(row["source_column"])
        target_col = str(row["target_column"])
        rename_map = renames.get(source, {})
        # If K8 renamed this column, prefer the post-K8 name; else
        # leave as authored.
        post_k8_col = rename_map.get(source_col, source_col)
        index.setdefault((source, target_col), []).append(post_k8_col)
    return index


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class NormCommitteeRunner(CommitteeRunner):
    """Run every normalization roster member against a variant.

    Parameters
    ----------
    roster_path : Path
        Per-domain ``normalization_committee_<domain>.yaml``.
    with_llm : bool
        Force-enable LLM-typed members. Default ``False``.

    Notes
    -----
    Dispatches on YAML shape: a top-level ``rule_normalizers:`` block
    (and a ``members:`` list whose entries name one of the C12 roster
    names) routes to :class:`C12NormCommitteeRunner`; otherwise the
    legacy per-(member, applies_to) runner takes over. See
    [plan_revision.md §C12](../../plans/plan_revision.md#L1058).
    """

    stage: Literal["norm"] = "norm"

    def __new__(
        cls,
        roster_path: Path,
        *,
        with_llm: bool = False,
        scoring_surface: str = "xml_targets",
    ) -> "NormCommitteeRunner":
        """Dispatch on YAML shape: C12 schema → C12 runner."""
        raw = _load_roster_yaml(roster_path)
        if isinstance(raw, dict) and _is_c12_shape(raw):
            from .committee_norm_c12 import C12NormCommitteeRunner

            return C12NormCommitteeRunner(  # type: ignore[return-value]
                roster_path,
                with_llm=with_llm,
                scoring_surface=scoring_surface,
            )
        return super().__new__(cls)

    def __init__(
        self,
        roster_path: Path,
        *,
        with_llm: bool = False,
        scoring_surface: str = "xml_targets",
    ) -> None:
        raw = _load_roster_yaml(roster_path)
        if _is_c12_shape(raw):
            # __new__ already returned a C12 runner; skip legacy init.
            return
        # scoring_surface only meaningful for the C12 path; legacy
        # members preserve the historical xml_targets behavior.
        del scoring_surface
        members = _parse_roster(raw["members"], with_llm=with_llm)
        self._specs = members
        self._seed = raw.get("seed", 42)
        self._tolerance_overrides = raw.get("fusion_protection_tolerance", None)
        self._normalizers: list[_NormalizerLike] = [
            _instantiate_member(spec) for spec in members
        ]
        super().__init__(
            roster=list(self._normalizers),
            config={"seed": self._seed, "with_llm": with_llm},
        )

    @property
    def roster_names(self) -> list[str]:
        return [spec.name for spec in self._specs]

    def run(self, bundle: VariantBundle) -> CommitteeResult:
        domain = bundle.domain
        sm_mapping = bundle.sm_mapping
        if sm_mapping is None or sm_mapping.empty:
            raise ValueError(
                f"No SM mapping for {domain}/{bundle.level}; cannot run "
                "Normalization without a source→canonical mapping."
            )

        # Index source columns by (source, canonical_attribute).
        attr_index = _build_source_attribute_index(sm_mapping, bundle.knob_08_renames)
        if not attr_index:
            raise ValueError(
                f"SM mapping for {domain}/{bundle.level} produced an empty "
                "attribute index. Check the gold file."
            )

        # Load fusion target values (authoritative reference for scoring).
        fusion_targets = load_fusion_target_values(domain)
        if not fusion_targets:
            raise ValueError(
                f"Fusion val/test reference values empty for {domain}; "
                "the Normalization committee has no signal to score."
            )

        # Wire LLM-canonicalizer in-context examples per (domain, attribute)
        # before the first call (deterministic so cache key is stable).
        self._wire_llm_examples(domain, fusion_targets)

        # Determine eligible attributes: intersection of (a) SM-resolved
        # canonical attributes, (b) attributes appearing as a tag in the
        # fusion XML, (c) attributes registered in the per-domain kind map.
        sm_attributes = {ca for (_, ca) in attr_index.keys()}
        gold_attributes: set[str] = set()
        for entity_attrs in fusion_targets.values():
            gold_attributes.update(entity_attrs.keys())
        kind_map = kind_map_for_domain(domain)
        kind_attributes = set(kind_map.keys())
        eligible_attributes = sorted(sm_attributes & gold_attributes & kind_attributes)

        if not eligible_attributes:
            raise ValueError(
                f"No eligible attributes for {domain}: SM∩fusion∩kind is "
                f"empty (sm={sorted(sm_attributes)}, "
                f"fusion={sorted(gold_attributes)}, "
                f"kind={sorted(kind_attributes)})."
            )

        # Per-source id index for fast (source, entity_id) → row lookup.
        source_id_index: dict[str, dict[str, int]] = {}
        for source_name, source_df in bundle.sources.items():
            id_col = "id" if "id" in source_df.columns else None
            if id_col is None:
                logger.warning(
                    "Source %s lacks an 'id' column; Normalization will "
                    "skip its rows.",
                    source_name,
                )
                continue
            source_id_index[source_name] = {
                str(eid): int(idx) for idx, eid in enumerate(source_df[id_col].tolist())
            }

        # Build the cross-source linkage from EM gold so non-primary
        # sources (whose record IDs don't match the fusion target IDs)
        # can still be scored. Empty linkage falls through to the
        # primary-source-only path.
        linkage = _build_entity_linkage(bundle)

        # Score each member.
        per_member: dict[str, MemberResult] = {}
        member_scores_lookup: dict[str, MemberPerAttributeScores] = {}
        all_per_attr: dict[str, dict[str, float]] = {}
        t0_total = time.monotonic()

        for spec, member in zip(self._specs, self._normalizers, strict=True):
            t0 = time.monotonic()
            scores = MemberPerAttributeScores(member=spec.name)
            applies_to = set(spec.applies_to) if spec.applies_to is not None else None

            for attribute in eligible_attributes:
                if applies_to is not None and attribute not in applies_to:
                    continue
                kind = kind_map.get(attribute, "long_string")
                tolerance = fusion_cell_tolerance(
                    domain,
                    attribute,
                    config_overrides=self._tolerance_overrides,
                )
                self._score_attribute(
                    member=member,
                    spec=spec,
                    attribute=attribute,
                    kind=kind,
                    tolerance=tolerance,
                    domain=domain,
                    bundle=bundle,
                    fusion_targets=fusion_targets,
                    attr_index=attr_index,
                    source_id_index=source_id_index,
                    linkage=linkage,
                    scores=scores,
                )

            elapsed = time.monotonic() - t0
            metrics = scores.macro_metrics()
            # Promote macro_f1 → "f1" so the base CommitteeResult shape stays
            # uniform across stages (SM/EM/Fusion all carry an "f1" per
            # member; we report macro_f1 here under both keys).
            metrics["f1"] = metrics["macro_f1"]
            metrics["precision"] = metrics["macro_precision"]
            metrics["recall"] = metrics["macro_recall"]

            per_member[spec.name] = MemberResult(
                name=spec.name,
                predictions=_member_predictions_frame(scores),
                metrics=metrics,
                runtime_s=elapsed,
                notes={
                    "signal_type": spec.signal_type,
                    "applies_to": spec.applies_to,
                },
            )
            member_scores_lookup[spec.name] = scores

            for attr_name, attr_score in scores.by_attribute.items():
                bucket = all_per_attr.setdefault(attr_name, {})
                bucket[spec.name] = attr_score.f1

        total_runtime = time.monotonic() - t0_total

        # Aggregated metrics: macro F1 across members, plus best-member F1
        # (Pending #8 ceiling).
        f1_values = [m.metrics["macro_f1"] for m in per_member.values()]
        precisions = [m.metrics["macro_precision"] for m in per_member.values()]
        recalls = [m.metrics["macro_recall"] for m in per_member.values()]
        best_member_name = ""
        best_member_f1 = 0.0
        if f1_values:
            for name, m in per_member.items():
                f1 = m.metrics["macro_f1"]
                if f1 > best_member_f1:
                    best_member_f1 = f1
                    best_member_name = name
        n = max(len(f1_values), 1)
        aggregated = {
            "macro_f1": sum(f1_values) / n if f1_values else 0.0,
            "min_f1": min(f1_values) if f1_values else 0.0,
            "max_f1": max(f1_values) if f1_values else 0.0,
            "macro_precision": sum(precisions) / n if precisions else 0.0,
            "macro_recall": sum(recalls) / n if recalls else 0.0,
            "best_member_f1": best_member_f1,
            "best_member_name_f1": 1.0 if best_member_name else 0.0,
        }

        # Per-attribute: F1 per (member, attribute) plus best-of-members.
        per_attribute_out: dict[str, dict[str, float]] = {}
        for attr_name, member_f1s in all_per_attr.items():
            entry: dict[str, float] = dict(member_f1s)
            entry["any_correct"] = (
                1.0 if any(v > 0.0 for v in member_f1s.values()) else 0.0
            )
            entry["best_member_f1"] = max(member_f1s.values(), default=0.0)
            per_attribute_out[attr_name] = entry

        # Per-partition: per-source rollup. For each source, mean F1 across
        # members across attributes that source carries.
        per_partition = self._per_source_rollup(
            member_scores_lookup, attr_index, eligible_attributes
        )

        result = CommitteeResult(
            stage="norm",
            domain=domain,
            level=bundle.level,
            per_member=per_member,
            aggregated=aggregated,
            per_attribute=per_attribute_out,
            per_partition=per_partition,
            runtime_s=total_runtime,
            roster=self.roster_names,
        )
        # Stash best-member name in a side-channel since aggregated only
        # accepts floats.
        if best_member_name:
            for member in per_member.values():
                if member.name == best_member_name:
                    member.notes.setdefault("best_member", True)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wire_llm_examples(
        self,
        domain: str,
        fusion_targets: dict[str, dict[str, list[str]]],
    ) -> None:
        """Pass canonical examples to LLM members (deterministic per-domain)."""
        llm_members = [
            (spec, m)
            for spec, m in zip(self._specs, self._normalizers, strict=True)
            if spec.signal_type == "llm" and hasattr(m, "set_examples")
        ]
        if not llm_members:
            return
        per_attribute: dict[str, list[str]] = {}
        for entity_attrs in fusion_targets.values():
            for attr, vals in entity_attrs.items():
                bucket = per_attribute.setdefault(attr, [])
                bucket.extend(str(v) for v in vals)
        examples_input = {domain: per_attribute}
        for _, m in llm_members:
            m.set_examples(examples_input)  # type: ignore[attr-defined]

    def _score_attribute(
        self,
        *,
        member: _NormalizerLike,
        spec: _NormRosterMember,
        attribute: str,
        kind: str,
        tolerance: ToleranceSpec,
        domain: str,
        bundle: VariantBundle,
        fusion_targets: dict[str, dict[str, list[str]]],
        attr_index: dict[tuple[str, str], list[str]],
        source_id_index: dict[str, dict[str, int]],
        linkage: dict[str, dict[str, str]],
        scores: MemberPerAttributeScores,
    ) -> None:
        """Iterate fusion-protected entities and score the (member, attribute) cell.

        For each fusion entity, resolve every source's record_id via the
        EM-gold linkage map (built once at run-time). Sources whose IDs
        match the fusion target IDs directly fall through to the
        same-id path; non-primary sources are reached via the linkage.
        """
        for entity_id, entity_attrs in fusion_targets.items():
            target_values = entity_attrs.get(attribute)
            if not target_values:
                continue
            entity_linkage = linkage.get(str(entity_id), {})
            for source_name, _ in bundle.sources.items():
                source_cols = attr_index.get((source_name, attribute), [])
                if not source_cols:
                    continue
                source_df = bundle.sources[source_name]
                id_lookup = source_id_index.get(source_name)
                if id_lookup is None:
                    continue
                # Two-step ID resolution: first try direct match (the
                # fusion-target convention is one source's IDs); if that
                # misses, fall back to the EM-gold-derived linkage.
                source_record_id = entity_linkage.get(source_name, str(entity_id))
                row_idx = id_lookup.get(source_record_id)
                if row_idx is None and source_record_id != str(entity_id):
                    row_idx = id_lookup.get(str(entity_id))
                if row_idx is None:
                    continue
                # If the SM mapping resolves multiple source columns to the
                # same canonical attribute, score each independently.
                for source_col in source_cols:
                    if source_col not in source_df.columns:
                        continue
                    raw_value = source_df.iat[
                        row_idx, source_df.columns.get_loc(source_col)
                    ]
                    try:
                        normalized = member.normalize(
                            raw_value,
                            attribute=attribute,
                            kind=kind,
                            domain=domain,
                        )
                    except Exception:
                        logger.exception(
                            "Member %s raised on (%s, %s, %s, %r)",
                            spec.name,
                            domain,
                            source_name,
                            attribute,
                            raw_value,
                        )
                        normalized = None
                    scores.record(attribute, normalized, target_values, tolerance)

    def _per_source_rollup(
        self,
        member_scores_lookup: dict[str, MemberPerAttributeScores],
        attr_index: dict[tuple[str, str], list[str]],
        eligible_attributes: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute per-source macro F1 across members."""
        sources = sorted({src for (src, _) in attr_index.keys()})
        out: dict[str, dict[str, float]] = {}
        for source in sources:
            source_attrs = [a for a in eligible_attributes if (source, a) in attr_index]
            if not source_attrs:
                out[source] = {"macro_f1": 0.0, "n_attributes": 0.0}
                continue
            member_f1s: list[float] = []
            for scores in member_scores_lookup.values():
                attr_f1s = [
                    scores.by_attribute[a].f1
                    for a in source_attrs
                    if a in scores.by_attribute
                ]
                if attr_f1s:
                    member_f1s.append(sum(attr_f1s) / len(attr_f1s))
            macro = sum(member_f1s) / len(member_f1s) if member_f1s else 0.0
            out[source] = {
                "macro_f1": macro,
                "n_attributes": float(len(source_attrs)),
            }
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_roster_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_c12_shape(raw: dict[str, Any]) -> bool:
    """Return True iff *raw* parses as a C12 norm roster.

    C12 YAMLs declare a top-level ``rule_normalizers:`` block (or
    ``llm_normalizer:`` block) and ``members:`` entries that name one
    of the C12 roster names. Legacy YAMLs declare ``members:`` with
    per-member ``module`` / ``class`` / ``applies_to`` fields and
    name them after the underlying normalizer class
    (e.g. ``text_clean`` / ``date_iso``).
    """
    if not isinstance(raw, dict):
        return False
    if "rule_normalizers" in raw or "llm_normalizer" in raw:
        return True
    members = raw.get("members") or []
    # Roster is C12 when every member name matches the C12 supported set.
    from .committee_norm_c12 import SUPPORTED_MEMBERS

    names = [m.get("name") for m in members if isinstance(m, dict)]
    if names and all(n in SUPPORTED_MEMBERS for n in names):
        return True
    return False


def _member_predictions_frame(
    scores: MemberPerAttributeScores,
) -> pd.DataFrame:
    """Compact per-attribute scoreboard for downstream reporting."""
    rows: list[dict[str, Any]] = []
    for attribute, score in scores.by_attribute.items():
        rows.append(
            {
                "attribute": attribute,
                "correct": score.correct,
                "wrong": score.wrong,
                "abstained": score.abstained,
                "total": score.total,
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "attribute",
                "correct",
                "wrong",
                "abstained",
                "total",
                "precision",
                "recall",
                "f1",
            ]
        )
    return pd.DataFrame(rows)


__all__ = ["NormCommitteeRunner"]

"""C12 normalization committee runner — coherent end-to-end members.

Replaces the legacy per-(member, applies_to) shape with a roster of
coherent normalization approaches. Each member produces a full per-cell
normalized output across every eligible attribute and is scored
end-to-end with macro_f1.

Members (per [plan_revision.md §C12](../../plans/plan_revision.md#L1058)):

* ``rule_per_attribute_optimal`` — for each attribute, sweep the
  type-applicable rule normalizers (TextCleanNormalizer for string;
  DateIsoNormalizer for date / year; NumberLocaleNormalizer for
  numeric; CountryIsoNormalizer for codelist; TaxonomyLookupNormalizer
  for nominal-with-taxonomy) on the fusion **validation** set; lock
  the per-attribute winner; apply across baseline + variant levels.
* ``llm_only`` — :class:`LLMCanonicalizer` (prompt v2) on every
  eligible attribute, no fallback. Synthesis explicitly allowed.
* ``passthrough`` — identity; the raw source value is the
  normalization output. Coherent baseline that surfaces "what F1 do I
  get without any normalization."

The per-attribute rule choice is locked once via a val-set sweep,
cached to ``baselines/<domain>/norm_committee_selection.json``, and
replayed at every level so K5/K6 corruption can't quietly change a
member's identity.

YAML schema is documented in :func:`_parse_roster` and the per-domain
``normalization_committee_<domain>.yaml`` files.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from .committee import CommitteeResult, CommitteeRunner, MemberResult
from .committee_norm_scoring import MemberPerAttributeScores
from .domain_config import load_domain_config
from .protection import (
    ToleranceSpec,
    fusion_cell_tolerance,
    kind_map_for_domain,
)
from .variant_loader import VariantBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


SUPPORTED_MEMBERS: frozenset[str] = frozenset(
    {
        "rule_per_attribute_optimal",
        "llm_only",
        "passthrough",
    }
)


# ---------------------------------------------------------------------------
# Roster types
# ---------------------------------------------------------------------------


@dataclass
class _RuleCandidate:
    """Single rule normalizer candidate available to the val sweep.

    Attributes
    ----------
    name : str
        Short label written to the selection cache (e.g. ``"text_clean"``,
        ``"country_iso"``).
    module : str
        Dotted module path.
    cls_name : str
        Class name within ``module``.
    applies_to : list[str]
        Attribute names this rule can natively handle.
    params : dict[str, Any]
        Constructor kwargs.
    """

    name: str
    module: str
    cls_name: str
    applies_to: list[str]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _LLMConfig:
    """LLM normalizer (LLMCanonicalizer) configuration.

    Attributes
    ----------
    module : str
        Dotted module path. Defaults to
        ``usecases_synthetic.lib.llm_normalizer``.
    cls_name : str
        Class name (``LLMCanonicalizer``).
    params : dict[str, Any]
        Constructor kwargs (model_name, num_examples, temperature,
        max_tokens, cache_dir, prompt_version).
    """

    module: str
    cls_name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _MemberSpec:
    """Single member entry in the C12 norm roster."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _C12NormRoster:
    """Parsed C12 normalization committee roster."""

    seed: int
    rule_candidates: list[_RuleCandidate]
    llm_config: _LLMConfig | None
    members: list[_MemberSpec]
    fusion_protection_tolerance: dict[str, dict[str, Any]] | None


# ---------------------------------------------------------------------------
# YAML parser
# ---------------------------------------------------------------------------


def _parse_roster(raw: dict[str, Any]) -> _C12NormRoster:
    """Parse a C12 normalization roster YAML.

    The YAML schema is::

        seed: 42
        fusion_protection_tolerance:
          ...  # optional per-attribute overrides

        # Rule normalizer candidates. Each declares which attributes it
        # can natively handle. The runner inverts this to a per-attribute
        # candidate list for the val sweep.
        rule_normalizers:
          - name: text_clean
            module: usecases_synthetic.lib.text_clean_normalizer
            class: TextCleanNormalizer
            applies_to: [name, artist, release-country]
            params: {}
          - name: country_iso
            module: usecases_synthetic.lib.country_iso_normalizer
            class: CountryIsoNormalizer
            applies_to: [release-country]
            params: {}
          ...

        # LLM normalizer config (used by llm_only).
        llm_normalizer:
          module: usecases_synthetic.lib.llm_normalizer
          class: LLMCanonicalizer
          params:
            model_name: gpt-5.4-mini
            num_examples: 5
            ...

        # C12 member roster (always 3).
        members:
          - name: rule_per_attribute_optimal
            params: {}
          - name: llm_only
            params: {}
          - name: passthrough
            params: {}

    Raises
    ------
    ValueError
        On unknown member name, missing required keys, or malformed
        ``rule_normalizers`` entries.
    """
    seed = int(raw.get("seed", 42))

    rule_normalizers_raw = raw.get("rule_normalizers") or []
    rule_candidates: list[_RuleCandidate] = []
    for entry in rule_normalizers_raw:
        for required_key in ("name", "module", "class", "applies_to"):
            if required_key not in entry:
                raise ValueError(
                    f"rule_normalizers entry missing required key "
                    f"{required_key!r}: {entry!r}"
                )
        applies_to = entry["applies_to"]
        if not isinstance(applies_to, list) or not applies_to:
            raise ValueError(
                f"rule_normalizers[{entry['name']!r}].applies_to must be a "
                f"non-empty list, got {applies_to!r}."
            )
        rule_candidates.append(
            _RuleCandidate(
                name=str(entry["name"]),
                module=str(entry["module"]),
                cls_name=str(entry["class"]),
                applies_to=[str(a) for a in applies_to],
                params=dict(entry.get("params") or {}),
            )
        )

    llm_raw = raw.get("llm_normalizer")
    llm_config: _LLMConfig | None = None
    if llm_raw is not None:
        for required_key in ("module", "class"):
            if required_key not in llm_raw:
                raise ValueError(
                    f"llm_normalizer entry missing required key "
                    f"{required_key!r}: {llm_raw!r}"
                )
        llm_config = _LLMConfig(
            module=str(llm_raw["module"]),
            cls_name=str(llm_raw["class"]),
            params=dict(llm_raw.get("params") or {}),
        )

    raw_members = raw.get("members") or []
    if not raw_members:
        raise ValueError("C12 norm roster requires a non-empty ``members`` list.")
    members: list[_MemberSpec] = []
    for entry in raw_members:
        if "name" not in entry:
            raise ValueError(f"member entry missing 'name': {entry!r}")
        name = str(entry["name"])
        if name not in SUPPORTED_MEMBERS:
            raise ValueError(
                f"Unknown norm member {name!r}. Supported: "
                f"{sorted(SUPPORTED_MEMBERS)}."
            )
        members.append(_MemberSpec(name=name, params=dict(entry.get("params") or {})))

    # Cross-validation: if llm_only is in roster, llm_normalizer must be set.
    if any(m.name == "llm_only" for m in members) and llm_config is None:
        raise ValueError(
            "Roster includes ``llm_only`` but no ``llm_normalizer:`` block "
            "is configured. Add one or remove llm_only from members."
        )
    # If rule_per_attribute_optimal is in roster, rule_normalizers must be set.
    if any(m.name == "rule_per_attribute_optimal" for m in members):
        if not rule_candidates:
            raise ValueError(
                "Roster includes ``rule_per_attribute_optimal`` but no "
                "``rule_normalizers:`` candidates are declared."
            )

    return _C12NormRoster(
        seed=seed,
        rule_candidates=rule_candidates,
        llm_config=llm_config,
        members=members,
        fusion_protection_tolerance=raw.get("fusion_protection_tolerance"),
    )


# ---------------------------------------------------------------------------
# Selection cache I/O
# ---------------------------------------------------------------------------


def _selection_cache_path(domain: str) -> Path:
    """Return the per-domain norm val-selection cache path."""
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "baselines" / domain / "norm_committee_selection.json"


def _load_selection_cache(domain: str) -> dict[str, dict[str, str]]:
    """Load the per-domain val-selection cache, or ``{}`` if absent."""
    path = _selection_cache_path(domain)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for member_name, mapping in data.items():
        if isinstance(mapping, dict):
            out[str(member_name)] = {str(k): str(v) for k, v in mapping.items()}
    return out


def _save_selection_cache(
    domain: str,
    cache: dict[str, dict[str, str]],
) -> None:
    """Persist the per-domain norm val-selection cache."""
    path = _selection_cache_path(domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# Val-only target value loading
# ---------------------------------------------------------------------------


def _load_targets_from_xml(path: Path) -> dict[str, dict[str, list[str]]]:
    """Parse a single fusion XML file into the entity-attribute target shape.

    Mirrors :func:`protection.load_fusion_target_values` per-file: each
    ``<entity>`` becomes a dict keyed by tag name; multi-valued elements
    aggregate into a list.
    """
    out: dict[str, dict[str, list[str]]] = {}
    if not path.exists():
        return out
    tree = ET.parse(path)
    root = tree.getroot()
    for entity_elem in root:
        id_elem = entity_elem.find("id")
        if id_elem is None or not id_elem.text:
            continue
        eid = id_elem.text.strip()
        attrs: dict[str, list[str]] = {}
        for child in entity_elem:
            if child.tag == "id":
                continue
            inner_values: list[str] = []
            if list(child):
                for sub in child:
                    if sub.text and sub.text.strip():
                        inner_values.append(sub.text.strip())
            elif child.text and child.text.strip():
                inner_values.append(child.text.strip())
            if inner_values:
                attrs[child.tag] = inner_values
        if attrs:
            out[eid] = attrs
    return out


def _load_val_and_test_targets(
    domain: str,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, list[str]]]]:
    """Return ``(val_targets, test_targets)`` for the domain.

    Empty dicts when the corresponding XML file is missing. ``val``
    drives val-selection; ``test`` drives the headline per-member F1.
    """
    cfg = load_domain_config(domain)
    val = _load_targets_from_xml(cfg.fusion_validation_path())
    test = _load_targets_from_xml(cfg.fusion_test_path())
    return val, test


def _load_canonical_schema_constraints(domain: str, bundle: Any) -> dict:
    """Load the canonical target-schema constraint map for ``domain``.

    Reads ``usecases/<domain>/input/schemamatching/<domain>_target_schema.json``
    (preferred) or ``target_schema.json`` and parses JSON Schema +
    ``x-pydi-consistency`` extensions into per-attribute constraints.
    """
    from pathlib import Path as _Path

    from pipelines.lib.schema_constraint_scorer import parse_target_schema

    repo_root = _Path(__file__).resolve().parents[2]
    domain_root = repo_root / "usecases" / domain / "input" / "schemamatching"
    candidates = [
        domain_root / f"{domain}_target_schema.json",
        domain_root / "target_schema.json",
    ]
    for p in candidates:
        if p.exists():
            return parse_target_schema(p)
    raise FileNotFoundError(
        f"No target schema for {domain} under {domain_root}; expected one of "
        f"{[c.name for c in candidates]}."
    )


# ---------------------------------------------------------------------------
# Coherent member objects
# ---------------------------------------------------------------------------


class _PassthroughNormalizer:
    """Coherent ``passthrough`` member: identity per cell.

    Used as a baseline that shows "what F1 do I get without any
    normalization." Returns the raw source value unchanged (cast to
    str), or ``None`` for null inputs.
    """

    name = "passthrough"

    def normalize(
        self,
        value: Any,
        *,
        attribute: str,  # noqa: ARG002 — protocol-required
        kind: str,  # noqa: ARG002
        domain: str,  # noqa: ARG002
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return text


class _CompositeRuleNormalizer:
    """Coherent ``rule_per_attribute_optimal`` member.

    Routes per-attribute calls to the val-best rule normalizer locked
    via the selection map. Attributes with no candidate (e.g. an
    attribute that no rule normalizer's ``applies_to`` covers) fall
    through to passthrough.
    """

    name = "rule_per_attribute_optimal"

    def __init__(
        self,
        *,
        per_attribute_normalizers: dict[str, Any],
        passthrough: _PassthroughNormalizer,
        selection_map: dict[str, str],
    ) -> None:
        self._per_attribute = per_attribute_normalizers
        self._passthrough = passthrough
        self.selection_map = dict(selection_map)

    def normalize(
        self,
        value: Any,
        *,
        attribute: str,
        kind: str,
        domain: str,
    ) -> str | None:
        normalizer = self._per_attribute.get(attribute)
        if normalizer is None:
            return self._passthrough.normalize(
                value, attribute=attribute, kind=kind, domain=domain
            )
        return normalizer.normalize(
            value, attribute=attribute, kind=kind, domain=domain
        )


# ---------------------------------------------------------------------------
# Normalizer loading
# ---------------------------------------------------------------------------


_INSTANCE_CACHE: dict[tuple[str, str, str], Any] = {}


def _build_rule_instance(candidate: _RuleCandidate, *, name: str) -> Any:
    """Instantiate a rule normalizer; cache so the val sweep does not
    construct the same class many times.

    Cache key is ``(module, class, params_repr)`` so two candidates with
    the same class but different params get distinct instances.
    """
    params_repr = json.dumps(candidate.params, sort_keys=True, default=str)
    key = (candidate.module, candidate.cls_name, params_repr)
    cached = _INSTANCE_CACHE.get(key)
    if cached is not None:
        # Rebind the requested instance name so logs / op-logs read
        # naturally (the cache stays a singleton).
        return cached
    mod = importlib.import_module(candidate.module)
    cls = getattr(mod, candidate.cls_name)
    instance = cls(name=name, **candidate.params)
    _INSTANCE_CACHE[key] = instance
    return instance


def _build_llm_instance(llm_config: _LLMConfig, *, op_log_path: Path | None) -> Any:
    """Instantiate the LLMCanonicalizer with optional op-log path.

    Not cached across calls — the op_log_path varies per (domain, level)
    and writing through one cached instance would mis-route logs.
    """
    mod = importlib.import_module(llm_config.module)
    cls = getattr(mod, llm_config.cls_name)
    params = dict(llm_config.params)
    if op_log_path is not None:
        params["op_log_path"] = str(op_log_path)
    return cls(**params)


# ---------------------------------------------------------------------------
# Val-selection sweep
# ---------------------------------------------------------------------------


def _candidates_for_attribute(
    attr: str, rule_candidates: list[_RuleCandidate]
) -> list[_RuleCandidate]:
    """Return rule candidates whose ``applies_to`` contains *attr*."""
    return [c for c in rule_candidates if attr in c.applies_to]


def _score_normalizer_on_val(
    *,
    normalizer: Any,
    attribute: str,
    kind: str,
    tolerance: ToleranceSpec,
    domain: str,
    val_targets: dict[str, dict[str, list[str]]],
    bundle: VariantBundle,
    attr_index: dict[tuple[str, str], list[str]],
    source_id_index: dict[str, dict[str, int]],
    linkage: dict[str, dict[str, str]],
) -> float:
    """Score a single normalizer on ``attribute`` against the val targets.

    Returns the F1 score (correct / (correct + wrong + abstained)).
    """
    from .committee_norm_scoring import MemberPerAttributeScores

    scores = MemberPerAttributeScores(member=f"_val_sweep__{attribute}")
    for entity_id, entity_attrs in val_targets.items():
        target_values = entity_attrs.get(attribute)
        if not target_values:
            continue
        entity_linkage = linkage.get(str(entity_id), {})
        for source_name in bundle.sources:
            source_cols = attr_index.get((source_name, attribute), [])
            if not source_cols:
                continue
            source_df = bundle.sources[source_name]
            id_lookup = source_id_index.get(source_name)
            if id_lookup is None:
                continue
            source_record_id = entity_linkage.get(source_name, str(entity_id))
            row_idx = id_lookup.get(source_record_id)
            if row_idx is None and source_record_id != str(entity_id):
                row_idx = id_lookup.get(str(entity_id))
            if row_idx is None:
                continue
            for source_col in source_cols:
                if source_col not in source_df.columns:
                    continue
                raw_value = source_df.iat[
                    row_idx, source_df.columns.get_loc(source_col)
                ]
                try:
                    normalized = normalizer.normalize(
                        raw_value,
                        attribute=attribute,
                        kind=kind,
                        domain=domain,
                    )
                except Exception:
                    logger.exception(
                        "val-sweep normalizer raised on (%s, %s, %s, %r)",
                        domain,
                        source_name,
                        attribute,
                        raw_value,
                    )
                    normalized = None
                scores.record(attribute, normalized, target_values, tolerance)
    score = scores.by_attribute.get(attribute)
    return float(score.f1) if score is not None else 0.0


def _run_val_selection(
    *,
    roster: _C12NormRoster,
    eligible_attributes: list[str],
    domain: str,
    val_targets: dict[str, dict[str, list[str]]],
    bundle: VariantBundle,
    attr_index: dict[tuple[str, str], list[str]],
    source_id_index: dict[str, dict[str, int]],
    linkage: dict[str, dict[str, str]],
    kind_map: dict[str, str],
) -> dict[str, str]:
    """Run the rule_per_attribute_optimal val sweep and return
    ``{attribute: winning_candidate_name}``.

    Attributes with zero candidates are mapped to the sentinel
    ``"_passthrough"`` (no rule applies; passthrough takes over at
    inference time).

    Ties broken by ``rule_normalizers`` declaration order (stable
    across runs).
    """
    out: dict[str, str] = {}
    for attr in eligible_attributes:
        candidates = _candidates_for_attribute(attr, roster.rule_candidates)
        if not candidates:
            out[attr] = "_passthrough"
            continue
        if len(candidates) == 1:
            # Single-candidate attrs lock without a sweep — they would
            # win unconditionally so the val pass is wasted compute.
            out[attr] = candidates[0].name
            continue
        kind = kind_map.get(attr, "long_string")
        tolerance = fusion_cell_tolerance(
            domain, attr, config_overrides=roster.fusion_protection_tolerance
        )
        best_name = candidates[0].name
        best_score = -1.0
        for cand in candidates:
            normalizer = _build_rule_instance(cand, name=cand.name)
            score = _score_normalizer_on_val(
                normalizer=normalizer,
                attribute=attr,
                kind=kind,
                tolerance=tolerance,
                domain=domain,
                val_targets=val_targets,
                bundle=bundle,
                attr_index=attr_index,
                source_id_index=source_id_index,
                linkage=linkage,
            )
            if score > best_score:
                best_score = score
                best_name = cand.name
        out[attr] = best_name
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _op_log_dir(domain: str, level: str) -> Path:
    """Return the per-(domain, level) LLM operation-log directory.

    Path: ``usecases_synthetic/output/norm_diagnostics/<domain>/<level>/``.
    """
    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / "output" / "norm_diagnostics" / domain / level
    out.mkdir(parents=True, exist_ok=True)
    return out


def _canonical_domain(domain: str) -> str:
    """Resolve ``music-small`` → ``music`` etc. so val-selection caches
    are shared across variant sizes.
    """
    from .domain_config import _resolve_knob_config_alias

    canonical = _resolve_knob_config_alias(domain)
    return canonical if canonical else domain


class C12NormCommitteeRunner(CommitteeRunner):
    """C12 normalization committee runner — coherent end-to-end members.

    Parameters
    ----------
    roster_path : Path
        Per-domain C12 ``normalization_committee_<domain>.yaml``.
    with_llm : bool
        When ``False`` (default), the ``llm_only`` member is skipped
        even if it appears in the roster. Mirrors the SM/EM/legacy-norm
        runner convention.
    """

    stage: Literal["norm"] = "norm"  # type: ignore[assignment]

    def __init__(
        self,
        roster_path: Path,
        *,
        with_llm: bool = False,
        scoring_surface: str = "xml_targets",
    ) -> None:
        with open(roster_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._roster = _parse_roster(raw)
        self._roster_path = roster_path
        self._with_llm = with_llm
        if scoring_surface not in {"xml_targets", "schema_constraints"}:
            raise ValueError(
                f"Unknown scoring_surface {scoring_surface!r}; expected "
                "'xml_targets' or 'schema_constraints'."
            )
        self._scoring_surface = scoring_surface

        # Active members = the roster minus llm_only when with_llm=False.
        self._active_members: list[_MemberSpec] = [
            m for m in self._roster.members if m.name != "llm_only" or with_llm
        ]
        super().__init__(
            roster=list(self._active_members),
            config={
                "seed": self._roster.seed,
                "with_llm": with_llm,
                "scoring_surface": scoring_surface,
            },
        )

    @property
    def roster_names(self) -> list[str]:
        return [m.name for m in self._active_members]

    def run(self, bundle: VariantBundle) -> CommitteeResult:
        from .committee_norm import _build_entity_linkage, _build_source_attribute_index

        domain = bundle.domain
        sm_mapping = bundle.sm_mapping
        if sm_mapping is None or sm_mapping.empty:
            raise ValueError(
                f"No SM mapping for {domain}/{bundle.level}; cannot run "
                "Normalization without a source→canonical mapping."
            )

        attr_index = _build_source_attribute_index(sm_mapping, bundle.knob_08_renames)
        if not attr_index:
            raise ValueError(
                f"SM mapping for {domain}/{bundle.level} produced an empty "
                "attribute index."
            )

        # Load schema constraints once when the schema_constraints
        # scoring surface is active; XML targets are only required for
        # the xml_targets surface AND for the rule_per_attribute_optimal
        # val sweep. Catch-and-ignore on the XML load so domains with
        # JSONL-only fusion gold (e.g. papers) don't break the
        # schema_constraints path.
        schema_constraints = None
        if self._scoring_surface == "schema_constraints":
            schema_constraints = _load_canonical_schema_constraints(domain, bundle)
            try:
                val_targets, test_targets = _load_val_and_test_targets(domain)
            except Exception:
                logger.info(
                    "No XML fusion targets available for %s; schema_constraints "
                    "surface does not require them, val-sweep rule selection "
                    "will fall back to defaults.",
                    domain,
                )
                val_targets, test_targets = {}, {}
        else:
            val_targets, test_targets = _load_val_and_test_targets(domain)

        if not test_targets and self._scoring_surface == "xml_targets":
            raise ValueError(
                f"Fusion test reference values empty for {domain}; the "
                "Normalization committee has no signal to score."
            )

        # Eligible attributes: intersection of SM-resolved attrs, gold-attrs,
        # and per-domain kind map.
        sm_attributes = {ca for (_, ca) in attr_index.keys()}
        kind_map = kind_map_for_domain(domain)
        kind_attributes = set(kind_map.keys())
        if self._scoring_surface == "schema_constraints":
            # Constrained attributes come from the JSON Schema, not from
            # the fusion XML — the latter only covers 6 attributes for
            # products which is too narrow.
            gold_attributes = {
                a for a, c in (schema_constraints or {}).items() if c.has_any_constraint
            }
            # The schema-constraint scorer doesn't consult the kind_map
            # (it uses its own AttributeConstraints type system). For
            # domains not registered in protection._DEFAULT_KIND_BY_DOMAIN_ATTR
            # (e.g. papers), an empty kind_attributes set would zero out
            # the intersection. Skip the kind filter for this surface.
            eligible_attributes = sorted(sm_attributes & gold_attributes)
        else:
            gold_attributes = set()
            for entity_attrs in test_targets.values():
                gold_attributes.update(entity_attrs.keys())
            eligible_attributes = sorted(
                sm_attributes & gold_attributes & kind_attributes
            )

        if not eligible_attributes:
            raise ValueError(
                f"No eligible attributes for {domain}: SM∩"
                f"{'schema' if self._scoring_surface == 'schema_constraints' else 'fusion'}"
                f"∩kind is empty (sm={sorted(sm_attributes)}, "
                f"gold={sorted(gold_attributes)}, "
                f"kind={sorted(kind_attributes)})."
            )

        # Per-source id index for fast (source, entity_id) → row lookup.
        source_id_index: dict[str, dict[str, int]] = {}
        for source_name, source_df in bundle.sources.items():
            if "id" not in source_df.columns:
                logger.warning(
                    "Source %s lacks an 'id' column; Normalization will "
                    "skip its rows.",
                    source_name,
                )
                continue
            source_id_index[source_name] = {
                str(eid): int(idx) for idx, eid in enumerate(source_df["id"].tolist())
            }

        linkage = _build_entity_linkage(bundle)

        # ---- Val-selection (rule_per_attribute_optimal only) -----------
        canonical_domain = _canonical_domain(domain)
        selection_cache = _load_selection_cache(canonical_domain)
        needs_rule_optimal = any(
            m.name == "rule_per_attribute_optimal" for m in self._active_members
        )

        rule_selection_map: dict[str, str] = {}
        if needs_rule_optimal:
            cached = selection_cache.get("rule_per_attribute_optimal", {})
            need_sweep_attrs = [a for a in eligible_attributes if a not in cached]
            if need_sweep_attrs:
                if not val_targets:
                    if self._scoring_surface == "schema_constraints":
                        # Schema-constraint scoring doesn't need a
                        # per-attribute val selection — the constraint
                        # set IS the gold. Fall back to ``_passthrough``
                        # (identity normalization) for every attribute
                        # so rule_per_attribute_optimal scores the
                        # passthrough output against the schema.
                        logger.info(
                            "rule_per_attribute_optimal val-selection skipped for "
                            "%s under schema_constraints surface (no fusion val "
                            "targets); all attributes routed to _passthrough.",
                            domain,
                        )
                        # Populate selection_cache so the downstream
                        # rebuild at "rule_selection_map = {a:
                        # selection_cache[...][a] for a in eligible}"
                        # finds an entry per attribute.
                        selection_cache["rule_per_attribute_optimal"] = {
                            a: "_passthrough" for a in eligible_attributes
                        }
                        # Skip the rest of the val-sweep block.
                        need_sweep_attrs = []
                    else:
                        raise ValueError(
                            f"rule_per_attribute_optimal needs val-selection but "
                            f"{domain} has no fusion validation set."
                        )
            if need_sweep_attrs:
                fresh = _run_val_selection(
                    roster=self._roster,
                    eligible_attributes=need_sweep_attrs,
                    domain=domain,
                    val_targets=val_targets,
                    bundle=bundle,
                    attr_index=attr_index,
                    source_id_index=source_id_index,
                    linkage=linkage,
                    kind_map=kind_map,
                )
                merged = dict(cached)
                merged.update(fresh)
                selection_cache["rule_per_attribute_optimal"] = merged
                _save_selection_cache(canonical_domain, selection_cache)
            rule_selection_map = {
                a: selection_cache["rule_per_attribute_optimal"][a]
                for a in eligible_attributes
            }

        # ---- Build per-member normalizer objects -----------------------
        op_log_dir = _op_log_dir(domain, bundle.level)
        passthrough = _PassthroughNormalizer()

        member_instances: dict[str, Any] = {}
        member_notes: dict[str, dict[str, Any]] = {}
        for spec in self._active_members:
            if spec.name == "passthrough":
                member_instances[spec.name] = passthrough
                member_notes[spec.name] = {}
            elif spec.name == "rule_per_attribute_optimal":
                per_attr: dict[str, Any] = {}
                for attr, cand_name in rule_selection_map.items():
                    if cand_name == "_passthrough":
                        continue  # falls through inside _CompositeRuleNormalizer
                    cand = next(
                        (
                            c
                            for c in self._roster.rule_candidates
                            if c.name == cand_name and attr in c.applies_to
                        ),
                        None,
                    )
                    if cand is None:
                        # Candidate name in cache no longer matches roster
                        # (e.g. yaml edited after a cache write). Skip this
                        # attribute; the composite normalizer will fall back
                        # to passthrough.
                        logger.warning(
                            "rule_per_attribute_optimal: cached candidate "
                            "%s for attribute %s not in current roster; "
                            "falling back to passthrough for this attr.",
                            cand_name,
                            attr,
                        )
                        continue
                    per_attr[attr] = _build_rule_instance(cand, name=cand.name)
                member_instances[spec.name] = _CompositeRuleNormalizer(
                    per_attribute_normalizers=per_attr,
                    passthrough=passthrough,
                    selection_map=rule_selection_map,
                )
                member_notes[spec.name] = {"selection_map": dict(rule_selection_map)}
            elif spec.name == "llm_only":
                if self._roster.llm_config is None:
                    raise ValueError("llm_only requires `llm_normalizer:` block")
                llm_instance = _build_llm_instance(
                    self._roster.llm_config,
                    op_log_path=op_log_dir / "llm_only_operations.csv",
                )
                # Wire in-context examples (deterministic) from test targets;
                # mirrors legacy norm runner's _wire_llm_examples.
                if hasattr(llm_instance, "set_examples"):
                    per_attribute: dict[str, list[str]] = {}
                    for entity_attrs in test_targets.values():
                        for attr, vals in entity_attrs.items():
                            per_attribute.setdefault(attr, []).extend(
                                str(v) for v in vals
                            )
                    llm_instance.set_examples({domain: per_attribute})
                member_instances[spec.name] = llm_instance
                member_notes[spec.name] = {
                    "model_name": self._roster.llm_config.params.get(
                        "model_name", "<default>"
                    ),
                }
            else:
                raise AssertionError(f"Unhandled C12 norm member {spec.name!r}")

        # ---- Per-member test scoring ----------------------------------
        per_member: dict[str, MemberResult] = {}
        member_scores_lookup: dict[str, MemberPerAttributeScores] = {}
        all_per_attr: dict[str, dict[str, float]] = {}
        t0_total = time.monotonic()

        for spec in self._active_members:
            t0 = time.monotonic()
            normalizer = member_instances[spec.name]
            if self._scoring_surface == "schema_constraints":
                from pipelines.lib.schema_constraint_scorer import (
                    SchemaConstraintScores,
                )

                schema_scores = SchemaConstraintScores(member=spec.name)
                self._score_member_against_schema(
                    normalizer=normalizer,
                    eligible_attributes=eligible_attributes,
                    kind_map=kind_map,
                    domain=domain,
                    bundle=bundle,
                    attr_index=attr_index,
                    schema_constraints=schema_constraints or {},
                    scores=schema_scores,
                )
                scores_obj: Any = schema_scores
            else:
                scores_obj = MemberPerAttributeScores(member=spec.name)
                self._score_member_on_targets(
                    normalizer=normalizer,
                    eligible_attributes=eligible_attributes,
                    kind_map=kind_map,
                    domain=domain,
                    test_targets=test_targets,
                    bundle=bundle,
                    attr_index=attr_index,
                    source_id_index=source_id_index,
                    linkage=linkage,
                    scores=scores_obj,
                )
            scores = scores_obj
            elapsed = time.monotonic() - t0
            metrics = scores.macro_metrics()
            metrics["f1"] = metrics["macro_f1"]
            metrics["precision"] = metrics["macro_precision"]
            metrics["recall"] = metrics["macro_recall"]
            metrics["scoring_surface"] = self._scoring_surface

            per_member[spec.name] = MemberResult(
                name=spec.name,
                predictions=_member_predictions_frame(scores),
                metrics=metrics,
                runtime_s=elapsed,
                notes=member_notes[spec.name],
            )
            member_scores_lookup[spec.name] = scores

            for attr_name, attr_score in scores.by_attribute.items():
                all_per_attr.setdefault(attr_name, {})[spec.name] = attr_score.f1

        total_runtime = time.monotonic() - t0_total

        # ---- Aggregation ----------------------------------------------
        f1_values = [m.metrics["macro_f1"] for m in per_member.values()]
        precisions = [m.metrics["macro_precision"] for m in per_member.values()]
        recalls = [m.metrics["macro_recall"] for m in per_member.values()]
        best_member_name = ""
        best_member_f1 = 0.0
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

        per_attribute_out: dict[str, dict[str, float]] = {}
        for attr_name, member_f1s in all_per_attr.items():
            entry: dict[str, float] = dict(member_f1s)
            entry["any_correct"] = (
                1.0 if any(v > 0.0 for v in member_f1s.values()) else 0.0
            )
            entry["best_member_f1"] = max(member_f1s.values(), default=0.0)
            per_attribute_out[attr_name] = entry

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
        if best_member_name:
            for member in per_member.values():
                if member.name == best_member_name:
                    member.notes.setdefault("best_member", True)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_member_on_targets(
        self,
        *,
        normalizer: Any,
        eligible_attributes: list[str],
        kind_map: dict[str, str],
        domain: str,
        test_targets: dict[str, dict[str, list[str]]],
        bundle: VariantBundle,
        attr_index: dict[tuple[str, str], list[str]],
        source_id_index: dict[str, dict[str, int]],
        linkage: dict[str, dict[str, str]],
        scores: MemberPerAttributeScores,
    ) -> None:
        """Score one member end-to-end across all eligible attributes."""
        for attribute in eligible_attributes:
            kind = kind_map.get(attribute, "long_string")
            tolerance = fusion_cell_tolerance(
                domain,
                attribute,
                config_overrides=self._roster.fusion_protection_tolerance,
            )
            for entity_id, entity_attrs in test_targets.items():
                target_values = entity_attrs.get(attribute)
                if not target_values:
                    continue
                entity_linkage = linkage.get(str(entity_id), {})
                for source_name in bundle.sources:
                    source_cols = attr_index.get((source_name, attribute), [])
                    if not source_cols:
                        continue
                    source_df = bundle.sources[source_name]
                    id_lookup = source_id_index.get(source_name)
                    if id_lookup is None:
                        continue
                    source_record_id = entity_linkage.get(source_name, str(entity_id))
                    row_idx = id_lookup.get(source_record_id)
                    if row_idx is None and source_record_id != str(entity_id):
                        row_idx = id_lookup.get(str(entity_id))
                    if row_idx is None:
                        continue
                    for source_col in source_cols:
                        if source_col not in source_df.columns:
                            continue
                        raw_value = source_df.iat[
                            row_idx, source_df.columns.get_loc(source_col)
                        ]
                        try:
                            normalized = normalizer.normalize(
                                raw_value,
                                attribute=attribute,
                                kind=kind,
                                domain=domain,
                            )
                        except Exception:
                            logger.exception(
                                "C12 norm member %s raised on " "(%s, %s, %s, %r)",
                                getattr(normalizer, "name", "?"),
                                domain,
                                source_name,
                                attribute,
                                raw_value,
                            )
                            normalized = None
                        scores.record(attribute, normalized, target_values, tolerance)

    def _score_member_against_schema(
        self,
        *,
        normalizer: Any,
        eligible_attributes: list[str],
        kind_map: dict[str, str],
        domain: str,
        bundle: VariantBundle,
        attr_index: dict[tuple[str, str], list[str]],
        schema_constraints: dict[str, Any],
        scores: Any,
    ) -> None:
        """Score a member by running its normalizer on every (source,
        canonical_attribute, row) cell and asking whether the output
        satisfies the target-schema constraints. Unlike
        :meth:`_score_member_on_targets`, this surface does not need
        per-entity gold values — the constraint set IS the gold.
        """
        for attribute in eligible_attributes:
            constraints = schema_constraints.get(attribute)
            if constraints is None or not constraints.has_any_constraint:
                continue
            kind = kind_map.get(attribute, "long_string")
            for source_name, source_df in bundle.sources.items():
                source_cols = attr_index.get((source_name, attribute), [])
                if not source_cols:
                    continue
                # Pre-resolve column locations for fast iat() access.
                col_locs: dict[str, int] = {}
                for c in source_cols:
                    if c in source_df.columns:
                        col_locs[c] = source_df.columns.get_loc(c)
                if not col_locs:
                    continue
                # field_applicability needs the row's product_type
                # (or analogue) — pre-resolve that column once. The
                # source DataFrame may carry RAW per-source column
                # vocabularies (e.g. ``category`` in products_1,
                # ``productCategory`` in products_2, ``Type`` in
                # products_3, ``cat`` in products_4) rather than the
                # canonical attribute name. Map canonical -> raw via
                # ``attr_index`` (built from sm_mapping) so the gate
                # finds the right column regardless of source schema.
                ctx_col_locs: dict[str, int] = {}
                for ctx_col in ("product_type",):
                    # Direct hit: source already uses canonical name.
                    if ctx_col in source_df.columns:
                        ctx_col_locs[ctx_col] = source_df.columns.get_loc(ctx_col)
                        continue
                    # Otherwise look up the raw column for this
                    # canonical attribute via the SM gold-derived
                    # attr_index.
                    for raw_col in attr_index.get((source_name, ctx_col), []):
                        if raw_col in source_df.columns:
                            ctx_col_locs[ctx_col] = source_df.columns.get_loc(raw_col)
                            break
                # Iterate rows once per (source, attribute, source_col).
                for source_col, col_loc in col_locs.items():
                    for row_idx in range(len(source_df)):
                        raw_value = source_df.iat[row_idx, col_loc]
                        try:
                            normalized = normalizer.normalize(
                                raw_value,
                                attribute=attribute,
                                kind=kind,
                                domain=domain,
                            )
                        except Exception:
                            logger.exception(
                                "C12 schema-norm member %s raised on (%s, %s, %s, %r)",
                                getattr(normalizer, "name", "?"),
                                domain,
                                source_name,
                                attribute,
                                raw_value,
                            )
                            normalized = None
                        row_ctx = {
                            c: source_df.iat[row_idx, ctx_col_locs[c]]
                            for c in ctx_col_locs
                        }
                        scores.record(attribute, normalized, constraints, row_ctx)

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


__all__ = [
    "C12NormCommitteeRunner",
    "SUPPORTED_MEMBERS",
    "_C12NormRoster",
    "_MemberSpec",
    "_RuleCandidate",
    "_LLMConfig",
    "_parse_roster",
    "_selection_cache_path",
    "_load_selection_cache",
    "_save_selection_cache",
]

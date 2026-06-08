"""C12 fusion committee runner — coherent end-to-end members.

Replaces the legacy per-(attribute, strategy) shape with a roster of
coherent fusion approaches. Each member produces one fused DataFrame
across all attributes via ``DataFusionEngine.run()`` and is scored
end-to-end with macro_accuracy.

Members (per [plan_revision.md §C12](../../plans/plan_revision.md#L1058)):

* ``pydi_per_attribute_optimal`` — per-attribute val-best PyDI function.
* ``llm_only`` — LLM judge (prompt v2) on every attribute, no fallback.
* ``fusionquery_only`` — FusionQuery on string/categorical/date;
  val-best PyDI on numeric + list.
* ``truthfinder_only`` — TruthFinder on string/categorical/date;
  val-best PyDI on numeric + list.
* ``ltm_only`` — LTM on string/categorical/date + list;
  val-best PyDI on numeric.
* ``casefusion_only`` — CaseFusion on string/categorical/date;
  val-best PyDI on numeric + list.
* ``accusim_only`` — AccuSim on every attribute via type-aware
  similarity (no fallback).
* ``voting_only`` — voting on every attribute (coherent baseline).
* ``prefer_higher_trust_only`` — prefer_higher_trust on every attribute
  (coherent baseline).

The per-(member, attribute) PyDI fallback choice is locked once via a
val-set sweep, cached to
``baselines/<domain>/fusion_committee_selection.json``, and replayed at
every level so K5/K6 corruption can't quietly change a member's
identity.

YAML schema is documented in :func:`_parse_roster` and the per-domain
``fusion_committee_<domain>.yaml`` files.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from PyDI.fusion.engine import DataFusionEngine
from PyDI.fusion.strategy import DataFusionStrategy

from .committee import CommitteeResult, CommitteeRunner, MemberResult
from .committee_fusion_scoring import _resolve_eval_fn, score_fusion
from .variant_loader import VariantBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# All members supported under C12. Used by the parser to reject typos.
SUPPORTED_MEMBERS: frozenset[str] = frozenset(
    {
        "pydi_per_attribute_optimal",
        "llm_only",
        "fusionquery_only",
        "truthfinder_only",
        "ltm_only",
        "casefusion_only",
        "accusim_only",
        "voting_only",
        "prefer_higher_trust_only",
    }
)

# Per-member native type sets. Attributes whose type lies in the member's
# native set are handled by the member's main method; the rest fall back
# to val-best PyDI (or, for ``llm_only`` / ``accusim_only``, are still
# handled natively).
_NATIVE_TYPES_BY_MEMBER: dict[str, frozenset[str]] = {
    "pydi_per_attribute_optimal": frozenset(),  # always selects PyDI
    "llm_only": frozenset({"string", "categorical", "date", "numeric", "list"}),
    "fusionquery_only": frozenset({"string", "categorical", "date"}),
    "truthfinder_only": frozenset({"string", "categorical", "date"}),
    "ltm_only": frozenset({"string", "categorical", "date", "list"}),
    "casefusion_only": frozenset({"string", "categorical", "date"}),
    "accusim_only": frozenset({"string", "categorical", "date", "numeric", "list"}),
    "voting_only": frozenset({"string", "categorical", "date", "numeric", "list"}),
    "prefer_higher_trust_only": frozenset(
        {"string", "categorical", "date", "numeric", "list"}
    ),
}


# ---------------------------------------------------------------------------
# Roster types
# ---------------------------------------------------------------------------


@dataclass
class _PydiCandidate:
    """Single PyDI conflict-resolution function for val-selection sweeps.

    Attributes
    ----------
    name : str
        Short label written to the selection cache (e.g. ``"voting"``,
        ``"median"``, ``"union"``).
    function_name : str
        Function name within ``module``.
    module : str
        Dotted module path
        (e.g. ``"PyDI.fusion.conflict_resolution.general"``).
    params : dict[str, Any]
        Optional kwargs passed to the resolver at engine-strategy build
        time.
    """

    name: str
    function_name: str
    module: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _MemberSpec:
    """Single member entry in the C12 fusion roster.

    Attributes
    ----------
    name : str
        Roster member name (one of :data:`SUPPORTED_MEMBERS`).
    params : dict[str, Any]
        Member-level parameters. For TD members this carries the
        adapter's factory params (e.g. ``init_trust``, ``gamma`` for
        TruthFinder). For ``llm_only`` it carries
        ``llm_model`` / ``temperature`` / ``max_tokens`` /
        ``prompt_version`` / ``cache_dir``.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _C12FusionRoster:
    """Parsed C12 fusion committee roster.

    Carries every datum the runner needs to build per-member
    ``DataFusionStrategy`` objects and the val-selection sweep harness.

    Attributes
    ----------
    seed : int
        Reproducibility seed.
    trust_scores : dict[str, float]
        Per-source trust score for ``prefer_higher_trust`` and TD priors.
    eval_specs : dict[str, str]
        Per-attribute evaluation-function name.
    eval_params : dict[str, dict[str, Any]]
        Per-attribute evaluation-function kwargs.
    attribute_types : dict[str, str]
        Per-attribute semantic type
        (``string`` / ``categorical`` / ``date`` / ``numeric`` /
        ``list``). Drives ``_NATIVE_TYPES_BY_MEMBER`` dispatch.
    pydi_candidates_by_type : dict[str, list[_PydiCandidate]]
        Per-type list of PyDI candidates to sweep on val. Populated from
        the YAML's ``pydi_candidates`` block.
    members : list[_MemberSpec]
        Roster members in declaration order. Order is preserved in
        ``CommitteeResult.roster``.
    fused_id_column : str
        Engine output ID column.
    gold_id_column : str
        Gold-standard ID column.
    column_mapping : dict[str, dict[str, str]]
        Per-source pre-engine column rename.
    gold_column_aliases : dict[str, str]
        Pre-evaluation gold-column rename.
    gold_list_columns : list[str]
        Gold columns that ship as Python-list literal strings and need
        ``ast.literal_eval``.
    """

    seed: int
    trust_scores: dict[str, float]
    eval_specs: dict[str, str]
    eval_params: dict[str, dict[str, Any]]
    attribute_types: dict[str, str]
    pydi_candidates_by_type: dict[str, list[_PydiCandidate]]
    members: list[_MemberSpec]
    fused_id_column: str
    gold_id_column: str
    column_mapping: dict[str, dict[str, str]]
    gold_column_aliases: dict[str, str]
    gold_list_columns: list[str]


# ---------------------------------------------------------------------------
# YAML parser
# ---------------------------------------------------------------------------


def _parse_roster(raw: dict[str, Any]) -> _C12FusionRoster:
    """Parse a C12 fusion roster YAML.

    The YAML schema is::

        seed: 42
        fused_id_column: id
        gold_id_column: id
        gold_list_columns: [tracks]
        trust_scores: {musicbrainz: 3, ...}
        column_mapping: {musicbrainz: {old: new}}
        gold_column_aliases: {...}

        evaluation_functions: {name: tokenized_match, ...}
        evaluation_params: {duration: {tolerance: 0.05}}

        attribute_types:
          name: string
          duration: numeric
          tracks: list
          ...

        pydi_candidates:
          string:
            - {name: voting, function: voting, module: PyDI.fusion.conflict_resolution.general}
            - {name: longest_string, function: longest_string, module: ...}
            ...
          numeric:
            - {name: median, function: median, module: ...}
            ...

        members:
          - name: pydi_per_attribute_optimal
            params: {}
          - name: llm_only
            params:
              llm_model: gpt-5.4-mini
              ...
          - name: fusionquery_only
            params:
              max_iters: 5
              ...
          ...

    Raises
    ------
    ValueError
        On unknown member name, missing required keys, or malformed
        ``pydi_candidates`` entries.
    """
    seed = int(raw.get("seed", 42))

    trust_scores = {k: float(v) for k, v in (raw.get("trust_scores") or {}).items()}
    eval_specs = dict(raw.get("evaluation_functions") or {})
    eval_params = {k: dict(v) for k, v in (raw.get("evaluation_params") or {}).items()}

    attribute_types_raw = raw.get("attribute_types") or {}
    if not attribute_types_raw:
        raise ValueError(
            "C12 fusion roster requires an ``attribute_types`` block "
            "(maps each attribute to one of string/categorical/date/"
            "numeric/list)."
        )
    attribute_types: dict[str, str] = {}
    valid_types = {"string", "categorical", "date", "numeric", "list"}
    for attr, t in attribute_types_raw.items():
        t_str = str(t)
        if t_str not in valid_types:
            raise ValueError(
                f"attribute_types[{attr!r}]={t_str!r} is not one of "
                f"{sorted(valid_types)}."
            )
        attribute_types[str(attr)] = t_str

    pydi_candidates_raw = raw.get("pydi_candidates") or {}
    pydi_candidates_by_type: dict[str, list[_PydiCandidate]] = {}
    for type_name, entries in pydi_candidates_raw.items():
        type_str = str(type_name)
        if type_str not in valid_types:
            raise ValueError(
                f"pydi_candidates key {type_str!r} is not one of "
                f"{sorted(valid_types)}."
            )
        parsed_entries: list[_PydiCandidate] = []
        for entry in entries or []:
            if "name" not in entry or "function" not in entry or "module" not in entry:
                raise ValueError(
                    f"pydi_candidates[{type_str!r}] entry missing required "
                    f"keys (name/function/module): {entry!r}"
                )
            parsed_entries.append(
                _PydiCandidate(
                    name=str(entry["name"]),
                    function_name=str(entry["function"]),
                    module=str(entry["module"]),
                    params=dict(entry.get("params") or {}),
                )
            )
        if not parsed_entries:
            raise ValueError(
                f"pydi_candidates[{type_str!r}] is empty — must list at "
                "least one candidate so val-selection has something to "
                "pick from."
            )
        pydi_candidates_by_type[type_str] = parsed_entries

    raw_members = raw.get("members") or []
    if not raw_members:
        raise ValueError("C12 fusion roster requires a non-empty ``members`` list.")
    members: list[_MemberSpec] = []
    for entry in raw_members:
        if "name" not in entry:
            raise ValueError(f"member entry missing 'name': {entry!r}")
        name = str(entry["name"])
        if name not in SUPPORTED_MEMBERS:
            raise ValueError(
                f"Unknown member {name!r}. Supported: {sorted(SUPPORTED_MEMBERS)}."
            )
        members.append(_MemberSpec(name=name, params=dict(entry.get("params") or {})))

    fused_id_column = str(raw.get("fused_id_column", "_id"))
    gold_id_column = str(raw.get("gold_id_column", "id"))
    column_mapping: dict[str, dict[str, str]] = {
        k: dict(v) for k, v in (raw.get("column_mapping") or {}).items()
    }
    gold_column_aliases: dict[str, str] = {
        k: str(v) for k, v in (raw.get("gold_column_aliases") or {}).items()
    }
    gold_list_columns: list[str] = [
        str(c) for c in (raw.get("gold_list_columns") or [])
    ]

    # Sanity: every attribute referenced in evaluation_functions must
    # have a type. The runner relies on this for dispatch.
    missing = [a for a in eval_specs if a not in attribute_types]
    if missing:
        raise ValueError(
            "attribute_types is missing entries for evaluation attributes: "
            f"{missing}"
        )

    # Sanity: pydi_candidates must cover every type that any fallback
    # member might encounter. Only types that actually appear in
    # attribute_types are required.
    used_types = set(attribute_types.values())
    missing_candidates = used_types - set(pydi_candidates_by_type.keys())
    if missing_candidates:
        raise ValueError(
            "pydi_candidates is missing entries for types used by "
            f"attribute_types: {sorted(missing_candidates)}. Every type "
            "that appears in attribute_types needs a candidate list so "
            "fallback members + pydi_per_attribute_optimal have something "
            "to sweep."
        )

    return _C12FusionRoster(
        seed=seed,
        trust_scores=trust_scores,
        eval_specs=eval_specs,
        eval_params=eval_params,
        attribute_types=attribute_types,
        pydi_candidates_by_type=pydi_candidates_by_type,
        members=members,
        fused_id_column=fused_id_column,
        gold_id_column=gold_id_column,
        column_mapping=column_mapping,
        gold_column_aliases=gold_column_aliases,
        gold_list_columns=gold_list_columns,
    )


# ---------------------------------------------------------------------------
# Resolver loading + LLM callable
# ---------------------------------------------------------------------------


_RESOLVER_CACHE: dict[tuple[str, str], Callable[..., Any]] = {}


def _load_resolver(module: str, fn_name: str) -> Callable[..., Any]:
    """Import a conflict-resolution function (cached)."""
    key = (module, fn_name)
    if key not in _RESOLVER_CACHE:
        import importlib

        mod = importlib.import_module(module)
        _RESOLVER_CACHE[key] = getattr(mod, fn_name)
    return _RESOLVER_CACHE[key]


# ---------------------------------------------------------------------------
# Selection cache I/O
# ---------------------------------------------------------------------------


def _selection_cache_path(domain: str) -> Path:
    """Return the per-domain val-selection cache path.

    Path: ``usecases_synthetic/baselines/<domain>/fusion_committee_selection.json``.
    """
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "baselines" / domain / "fusion_committee_selection.json"


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
    """Persist the per-domain val-selection cache."""
    path = _selection_cache_path(domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# Strategy construction
# ---------------------------------------------------------------------------


def _build_llm_callable(
    member_params: dict[str, Any],
) -> tuple[Callable[..., Any] | None, dict[str, Any]]:
    """Build the OpenAI llm callable for ``llm_only`` from member params.

    Imports + delegates to the legacy committee_fusion._build_openai_llm_callable
    so cache + ChatOpenAI client are shared. Returns ``(callable, resolved_kwargs)``
    where ``resolved_kwargs`` is the dict to pass through to ``llm_judge``
    minus the ``llm_callable``-spec metadata (``llm_model`` / ``temperature``
    / ``max_tokens``).
    """
    from .committee_fusion import _build_openai_llm_callable

    resolved = dict(member_params)
    spec = resolved.get("llm_callable")
    callable_ = None
    if isinstance(spec, str) and spec.lower() == "openai":
        callable_ = _build_openai_llm_callable(
            model=str(resolved.pop("llm_model", "gpt-5.4-mini")),
            temperature=float(resolved.pop("temperature", 0.0)),
            max_tokens=int(resolved.pop("max_tokens", 2048)),
        )
        if "model_id" not in resolved:
            resolved["model_id"] = "gpt-5.4-mini"
    resolved.pop("llm_callable", None)
    return callable_, resolved


def _build_member_strategy(
    *,
    member: _MemberSpec,
    roster: _C12FusionRoster,
    datasets: list[pd.DataFrame],
    correspondences: pd.DataFrame,
    id_column: Any,
    selection_map: dict[str, str],
    op_log_path: Path | None,
) -> DataFusionStrategy:
    """Build the full per-member DataFusionStrategy.

    For every attribute in :attr:`_C12FusionRoster.attribute_types`,
    register the resolver that the member produces — either the
    member's native method (when the attribute's type is in the
    member's :data:`_NATIVE_TYPES_BY_MEMBER`) or the val-best PyDI
    function read from ``selection_map``.

    Parameters
    ----------
    member
        The roster member.
    roster
        The parsed C12 roster.
    datasets
        Stamped + renamed source DataFrames passed to TD adapters'
        batch-fit factories.
    correspondences
        Engine-input correspondences passed to factories.
    id_column
        Engine id column passed to factories.
    selection_map
        Per-attribute method-choice for this member's PyDI fallback (or
        for ``pydi_per_attribute_optimal``, the per-attribute winning
        candidate name). Empty for ``llm_only``, ``accusim_only``,
        ``voting_only``, ``prefer_higher_trust_only`` (no selection
        needed).
    op_log_path
        When non-None for ``llm_only``, threaded into ``llm_judge``
        as the operation-log destination. Ignored for other members.
    """
    strategy = DataFusionStrategy(name=member.name)
    native_types = _NATIVE_TYPES_BY_MEMBER[member.name]

    for attr, attr_type in roster.attribute_types.items():
        if attr_type in native_types:
            resolver, params = _resolver_for_native(
                member=member,
                attr=attr,
                attr_type=attr_type,
                datasets=datasets,
                correspondences=correspondences,
                id_column=id_column,
                op_log_path=op_log_path,
            )
            if params:
                # Factory-built resolvers are already bound callables —
                # no kwargs at strategy-register time.
                if member.name in {
                    "fusionquery_only",
                    "truthfinder_only",
                    "ltm_only",
                    "casefusion_only",
                    "accusim_only",
                }:
                    strategy.add_attribute_fuser(attr, resolver)
                else:
                    strategy.add_attribute_fuser(attr, resolver, **params)
            else:
                strategy.add_attribute_fuser(attr, resolver)
        else:
            # Val-best PyDI fallback for this attribute.
            candidate_name = selection_map.get(attr)
            if candidate_name is None:
                raise ValueError(
                    f"Member {member.name!r}: no selection-map entry for "
                    f"attribute {attr!r} (type {attr_type}). Run the "
                    "val sweep before fusion."
                )
            candidate = _find_candidate(
                roster=roster, attr_type=attr_type, name=candidate_name
            )
            resolver = _load_resolver(candidate.module, candidate.function_name)
            strategy.add_attribute_fuser(attr, resolver, **candidate.params)

    # ``id`` column resolver (stable for evaluator alignment).
    id_resolver = _load_resolver(
        "PyDI.fusion.conflict_resolution.general", "prefer_higher_trust"
    )
    strategy.add_attribute_fuser("id", id_resolver)

    # Evaluation functions.
    for attr, fn_name in roster.eval_specs.items():
        fn = _resolve_eval_fn(fn_name)
        params = roster.eval_params.get(attr, {})
        if params:
            strategy.add_evaluation_function(attr, fn, **params)
        else:
            strategy.add_evaluation_function(attr, fn)

    return strategy


def _resolver_for_native(
    *,
    member: _MemberSpec,
    attr: str,
    attr_type: str,
    datasets: list[pd.DataFrame],
    correspondences: pd.DataFrame,
    id_column: Any,
    op_log_path: Path | None,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Resolve the (resolver, kwargs) pair for a native-type attribute.

    For TD members + accusim_only, calls the per-attribute factory
    (``make_<method>_resolver``) once with the member's params and
    returns the bound resolver (kwargs become ``{}``).

    For ``llm_only``, returns ``llm_judge`` with the merged params dict.
    For ``voting_only`` / ``prefer_higher_trust_only``, returns the
    matching PyDI function with no kwargs.
    """
    if member.name == "voting_only":
        return (
            _load_resolver("PyDI.fusion.conflict_resolution.general", "voting"),
            {},
        )
    if member.name == "prefer_higher_trust_only":
        return (
            _load_resolver(
                "PyDI.fusion.conflict_resolution.general", "prefer_higher_trust"
            ),
            {},
        )
    if member.name == "llm_only":
        callable_, resolved = _build_llm_callable(member.params)
        from .llm_judge_fusion import llm_judge

        resolved["llm_callable"] = callable_
        if op_log_path is not None:
            resolved["op_log_path"] = str(op_log_path)
        return llm_judge, resolved
    if member.name == "accusim_only":
        from .td_batch_fusion import make_accusim_resolver

        resolver = make_accusim_resolver(
            datasets=datasets,
            correspondences=correspondences,
            target_attr=attr,
            id_column=id_column,
            **member.params,
        )
        return resolver, {}
    if member.name == "fusionquery_only":
        from .td_batch_fusion import make_fusionquery_resolver

        resolver = make_fusionquery_resolver(
            datasets=datasets,
            correspondences=correspondences,
            target_attr=attr,
            id_column=id_column,
            **member.params,
        )
        return resolver, {}
    if member.name == "truthfinder_only":
        from .td_batch_fusion import make_truthfinder_resolver

        resolver = make_truthfinder_resolver(
            datasets=datasets,
            correspondences=correspondences,
            target_attr=attr,
            id_column=id_column,
            **member.params,
        )
        return resolver, {}
    if member.name == "ltm_only":
        from .td_batch_fusion import make_ltm_resolver

        resolver = make_ltm_resolver(
            datasets=datasets,
            correspondences=correspondences,
            target_attr=attr,
            id_column=id_column,
            **member.params,
        )
        return resolver, {}
    if member.name == "casefusion_only":
        from .td_batch_fusion import make_casefusion_resolver

        resolver = make_casefusion_resolver(
            datasets=datasets,
            correspondences=correspondences,
            target_attr=attr,
            id_column=id_column,
            **member.params,
        )
        return resolver, {}
    # pydi_per_attribute_optimal — handled via selection_map path, not
    # native.
    raise AssertionError(
        f"_resolver_for_native called for {member.name!r} which has no "
        f"native type set."
    )


def _find_candidate(
    *,
    roster: _C12FusionRoster,
    attr_type: str,
    name: str,
) -> _PydiCandidate:
    """Look up a PyDI candidate by ``(type, name)``."""
    for cand in roster.pydi_candidates_by_type.get(attr_type, []):
        if cand.name == name:
            return cand
    raise ValueError(
        f"PyDI candidate {name!r} not registered for type {attr_type!r}. "
        f"Candidates: {[c.name for c in roster.pydi_candidates_by_type.get(attr_type, [])]}"
    )


# ---------------------------------------------------------------------------
# Val-selection sweep
# ---------------------------------------------------------------------------


def _selection_attrs_for_member(
    member: _MemberSpec,
    roster: _C12FusionRoster,
) -> list[str]:
    """Return attributes that need val-selection for this member.

    For ``pydi_per_attribute_optimal``: every attribute (always selects
    PyDI). For TD members: attributes whose type is NOT in their native
    set. For the others (``llm_only`` / ``accusim_only`` / ``voting_only``
    / ``prefer_higher_trust_only``): empty.
    """
    if member.name == "pydi_per_attribute_optimal":
        return list(roster.attribute_types.keys())
    native = _NATIVE_TYPES_BY_MEMBER[member.name]
    if native == frozenset({"string", "categorical", "date", "numeric", "list"}):
        return []
    return [
        attr
        for attr, attr_type in roster.attribute_types.items()
        if attr_type not in native
    ]


def _val_sweep_attribute(
    *,
    attr: str,
    attr_type: str,
    candidates: list[_PydiCandidate],
    roster: _C12FusionRoster,
    datasets: list[pd.DataFrame],
    val_correspondences: pd.DataFrame,
    val_gold_df: pd.DataFrame,
) -> str:
    """Run a per-attribute val sweep and return the winning candidate name.

    For each candidate, builds a DataFusionStrategy with the candidate
    resolver on ``attr`` and ``voting`` on every other attribute, runs
    the engine on the val correspondences, and scores against val gold.
    The winner is the candidate with the highest accuracy on ``attr``.

    Ties broken by candidate-list declaration order (stable across runs).
    """
    voting = _load_resolver("PyDI.fusion.conflict_resolution.general", "voting")
    id_resolver = _load_resolver(
        "PyDI.fusion.conflict_resolution.general", "prefer_higher_trust"
    )
    eval_fn = _resolve_eval_fn(roster.eval_specs.get(attr, "exact_match"))
    eval_params = roster.eval_params.get(attr, {})

    best_name: str | None = None
    best_score = -1.0

    for cand in candidates:
        try:
            resolver = _load_resolver(cand.module, cand.function_name)
            strategy = DataFusionStrategy(name=f"val_sweep__{attr}__{cand.name}")
            strategy.add_attribute_fuser(attr, resolver, **cand.params)
            for other_attr in roster.attribute_types:
                if other_attr == attr:
                    continue
                strategy.add_attribute_fuser(other_attr, voting)
            strategy.add_attribute_fuser("id", id_resolver)
            if eval_params:
                strategy.add_evaluation_function(attr, eval_fn, **eval_params)
            else:
                strategy.add_evaluation_function(attr, eval_fn)

            engine = DataFusionEngine(strategy)
            fused = engine.run(
                datasets=datasets,
                correspondences=val_correspondences,
                id_column="id",
                include_singletons=True,
            )
            metrics = score_fusion(
                fused_df=fused,
                gold_df=val_gold_df,
                eval_specs={attr: roster.eval_specs.get(attr, "exact_match")},
                eval_params={attr: eval_params} if eval_params else None,
                fused_id_column=roster.fused_id_column,
                gold_id_column=roster.gold_id_column,
            )
            score = float(metrics.get(f"{attr}_accuracy", 0.0))
        except Exception:
            logger.exception(
                "val sweep failed for attr=%s candidate=%s — score 0",
                attr,
                cand.name,
            )
            score = 0.0

        if score > best_score:
            best_score = score
            best_name = cand.name

    if best_name is None:
        # Defensive — candidates is non-empty by parser guarantee, so
        # this only fires if every candidate raised.
        best_name = candidates[0].name
    return best_name


def _run_val_selection(
    *,
    member: _MemberSpec,
    roster: _C12FusionRoster,
    datasets: list[pd.DataFrame],
    val_correspondences: pd.DataFrame,
    val_gold_df: pd.DataFrame,
) -> dict[str, str]:
    """Run val selection for every attribute the member needs and return
    ``{attribute: winning_candidate_name}``.
    """
    selection_attrs = _selection_attrs_for_member(member, roster)
    out: dict[str, str] = {}
    for attr in selection_attrs:
        attr_type = roster.attribute_types[attr]
        candidates = roster.pydi_candidates_by_type[attr_type]
        winner = _val_sweep_attribute(
            attr=attr,
            attr_type=attr_type,
            candidates=candidates,
            roster=roster,
            datasets=datasets,
            val_correspondences=val_correspondences,
            val_gold_df=val_gold_df,
        )
        out[attr] = winner
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _apply_gold_aliases_and_lists(
    roster: _C12FusionRoster,
    gold_df: pd.DataFrame,
    bundle: VariantBundle | None = None,
) -> pd.DataFrame:
    """Apply gold_column_aliases + gold_list_columns to a gold DataFrame.

    When the gold is DOI-keyed (papers: ``gold_id_column == 'doi'``) and a
    ``bundle`` is supplied, also attach a ``source_ids`` column mapping each
    gold doi -> its comma-joined source-record ids. The C12 fused output is
    keyed by source ids (the engine fuses ``id`` to one source id and records
    ``_fusion_sources``), never by a doi, so ``DataFusionEvaluator`` can only
    align the doi gold through its ``source_ids`` fallback
    (``PyDI/fusion/evaluation.py`` ~line 529/543-546, which splits on ',').
    Without it papers fusion silently scores 0 (no fused row aligns to gold).
    The block is a no-op for id-keyed domains (``gold_id_column != 'doi'``),
    so companies/games/music/products are unchanged.
    """
    out = gold_df
    if roster.gold_column_aliases:
        out = out.rename(
            columns={
                k: v for k, v in roster.gold_column_aliases.items() if k in out.columns
            }
        )
    if (
        bundle is not None
        and roster.gold_id_column == "doi"
        and "doi" in out.columns
        and "source_ids" not in out.columns
    ):
        from .fusion_perfect_clusters import _doi_to_record_ids, _normalize_doi

        doi_to_records = _doi_to_record_ids(bundle)
        out = out.copy()
        out["source_ids"] = [
            ",".join(doi_to_records.get(_normalize_doi(d) or "", []))
            for d in out["doi"]
        ]
    if roster.gold_list_columns:
        import ast as _ast

        def _safe_literal(value: Any) -> Any:
            if value is None:
                return value
            if isinstance(value, (list, tuple, set)):
                return list(value)
            if isinstance(value, float) and pd.isna(value):
                return value
            text = str(value).strip()
            if not text:
                return value
            try:
                parsed = _ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return value
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
            return value

        out = out.copy()
        for col in roster.gold_list_columns:
            if col in out.columns:
                out[col] = out[col].map(_safe_literal)
    return out


def _build_val_correspondences(
    bundle: VariantBundle,
    val_entity_ids: set[str],
) -> pd.DataFrame:
    """Build correspondences restricted to val entities.

    Walks every EM gold positive in the bundle, keeps rows where the
    fusion-id-side (``id1`` by convention) is in ``val_entity_ids``.
    """
    from .committee_fusion import _build_correspondences_from_bundle

    full = _build_correspondences_from_bundle(bundle)
    if full.empty or not val_entity_ids:
        return full.iloc[0:0].copy()
    id1_in = full["id1"].astype(str).isin(val_entity_ids)
    id2_in = full["id2"].astype(str).isin(val_entity_ids)
    return full[id1_in | id2_in].copy()


class C12FusionCommitteeRunner(CommitteeRunner):
    """C12 fusion committee runner — coherent end-to-end members.

    Parameters
    ----------
    roster_path
        Path to the C12 ``fusion_committee_<domain>.yaml``.
    """

    stage = "fusion"  # type: ignore[assignment]

    def __init__(self, roster_path: Path) -> None:
        import yaml

        with open(roster_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._roster = _parse_roster(raw)
        self._roster_path = roster_path
        super().__init__(
            roster=list(self._roster.members),
            config={
                "seed": self._roster.seed,
                "trust_scores": self._roster.trust_scores,
            },
        )

    @property
    def roster_names(self) -> list[str]:
        return [m.name for m in self._roster.members]

    def run(
        self,
        bundle: VariantBundle,
        correspondences: pd.DataFrame | None = None,
        *,
        reselect: bool = False,
    ) -> CommitteeResult:
        """Run every C12 member end-to-end.

        Each member produces a single fused DataFrame via
        ``DataFusionEngine.run()`` (one call per member, not per
        (member, attribute)); ``score_fusion`` then yields the
        per-member macro_accuracy that drives monotonicity.

        Val-selection (for ``pydi_per_attribute_optimal`` + the TD
        members with PyDI fallback) is run once per (domain, member,
        attribute) and cached at
        ``baselines/<domain>/fusion_committee_selection.json``.
        Subsequent levels read the cache so K5/K6 corruption can't
        change a member's identity across the level ladder.

        Parameters
        ----------
        reselect : bool, default False
            Normal runs (measure_baseline / validate_variant) leave this
            ``False``: the persisted selection cache is read and reused for
            stable, level-consistent member identities. The HPO sweep
            (``_tune_fusion_committee``) sets it ``True`` so val-selection
            re-runs fresh on every sweep cell against the mutated candidate
            params, computed in-memory and never persisted (so a sweep
            never clobbers the real ``fusion_committee_selection.json``).
            Without this, the ``(domain, member, attr)`` cache key ignores
            swept params and freezes selection, making e.g. the ``trim``
            sub-sweep inert.
        """
        from .committee_fusion import (
            _build_correspondences_from_bundle,
            _parse_source_list_columns,
            _stamp_trust_scores,
        )

        gold_df = bundle.fusion_gold
        if gold_df is None or gold_df.empty:
            raise ValueError(
                f"No fusion gold for {bundle.domain}/{bundle.level}. "
                "Ensure test_set.xml exists in the fusion directory."
            )
        gold_df = _apply_gold_aliases_and_lists(self._roster, gold_df, bundle)

        if correspondences is None:
            correspondences = _build_correspondences_from_bundle(bundle)

        effective_column_mapping = bundle.resolve_column_mapping(
            self._roster.column_mapping
        )
        parsed_sources = _parse_source_list_columns(bundle.sources, bundle.domain)
        datasets = _stamp_trust_scores(
            parsed_sources,
            self._roster.trust_scores,
            column_mapping=effective_column_mapping,
        )

        # Memory (papers): the variant sources (load_variant) ship raw bloat
        # columns the fusion never uses — notably crossref ``abstract_text``
        # (full abstracts). DataFusionEngine rebuilds ~182k record groups
        # holding full record copies once PER member (~50 members), so carrying
        # the bloat grows host RSS past 512G and OOM-kills the job. The baseline
        # path (canonical_loader) already ships clean canonical sources and
        # stays ~9G — so restrict each variant source to the columns fusion
        # actually needs (id/doi + provenance + the canonical target attrs),
        # making the engine process slim records like baseline. attrs (the
        # per-source trust_score) are preserved. Gated to papers, so the
        # committed 4-domain fusion is unchanged.
        if bundle.domain in ("papers", "papers-small"):
            _keep = {
                "id",
                "_id",
                "doi",
                "cluster_id",
                "source",
                "_source",
                "source_ids",
                "_fusion_sources",
            }
            _keep |= set(self._roster.eval_specs)
            _keep |= set(self._roster.attribute_types)
            # ``datasets`` is a LIST of DataFrames (each carries its source name
            # + trust in ``df.attrs`` — dataset_name/trust_score), not a dict.
            # Slim each and copy attrs back (indexing drops .attrs).
            _slim: list[pd.DataFrame] = []
            for _df in datasets:
                _cols = [c for c in _df.columns if c in _keep]
                _sd = _df[_cols].copy()
                _sd.attrs = dict(_df.attrs)
                _slim.append(_sd)
            datasets = _slim

        # ---- Val-selection: pull from cache or run sweep -----------------
        canonical_domain = _canonical_domain(bundle.domain)
        # reselect=True (HPO sweep) starts from an empty in-memory cache so
        # selection always re-runs under the cell's mutated params and the
        # persisted cache on disk is never read or written.
        selection_cache = {} if reselect else _load_selection_cache(canonical_domain)

        val_gold_df: pd.DataFrame | None = None
        val_correspondences: pd.DataFrame | None = None
        for member in self._roster.members:
            attrs_needed = _selection_attrs_for_member(member, self._roster)
            if not attrs_needed:
                continue
            existing = selection_cache.get(member.name, {})
            if all(attr in existing for attr in attrs_needed):
                # With reselect, ``existing`` is always empty so this never
                # short-circuits — selection is recomputed every cell.
                continue
            # Lazy-build val materials on first miss.
            if val_gold_df is None:
                if bundle.fusion_validation is None or bundle.fusion_validation.empty:
                    raise ValueError(
                        f"Member {member.name!r} needs val-selection but "
                        f"{bundle.domain}/{bundle.level} has no "
                        "fusion_validation set."
                    )
                val_gold_df = _apply_gold_aliases_and_lists(
                    self._roster, bundle.fusion_validation, bundle
                )
                val_entity_ids = {
                    str(v)
                    for v in val_gold_df[self._roster.gold_id_column].tolist()
                    if pd.notna(v)
                }
                val_correspondences = _build_val_correspondences(bundle, val_entity_ids)
            assert val_correspondences is not None
            assert val_gold_df is not None
            new_picks = _run_val_selection(
                member=member,
                roster=self._roster,
                datasets=datasets,
                val_correspondences=val_correspondences,
                val_gold_df=val_gold_df,
            )
            merged = dict(existing)
            merged.update(new_picks)
            selection_cache[member.name] = merged
            if not reselect:
                _save_selection_cache(canonical_domain, selection_cache)

        # ---- Per-member test run ----------------------------------------
        per_member: dict[str, MemberResult] = {}
        per_attribute_acc: dict[str, dict[str, float]] = {}
        # accumulate {member_name: {attr: accuracy}} for selection-map
        # surface
        t0_total = time.monotonic()

        op_log_dir = _op_log_dir(bundle.domain, bundle.level)

        # Memory (papers): keep_only_winner trackers. We retain the fused frame
        # for only the running best-by-val member (see store block below).
        _best_fused_name: str | None = None
        _best_fused_val = float("-inf")

        for member in self._roster.members:
            t0 = time.monotonic()
            selection_map = selection_cache.get(member.name, {})
            op_log_path: Path | None = None
            if member.name == "llm_only":
                op_log_path = op_log_dir / "llm_only_operations.csv"

            try:
                strategy = _build_member_strategy(
                    member=member,
                    roster=self._roster,
                    datasets=datasets,
                    correspondences=correspondences,
                    id_column="id",
                    selection_map=selection_map,
                    op_log_path=op_log_path,
                )
                engine = DataFusionEngine(strategy)
                fused = engine.run(
                    datasets=datasets,
                    correspondences=correspondences,
                    id_column="id",
                    include_singletons=True,
                )
                metrics = score_fusion(
                    fused_df=fused,
                    gold_df=gold_df,
                    eval_specs=self._roster.eval_specs,
                    eval_params=self._roster.eval_params,
                    fused_id_column=self._roster.fused_id_column,
                    gold_id_column=self._roster.gold_id_column,
                )
            except Exception:
                logger.exception("C12 fusion member %s failed", member.name)
                fused = pd.DataFrame()
                metrics = {"overall_accuracy": 0.0, "macro_accuracy": 0.0}

            elapsed = time.monotonic() - t0
            # Promote macro/overall accuracy to "f1" for cross-stage
            # consistency (best-member ceiling reporting reads "f1").
            metrics.setdefault("macro_accuracy", metrics.get("overall_accuracy", 0.0))
            metrics["f1"] = metrics["macro_accuracy"]

            # VAL surface (user spec: score fusion val AND test). Score the
            # SAME fused output against the fusion VALIDATION gold — no
            # re-fusion, just a second scoring pass. For a variant,
            # bundle.fusion_validation is the variant's own val set.
            metrics["f1_test"] = metrics["f1"]
            if (
                bundle.fusion_validation is not None
                and not bundle.fusion_validation.empty
                and not fused.empty
            ):
                # DOI-keyed gold (papers) needs the doi->source_ids bridge (and
                # its alias/list prep) for this val surface too, else it scores
                # 0. id-keyed domains keep the RAW val gold here exactly as
                # before — gating on gold_id_column=="doi" guarantees no change
                # to committed companies/games/music/products val metrics.
                val_gold_for_scoring = (
                    _apply_gold_aliases_and_lists(
                        self._roster, bundle.fusion_validation, bundle
                    )
                    if self._roster.gold_id_column == "doi"
                    else bundle.fusion_validation
                )
                try:
                    val_metrics = score_fusion(
                        fused_df=fused,
                        gold_df=val_gold_for_scoring,
                        eval_specs=self._roster.eval_specs,
                        eval_params=self._roster.eval_params,
                        fused_id_column=self._roster.fused_id_column,
                        gold_id_column=self._roster.gold_id_column,
                    )
                    val_macro = val_metrics.get(
                        "macro_accuracy", val_metrics.get("overall_accuracy", 0.0)
                    )
                except Exception:
                    logger.exception(
                        "C12 fusion member %s val scoring failed", member.name
                    )
                    val_macro = float("nan")
                metrics["macro_accuracy_val"] = float(val_macro)
                metrics["f1_val"] = float(val_macro)

            # Per-attribute accuracies for this member.
            attr_accs: dict[str, float] = {}
            for attr in self._roster.attribute_types:
                key = f"{attr}_accuracy"
                if key in metrics:
                    attr_accs[attr] = float(metrics[key])
                    bucket = per_attribute_acc.setdefault(attr, {})
                    bucket[member.name] = attr_accs[attr]

            notes: dict[str, Any] = {
                "selection_map": dict(selection_map),
                "native_types": sorted(_NATIVE_TYPES_BY_MEMBER[member.name]),
            }

            # Memory (papers): retaining every member's full fused frame across
            # the ~50 attribute×strategy members blows past 256/512G — each frame
            # carries ~30 object-dtype text columns incl huge raw text
            # (abstract_text, raw author/keyword lists) that neither fusion
            # scoring nor the e2e panel use. Slim each RETAINED frame to the
            # columns scoring needs (id/doi/source_ids/_fusion_sources + the
            # target attributes) AFTER test+val scoring above consumed the full
            # frame. Gated to papers so the committed 4-domain fusion outputs are
            # byte-identical.
            if bundle.domain in ("papers", "papers-small") and not fused.empty:
                _keep = {
                    "id",
                    "_id",
                    "doi",
                    "cluster_id",
                    "source_ids",
                    "_fusion_sources",
                    "_source",
                    "source",
                }
                _keep |= set(self._roster.eval_specs)
                _keep |= set(self._roster.attribute_types)
                _cols = [c for c in fused.columns if c in _keep]
                fused = fused[_cols].copy()

            per_member[member.name] = MemberResult(
                name=member.name,
                predictions=fused,
                metrics=metrics,
                runtime_s=elapsed,
                notes=notes,
            )

            # Memory (papers): keep_only_winner. Even after the per-frame slim
            # above, retaining all ~50 attribute×strategy fused frames at once
            # OOMs past 256/512G. stage_runners selects the fusion winner by val
            # macro_accuracy and treats empty predictions as val 0, so only the
            # single best-by-val frame must survive. Stream-drop every non-best
            # frame's predictions, bounding peak retention to ~one frame. The
            # per-member metrics (incl. macro_accuracy/_val used by aggregation)
            # are scalars and are always kept. Gated to papers so the committed
            # 4-domain fusion outputs are unchanged.
            if bundle.domain in ("papers", "papers-small"):
                _v = metrics.get(
                    "macro_accuracy_val", metrics.get("macro_accuracy", 0.0)
                )
                if _best_fused_name is None or _v > _best_fused_val:
                    if _best_fused_name is not None:
                        per_member[_best_fused_name].predictions = pd.DataFrame()
                    _best_fused_name = member.name
                    _best_fused_val = float(_v)
                else:
                    per_member[member.name].predictions = pd.DataFrame()

        total_runtime = time.monotonic() - t0_total

        # ---- Aggregation -----------------------------------------------
        macro_values = [m.metrics["macro_accuracy"] for m in per_member.values()]
        n = max(len(macro_values), 1)
        best_member_name = ""
        best_member_macro = 0.0
        for name, m in per_member.items():
            v = m.metrics["macro_accuracy"]
            if v > best_member_macro:
                best_member_macro = v
                best_member_name = name
        aggregated: dict[str, float] = {
            "macro_accuracy": sum(macro_values) / n if macro_values else 0.0,
            "min_accuracy": min(macro_values) if macro_values else 0.0,
            "max_accuracy": max(macro_values) if macro_values else 0.0,
            "best_member_macro_accuracy": best_member_macro,
        }
        # Legacy compatibility: older readers (analyze_monotonicity,
        # build_statistics) referenced ``overall_accuracy`` as the
        # fusion headline. Promote best_member_macro as the new headline
        # (parallels EM/SM best-member reporting) and keep the legacy
        # key for backward-compat readers.
        aggregated["overall_accuracy"] = best_member_macro

        # Per-attribute: per-member accuracy + best-of-members.
        per_attribute_out: dict[str, dict[str, float]] = {}
        for attr, member_accs in per_attribute_acc.items():
            entry: dict[str, float] = dict(member_accs)
            entry["best_member_accuracy"] = max(member_accs.values(), default=0.0)
            entry["mean_member_accuracy"] = (
                sum(member_accs.values()) / len(member_accs) if member_accs else 0.0
            )
            per_attribute_out[attr] = entry

        return CommitteeResult(
            stage="fusion",
            domain=bundle.domain,
            level=bundle.level,
            per_member=per_member,
            aggregated=aggregated,
            per_attribute=per_attribute_out,
            per_partition={},
            runtime_s=total_runtime,
            roster=self.roster_names,
        )


# ---------------------------------------------------------------------------
# Domain canonicalization
# ---------------------------------------------------------------------------


def _canonical_domain(domain: str) -> str:
    """Resolve ``music-small`` → ``music`` etc. so val-selection caches
    are shared across variant sizes.
    """
    from .domain_config import _resolve_knob_config_alias

    canonical = _resolve_knob_config_alias(domain)
    return canonical if canonical else domain


def _op_log_dir(domain: str, level: str) -> Path:
    """Return the per-(domain, level) LLM operation-log directory.

    Path: ``usecases_synthetic/output/fusion_diagnostics/<domain>/<level>/``.
    """
    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / "output" / "fusion_diagnostics" / domain / level
    out.mkdir(parents=True, exist_ok=True)
    return out


__all__ = [
    "C12FusionCommitteeRunner",
    "SUPPORTED_MEMBERS",
    "_C12FusionRoster",
    "_MemberSpec",
    "_PydiCandidate",
    "_parse_roster",
    "_selection_cache_path",
    "_load_selection_cache",
    "_save_selection_cache",
    "_NATIVE_TYPES_BY_MEMBER",
]

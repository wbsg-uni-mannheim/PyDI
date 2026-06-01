"""Fusion committee runner.

For each attribute declared in ``fusion_committee.yaml``, runs every
listed conflict-resolution strategy independently, evaluates fused
output against the fusion gold standard, and reports per-attribute
per-strategy accuracy.

The aggregated metrics expose per-attribute *spread* (max - min across
strategies), which is the signal Module 8 uses to detect K10's
predicted monotone widening.

Design decisions (from ``module_04_fusion_committee.md``):

* Per-attribute strategies are evaluated **independently** — one full
  ``DataFusionEngine.run()`` call per (attribute, strategy) pair.
  Other attributes fall back to ``voting`` so the engine can still
  produce fused records.
* Trust scores come from ``fusion_committee.yaml`` (not the source
  DataFrames) and are stamped on ``df.attrs["trust_score"]`` before
  each engine run.
* On variants where K10 reshuffles source reliability the trust scores
  do **not** change — the committee measures how well each strategy
  adapts with fixed trust assignments.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import yaml

from PyDI.fusion.engine import DataFusionEngine
from PyDI.fusion.strategy import DataFusionStrategy

from .column_mapping import apply_column_mapping
from .committee import CommitteeResult, CommitteeRunner, MemberResult
from .committee_fusion_scoring import score_fusion
from .variant_loader import VariantBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Roster spec
# ---------------------------------------------------------------------------


@dataclass
class _StrategySpec:
    """Parsed single-strategy entry within an attribute."""

    name: str
    function_name: str
    module: str
    strategy_type: str
    params: dict[str, Any]
    factory: bool = False


@dataclass
class _AttributeSpec:
    """Parsed per-attribute block from the roster YAML."""

    name: str
    attribute_class: str
    strategies: list[_StrategySpec]


@dataclass
class _FusionRoster:
    """Parsed full fusion committee roster."""

    seed: int
    trust_scores: dict[str, float]
    eval_specs: dict[str, str]
    eval_params: dict[str, dict[str, Any]]
    attributes: list[_AttributeSpec]
    fused_id_column: str
    gold_id_column: str
    column_mapping: dict[str, dict[str, str]]
    # Rename map applied to ``bundle.fusion_gold`` columns before evaluation.
    # Bridges PyDI's XML loader's ``<parent>_<child>`` flattening
    # (e.g. ``<keypeople><name>...</name></keypeople>`` → column
    # ``keypeople_name``) back to the canonical schema column names that
    # the strategies + evaluation functions reference.
    gold_column_aliases: dict[str, str]
    # Columns in ``bundle.fusion_gold`` that ship as Python-list literal
    # strings (e.g. ``"['Track 1', 'Track 2']"`` for music's ``tracks``)
    # and must be ``ast.literal_eval``'d into actual lists before
    # ``tokenized_match`` (Jaccard over sets) can compare them. Applied
    # after ``gold_column_aliases`` so the names here refer to the
    # post-rename column.
    gold_list_columns: list[str]


def _parse_roster(raw: dict[str, Any]) -> _FusionRoster:
    """Parse the YAML roster into typed dataclasses.

    Parameters
    ----------
    raw : dict
        Parsed ``fusion_committee.yaml``.

    Returns
    -------
    _FusionRoster
        Typed roster.
    """
    seed = raw.get("seed", 42)
    trust_scores = dict(raw.get("trust_scores", {}))
    eval_specs = dict(raw.get("evaluation_functions", {}))
    eval_params = {k: dict(v) for k, v in (raw.get("evaluation_params") or {}).items()}

    attributes: list[_AttributeSpec] = []
    for attr_name, attr_block in raw.get("attributes", {}).items():
        strategies: list[_StrategySpec] = []
        for s in attr_block.get("strategies", []):
            strategies.append(
                _StrategySpec(
                    name=s["name"],
                    function_name=s["function"],
                    module=s["module"],
                    strategy_type=s.get("strategy_type", "cell_local"),
                    params=dict(s.get("params") or {}),
                    factory=bool(s.get("factory", False)),
                )
            )
        attributes.append(
            _AttributeSpec(
                name=attr_name,
                attribute_class=attr_block.get("attribute_class", "unknown"),
                strategies=strategies,
            )
        )

    fused_id_column = raw.get("fused_id_column", "_id")
    gold_id_column = raw.get("gold_id_column", "id")
    column_mapping: dict[str, dict[str, str]] = {
        k: dict(v) for k, v in (raw.get("column_mapping") or {}).items()
    }
    gold_column_aliases: dict[str, str] = {
        k: str(v) for k, v in (raw.get("gold_column_aliases") or {}).items()
    }
    gold_list_columns: list[str] = [
        str(c) for c in (raw.get("gold_list_columns") or [])
    ]

    return _FusionRoster(
        seed=seed,
        trust_scores=trust_scores,
        eval_specs=eval_specs,
        eval_params=eval_params,
        attributes=attributes,
        fused_id_column=fused_id_column,
        gold_id_column=gold_id_column,
        column_mapping=column_mapping,
        gold_column_aliases=gold_column_aliases,
        gold_list_columns=gold_list_columns,
    )


# ---------------------------------------------------------------------------
# Conflict resolution loader
# ---------------------------------------------------------------------------


_RESOLVER_CACHE: dict[tuple[str, str], Callable[..., Any]] = {}


def _load_resolver(module_path: str, fn_name: str) -> Callable[..., Any]:
    """Import a conflict-resolution function by module and name.

    Parameters
    ----------
    module_path : str
        Dotted module path (e.g.
        ``"PyDI.fusion.conflict_resolution.general"``).
    fn_name : str
        Function name within the module.

    Returns
    -------
    Callable
        The conflict-resolution function.
    """
    key = (module_path, fn_name)
    if key not in _RESOLVER_CACHE:
        mod = importlib.import_module(module_path)
        _RESOLVER_CACHE[key] = getattr(mod, fn_name)
    return _RESOLVER_CACHE[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_resolver() -> Callable[..., Any]:
    """Return the ``voting`` function for fallback attributes."""
    return _load_resolver("PyDI.fusion.conflict_resolution.general", "voting")


_LLM_CALLABLE_CACHE: dict[tuple[str, float, int], Callable[[str, str, str], str]] = {}


def _build_openai_llm_callable(
    model: str = "gpt-5.4-mini",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Callable[[str, str, str], str]:
    """Construct a 3-arg ``(system_prompt, user_prompt, model_id) -> str`` callable.

    Wraps ``langchain_openai.ChatOpenAI`` so the fusion ``llm_judge``
    strategy can consume real model output (with cache-aware short-
    circuiting in the judge itself). The instance is cached per
    ``(model, temperature, max_tokens)`` so repeated builds inside the
    sweep harness reuse a single client.

    Parameters
    ----------
    model
        Chat-completions model identifier (default ``gpt-5.4-mini`` per
        the §"LLM model defaults + per-run override" policy).
    temperature
        Sampling temperature — default ``0.0`` for determinism + stable
        cache keys.
    max_tokens
        Response cap. The judge expects a short JSON answer; 64 leaves
        slack for the structured payload.

    Returns
    -------
    Callable
        Signature ``(system_prompt, user_prompt, model_id) -> raw_text``.
        ``model_id`` argument is honoured by the wrapper (it overrides the
        cached client when distinct) so the per-cell cache key in the
        judge stays consistent with the actual model that produced the
        response.
    """
    key = (model, temperature, max_tokens)
    if key in _LLM_CALLABLE_CACHE:
        return _LLM_CALLABLE_CACHE[key]

    from langchain_core.messages import HumanMessage, SystemMessage

    from .llm_client import build_chat_openai

    chat = build_chat_openai(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    def _call(system_prompt: str, user_prompt: str, model_id: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        if model_id != model:
            override = build_chat_openai(
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = override.invoke(messages)
        else:
            response = chat.invoke(messages)
        content = getattr(response, "content", response)
        return str(content)

    _LLM_CALLABLE_CACHE[key] = _call
    return _call


def _build_strategy(
    target_attr: str,
    spec: _StrategySpec,
    all_attrs: list[str],
    eval_specs: dict[str, str],
    eval_params: dict[str, dict[str, Any]],
    *,
    datasets: list[pd.DataFrame] | None = None,
    correspondences: pd.DataFrame | None = None,
    id_column: Any = None,
) -> DataFusionStrategy:
    """Build a ``DataFusionStrategy`` for a single (attribute, strategy).

    Registers the chosen resolver for *target_attr* and ``voting`` for
    every other attribute so ``DataFusionEngine`` can produce a full
    fused record.

    Parameters
    ----------
    target_attr : str
        The attribute being tested with *spec*.
    spec : _StrategySpec
        The strategy spec for *target_attr*. When ``spec.factory`` is True,
        the loaded module-level callable is treated as a *factory* — it is
        called once with ``(datasets, correspondences, target_attr,
        id_column=..., **spec.params)`` and the returned object is used as
        the per-cell resolver. This is the entry point for batch-fit
        truth-discovery strategies that need to learn one source-trust
        vector across the entire attribute corpus before per-cell scoring.
    all_attrs : list[str]
        All attribute names in the roster.
    eval_specs : dict[str, str]
        Evaluation function names per attribute.
    eval_params : dict[str, dict[str, Any]]
        Optional eval-function kwargs per attribute.
    datasets : list[pd.DataFrame], optional
        Stamped + column-renamed source DataFrames. Required when
        ``spec.factory`` is True.
    correspondences : pd.DataFrame, optional
        Linking correspondences. Required when ``spec.factory`` is True.
    id_column : Any, optional
        Identifier column name(s) the engine will use; forwarded to
        factories so their corpus-walk uses the same group construction the
        engine will use at fusion time.

    Returns
    -------
    DataFusionStrategy
        Ready-to-use strategy.
    """
    strategy = DataFusionStrategy(name=f"{target_attr}_{spec.name}")

    # Materialise the OpenAI LLM callable when the YAML opts in via
    # ``params.llm_callable: openai``. Reads ``llm_model`` / ``temperature``
    # / ``max_tokens`` from the same params block (all optional).
    resolved_params = dict(spec.params)
    llm_callable_spec = resolved_params.get("llm_callable")
    if isinstance(llm_callable_spec, str) and llm_callable_spec.lower() == "openai":
        resolved_params["llm_callable"] = _build_openai_llm_callable(
            model=str(resolved_params.pop("llm_model", "gpt-5.4-mini")),
            temperature=float(resolved_params.pop("temperature", 0.0)),
            max_tokens=int(resolved_params.pop("max_tokens", 2048)),
        )
        if "model_id" not in resolved_params:
            resolved_params["model_id"] = "gpt-5.4-mini"

    # Register the target attribute with its specific resolver.
    loaded = _load_resolver(spec.module, spec.function_name)
    if spec.factory:
        if datasets is None or correspondences is None:
            raise ValueError(
                f"Strategy {spec.name!r} for attribute {target_attr!r} is "
                "marked factory: true but datasets/correspondences were not "
                "passed to _build_strategy."
            )
        resolver = loaded(
            datasets=datasets,
            correspondences=correspondences,
            target_attr=target_attr,
            id_column=id_column,
            **resolved_params,
        )
        strategy.add_attribute_fuser(target_attr, resolver)
    else:
        strategy.add_attribute_fuser(target_attr, loaded, **resolved_params)

    # Register all other attributes with a default resolver (voting).
    default = _default_resolver()
    for attr in all_attrs:
        if attr == target_attr:
            continue
        strategy.add_attribute_fuser(attr, default)

    # Register ``id`` with prefer_higher_trust so the fused output's
    # ``id`` column always holds the highest-trust source's ID (Forbes
    # for companies).  The engine overwrites ``_id`` during attribute
    # fusion, so we need a stable ID column for evaluator alignment.
    id_resolver = _load_resolver(
        "PyDI.fusion.conflict_resolution.general", "prefer_higher_trust"
    )
    strategy.add_attribute_fuser("id", id_resolver)

    # Register evaluation functions for every attribute.
    from .committee_fusion_scoring import _resolve_eval_fn

    for attr, fn_name in eval_specs.items():
        fn = _resolve_eval_fn(fn_name)
        params = eval_params.get(attr, {})
        if params:
            strategy.add_evaluation_function(attr, fn, **params)
        else:
            strategy.add_evaluation_function(attr, fn)

    return strategy


def _parse_source_list_columns(
    sources: dict[str, pd.DataFrame],
    domain: str,
) -> dict[str, pd.DataFrame]:
    """Parse list-typed source columns just before fusion engine invocation.

    Loaders emit raw string values for ``tracks`` (music) and
    ``genres`` / ``genre`` (games) so SM / EM / Norm matchers see plain
    strings (those matchers cannot tolerate list cells — they crash on
    ``pd.isna(list)`` or attempt to hash unhashable values). This
    helper re-parses those columns to Python lists immediately before
    the fusion engine consumes the DataFrames, where the list-aware
    strategies (``union``, ``intersection``, ``intersection_k_sources``,
    ``ltm``) + the ``tokenized_match`` (Jaccard) eval function require
    real list semantics.

    Resolves ``knob_config_alias`` so ``music-small`` / ``games-small``
    get the same transforms as their parent domains.

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Source DataFrames keyed by name.
    domain : str
        Variant domain name.

    Returns
    -------
    dict[str, DataFrame]
        New dict of copies with list-typed columns parsed. Sources
        whose columns don't need parsing are still copied so callers
        can mutate ``df.attrs`` without aliasing the bundle.
    """
    from .domain_config import _resolve_knob_config_alias
    from .loaders import parse_list_literal, split_comma_list

    canonical = _resolve_knob_config_alias(domain) or domain
    out: dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        new_df = df.copy()
        new_df.attrs = dict(df.attrs)
        if canonical == "music" and "tracks" in new_df.columns:
            new_df["tracks"] = new_df["tracks"].map(parse_list_literal)
        elif canonical == "games":
            for genre_col in ("genres", "genre"):
                if genre_col in new_df.columns:
                    new_df[genre_col] = new_df[genre_col].map(split_comma_list)
                    break
        out[name] = new_df
    return out


def _stamp_trust_scores(
    sources: dict[str, pd.DataFrame],
    trust_scores: dict[str, float],
    column_mapping: dict[str, dict[str, str]] | None = None,
) -> list[pd.DataFrame]:
    """Copy source DataFrames, stamp ``attrs["trust_score"]``, and
    optionally rename columns.

    Parameters
    ----------
    sources : dict[str, DataFrame]
        Source DataFrames keyed by name.
    trust_scores : dict[str, float]
        Trust scores from the roster YAML.
    column_mapping : dict, optional
        Per-source column rename mapping.  When provided, each source's
        columns are renamed before returning.

    Returns
    -------
    list[DataFrame]
        Shallow copies with ``trust_score`` set in ``attrs``.
    """
    mapping = column_mapping or {}
    out: list[pd.DataFrame] = []
    for name, df in sources.items():
        copy = df.copy()
        src_map = mapping.get(name, {})
        if src_map:
            copy = apply_column_mapping(copy, src_map)
        copy.attrs = dict(df.attrs)
        copy.attrs["trust_score"] = trust_scores.get(name, 1.0)
        copy.attrs["dataset_name"] = name
        out.append(copy)
    return out


# ---------------------------------------------------------------------------
# Fusion Committee Runner
# ---------------------------------------------------------------------------


class FusionCommitteeRunner(CommitteeRunner):
    """Fusion committee runner.

    Loads the roster from ``fusion_committee.yaml``, and for each
    (attribute, strategy) pair runs ``DataFusionEngine`` with that
    strategy for the target attribute (other attributes default to
    ``voting``), then evaluates against the fusion gold standard.

    Parameters
    ----------
    roster_path : Path
        Path to ``fusion_committee.yaml``.
    """

    stage: Literal["fusion"] = "fusion"

    def __new__(cls, roster_path: Path) -> "FusionCommitteeRunner":
        """Dispatch on YAML shape: ``members:`` key → C12 runner.

        The C12 restructure (plan_revision.md §C12) replaces the
        per-(attribute, strategy) shape with a roster of coherent
        end-to-end members. To keep call sites stable during the
        migration, ``FusionCommitteeRunner(path)`` continues to be the
        public entry point — it returns a
        :class:`C12FusionCommitteeRunner` when the YAML has a
        top-level ``members:`` list and the legacy runner otherwise.
        """
        raw = _load_roster_yaml(roster_path)
        if isinstance(raw, dict) and "members" in raw:
            from .committee_fusion_c12 import C12FusionCommitteeRunner

            return C12FusionCommitteeRunner(roster_path)  # type: ignore[return-value]
        return super().__new__(cls)

    def __init__(self, roster_path: Path) -> None:
        raw = _load_roster_yaml(roster_path)
        if "members" in raw:
            # __new__ already returned a C12 runner; skip legacy init
            # (Python still calls __init__ when __new__ returns an
            # instance of cls, but C12 runner is not — so this is
            # belt-and-braces).
            return
        self._roster = _parse_roster(raw)

        # Build a flat member-name list for roster_names.
        member_names: list[str] = []
        for attr_spec in self._roster.attributes:
            for strat in attr_spec.strategies:
                member_names.append(f"{attr_spec.name}_{strat.name}")
        self._member_names = member_names

        super().__init__(
            roster=list(self._roster.attributes),
            config={
                "seed": self._roster.seed,
                "trust_scores": self._roster.trust_scores,
            },
        )

    @property
    def roster_names(self) -> list[str]:
        """Return member names in declaration order."""
        return list(self._member_names)

    def run(
        self,
        bundle: VariantBundle,
        correspondences: pd.DataFrame | None = None,
    ) -> CommitteeResult:
        """Run every (attribute, strategy) pair and aggregate.

        Parameters
        ----------
        bundle : VariantBundle
            Loaded variant (baseline or augmented).
        correspondences : DataFrame or None
            EM correspondences to build record groups from.  When
            ``None`` the runner concatenates all test-gold pairs from
            the bundle (baseline path).

        Returns
        -------
        CommitteeResult
            Per-member and aggregated fusion metrics.
        """
        gold_df = bundle.fusion_gold
        if gold_df is None or gold_df.empty:
            raise ValueError(
                f"No fusion gold for {bundle.domain}/{bundle.level}. "
                "Ensure test_set.xml exists in the fusion directory."
            )
        # Apply YAML-declared gold column aliases. Bridges PyDI's
        # ``<parent>_<child>`` XML flattening (e.g. ``<keypeople><name>``
        # → column ``keypeople_name``) back to canonical schema names.
        if self._roster.gold_column_aliases:
            gold_df = gold_df.rename(
                columns={
                    k: v
                    for k, v in self._roster.gold_column_aliases.items()
                    if k in gold_df.columns
                }
            )
        # Parse Python-list literal columns (e.g. music ``tracks`` which
        # ships as ``"['Track 1', 'Track 2']"`` in the XML text content)
        # into actual lists so ``tokenized_match`` (set Jaccard) can
        # compare against list-typed fused values.
        if self._roster.gold_list_columns:
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

            gold_df = gold_df.copy()
            for col in self._roster.gold_list_columns:
                if col in gold_df.columns:
                    gold_df[col] = gold_df[col].map(_safe_literal)

        if correspondences is None:
            correspondences = _build_correspondences_from_bundle(bundle)

        # Resolve the static column_mapping through any K8 renames on
        # this bundle so _stamp_trust_scores renames the columns that
        # actually exist in the variant's DataFrames.
        effective_column_mapping = bundle.resolve_column_mapping(
            self._roster.column_mapping
        )
        # Parse list-typed source columns (music ``tracks`` Python-list
        # literals, games ``genres`` comma-separated strings) just
        # before fusion engine invocation. Done here rather than at
        # loader-time so SM / EM / Norm matchers — which crash on
        # ``pd.isna(list)`` or attempt to hash list values — keep
        # working. Alias-aware (music-small → music).
        parsed_sources = _parse_source_list_columns(bundle.sources, bundle.domain)
        datasets = _stamp_trust_scores(
            parsed_sources,
            self._roster.trust_scores,
            column_mapping=effective_column_mapping,
        )
        all_attr_names = [a.name for a in self._roster.attributes]

        per_member: dict[str, MemberResult] = {}
        # per_attribute accumulator: {attr: {strat_name: accuracy}}
        attr_strat_acc: dict[str, dict[str, float]] = {}

        t0_total = time.monotonic()

        for attr_spec in self._roster.attributes:
            attr_strat_acc[attr_spec.name] = {}

            for strat in attr_spec.strategies:
                member_key = f"{attr_spec.name}_{strat.name}"
                t0 = time.monotonic()

                try:
                    strategy = _build_strategy(
                        target_attr=attr_spec.name,
                        spec=strat,
                        all_attrs=all_attr_names,
                        eval_specs=self._roster.eval_specs,
                        eval_params=self._roster.eval_params,
                        datasets=datasets,
                        correspondences=correspondences,
                        id_column="id",
                    )

                    engine = DataFusionEngine(strategy)
                    fused = engine.run(
                        datasets=datasets,
                        correspondences=correspondences,
                        id_column="id",
                        # Perfect-cluster eval (R5 Fusion design 2026-05-12):
                        # gold-declared entities with no cross-source partner
                        # in their correspondence set still need a fused
                        # record so the evaluator can compare them. Singletons
                        # fuse trivially to their lone source's values.
                        include_singletons=True,
                    )

                    # Score the fused output against gold.
                    metrics = score_fusion(
                        fused_df=fused,
                        gold_df=gold_df,
                        eval_specs=self._roster.eval_specs,
                        eval_params=self._roster.eval_params,
                        fused_id_column=self._roster.fused_id_column,
                        gold_id_column=self._roster.gold_id_column,
                    )

                except Exception:
                    logger.exception("Fusion member %s failed", member_key)
                    fused = pd.DataFrame()
                    metrics = {"overall_accuracy": 0.0, "macro_accuracy": 0.0}

                elapsed = time.monotonic() - t0

                # Extract the target attribute's accuracy.
                attr_acc_key = f"{attr_spec.name}_accuracy"
                target_accuracy = metrics.get(attr_acc_key, 0.0)
                attr_strat_acc[attr_spec.name][strat.name] = target_accuracy

                per_member[member_key] = MemberResult(
                    name=member_key,
                    predictions=fused,
                    metrics=metrics,
                    runtime_s=elapsed,
                    notes={
                        "attribute": attr_spec.name,
                        "strategy": strat.name,
                        "strategy_type": strat.strategy_type,
                        "attribute_class": attr_spec.attribute_class,
                    },
                )

        total_runtime = time.monotonic() - t0_total

        # Per-attribute summary.
        per_attribute = _compute_per_attribute(attr_strat_acc)

        # Aggregated metrics.
        aggregated = _compute_aggregated(per_attribute)

        # Per-partition: group by attribute_class.
        per_partition = _compute_per_partition(self._roster.attributes, per_attribute)

        return CommitteeResult(
            stage="fusion",
            domain=bundle.domain,
            level=bundle.level,
            per_member=per_member,
            aggregated=aggregated,
            per_attribute=per_attribute,
            per_partition=per_partition,
            runtime_s=total_runtime,
            roster=self.roster_names,
        )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _compute_per_attribute(
    attr_strat_acc: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute per-attribute summary metrics.

    Parameters
    ----------
    attr_strat_acc : dict
        ``{attribute: {strategy_name: accuracy}}``.

    Returns
    -------
    dict[str, dict[str, float]]
        Per-attribute dict with ``best_strategy_accuracy``,
        ``mean_strategy_accuracy``, ``spread``, and per-strategy
        accuracy values.
    """
    result: dict[str, dict[str, float]] = {}

    for attr, strat_dict in attr_strat_acc.items():
        if not strat_dict:
            result[attr] = {
                "best_strategy_accuracy": 0.0,
                "mean_strategy_accuracy": 0.0,
                "spread": 0.0,
            }
            continue

        values = list(strat_dict.values())
        best = max(values)
        worst = min(values)
        mean = sum(values) / len(values)

        entry: dict[str, float] = {
            "best_strategy_accuracy": best,
            "mean_strategy_accuracy": mean,
            "spread": best - worst,
        }
        # Also include per-strategy accuracy under their names.
        for sname, acc in strat_dict.items():
            entry[sname] = acc

        result[attr] = entry

    return result


def _compute_aggregated(
    per_attribute: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute committee-level aggregated fusion metrics.

    Parameters
    ----------
    per_attribute : dict
        Output of :func:`_compute_per_attribute`.

    Returns
    -------
    dict[str, float]
        Flat aggregated dict with ``overall_accuracy`` (macro of best
        per attribute), ``overall_mean_accuracy`` (macro of mean per
        attribute), and ``overall_spread`` (macro of spread).
    """
    if not per_attribute:
        return {
            "overall_accuracy": 0.0,
            "overall_mean_accuracy": 0.0,
            "overall_spread": 0.0,
        }

    bests = [a["best_strategy_accuracy"] for a in per_attribute.values()]
    means = [a["mean_strategy_accuracy"] for a in per_attribute.values()]
    spreads = [a["spread"] for a in per_attribute.values()]
    n = len(bests)

    return {
        "overall_accuracy": sum(bests) / n,
        "overall_mean_accuracy": sum(means) / n,
        "overall_spread": sum(spreads) / n,
    }


def _compute_per_partition(
    attributes: list[_AttributeSpec],
    per_attribute: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Group per-attribute metrics by attribute class.

    Parameters
    ----------
    attributes : list[_AttributeSpec]
        Roster attribute specs (carry ``attribute_class``).
    per_attribute : dict
        Per-attribute summary metrics.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{attribute_class: {mean_best, mean_spread, n_attributes}}``.
    """
    class_bests: dict[str, list[float]] = {}
    class_spreads: dict[str, list[float]] = {}

    for attr_spec in attributes:
        cls = attr_spec.attribute_class
        pa = per_attribute.get(attr_spec.name, {})
        class_bests.setdefault(cls, []).append(pa.get("best_strategy_accuracy", 0.0))
        class_spreads.setdefault(cls, []).append(pa.get("spread", 0.0))

    result: dict[str, dict[str, float]] = {}
    for cls in class_bests:
        bvals = class_bests[cls]
        svals = class_spreads[cls]
        n = len(bvals)
        result[cls] = {
            "mean_best_accuracy": sum(bvals) / n if n else 0.0,
            "mean_spread": sum(svals) / n if n else 0.0,
            "n_attributes": float(n),
        }

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_roster_yaml(path: Path) -> dict[str, Any]:
    """Load and return the roster YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_correspondences_from_bundle(
    bundle: VariantBundle,
) -> pd.DataFrame:
    """Concatenate all test-gold correspondences into a single frame.

    Filters to positive labels only (``label == 1``) and ensures
    columns ``id1``, ``id2``, ``score`` exist.

    Parameters
    ----------
    bundle : VariantBundle
        Loaded variant.

    Returns
    -------
    DataFrame
        Combined correspondences.
    """
    frames: list[pd.DataFrame] = []
    for pair, gold_df in bundle.em_gold.items():
        if gold_df.empty:
            continue
        # Only keep positive matches.
        if "label" in gold_df.columns:
            label_col = gold_df["label"]
            if label_col.dtype == object:
                mask = label_col.str.lower() == "true"
            else:
                mask = label_col.astype(bool)
            pos = gold_df.loc[mask].copy()
        else:
            pos = gold_df.copy()
        # Ensure score column.
        if "score" not in pos.columns and "similarity" not in pos.columns:
            pos["score"] = 1.0
        frames.append(pos)

    if not frames:
        return pd.DataFrame(columns=["id1", "id2", "score"])

    combined = pd.concat(frames, ignore_index=True)
    if "score" not in combined.columns and "similarity" in combined.columns:
        combined = combined.rename(columns={"similarity": "score"})
    return combined

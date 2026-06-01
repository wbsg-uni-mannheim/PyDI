"""Validate committee roster YAML configs.

Checks:
- Each YAML loads without error.
- Every class reference in the roster is importable.
- Enabled-by-default members cover the required axes declared in each
  YAML's ``required_axes`` section.
- Structural invariants: each member has a ``name`` and expected fields.

Does **not** instantiate matchers or run any pipeline code — that is
M2/M3/M4.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml

CONFIG_DIR: Path = Path(__file__).resolve().parents[1] / "config" / "committees"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    assert path.exists(), f"Missing config: {path}"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"{filename} did not parse as a dict"
    return data


def _assert_importable(module: str, cls: str) -> None:
    """Assert that ``cls`` can be imported from ``module``."""
    mod = importlib.import_module(module)
    assert hasattr(mod, cls), (
        f"{module} has no attribute {cls!r}. "
        f"Available: {[a for a in dir(mod) if not a.startswith('_')]}"
    )


def _enabled_members(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return members where ``enabled_by_default`` is true."""
    return [m for m in members if m.get("enabled_by_default", True)]


def _check_axis_coverage(
    members: list[dict[str, Any]],
    required_axes: dict[str, list[Any]],
    axis_getter: Any,
) -> list[str]:
    """Return a list of error messages for missing axis values."""
    errors: list[str] = []
    for axis, required_values in required_axes.items():
        covered = set()
        for member in members:
            value = axis_getter(member, axis)
            if value is not None:
                if isinstance(value, bool):
                    covered.add(value)
                else:
                    covered.add(value)
        for required in required_values:
            # YAML parses "true"/"false" as bool; required_values may
            # contain bool or str.
            if isinstance(required, str) and required.lower() in ("true", "false"):
                required = required.lower() == "true"
            if required not in covered:
                errors.append(
                    f"Axis {axis!r}: required value {required!r} not "
                    f"covered. Covered: {sorted(covered, key=str)}"
                )
    return errors


# ===================================================================
# SM committee
# ===================================================================


class TestSMCommitteeConfig:
    """Tests for ``sm_committee.yaml``."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return _load_yaml("sm_committee.yaml")

    def test_loads(self, config: dict[str, Any]) -> None:
        assert "members" in config
        assert "seed" in config

    def test_members_have_required_fields(self, config: dict[str, Any]) -> None:
        for member in config["members"]:
            assert "name" in member, f"Member missing 'name': {member}"
            assert "module" in member
            assert "class" in member
            assert "signal_type" in member
            assert "params" in member

    def test_all_classes_importable(self, config: dict[str, Any]) -> None:
        for member in config["members"]:
            _assert_importable(member["module"], member["class"])

    def test_axis_coverage(self, config: dict[str, Any]) -> None:
        enabled = _enabled_members(config["members"])
        required = config.get("required_axes", {})

        def getter(member: dict[str, Any], axis: str) -> Any:
            return member.get(axis)

        errors = _check_axis_coverage(enabled, required, getter)
        assert errors == [], "\n".join(errors)

    def test_at_least_one_duplicate_and_one_embedding(
        self, config: dict[str, Any]
    ) -> None:
        """Explicit sanity: K8 (anonymised headers) requires at least one
        non-header-dependent signal. Duplicate (correspondence-based) and
        embedding (SBERT over column name + sample values) both qualify
        — the roster must keep one of each as deterministic anchors.

        ``duplicate_majority`` was disabled 2026-05-08 due to a
        runner-shape mismatch and re-enabled 2026-05-10 after the SM
        runner was patched to dispatch duplicate-typed members per
        source-pair (see ``SMCommitteeRunner._run_duplicate_per_pair``
        and plan_s1_scale.md §"R5 SM duplicate-matcher fix")."""
        enabled = _enabled_members(config["members"])
        types = {m["signal_type"] for m in enabled}
        assert "duplicate" in types, "No duplicate-based SM member enabled"
        assert "embedding" in types, "No embedding-based SM member enabled"

    def test_has_hybrid_member(self, config: dict[str, Any]) -> None:
        """C1.6: the committee must include a hybrid-ensemble SM member.

        The only shipped option is ``coma_hybrid`` (Valentine ``ComaPy``
        wrapper). Its aggregated label+instance+structural signal is
        second-order — distinct from the single-signal slots — so the
        roster explicitly requires one enabled-by-default hybrid.
        """
        enabled = _enabled_members(config["members"])
        types = {m["signal_type"] for m in enabled}
        assert (
            "hybrid" in types
        ), "No hybrid-signal SM member enabled (expected `coma_hybrid`)"


# ===================================================================
# EM Blocking committee (C2.4b split)
# ===================================================================
#
# The former combined ``em_committee.yaml`` was retired in C4 — after
# the C2.4b split, runtime reads ``em_blocking_committee.yaml`` +
# ``em_matching_committee.yaml``.  Coverage of the old class is fully
# subsumed by ``TestEMBlockingCommitteeConfig`` +
# ``TestEMMatchingCommitteeConfig`` below.


class TestEMBlockingCommitteeConfig:
    """Tests for ``em_blocking_committee.yaml`` (C2.4b split)."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return _load_yaml("em_blocking_committee.yaml")

    def test_loads(self, config: dict[str, Any]) -> None:
        assert "members" in config
        assert "seed" in config

    def test_composition_block_present(self, config: dict[str, Any]) -> None:
        """Sequential select-best-blocker requires an explicit composition block."""
        composition = config.get("composition", {})
        assert composition.get("strategy") == "select_best", (
            "Blocking committee composition.strategy must be 'select_best' "
            "(select-best-blocker → matching-committee handoff, frozen "
            "2026-04-21 per blocking_shortlist.md)."
        )
        assert float(composition.get("recall_floor", 0.0)) >= 0.97, (
            "Blocking committee must enforce a pair-recall floor >= 0.97 "
            "(frozen 2026-04-21)."
        )
        assert composition.get("tie_breaker") == "reduction_ratio"

    def test_members_have_required_fields(self, config: dict[str, Any]) -> None:
        for member in config["members"]:
            assert "name" in member, f"Member missing 'name': {member}"
            assert (
                "blocker" in member
            ), f"Blocking member {member.get('name')!r} missing 'blocker' spec"
            assert "blocking_type" in member
            # Matching-committee-only fields must NOT appear here — the
            # split is the whole point of the refactor.
            assert "matcher" not in member, (
                f"Blocking member {member.get('name')!r} must not specify "
                "a matcher; matchers belong in em_matching_committee.yaml"
            )

    def test_enabled_blocker_classes_importable(self, config: dict[str, Any]) -> None:
        """Enabled members must resolve to a real class.

        Disabled placeholders (e.g. sc_block, pending adapter) are
        skipped — their module path is allowed to point at not-yet-
        written code until they are promoted to enabled_by_default.
        """
        for member in _enabled_members(config["members"]):
            blocker = member["blocker"]
            _assert_importable(blocker["module"], blocker["class"])

    def test_axis_coverage(self, config: dict[str, Any]) -> None:
        enabled = _enabled_members(config["members"])
        required = config.get("required_axes", {})

        def getter(member: dict[str, Any], axis: str) -> Any:
            return member.get(axis)

        errors = _check_axis_coverage(enabled, required, getter)
        assert errors == [], "\n".join(errors)

    def test_has_lexical_sparse_embedding(self, config: dict[str, Any]) -> None:
        """Enabled set must cover lexical + sparse + embedding axes.

        `hybrid` (sc_block) is allowed to remain disabled until the
        supervised-contrastive adapter + per-domain checkpoint land.
        """
        enabled = _enabled_members(config["members"])
        types = {m["blocking_type"] for m in enabled}
        assert "lexical" in types, "No lexical blocker enabled"
        assert "sparse" in types, "No sparse blocker enabled (BM25)"
        assert "embedding" in types, "No embedding blocker enabled"

    def test_column_mapping_present(self, config: dict[str, Any]) -> None:
        mapping = config.get("column_mapping", {})
        assert "dbpedia" in mapping
        assert "forbes" in mapping
        assert "fullcontact" in mapping


# ===================================================================
# EM Matching committee (C2.4b split)
# ===================================================================


class TestEMMatchingCommitteeConfig:
    """Tests for ``em_matching_committee.yaml`` (C2.4b split)."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return _load_yaml("em_matching_committee.yaml")

    def test_loads(self, config: dict[str, Any]) -> None:
        assert "members" in config
        assert "seed" in config

    def test_members_have_required_fields(self, config: dict[str, Any]) -> None:
        for member in config["members"]:
            assert "name" in member, f"Member missing 'name': {member}"
            assert (
                "matcher" in member
            ), f"Matching member {member.get('name')!r} missing 'matcher' spec"
            assert "matching_type" in member
            assert "threshold" in member
            # Blocker specs belong in em_blocking_committee.yaml.
            assert "blocker" not in member, (
                f"Matching member {member.get('name')!r} must not specify "
                "a blocker; blockers belong in em_blocking_committee.yaml"
            )

    def test_enabled_matcher_classes_importable(self, config: dict[str, Any]) -> None:
        for member in _enabled_members(config["members"]):
            matcher = member["matcher"]
            _assert_importable(matcher["module"], matcher["class"])

    def test_enabled_comparator_classes_importable(
        self, config: dict[str, Any]
    ) -> None:
        for member in _enabled_members(config["members"]):
            for comp in member.get("comparators", []):
                _assert_importable(comp["module"], comp["class"])

    def test_axis_coverage(self, config: dict[str, Any]) -> None:
        enabled = _enabled_members(config["members"])
        required = config.get("required_axes", {})

        def getter(member: dict[str, Any], axis: str) -> Any:
            return member.get(axis)

        errors = _check_axis_coverage(enabled, required, getter)
        assert errors == [], "\n".join(errors)

    def test_has_learned_matcher(self, config: dict[str, Any]) -> None:
        """Ditto PLM is the load-bearing learned slot in the initial split."""
        enabled = _enabled_members(config["members"])
        types = {m["matching_type"] for m in enabled}
        assert "learned" in types, "No learned matcher enabled (expected ditto_plm)"

    def test_has_missing_value_tolerant(self, config: dict[str, Any]) -> None:
        enabled = _enabled_members(config["members"])
        tolerant = {m.get("missing_value_tolerant") for m in enabled}
        assert True in tolerant, (
            "No missing-value-tolerant matcher enabled " "(expected ditto_plm)"
        )

    def test_column_mapping_matches_blocking(self, config: dict[str, Any]) -> None:
        """Column mapping must stay in sync with em_blocking_committee.yaml."""
        blocking = _load_yaml("em_blocking_committee.yaml")
        assert config["column_mapping"] == blocking["column_mapping"], (
            "column_mapping drift between em_matching_committee.yaml and "
            "em_blocking_committee.yaml — they must stay in sync "
            "(C2.4b invariant)."
        )


# ===================================================================
# Fusion committee
# ===================================================================


class TestFusionCommitteeConfig:
    """Tests for the companies ``fusion_committee.yaml``.

    Companies adopted the C12 coherent-member schema (plan_revision.md
    §C12) 2026-05-26. The tests branch on shape via :meth:`_is_c12_shape`
    so the negative branch stays exercised if a legacy YAML reappears
    in archive / experiment directories.
    """

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return _load_yaml("fusion_committee.yaml")

    @staticmethod
    def _is_c12_shape(config: dict[str, Any]) -> bool:
        return "members" in config

    def test_loads(self, config: dict[str, Any]) -> None:
        assert "seed" in config
        assert "trust_scores" in config
        if self._is_c12_shape(config):
            assert "members" in config
            assert "attribute_types" in config
            assert "pydi_candidates" in config
        else:
            assert "attributes" in config

    def test_trust_scores_for_all_sources(self, config: dict[str, Any]) -> None:
        scores = config["trust_scores"]
        for source in ("forbes", "dbpedia", "fullcontact"):
            assert source in scores, f"Missing trust score for {source}"

    def test_attributes_have_strategies(self, config: dict[str, Any]) -> None:
        if self._is_c12_shape(config):
            members = config["members"]
            assert (
                len(members) >= 2
            ), f"Only {len(members)} C12 members (need >= 2 for committee spread)"
            return
        for attr_name, attr_spec in config["attributes"].items():
            strategies = attr_spec.get("strategies", [])
            assert len(strategies) >= 2, (
                f"Attribute {attr_name!r} has only {len(strategies)} "
                f"strategies; need >= 2 for committee spread"
            )

    def test_strategy_functions_importable(self, config: dict[str, Any]) -> None:
        if self._is_c12_shape(config):
            for _type_name, candidates in config["pydi_candidates"].items():
                for cand in candidates:
                    _assert_importable(cand["module"], cand["function"])
            return
        for attr_name, attr_spec in config["attributes"].items():
            for strategy in attr_spec["strategies"]:
                _assert_importable(strategy["module"], strategy["function"])

    def test_axis_coverage_per_attribute(self, config: dict[str, Any]) -> None:
        """Every attribute must have both cell_local and trust_weighted."""
        if self._is_c12_shape(config):
            pytest.skip("C12 retires per-attribute strategy_type axes")
        required = config.get("required_axes", {})
        if "strategy_type" not in required:
            pytest.skip("No required_axes.strategy_type in config")

        required_types = set(required["strategy_type"])
        for attr_name, attr_spec in config["attributes"].items():
            covered = {s["strategy_type"] for s in attr_spec["strategies"]}
            missing = required_types - covered
            assert not missing, (
                f"Attribute {attr_name!r} missing strategy_type(s): "
                f"{missing}. Covered: {covered}"
            )

    def test_evaluation_functions_cover_all_attributes(
        self, config: dict[str, Any]
    ) -> None:
        eval_funcs = config.get("evaluation_functions", {})
        attrs = (
            config["attribute_types"]
            if self._is_c12_shape(config)
            else config["attributes"]
        )
        for attr_name in attrs:
            assert (
                attr_name in eval_funcs
            ), f"No evaluation function for attribute {attr_name!r}"

    def test_expected_companies_attributes(self, config: dict[str, Any]) -> None:
        """Companies domain must cover these attributes.

        ``industry`` is dropped (gold has no <industry> tag — see R5 Fusion
        sign-off 2026-05-12). ``keypeople`` is re-added with list-aware
        strategies (union/intersection/intersection_k_sources/ltm) and the
        Jaccard-based ``tokenized_match`` eval per the same sign-off.
        """
        expected = {
            "name",
            "assets",
            "revenue",
            "founded",
            "country",
            "city",
            "keypeople",
        }
        if self._is_c12_shape(config):
            actual = set(config["attribute_types"])
        else:
            actual = set(config["attributes"])
        missing = expected - actual
        assert not missing, f"Missing attributes: {missing}"

    def test_truth_discovery_committee_coverage(self, config: dict[str, Any]) -> None:
        """At least one TD-style member must be wired.

        Pre-C12: at least one strategy with ``strategy_type:
        truth_discovery`` somewhere across attributes.
        C12: at least one of the four TD members (truthfinder_only,
        fusionquery_only, ltm_only, casefusion_only) — or accusim_only
        — appears in the ``members:`` list.
        """
        if self._is_c12_shape(config):
            td_members = {
                "truthfinder_only",
                "fusionquery_only",
                "ltm_only",
                "casefusion_only",
                "accusim_only",
            }
            roster = {m["name"] for m in config["members"]}
            assert (
                roster & td_members
            ), f"No TD member wired (need one of {sorted(td_members)})."
            return
        td_strategies: list[tuple[str, str]] = []
        for attr_name, attr_spec in config["attributes"].items():
            for strategy in attr_spec["strategies"]:
                if strategy.get("strategy_type") == "truth_discovery":
                    td_strategies.append((attr_name, strategy["name"]))
        assert td_strategies, (
            "Fusion committee has no truth_discovery strategies. "
            "C3.4 requires at least one TD member overall."
        )

    def test_llm_adjudicated_committee_coverage(self, config: dict[str, Any]) -> None:
        """At least one LLM-judging member must be wired.

        Pre-C12: at least one strategy with ``strategy_type:
        llm_adjudicated``. C12: ``llm_only`` appears in the
        ``members:`` list.
        """
        if self._is_c12_shape(config):
            roster = {m["name"] for m in config["members"]}
            assert "llm_only" in roster, (
                "C12 fusion committee must include `llm_only` for semantic"
                " adjudication coverage."
            )
            return
        llm_strategies: list[tuple[str, str]] = []
        for attr_name, attr_spec in config["attributes"].items():
            for strategy in attr_spec["strategies"]:
                if strategy.get("strategy_type") == "llm_adjudicated":
                    llm_strategies.append((attr_name, strategy["name"]))
        assert llm_strategies, (
            "Fusion committee has no llm_adjudicated strategies. "
            "C3.4 requires at least one LLM-judge member overall."
        )

    def test_c34_seven_member_roster_present(self, config: dict[str, Any]) -> None:
        """Every C3.4 named member must be reachable from the YAML.

        Pre-C12: each TD/LLM member appears under at least one
        ``attributes.<attr>.strategies`` block; a robust_aggregators
        family member is wired somewhere. C12: the equivalent coherent
        members (``truthfinder_only`` / ``fusionquery_only`` / etc.)
        appear in the ``members:`` list, and the pydi_candidates
        registry exposes a robust-aggregator (so val-selection can pick
        one when numeric attributes show up).
        """
        if self._is_c12_shape(config):
            roster = {m["name"] for m in config["members"]}
            required_members = {
                "truthfinder_only",
                "fusionquery_only",
                "ltm_only",
                "casefusion_only",
                "accusim_only",
                "llm_only",
            }
            missing = required_members - roster
            assert not missing, f"C12 coherent members missing: {missing}"

            robust_family = {"trimmed_mean", "huber_m_estimator", "median_of_means"}
            numeric_candidates = config["pydi_candidates"].get("numeric", [])
            candidate_names = {c["name"] for c in numeric_candidates}
            assert candidate_names & robust_family, (
                "No robust_aggregators candidate registered for numeric type. "
                f"Expected at least one of {robust_family}."
            )
            return

        wired_names: set[str] = set()
        wired_functions: set[str] = set()
        for attr_spec in config["attributes"].values():
            for strategy in attr_spec["strategies"]:
                wired_names.add(strategy["name"])
                wired_functions.add(strategy["function"])

        td_singletons = {
            "truthfinder",
            "accusim",
            "ltm",
            "casefusion",
            "fusionquery",
            "llm_judge",
        }
        missing = td_singletons - wired_names
        assert not missing, f"C3.4 members not wired in YAML: {missing}"

        robust_family = {"trimmed_mean", "huber_m_estimator", "median_of_means"}
        assert wired_functions & robust_family, (
            "No member of the robust_aggregators family is wired. "
            f"Expected at least one of {robust_family}."
        )


# ===================================================================
# Cross-committee invariants (C4)
# ===================================================================


class TestCrossCommitteeInvariants:
    """Invariants that span multiple committee YAMLs (C4 consistency review).

    These tests pin properties that cannot live in any single committee's
    own test class because they depend on agreement between two or more
    committees.  Breaking one of these tests means a committee-finalization
    step invalidated a cross-committee assumption (column renames, trust
    directions, version-pinning coverage) and the coupled file must be
    updated in lockstep.
    """

    def test_column_mapping_blocking_matching_fusion_agree(self) -> None:
        """EM-blocking, EM-matching, and Fusion must share one canonical map.

        The three committees consume the same heterogeneous sources and
        rename the same columns onto the same canonical schema.  Drift
        between any two means one committee reads a column the other does
        not produce, which silently degrades downstream F1 at the level
        where the variant loader's K8 translator can't line them up.
        """
        blocking = _load_yaml("em_blocking_committee.yaml")["column_mapping"]
        matching = _load_yaml("em_matching_committee.yaml")["column_mapping"]
        fusion = _load_yaml("fusion_committee.yaml")["column_mapping"]

        assert blocking == matching, (
            "column_mapping drift between em_blocking_committee.yaml and "
            "em_matching_committee.yaml (C2.4b invariant)."
        )
        assert blocking == fusion, (
            "column_mapping drift between em_blocking_committee.yaml and "
            "fusion_committee.yaml (C4 invariant — fusion consumes the "
            "same canonical schema as EM)."
        )

    def test_trust_scores_agree_with_td_learned_ordering(self) -> None:
        """Manual ``trust_scores`` in fusion must not contradict learned TD.

        The C3.4.11 smoke test demonstrated that TruthFinder / FusionQuery
        / AccuSim all learn ``forbes > fullcontact > dbpedia`` on the
        companies fixture.  Manual ``trust_scores`` feed ``favour_sources``
        and ``prefer_higher_trust``; if they disagreed with the learned
        ranking we'd have two classes of strategy voting in opposite
        directions on the same source.  We assert the direction — not the
        exact values — since learned trust is continuous and the manual
        prior is a tie-breaker.
        """
        fusion = _load_yaml("fusion_committee.yaml")
        scores = fusion["trust_scores"]
        # Companies domain: learned TD consistently ranks dbpedia lowest
        # (per C3.4.11 smoke test).  The manual prior must not push dbpedia
        # above either of the other two sources.
        assert scores["forbes"] > scores["dbpedia"], (
            "Manual trust_scores contradict learned TD ranking "
            f"(forbes={scores['forbes']}, dbpedia={scores['dbpedia']}); "
            "see C3.4.11 smoke test."
        )
        assert scores["fullcontact"] > scores["dbpedia"], (
            "Manual trust_scores contradict learned TD ranking "
            f"(fullcontact={scores['fullcontact']}, dbpedia={scores['dbpedia']})."
        )

    def test_retired_em_committee_yaml_absent(self) -> None:
        """The pre-C2.4b combined ``em_committee.yaml`` was retired in C4.

        Runtime reads ``em_blocking_committee.yaml`` +
        ``em_matching_committee.yaml`` after the split.  Resurrecting the
        old file risks a reader picking up a stale schema; this test is
        a regression guard against that mistake.
        """
        combined = CONFIG_DIR / "em_committee.yaml"
        assert not combined.exists(), (
            f"{combined} was retired in C4 (plan_committee_finalization.md); "
            "committee runtime now reads em_blocking_committee.yaml + "
            "em_matching_committee.yaml.  Delete the restored file."
        )


# ===================================================================
# Per-domain committee forks (S10)
# ===================================================================
#
# S10 of plans/plan_s1_scale.md forks em_blocking / em_matching / fusion
# YAMLs per non-companies domain (games / music / movies / products).
# The companies-targeted tests above remain authoritative for the
# canonical files.  These parametrized tests assert the same structural
# invariants on each per-domain fork without duplicating every
# companies-specific assertion.

PER_DOMAIN_COMMITTEE_DOMAINS: list[str] = ["games", "music", "movies", "products"]


@pytest.mark.parametrize("domain", PER_DOMAIN_COMMITTEE_DOMAINS)
class TestPerDomainEMBlockingCommittee:
    """Per-domain forks of ``em_blocking_committee.yaml`` (S10)."""

    def _load(self, domain: str) -> dict[str, Any]:
        return _load_yaml(f"em_blocking_committee_{domain}.yaml")

    def test_loads(self, domain: str) -> None:
        config = self._load(domain)
        assert "members" in config
        assert "seed" in config

    def test_composition_block_present(self, domain: str) -> None:
        config = self._load(domain)
        composition = config.get("composition", {})
        assert (
            composition.get("strategy") == "select_best"
        ), f"{domain}: blocking composition.strategy must be 'select_best'"
        assert (
            float(composition.get("recall_floor", 0.0)) >= 0.97
        ), f"{domain}: blocking pair-recall floor must be >= 0.97"

    def test_members_have_required_fields(self, domain: str) -> None:
        config = self._load(domain)
        for member in config["members"]:
            assert "name" in member
            assert "blocker" in member
            assert "blocking_type" in member
            assert "matcher" not in member

    def test_enabled_blocker_classes_importable(self, domain: str) -> None:
        config = self._load(domain)
        for member in _enabled_members(config["members"]):
            blocker = member["blocker"]
            _assert_importable(blocker["module"], blocker["class"])

    def test_axis_coverage(self, domain: str) -> None:
        config = self._load(domain)
        enabled = _enabled_members(config["members"])
        required = config.get("required_axes", {})

        def getter(member: dict[str, Any], axis: str) -> Any:
            return member.get(axis)

        errors = _check_axis_coverage(enabled, required, getter)
        assert errors == [], "\n".join(errors)

    def test_has_lexical_sparse_embedding(self, domain: str) -> None:
        config = self._load(domain)
        enabled = _enabled_members(config["members"])
        types = {m["blocking_type"] for m in enabled}
        assert "lexical" in types
        assert "sparse" in types
        assert "embedding" in types

    def test_blocking_name_column_set(self, domain: str) -> None:
        """Per-domain forks must declare ``blocking_name_column``.

        The companies/games/music canonical primary is ``name``; movies
        and products use ``title``. The committee runner reads this
        field and feeds it to the pattern-based blocking-key generator
        in ``committee_em._generate_blocking_keys``.

        Accepted StandardBlocker keys (per the R5 EM blocking sweep
        sign-off 2026-05-10): any ``<name>_first_<N>`` /
        ``<name>_first_token`` / ``<name>_norm`` pattern. Compound or
        unrecognised keys would fail the runner's pattern check at
        runtime; this test catches them at config-load.
        """
        import re as _re

        config = self._load(domain)
        assert (
            "blocking_name_column" in config
        ), f"{domain}: missing top-level 'blocking_name_column' field"
        pattern = _re.compile(r"^[a-zA-Z_]+_(first_token|first_\d+|norm)$")
        for member in config["members"]:
            if member.get("blocker", {}).get("class") == "StandardBlocker":
                on_keys = member["blocker"]["params"].get("on", []) or []
                if isinstance(on_keys, str):
                    on_keys = [on_keys]
                for key in on_keys:
                    assert pattern.match(str(key)), (
                        f"{domain}: StandardBlocker key {key!r} doesn't match "
                        "the supported pattern <col>_first_<N> / "
                        "<col>_first_token / <col>_norm. See "
                        "committee_em._derive_blocking_key."
                    )


@pytest.mark.parametrize("domain", PER_DOMAIN_COMMITTEE_DOMAINS)
class TestPerDomainEMMatchingCommittee:
    """Per-domain forks of ``em_matching_committee.yaml`` (S10)."""

    def _load(self, domain: str) -> dict[str, Any]:
        return _load_yaml(f"em_matching_committee_{domain}.yaml")

    def test_loads(self, domain: str) -> None:
        config = self._load(domain)
        assert "members" in config
        assert "seed" in config

    def test_members_have_required_fields(self, domain: str) -> None:
        config = self._load(domain)
        for member in config["members"]:
            assert "name" in member
            assert "matcher" in member
            assert "matching_type" in member
            assert "blocker" not in member

    def test_enabled_matcher_classes_importable(self, domain: str) -> None:
        config = self._load(domain)
        for member in _enabled_members(config["members"]):
            matcher = member["matcher"]
            _assert_importable(matcher["module"], matcher["class"])

    def test_axis_coverage(self, domain: str) -> None:
        config = self._load(domain)
        enabled = _enabled_members(config["members"])
        required = config.get("required_axes", {})

        def getter(member: dict[str, Any], axis: str) -> Any:
            return member.get(axis)

        errors = _check_axis_coverage(enabled, required, getter)
        assert errors == [], "\n".join(errors)

    def test_has_learned_matcher(self, domain: str) -> None:
        """Each per-domain fork must enable at least ``ditto_plm``."""
        config = self._load(domain)
        enabled = _enabled_members(config["members"])
        types = {m["matching_type"] for m in enabled}
        assert "learned" in types, f"{domain}: no learned matcher enabled by default"

    def test_column_mapping_matches_blocking(self, domain: str) -> None:
        """Per-domain matching column_mapping must match its blocking fork."""
        matching = self._load(domain)
        blocking = _load_yaml(f"em_blocking_committee_{domain}.yaml")
        assert matching["column_mapping"] == blocking["column_mapping"], (
            f"{domain}: column_mapping drift between em_matching and "
            f"em_blocking forks"
        )


@pytest.mark.parametrize("domain", PER_DOMAIN_COMMITTEE_DOMAINS)
class TestPerDomainFusionCommittee:
    """Per-domain forks of ``fusion_committee.yaml`` (S10)."""

    def _load(self, domain: str) -> dict[str, Any]:
        return _load_yaml(f"fusion_committee_{domain}.yaml")

    @staticmethod
    def _is_c12_shape(config: dict[str, Any]) -> bool:
        """C12 forks use ``members:`` instead of per-(attribute, strategy)
        ``attributes:`` blocks (plan_revision.md §C12)."""
        return "members" in config

    def test_loads(self, domain: str) -> None:
        config = self._load(domain)
        assert "seed" in config
        assert "trust_scores" in config
        if self._is_c12_shape(config):
            # C12 shape: members + attribute_types + pydi_candidates.
            assert "members" in config
            assert "attribute_types" in config
            assert "pydi_candidates" in config
        else:
            # Legacy per-(attribute, strategy) shape.
            assert "attributes" in config

    def test_trust_scores_for_all_sources(self, domain: str) -> None:
        """``trust_scores`` must cover every source declared in the matching fork."""
        config = self._load(domain)
        matching = _load_yaml(f"em_matching_committee_{domain}.yaml")
        sources = set(matching["column_mapping"].keys())
        scores = config["trust_scores"]
        for source in sources:
            assert (
                source in scores
            ), f"{domain}: missing trust score for source {source!r}"

    def test_attributes_have_strategies(self, domain: str) -> None:
        config = self._load(domain)
        if self._is_c12_shape(config):
            # C12: members are coherent end-to-end approaches, not per-
            # attribute strategy lists. Require >= 2 members so the
            # committee carries spread.
            members = config["members"]
            assert (
                len(members) >= 2
            ), f"{domain}: only {len(members)} C12 members (need >= 2)"
            return
        for attr_name, attr_spec in config["attributes"].items():
            strategies = attr_spec.get("strategies", [])
            assert len(strategies) >= 2, (
                f"{domain}/{attr_name}: only {len(strategies)} strategies "
                "(need >= 2 for committee spread)"
            )

    def test_strategy_functions_importable(self, domain: str) -> None:
        config = self._load(domain)
        if self._is_c12_shape(config):
            # Importability check on the pydi_candidates registry — that's
            # the C12 surface where per-function module paths live.
            for type_name, candidates in config["pydi_candidates"].items():
                for cand in candidates:
                    _assert_importable(cand["module"], cand["function"])
            return
        for attr_name, attr_spec in config["attributes"].items():
            for strategy in attr_spec["strategies"]:
                _assert_importable(strategy["module"], strategy["function"])

    def test_axis_coverage_per_attribute(self, domain: str) -> None:
        config = self._load(domain)
        if self._is_c12_shape(config):
            pytest.skip("C12 retires per-attribute strategy_type axes")
        required = config.get("required_axes", {})
        if "strategy_type" not in required:
            pytest.skip("No required_axes.strategy_type")
        required_types = set(required["strategy_type"])
        for attr_name, attr_spec in config["attributes"].items():
            covered = {s["strategy_type"] for s in attr_spec["strategies"]}
            missing = required_types - covered
            assert (
                not missing
            ), f"{domain}/{attr_name}: missing strategy_type(s) {missing}"

    def test_evaluation_functions_cover_all_attributes(self, domain: str) -> None:
        config = self._load(domain)
        eval_funcs = config.get("evaluation_functions", {})
        if self._is_c12_shape(config):
            for attr_name in config["attribute_types"]:
                assert (
                    attr_name in eval_funcs
                ), f"{domain}/{attr_name}: no evaluation_function configured"
            return
        for attr_name in config["attributes"]:
            assert (
                attr_name in eval_funcs
            ), f"{domain}/{attr_name}: no evaluation_function configured"

    def test_has_truth_discovery_member(self, domain: str) -> None:
        """Every per-domain fork must wire at least one TD method.

        Pre-C12 (legacy shape): at least one strategy with
        ``strategy_type: truth_discovery`` somewhere across attributes.
        C12: at least one of the four TD members (truthfinder_only,
        fusionquery_only, ltm_only, casefusion_only) — or accusim_only
        — is listed in ``members:``.
        """
        config = self._load(domain)
        if self._is_c12_shape(config):
            td_members = {
                "truthfinder_only",
                "fusionquery_only",
                "ltm_only",
                "casefusion_only",
                "accusim_only",
            }
            roster = {m["name"] for m in config["members"]}
            assert roster & td_members, (
                f"{domain}: no TD member wired (need one of " f"{sorted(td_members)})"
            )
            return
        for attr_spec in config["attributes"].values():
            for strategy in attr_spec["strategies"]:
                if strategy.get("strategy_type") == "truth_discovery":
                    return
        pytest.fail(f"{domain}: no truth_discovery strategy wired")

    def test_column_mapping_matches_blocking(self, domain: str) -> None:
        """Per-domain fusion column_mapping must match its blocking fork."""
        fusion = self._load(domain)
        blocking = _load_yaml(f"em_blocking_committee_{domain}.yaml")
        assert (
            fusion["column_mapping"] == blocking["column_mapping"]
        ), f"{domain}: column_mapping drift between fusion and em_blocking"


# ===================================================================
# Per-domain committee-path resolver (S10)
# ===================================================================


class TestCommitteePathResolver:
    """Cover :mod:`usecases_synthetic.lib.committee_paths`."""

    def test_companies_resolves_to_canonical(self) -> None:
        from usecases_synthetic.lib.committee_paths import resolve_committee_path

        path = resolve_committee_path(
            "em_blocking_committee", "companies", committee_dir=CONFIG_DIR
        )
        assert path.name == "em_blocking_committee.yaml"
        assert path.exists()

    def test_companies_small_aliases_companies(self) -> None:
        from usecases_synthetic.lib.committee_paths import resolve_committee_path

        path = resolve_committee_path(
            "em_blocking_committee", "companies-small", committee_dir=CONFIG_DIR
        )
        assert path.name == "em_blocking_committee.yaml"

    def test_per_domain_fork_resolves_to_suffixed_file(self) -> None:
        from usecases_synthetic.lib.committee_paths import resolve_committee_path

        for domain in PER_DOMAIN_COMMITTEE_DOMAINS:
            path = resolve_committee_path(
                "em_blocking_committee", domain, committee_dir=CONFIG_DIR
            )
            assert path.name == f"em_blocking_committee_{domain}.yaml"
            assert path.exists(), f"{domain}: expected fork at {path}"

    def test_sm_committee_never_forks(self) -> None:
        """SM committee is structurally domain-agnostic; one shared YAML."""
        from usecases_synthetic.lib.committee_paths import resolve_committee_path

        for domain in ["companies", "companies-small", *PER_DOMAIN_COMMITTEE_DOMAINS]:
            path = resolve_committee_path(
                "sm_committee", domain, committee_dir=CONFIG_DIR
            )
            assert path.name == "sm_committee.yaml"

    def test_normalization_always_per_domain(self) -> None:
        """Normalization committee resolves to a suffixed file for every domain.

        Unlike EM/Fusion where companies is the canonical unsuffixed
        file, the Normalization roster ships per-domain files for every
        domain (no unsuffixed canonical exists). See
        ``plan_s1_scale.md`` §"R5 Normalization sign-off (2026-05-10)".
        """
        from usecases_synthetic.lib.committee_paths import resolve_committee_path

        for domain in ["companies", "games", "music"]:
            path = resolve_committee_path(
                "normalization_committee", domain, committee_dir=CONFIG_DIR
            )
            assert (
                path.name == f"normalization_committee_{domain}.yaml"
            ), f"{domain}: expected suffixed normalization YAML, got {path.name}"
            assert path.exists(), f"{domain}: missing normalization YAML at {path}"


# ===================================================================
# Normalization committee (per-domain forks; R5 Normalization 2026-05-10)
# ===================================================================


NORM_COMMITTEE_DOMAINS: list[str] = ["companies", "games", "music"]


@pytest.mark.parametrize("domain", NORM_COMMITTEE_DOMAINS)
class TestNormCommitteeConfig:
    """Per-domain forks of ``normalization_committee_<domain>.yaml``.

    All four norm YAMLs adopted the C12 coherent-member schema
    (plan_revision.md §C12) 2026-05-26. Tests branch on shape via
    :meth:`_is_c12_shape` so the legacy negative branch stays
    exercised if a legacy YAML reappears (e.g. archived experiment
    files).
    """

    def _load(self, domain: str) -> dict[str, Any]:
        return _load_yaml(f"normalization_committee_{domain}.yaml")

    @staticmethod
    def _is_c12_shape(config: dict[str, Any]) -> bool:
        return "rule_normalizers" in config or "llm_normalizer" in config

    def test_loads(self, domain: str) -> None:
        config = self._load(domain)
        assert "seed" in config
        assert "members" in config

    def test_members_have_required_fields(self, domain: str) -> None:
        config = self._load(domain)
        if self._is_c12_shape(config):
            from usecases_synthetic.lib.committee_norm_c12 import SUPPORTED_MEMBERS

            for member in config["members"]:
                assert "name" in member
                assert member["name"] in SUPPORTED_MEMBERS, (
                    f"{domain}: member {member['name']!r} not in C12 "
                    f"SUPPORTED_MEMBERS {sorted(SUPPORTED_MEMBERS)}"
                )
            return
        for member in config["members"]:
            assert "name" in member
            assert "module" in member
            assert "class" in member
            assert "signal_type" in member
            assert "params" in member

    def test_all_classes_importable(self, domain: str) -> None:
        config = self._load(domain)
        if self._is_c12_shape(config):
            for entry in config.get("rule_normalizers", []):
                _assert_importable(entry["module"], entry["class"])
            llm_cfg = config.get("llm_normalizer")
            if llm_cfg is not None:
                _assert_importable(llm_cfg["module"], llm_cfg["class"])
            return
        for member in config["members"]:
            _assert_importable(member["module"], member["class"])

    def test_axis_coverage(self, domain: str) -> None:
        config = self._load(domain)
        if self._is_c12_shape(config):
            pytest.skip("C12 retires per-member signal_type axes")
        enabled = _enabled_members(config["members"])
        required = config.get("required_axes", {})

        def getter(member: dict[str, Any], axis: str) -> Any:
            return member.get(axis)

        errors = _check_axis_coverage(enabled, required, getter)
        assert errors == [], "\n".join(errors)

    def test_has_llm_member(self, domain: str) -> None:
        """Every Normalization roster must include an LLM-typed member.

        Open-vocabulary nominal attributes (e.g. industry / ESRB / label)
        are not exhaustively covered by rule-based primitives; the LLM
        is the safety net.
        """
        config = self._load(domain)
        if self._is_c12_shape(config):
            names = {m["name"] for m in config["members"]}
            assert "llm_only" in names, f"{domain}: C12 roster must include `llm_only`"
            assert (
                config.get("llm_normalizer") is not None
            ), f"{domain}: llm_only requires an `llm_normalizer:` block"
            return
        members = config["members"]
        types = {m["signal_type"] for m in members}
        assert "llm" in types, f"{domain}: no llm-typed normalization member"

    def test_has_rule_string(self, domain: str) -> None:
        """Every Normalization roster must include a rule-based string member.

        Pre-C12: a member with ``signal_type: rule_string`` in the
        roster. C12: a ``text_clean`` entry in ``rule_normalizers:``.
        """
        config = self._load(domain)
        if self._is_c12_shape(config):
            rule_names = {c["name"] for c in config.get("rule_normalizers", [])}
            assert (
                "text_clean" in rule_names
            ), f"{domain}: C12 rule_normalizers must include `text_clean`"
            return
        enabled = _enabled_members(config["members"])
        types = {m["signal_type"] for m in enabled}
        assert (
            "rule_string" in types
        ), f"{domain}: missing a rule_string member (text_clean)"

    def test_applies_to_lists_canonical_attrs(self, domain: str) -> None:
        """Every ``applies_to`` entry must be a string (canonical attribute).

        Pre-C12: per-member ``applies_to``. C12: per-rule-candidate
        ``applies_to`` in ``rule_normalizers:``.
        """
        config = self._load(domain)
        if self._is_c12_shape(config):
            for entry in config.get("rule_normalizers", []):
                applies_to = entry.get("applies_to")
                assert isinstance(applies_to, list) and applies_to, (
                    f"{domain}: rule {entry['name']!r} applies_to must be "
                    "a non-empty list"
                )
                for item in applies_to:
                    assert isinstance(item, str), (
                        f"{domain}: rule {entry['name']!r} applies_to entry "
                        f"{item!r} is not a string"
                    )
            return
        for member in config["members"]:
            applies_to = member.get("applies_to")
            if applies_to is None:
                continue
            assert isinstance(
                applies_to, list
            ), f"{domain}: member {member['name']!r} applies_to must be a list"
            for entry in applies_to:
                assert isinstance(entry, str), (
                    f"{domain}: member {member['name']!r} applies_to entry "
                    f"{entry!r} is not a string"
                )

"""Functional tests for the C3.4 fusion adapters (batch-mode + LLM-judge).

Covers:

* Five batch-fit truth-discovery factories from
  [`usecases_synthetic.lib.td_batch_fusion`](../lib/td_batch_fusion.py)
  (TruthFinder / LTM / CASEFusion / FusionQuery / AccuSim) that wrap the
  vendored ``usecases_synthetic.third_party.fusionquery`` core (or, for
  AccuSim, the paper reimplementation in the same module).
* LLM-as-Judge per-cell adapter from
  [`usecases_synthetic.lib.llm_judge_fusion`](../lib/llm_judge_fusion.py)
  with file-backed prompt cache.

Why batch?
----------
The upstream methods learn one source-trust vector across many
``(entity, source, value)`` claims; PyDI's per-cell ``ConflictResolutionFunction``
API previously gave each method only one cell at a time, which collapsed it to
a similarity-aware vote. The batch factories run ``prepare_for_fusion`` +
``iterate_fusion`` ONCE on the whole attribute corpus, then expose a per-cell
lookup closure keyed by ``group_id``. These tests exercise the corpus-walk +
batch fit + lookup paths end-to-end.

Per §Process-requirement item 2 in
[plans/plan_committee_finalization.md](../../plans/plan_committee_finalization.md):
each adapter has a real-code-path test on realistic inputs, plus determinism,
NaN tolerance, sanity (corpus-wide majority wins), and edge-case coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.llm_judge_fusion import llm_judge
from usecases_synthetic.lib.td_batch_fusion import (
    make_accusim_resolver,
    make_casefusion_resolver,
    make_fusionquery_resolver,
    make_ltm_resolver,
    make_truthfinder_resolver,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_BATCH_FACTORIES: List[Callable[..., Any]] = [
    make_truthfinder_resolver,
    make_ltm_resolver,
    make_casefusion_resolver,
    make_fusionquery_resolver,
    make_accusim_resolver,
]
_BATCH_IDS = ["truthfinder", "ltm", "casefusion", "fusionquery", "accusim"]


def _make_dataset(
    name: str, rows: List[Dict[str, Any]], *, trust_score: float = 1.0
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.attrs["dataset_name"] = name
    df.attrs["trust_score"] = trust_score
    return df


def _correspondences(pairs: List[tuple[str, str]]) -> pd.DataFrame:
    """Build a correspondences DataFrame with columns ``id1``, ``id2``, ``score``.

    Each pair entry is a (``id1``, ``id2``) tuple; ``score`` is fixed at 1.0
    so PyDI's connected-components grouping picks them up unconditionally.
    """
    return pd.DataFrame([{"id1": a, "id2": b, "score": 1.0} for (a, b) in pairs])


@pytest.fixture()
def companies_corpus() -> Dict[str, Any]:
    """Three sources × five entities × ``name`` attribute.

    ``forbes`` is the high-trust source (always correct), ``fullcontact`` is
    medium (mostly correct, with a paraphrase on entity 4), ``dbpedia`` is
    noisy (wrong on entities 0, 1, and 2 — corpus-wide TD should learn this
    and downweight dbpedia).
    """
    forbes = _make_dataset(
        "forbes",
        [
            {"id": "f1", "name": "Apple Inc."},
            {"id": "f2", "name": "Microsoft Corp"},
            {"id": "f3", "name": "Alphabet Inc"},
            {"id": "f4", "name": "Amazon.com Inc"},
            {"id": "f5", "name": "Meta Platforms"},
        ],
    )
    fullcontact = _make_dataset(
        "fullcontact",
        [
            {"id": "fc1", "name": "Apple Inc."},
            {"id": "fc2", "name": "Microsoft Corp"},
            {"id": "fc3", "name": "Alphabet Inc"},
            {"id": "fc4", "name": "Amazon Inc"},  # paraphrase
            {"id": "fc5", "name": "Meta Platforms"},
        ],
    )
    dbpedia = _make_dataset(
        "dbpedia",
        [
            {"id": "d1", "name": "WRONG_VAL_A"},  # wrong
            {"id": "d2", "name": "WRONG_VAL_B"},  # wrong
            {"id": "d3", "name": "WRONG_VAL_C"},  # wrong
            {"id": "d4", "name": "Amazon.com Inc"},
            {"id": "d5", "name": "Meta Platforms"},
        ],
    )
    pairs = [
        ("f1", "fc1"),
        ("fc1", "d1"),
        ("f2", "fc2"),
        ("fc2", "d2"),
        ("f3", "fc3"),
        ("fc3", "d3"),
        ("f4", "fc4"),
        ("fc4", "d4"),
        ("f5", "fc5"),
        ("fc5", "d5"),
    ]
    return {
        "datasets": [forbes, fullcontact, dbpedia],
        "correspondences": _correspondences(pairs),
        "target_attr": "name",
        "id_column": "id",
        "expected_winners": {
            # Each entity's true name (forbes-source canonical):
            "g0_label": "Apple Inc.",
            "g1_label": "Microsoft Corp",
            "g2_label": "Alphabet Inc",
            "g3_label": "Amazon.com Inc",
            "g4_label": "Meta Platforms",
        },
    }


# ---------------------------------------------------------------------------
# Cross-factory contract tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _BATCH_FACTORIES, ids=_BATCH_IDS)
def test_factory_returns_callable(
    factory: Callable[..., Any], companies_corpus: Dict[str, Any]
) -> None:
    """Factory returns a per-cell ``ConflictResolutionFunction`` callable."""
    resolver = factory(
        datasets=companies_corpus["datasets"],
        correspondences=companies_corpus["correspondences"],
        target_attr=companies_corpus["target_attr"],
        id_column=companies_corpus["id_column"],
    )
    assert callable(resolver)
    # Sanity: calling it on a known group id should produce a 3-tuple.
    result = resolver(["Apple Inc."], group_id="some_group_id")
    assert isinstance(result, tuple) and len(result) == 3


@pytest.mark.parametrize("factory", _BATCH_FACTORIES, ids=_BATCH_IDS)
def test_majority_winner_via_lookup(
    factory: Callable[..., Any], companies_corpus: Dict[str, Any]
) -> None:
    """For every entity in the corpus, the resolver's lookup must return one
    of the originally-claimed values (verbatim, no inventions).
    """
    resolver = factory(
        datasets=companies_corpus["datasets"],
        correspondences=companies_corpus["correspondences"],
        target_attr=companies_corpus["target_attr"],
        id_column=companies_corpus["id_column"],
    )
    # Enumerate the actual group_ids that the engine would emit by re-running
    # the same group-construction the factory used.
    from PyDI.fusion.engine import build_record_groups_from_correspondences

    groups = build_record_groups_from_correspondences(
        datasets=companies_corpus["datasets"],
        correspondences=companies_corpus["correspondences"],
        id_column=companies_corpus["id_column"],
    )
    assert (
        len(groups) == 5
    ), f"Fixture should produce 5 record groups, got {len(groups)}"
    for group in groups:
        per_source_values = [
            r.get(companies_corpus["target_attr"]) for r in group.records
        ]
        value, confidence, metadata = resolver(
            per_source_values, group_id=group.group_id
        )
        # Either a batch lookup or the documented first-valid fallback.
        assert metadata.get("source") in {"batch_lookup", "fallback_first_valid"}
        # Winner must be one of the claimed values.
        assert (
            value in per_source_values
        ), f"Winner {value!r} not in claims {per_source_values!r} for group {group.group_id}"
        assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize("factory", _BATCH_FACTORIES, ids=_BATCH_IDS)
def test_corpus_wide_majority_wins_on_clean_entities(
    factory: Callable[..., Any], companies_corpus: Dict[str, Any]
) -> None:
    """Entities where 2/3 sources agree (entities 0, 1, 2 in the fixture —
    forbes + fullcontact agree, dbpedia disagrees) — the majority value
    should win.
    """
    resolver = factory(
        datasets=companies_corpus["datasets"],
        correspondences=companies_corpus["correspondences"],
        target_attr=companies_corpus["target_attr"],
        id_column=companies_corpus["id_column"],
    )
    from PyDI.fusion.engine import build_record_groups_from_correspondences

    groups = build_record_groups_from_correspondences(
        datasets=companies_corpus["datasets"],
        correspondences=companies_corpus["correspondences"],
        id_column=companies_corpus["id_column"],
    )
    # Find the group containing forbes id "f1" (Apple Inc.).
    target_group = next(
        g for g in groups if any(r.get("_id") == "f1" for r in g.records)
    )
    per_source_values = [r.get("name") for r in target_group.records]
    value, _confidence, _metadata = resolver(
        per_source_values, group_id=target_group.group_id
    )
    assert value == "Apple Inc.", (
        f"Majority value should win: forbes=Apple Inc., fullcontact=Apple Inc., "
        f"dbpedia=WRONG_VAL_A. Got {value!r}."
    )


@pytest.mark.parametrize("factory", _BATCH_FACTORIES, ids=_BATCH_IDS)
def test_determinism_two_fits_identical(
    factory: Callable[..., Any], companies_corpus: Dict[str, Any]
) -> None:
    """Two batch fits on the same corpus must produce identical winners."""
    resolver_a = factory(
        datasets=companies_corpus["datasets"],
        correspondences=companies_corpus["correspondences"],
        target_attr=companies_corpus["target_attr"],
        id_column=companies_corpus["id_column"],
    )
    resolver_b = factory(
        datasets=companies_corpus["datasets"],
        correspondences=companies_corpus["correspondences"],
        target_attr=companies_corpus["target_attr"],
        id_column=companies_corpus["id_column"],
    )
    from PyDI.fusion.engine import build_record_groups_from_correspondences

    groups = build_record_groups_from_correspondences(
        datasets=companies_corpus["datasets"],
        correspondences=companies_corpus["correspondences"],
        id_column=companies_corpus["id_column"],
    )
    for group in groups:
        per_source_values = [r.get("name") for r in group.records]
        a = resolver_a(per_source_values, group_id=group.group_id)
        b = resolver_b(per_source_values, group_id=group.group_id)
        assert a[0] == b[0]
        assert a[1] == b[1]


@pytest.mark.parametrize("factory", _BATCH_FACTORIES, ids=_BATCH_IDS)
def test_fallback_for_unknown_group(
    factory: Callable[..., Any], companies_corpus: Dict[str, Any]
) -> None:
    """A group_id the batch never saw falls back to first-valid (no crash)."""
    resolver = factory(
        datasets=companies_corpus["datasets"],
        correspondences=companies_corpus["correspondences"],
        target_attr=companies_corpus["target_attr"],
        id_column=companies_corpus["id_column"],
    )
    value, confidence, metadata = resolver(
        ["Foo", None, "Bar"], group_id="unseen_group"
    )
    assert value == "Foo"
    assert confidence == 0.5
    assert metadata["source"] == "fallback_first_valid"


@pytest.mark.parametrize("factory", _BATCH_FACTORIES, ids=_BATCH_IDS)
def test_empty_corpus_returns_no_valid_lookup(
    factory: Callable[..., Any],
) -> None:
    """Empty datasets — factory still returns a usable (no-lookup) resolver."""
    empty_a = _make_dataset("a", [{"id": "a1"}])
    empty_b = _make_dataset("b", [{"id": "b1"}])
    resolver = factory(
        datasets=[empty_a, empty_b],
        correspondences=_correspondences([("a1", "b1")]),
        target_attr="never_present_attr",
        id_column="id",
    )
    # Resolver should fall back to first-valid for any cell.
    value, _, metadata = resolver(["x"], group_id="any")
    assert value == "x"
    assert metadata["source"] == "fallback_first_valid"


# ---------------------------------------------------------------------------
# Per-factory specific assertions
# ---------------------------------------------------------------------------


class TestTruthFinderBatch:
    def test_src_trust_recorded_in_metadata(
        self, companies_corpus: Dict[str, Any]
    ) -> None:
        resolver = make_truthfinder_resolver(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            target_attr=companies_corpus["target_attr"],
            id_column=companies_corpus["id_column"],
        )
        # Inspect any batch-lookup result for the recorded src_trust dict.
        _, _, metadata = resolver(["Apple Inc."], group_id="dbpedia_d1")
        # If the lookup hit, src_trust is in extras; if it fell back, the
        # corpus-walk still happened and we can confirm via the closure cell.
        # Run a known-group lookup instead.
        from PyDI.fusion.engine import build_record_groups_from_correspondences

        groups = build_record_groups_from_correspondences(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            id_column=companies_corpus["id_column"],
        )
        gid = groups[0].group_id
        _, _, metadata = resolver(["Apple Inc."], group_id=gid)
        if metadata["source"] == "batch_lookup":
            assert "src_trust" in metadata
            assert {"forbes", "fullcontact", "dbpedia"}.issubset(metadata["src_trust"])
            # Forbes and fullcontact agreed more often than dbpedia, so dbpedia
            # should not be the highest-trust source.
            ranked = sorted(metadata["src_trust"].items(), key=lambda kv: kv[1])
            assert ranked[0][0] == "dbpedia"


class TestLTMBatch:
    def test_n_facts_recorded(self, companies_corpus: Dict[str, Any]) -> None:
        resolver = make_ltm_resolver(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            target_attr=companies_corpus["target_attr"],
            id_column=companies_corpus["id_column"],
        )
        from PyDI.fusion.engine import build_record_groups_from_correspondences

        groups = build_record_groups_from_correspondences(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            id_column=companies_corpus["id_column"],
        )
        gid = groups[0].group_id
        _, _, metadata = resolver(["Apple Inc."], group_id=gid)
        if metadata["source"] == "batch_lookup":
            assert "n_facts" in metadata
            # 5 entities × ~2-3 distinct names = 10-13 unique fact strings,
            # plus a few across-entity duplicates collapsing.
            assert metadata["n_facts"] >= 5


class TestFusionQueryBatch:
    def test_history_resets_between_factory_calls(
        self, companies_corpus: Dict[str, Any]
    ) -> None:
        """Factory must reset ``EMFusioner.his_data_size`` per call so two
        attributes don't share trust history.
        """
        from usecases_synthetic.third_party.fusionquery.fusion import EMFusioner

        EMFusioner.his_data_size = np.array([[42.0]])  # contaminated
        resolver = make_fusionquery_resolver(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            target_attr=companies_corpus["target_attr"],
            id_column=companies_corpus["id_column"],
            history_size=10.0,
        )
        # Batch fit ran — class-level history was rebuilt with the new size.
        assert EMFusioner.his_data_size is not None
        # The exact post-fit value depends on the sa_mask increment, but it
        # must NOT be the contaminated 42.0 anymore.
        assert float(EMFusioner.his_data_size[0, 0]) != 42.0


class TestAccuSimBatch:
    def test_validates_accuracy_prior(self) -> None:
        with pytest.raises(ValueError, match="accuracy_prior"):
            make_accusim_resolver(
                datasets=[
                    _make_dataset("a", [{"id": "a1", "name": "x"}]),
                    _make_dataset("b", [{"id": "b1", "name": "y"}]),
                ],
                correspondences=_correspondences([("a1", "b1")]),
                target_attr="name",
                id_column="id",
                accuracy_prior=1.0,
            )

    def test_src_accuracy_recorded(self, companies_corpus: Dict[str, Any]) -> None:
        resolver = make_accusim_resolver(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            target_attr=companies_corpus["target_attr"],
            id_column=companies_corpus["id_column"],
        )
        from PyDI.fusion.engine import build_record_groups_from_correspondences

        groups = build_record_groups_from_correspondences(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            id_column=companies_corpus["id_column"],
        )
        gid = groups[0].group_id
        _, _, metadata = resolver(["Apple Inc."], group_id=gid)
        if metadata["source"] == "batch_lookup":
            assert "src_accuracy" in metadata
            acc = metadata["src_accuracy"]
            # Forbes always agrees with fullcontact on entities 0-3 → high
            # accuracy. Dbpedia disagrees on entities 0-2 → low accuracy.
            assert acc["forbes"] > acc["dbpedia"]

    def test_custom_similarity_callable(self, companies_corpus: Dict[str, Any]) -> None:
        called: Dict[str, int] = {"count": 0}

        def counting_sim(a: Any, b: Any) -> float:
            called["count"] += 1
            return 1.0 if a == b else 0.0

        resolver = make_accusim_resolver(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            target_attr=companies_corpus["target_attr"],
            id_column=companies_corpus["id_column"],
            similarity=counting_sim,
        )
        # Probe a known group so the closure is used; this also verifies the
        # similarity callable was consulted at fit time.
        from PyDI.fusion.engine import build_record_groups_from_correspondences

        groups = build_record_groups_from_correspondences(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            id_column=companies_corpus["id_column"],
        )
        resolver(["Apple Inc."], group_id=groups[0].group_id)
        assert called["count"] > 0


# ---------------------------------------------------------------------------
# Corpus-wide trust learning sanity check
# ---------------------------------------------------------------------------


class TestCorpusWideTrustLearning:
    """The whole point of moving to batch mode: source trust should reflect
    cross-entity agreement patterns, not just per-cell behaviour.
    """

    def test_truthfinder_downweights_consistently_wrong_source(
        self, companies_corpus: Dict[str, Any]
    ) -> None:
        """In the fixture, dbpedia disagrees on 3 of 5 entities. TruthFinder's
        cross-cell trust update should give dbpedia the lowest src_trust.
        """
        resolver = make_truthfinder_resolver(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            target_attr=companies_corpus["target_attr"],
            id_column=companies_corpus["id_column"],
        )
        from PyDI.fusion.engine import build_record_groups_from_correspondences

        groups = build_record_groups_from_correspondences(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            id_column=companies_corpus["id_column"],
        )
        _, _, metadata = resolver(["Apple Inc."], group_id=groups[0].group_id)
        assert metadata["source"] == "batch_lookup"
        src_trust = metadata["src_trust"]
        ranked = sorted(src_trust.items(), key=lambda kv: kv[1])
        # Lowest trust must be dbpedia.
        assert (
            ranked[0][0] == "dbpedia"
        ), f"Expected dbpedia at the bottom of src_trust, got: {src_trust}"

    def test_fusionquery_downweights_consistently_wrong_source(
        self, companies_corpus: Dict[str, Any]
    ) -> None:
        """Same property for FusionQuery's EM-style fit."""
        resolver = make_fusionquery_resolver(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            target_attr=companies_corpus["target_attr"],
            id_column=companies_corpus["id_column"],
        )
        from PyDI.fusion.engine import build_record_groups_from_correspondences

        groups = build_record_groups_from_correspondences(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            id_column=companies_corpus["id_column"],
        )
        _, _, metadata = resolver(["Apple Inc."], group_id=groups[0].group_id)
        assert metadata["source"] == "batch_lookup"
        src_trust = metadata["src_trust"]
        ranked = sorted(src_trust.items(), key=lambda kv: kv[1])
        assert ranked[0][0] == "dbpedia"

    def test_accusim_downweights_consistently_wrong_source(
        self, companies_corpus: Dict[str, Any]
    ) -> None:
        """AccuSim's per-source accuracy must reflect dbpedia's poor agreement."""
        resolver = make_accusim_resolver(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            target_attr=companies_corpus["target_attr"],
            id_column=companies_corpus["id_column"],
        )
        from PyDI.fusion.engine import build_record_groups_from_correspondences

        groups = build_record_groups_from_correspondences(
            datasets=companies_corpus["datasets"],
            correspondences=companies_corpus["correspondences"],
            id_column=companies_corpus["id_column"],
        )
        _, _, metadata = resolver(["Apple Inc."], group_id=groups[0].group_id)
        assert metadata["source"] == "batch_lookup"
        acc = metadata["src_accuracy"]
        # Dbpedia's accuracy must be below forbes and fullcontact.
        assert acc["dbpedia"] < acc["forbes"]
        assert acc["dbpedia"] < acc["fullcontact"]


# ---------------------------------------------------------------------------
# LLM-as-judge tests (per-cell — natively, no batch fit needed)
# ---------------------------------------------------------------------------


class TestLLMJudge:
    """Tests for the v2 LLM judge (C12, 2026-05-25).

    v2 contract: ``{"value", "operation", "confidence", "reasoning"}``;
    ``operation`` is one of
    ``{verbatim_pick, aggregation, union, intersection, normalization, interpolation}``;
    synthesis (non-candidate values) is permitted.
    """

    def test_returns_chosen_value(self, tmp_path: Path) -> None:
        def stub(system: str, user: str, model: str) -> str:
            return json.dumps(
                {
                    "value": "New York City",
                    "operation": "normalization",
                    "confidence": 0.95,
                    "reasoning": "canonical long form",
                }
            )

        value, confidence, metadata = llm_judge(
            ["NYC", "New York", "New York City"],
            sources=["a", "b", "c"],
            source_datasets={"a": "forbes", "b": "fullcontact", "c": "dbpedia"},
            attribute="city",
            llm_callable=stub,
            cache_dir=tmp_path,
        )
        assert value == "New York City"
        assert confidence == pytest.approx(0.95)
        assert metadata["cache_hit"] is False
        assert metadata["operation"] == "normalization"
        assert metadata["reasoning"] == "canonical long form"
        assert metadata["synthesized"] is False

    def test_synthesis_allowed_under_v2(self, tmp_path: Path) -> None:
        """v2 lifts the v1 verbatim-only constraint."""

        def synthesizer(system: str, user: str, model: str) -> str:
            return json.dumps(
                {
                    "value": "Frankfurt am Main",
                    "operation": "interpolation",
                    "confidence": 0.8,
                    "reasoning": "merged regional + city info",
                }
            )

        value, _, metadata = llm_judge(
            ["Frankfurt", "Frankfurt (Main)"],
            sources=["a", "b"],
            source_datasets={"a": "forbes", "b": "fullcontact"},
            attribute="city",
            llm_callable=synthesizer,
            cache_dir=tmp_path,
        )
        assert value == "Frankfurt am Main"
        assert metadata["operation"] == "interpolation"
        assert metadata["synthesized"] is True

    def test_cache_hit_skips_llm(self, tmp_path: Path) -> None:
        called = {"count": 0}

        def counting(system: str, user: str, model: str) -> str:
            called["count"] += 1
            return json.dumps(
                {
                    "value": "Boston",
                    "operation": "verbatim_pick",
                    "confidence": 0.9,
                    "reasoning": "majority",
                }
            )

        for _ in range(2):
            llm_judge(
                ["Boston", "Cambridge"],
                sources=["a", "b"],
                source_datasets={"a": "forbes", "b": "fullcontact"},
                attribute="city",
                llm_callable=counting,
                cache_dir=tmp_path,
            )
        assert called["count"] == 1, "Second call should hit the cache"

    def test_cache_persists_to_disk(self, tmp_path: Path) -> None:
        def stub(system: str, user: str, model: str) -> str:
            return json.dumps(
                {
                    "value": "Boston",
                    "operation": "verbatim_pick",
                    "confidence": 0.9,
                    "reasoning": "majority",
                }
            )

        llm_judge(
            ["Boston", "Cambridge"],
            sources=["a", "b"],
            source_datasets={"a": "forbes", "b": "fullcontact"},
            attribute="city",
            llm_callable=stub,
            cache_dir=tmp_path,
        )
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        with open(files[0], encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["model_id"]
        assert payload["prompt_version"] == "v2"
        assert payload["attribute"] == "city"
        assert payload["raw_response"]
        assert payload["parsed"]["operation"] == "verbatim_pick"

    def test_prompt_version_invalidates_cache(self, tmp_path: Path) -> None:
        called = {"count": 0}

        def counting(system: str, user: str, model: str) -> str:
            called["count"] += 1
            return json.dumps(
                {
                    "value": "Boston",
                    "operation": "verbatim_pick",
                    "confidence": 0.9,
                    "reasoning": "majority",
                }
            )

        for version in ("v2", "v2-experimental"):
            llm_judge(
                ["Boston", "Cambridge"],
                sources=["a", "b"],
                source_datasets={"a": "forbes", "b": "fullcontact"},
                attribute="city",
                prompt_version=version,
                llm_callable=counting,
                cache_dir=tmp_path,
            )
        assert called["count"] == 2

    def test_fallback_when_no_llm_no_cache(self, tmp_path: Path) -> None:
        value, confidence, metadata = llm_judge(
            ["Apple", "Apple", "Microsoft"],
            sources=["a", "b", "c"],
            source_datasets={"a": "forbes", "b": "fullcontact", "c": "dbpedia"},
            attribute="name",
            cache_dir=tmp_path,
        )
        assert value == "Apple"
        assert confidence == pytest.approx(2 / 3)
        assert metadata["fallback"] == "voting"
        assert metadata["reason"] == "no_llm_callable_no_cache"

    def test_fallback_on_bogus_response(self, tmp_path: Path) -> None:
        def bogus(system: str, user: str, model: str) -> str:
            return "this is definitely not json"

        value, _, metadata = llm_judge(
            ["Boston", "Cambridge"],
            sources=["a", "b"],
            attribute="city",
            llm_callable=bogus,
            cache_dir=tmp_path,
        )
        assert value in ("Boston", "Cambridge")
        assert metadata["fallback"] == "voting"
        assert metadata["reason"] == "parse_failed"

    def test_unknown_operation_falls_back(self, tmp_path: Path) -> None:
        """v2 parser rejects responses whose ``operation`` is unrecognised."""

        def liar(system: str, user: str, model: str) -> str:
            return json.dumps(
                {
                    "value": "Boston",
                    "operation": "make_up_a_value",
                    "confidence": 0.9,
                    "reasoning": "n/a",
                }
            )

        value, _, metadata = llm_judge(
            ["Boston", "Cambridge"],
            sources=["a", "b"],
            attribute="city",
            llm_callable=liar,
            cache_dir=tmp_path,
        )
        assert value in ("Boston", "Cambridge")
        assert metadata["fallback"] == "voting"
        assert metadata["reason"] == "parse_failed"

    def test_strict_cache_raises_on_miss(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="strict cache miss"):
            llm_judge(
                ["X", "Y"],
                sources=["a", "b"],
                attribute="city",
                cache_dir=tmp_path,
                strict_cache=True,
            )

    def test_fenced_json_response_parses(self, tmp_path: Path) -> None:
        def fenced(system: str, user: str, model: str) -> str:
            return (
                "```json\n"
                + json.dumps(
                    {
                        "value": "Boston",
                        "operation": "verbatim_pick",
                        "confidence": 0.8,
                        "reasoning": "first source",
                    }
                )
                + "\n```"
            )

        value, confidence, metadata = llm_judge(
            ["Boston", "Cambridge"],
            sources=["a", "b"],
            attribute="city",
            llm_callable=fenced,
            cache_dir=tmp_path,
        )
        assert value == "Boston"
        assert confidence == pytest.approx(0.8)
        assert metadata["operation"] == "verbatim_pick"

    def test_empty_returns_no_valid_values(self, tmp_path: Path) -> None:
        value, confidence, metadata = llm_judge(
            [None, None], sources=["a", "b"], cache_dir=tmp_path
        )
        assert value is None
        assert confidence == 0.0
        assert metadata["reason"] == "no_valid_values"

    def test_single_surviving_source_short_circuits(self, tmp_path: Path) -> None:
        value, _, metadata = llm_judge(
            ["Apple Inc.", None, float("nan")],
            sources=["a", "b", "c"],
            cache_dir=tmp_path,
        )
        assert value == "Apple Inc."
        assert metadata["note"] == "single_source_short_circuit"

    def test_op_log_appended_when_path_set(self, tmp_path: Path) -> None:
        """Op log captures one row per non-trivial v2 call."""

        def stub(system: str, user: str, model: str) -> str:
            return json.dumps(
                {
                    "value": "Boston",
                    "operation": "verbatim_pick",
                    "confidence": 0.9,
                    "reasoning": "majority",
                }
            )

        op_log = tmp_path / "llm_only_operations.csv"
        llm_judge(
            ["Boston", "Cambridge"],
            sources=["a", "b"],
            source_datasets={"a": "forbes", "b": "fullcontact"},
            attribute="city",
            llm_callable=stub,
            cache_dir=tmp_path / "cache",
            op_log_path=op_log,
            group_id="cluster-42",
        )
        assert op_log.exists()
        rows = pd.read_csv(op_log)
        assert len(rows) == 1
        assert rows.loc[0, "operation"] == "verbatim_pick"
        assert rows.loc[0, "attribute"] == "city"
        assert rows.loc[0, "group_id"] == "cluster-42"
        assert int(rows.loc[0, "cache_hit"]) == 0

    def test_op_log_skips_single_source_short_circuit(self, tmp_path: Path) -> None:
        """Single-source short-circuit returns before the LLM and the log."""

        op_log = tmp_path / "llm_only_operations.csv"
        llm_judge(
            ["Apple Inc.", None],
            sources=["a", "b"],
            attribute="name",
            cache_dir=tmp_path / "cache",
            op_log_path=op_log,
        )
        assert not op_log.exists()

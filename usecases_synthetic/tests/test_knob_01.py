"""Tests for Knob 01 — Surface Augmentation Intensity.

Acceptance criteria (from module_06_knob_01.md):
1. At easy, all outputs are canonical forms (consistent, normalized)
2. At medium, outputs show abbreviation / reordering / deletion but token
   overlap >= 50% with original
3. At hard, LLM paraphrases cached and never re-queried on rerun
4. Anchor-survivor floor: >=1 source retains non-paraphrased primary per
   fusion-gold entity
5. ``paraphrase_value_for_knob_04`` callable is deterministic given same RNG
6. Collision index: cells with prior provenance are skipped
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.collision_index import CollisionIndex
from usecases_synthetic.lib.llm_cache import LLMCache, LLMCacheMiss
from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog
from usecases_synthetic.lib.surface_operators import (
    UNCHANGED_SENTINEL,
    VALID_TRANSFORM_FNS,
    _is_near_identity,
    abbreviate,
    build_first_token_index,
    build_openai_paraphrase_client,
    contamination_first_token_probe_passed,
    contamination_ngram_passed,
    eda_random_delete,
    eda_random_swap,
    llm_paraphrase,
    ngram_overlap_tokens,
    normalize_to_canonical,
    paraphrase_value_for_knob_04,
)
from usecases_synthetic.scripts.apply_knob_01_surface import (
    REALISED_COLUMNS,
    SKIPPED_COLUMNS,
    _token_jaccard_drop,
    apply_knob_01,
    build_realised_df,
    load_knob_01_config,
    write_outputs,
)

# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def small_sources() -> dict[str, pd.DataFrame]:
    """Three companies-like sources with entities 0..9 linked across sources."""
    names = [
        "International Business Machines",
        "Microsoft Corporation",
        "Apple Incorporated",
        "The Coca-Cola Company",
        "Amazon.com Incorporated",
        "General Electric Company",
        "Bank of America Corporation",
        "Tesla Incorporated",
        "Alphabet Incorporated",
        "Meta Platforms Incorporated",
    ]
    countries_long = [
        "United States",
        "United States",
        "United States of America",
        "United States",
        "United States",
        "United States",
        "United States of America",
        "United States",
        "United States",
        "United States",
    ]
    countries_short = ["USA" for _ in names]
    countries_canonical = ["United States" for _ in names]

    dbpedia = pd.DataFrame(
        {
            "identifier": [f"db_{i}" for i in range(10)],
            "name": names,
            "countryName": countries_canonical,
            "cityName": [
                "Armonk",
                "Redmond",
                "Cupertino",
                "Atlanta",
                "Seattle",
                "Boston",
                "Charlotte",
                "Austin",
                "Mountain View",
                "Menlo Park",
            ],
            "industryName": ["Technology"] * 10,
        }
    )
    dbpedia.attrs["dataset_name"] = "dbpedia"

    forbes = pd.DataFrame(
        {
            "Identifier": [f"fb_{i}" for i in range(10)],
            "Company": [
                n.replace("Incorporated", "Inc.").replace("Corporation", "Corp.")
                for n in names
            ],
            "Country": countries_long,
            "Sector": ["Technology"] * 10,
            "Industry": ["Software"] * 10,
        }
    )
    forbes.attrs["dataset_name"] = "forbes"

    fullcontact = pd.DataFrame(
        {
            "id": [f"fc_{i}" for i in range(10)],
            "name": names,
            "country": countries_short,
            "locality": [
                "Armonk",
                "Redmond",
                "Cupertino",
                "Atlanta",
                "Seattle",
                "Boston",
                "Charlotte",
                "Austin",
                "Mountain View",
                "Menlo Park",
            ],
        }
    )
    fullcontact.attrs["dataset_name"] = "fullcontact"

    return {"dbpedia": dbpedia, "forbes": forbes, "fullcontact": fullcontact}


@pytest.fixture
def small_entity_groups() -> dict[str, list[tuple[str, str]]]:
    """Entity groups linking the 3 sources for entities 0-9."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for i in range(10):
        groups[f"group_{i}"] = [
            ("dbpedia", f"db_{i}"),
            ("forbes", f"fb_{i}"),
            ("fullcontact", f"fc_{i}"),
        ]
    return groups


@pytest.fixture
def small_config() -> dict[str, Any]:
    """Minimal config for the dispatcher."""
    return {
        "id_columns": {
            "dbpedia": "identifier",
            "forbes": "Identifier",
            "fullcontact": "id",
        },
        "attribute_classes": {
            "dbpedia": {
                "name": "primary",
                "countryName": "key",
                "cityName": "key",
                "industryName": "categorical",
            },
            "forbes": {
                "Company": "primary",
                "Country": "key",
                "Sector": "categorical",
                "Industry": "categorical",
            },
            "fullcontact": {
                "name": "primary",
                "country": "key",
                "locality": "key",
            },
        },
        "attribute_mapping": {
            "dbpedia": {
                "name": "name",
                "countryName": "country",
                "cityName": "city",
                "industryName": "industry",
            },
            "forbes": {
                "Company": "name",
                "Country": "country",
                "Sector": "industry",
                "Industry": "industry",
            },
            "fullcontact": {
                "name": "name",
                "country": "country",
                "locality": "city",
            },
        },
        "paraphrase_rate_primary": {"easy": 0.0, "medium": 0.5, "hard": 0.9},
        "paraphrase_rate_key": {"easy": 0.0, "medium": 0.5, "hard": 0.9},
        "paraphrase_rate_secondary": {"easy": 0.0, "medium": 0.5, "hard": 0.9},
        "paraphrase_rate_categorical": {"easy": 0.0, "medium": 0.5, "hard": 0.9},
        "operator_mix": {
            "easy": {"normalize_to_canonical": 1.0},
            "medium": {
                "abbreviate": 2.0,
                "eda_random_swap": 1.0,
                "eda_random_delete": 1.0,
            },
            "hard": {
                "abbreviate": 1.0,
                "eda_random_swap": 1.0,
                "eda_random_delete": 1.0,
                "llm_paraphrase": 2.0,
            },
        },
        "baseline_above_target_rules": [
            {
                "source": "forbes",
                "attribute": "Country",
                "canonical_from": "dbpedia",
                "canonical_attribute": "countryName",
                "strategy": "shortest",
            },
        ],
        "abbreviation_table": {
            "Incorporated": "Inc.",
            "Corporation": "Corp.",
            "Company": "Co.",
            "International Business Machines": "IBM",
        },
        "stopword_list": ["inc", "inc.", "corp", "corp.", "the", "of"],
        "key_token_skiplist": {},
        "key_token_skiplist_global": [],
        "llm_prompt_version": "v1",
        "llm_model_id": "mock-model",
        "llm_temperature": 0.0,
        "anchor_survivor_floor": {
            "primary": True,
            "key": False,
            "secondary": False,
            "categorical": False,
        },
    }


# ---- Operator unit tests ----------------------------------------------------


class TestNormalizeToCanonical:
    def test_picks_shortest(self) -> None:
        result = normalize_to_canonical(
            "United States of America", ["United States", "USA"]
        )
        assert result is not None
        canonical, params = result
        assert canonical == "USA"
        assert params["strategy"] == "shortest"

    def test_no_change_returns_none(self) -> None:
        assert normalize_to_canonical("USA", ["USA", "USA"]) is None

    def test_empty_siblings_returns_none(self) -> None:
        assert normalize_to_canonical("USA", []) is None

    def test_deterministic_tie_break(self) -> None:
        r1 = normalize_to_canonical("XYZ", ["AAA", "BBB"])
        r2 = normalize_to_canonical("XYZ", ["BBB", "AAA"])
        assert r1 == r2
        assert r1 is not None
        assert r1[0] == "AAA"

    def test_most_frequent(self) -> None:
        result = normalize_to_canonical(
            "XYZ",
            ["USA", "USA", "United States"],
            strategy="most_frequent",
        )
        assert result is not None
        assert result[0] == "USA"


class TestAbbreviate:
    def test_expand_to_contract(self) -> None:
        result = abbreviate("Apple Incorporated", {"Incorporated": "Inc."})
        assert result is not None
        assert result[0] == "Apple Inc."
        assert result[1]["direction"] == "expand_to_contract"

    def test_contract_to_expand(self) -> None:
        result = abbreviate("Apple Inc.", {"Incorporated": "Inc."})
        assert result is not None
        assert result[0] == "Apple Incorporated"
        assert result[1]["direction"] == "contract_to_expand"

    def test_no_match(self) -> None:
        assert abbreviate("Nothing Here", {"Incorporated": "Inc."}) is None

    def test_table_lookup(self) -> None:
        result = abbreviate(
            "International Business Machines",
            {"International Business Machines": "IBM"},
        )
        assert result is not None
        assert result[0] == "IBM"


class TestEdaRandomSwap:
    def test_basic_swap(self) -> None:
        rng = np.random.default_rng(42)
        result = eda_random_swap(
            "Alpha Beta Gamma Delta", rng, stopwords=set(), key_tokens=set()
        )
        assert result is not None
        new_val, params = result
        # Token set preserved.
        assert sorted(new_val.split()) == sorted("Alpha Beta Gamma Delta".split())
        assert new_val != "Alpha Beta Gamma Delta"
        assert "positions" in params

    def test_stopwords_untouched(self) -> None:
        """Swap never touches stopwords."""
        rng = np.random.default_rng(42)
        result = eda_random_swap(
            "the Alpha of Beta Gamma",
            rng,
            stopwords={"the", "of"},
            key_tokens=set(),
        )
        # 3 non-stopword positions; swap succeeds.
        assert result is not None
        new_val, _ = result
        tokens = new_val.split()
        # "the" always at position 0, "of" always at position 2.
        assert tokens[0] == "the"
        assert tokens[2] == "of"

    def test_too_few_tokens(self) -> None:
        rng = np.random.default_rng(42)
        result = eda_random_swap("Alpha", rng, stopwords=set(), key_tokens=set())
        assert result is None

    def test_all_stopwords_returns_none(self) -> None:
        rng = np.random.default_rng(42)
        result = eda_random_swap(
            "the of", rng, stopwords={"the", "of"}, key_tokens=set()
        )
        assert result is None

    def test_key_tokens_untouched(self) -> None:
        rng = np.random.default_rng(42)
        result = eda_random_swap(
            "Acme Foo Bar Baz",
            rng,
            stopwords=set(),
            key_tokens={"Acme"},
        )
        if result is not None:
            new_val, _ = result
            assert new_val.split()[0] == "Acme"

    def test_100_values(self) -> None:
        corrupted = 0
        for i in range(100):
            rng = np.random.default_rng(i)
            result = eda_random_swap(
                f"Alpha{i} Beta{i} Gamma{i} Delta{i}",
                rng,
                stopwords=set(),
                key_tokens=set(),
            )
            if (
                result is not None
                and result[0] != f"Alpha{i} Beta{i} Gamma{i} Delta{i}"
            ):
                corrupted += 1
        assert corrupted >= 90


class TestEdaRandomDelete:
    def test_basic_delete(self) -> None:
        rng = np.random.default_rng(42)
        result = eda_random_delete(
            "Alpha Beta Gamma Delta", rng, stopwords=set(), key_tokens=set()
        )
        assert result is not None
        new_val, params = result
        assert len(new_val.split()) == 3
        assert "token_removed" in params

    def test_stopword_not_deleted(self) -> None:
        rng = np.random.default_rng(42)
        for i in range(20):
            rng = np.random.default_rng(i)
            result = eda_random_delete(
                "Alpha the Beta",
                rng,
                stopwords={"the"},
                key_tokens=set(),
            )
            if result is not None:
                assert "the" in result[0].split()

    def test_single_token_returns_none(self) -> None:
        rng = np.random.default_rng(42)
        assert (
            eda_random_delete("Alpha", rng, stopwords=set(), key_tokens=set()) is None
        )

    def test_100_values(self) -> None:
        corrupted = 0
        for i in range(100):
            rng = np.random.default_rng(i)
            result = eda_random_delete(
                f"Alpha{i} Beta{i} Gamma{i}",
                rng,
                stopwords=set(),
                key_tokens=set(),
            )
            if result is not None and result[0] != f"Alpha{i} Beta{i} Gamma{i}":
                corrupted += 1
        assert corrupted >= 90


class TestContaminationChecks:
    def test_ngram_overlap_detected(self) -> None:
        a = "the quick brown fox jumps over the lazy dog today"
        b = "the quick brown fox jumps over the lazy dog yesterday"
        # 9-token contiguous overlap.
        assert ngram_overlap_tokens(a, b, n=8) is True

    def test_ngram_overlap_not_detected(self) -> None:
        a = "Apple Inc."
        b = "Apple Corporation"
        assert ngram_overlap_tokens(a, b, n=8) is False

    def test_ngram_too_short(self) -> None:
        # Values shorter than n cannot trigger overlap.
        assert ngram_overlap_tokens("short", "short", n=8) is False

    def test_contamination_passes_on_short(self) -> None:
        assert contamination_ngram_passed("Apple", "Apple") is True

    def test_contamination_fails_on_overlap(self) -> None:
        # Construct a 9-token overlap.
        a = "the quick brown fox jumps over the lazy dog"
        b = "the quick brown fox jumps over the lazy dog today"
        assert contamination_ngram_passed(b, a, n=8) is False

    def test_first_token_index(self) -> None:
        records = [("e1", "Acme Industries Inc"), ("e2", "Globex Industries Ltd")]
        idx = build_first_token_index(records, n_tokens=2)
        assert idx[("acme", "industries")] == "e1"

    def test_first_token_probe_self(self) -> None:
        idx = {("acme", "industries", "inc"): "e1"}
        # Self-alias passes.
        assert (
            contamination_first_token_probe_passed(
                "Acme Industries Inc Corp", "e1", idx
            )
            is True
        )

    def test_first_token_probe_alias_fails(self) -> None:
        idx = {("acme", "industries", "inc"): "e1"}
        assert (
            contamination_first_token_probe_passed(
                "Acme Industries Inc is here", "e2", idx
            )
            is False
        )


# ---- LLM cache tests -------------------------------------------------------


class TestLLMCache:
    def test_put_get_round_trip(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path, prompt_version="v1", model_id="mock-model")
        h = cache.make_cell_hash("src", "col", "value")
        assert cache.get(h) is None
        cache.put(h, {"result": {"paraphrase": "VaLuE"}})
        assert cache.get(h) == {"result": {"paraphrase": "VaLuE"}}

    def test_hash_deterministic(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path, prompt_version="v1", model_id="mock-model")
        h1 = cache.make_cell_hash("src", "col", "value")
        h2 = cache.make_cell_hash("src", "col", "value")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_changes_with_prompt_version(self, tmp_path: Path) -> None:
        c1 = LLMCache(tmp_path, prompt_version="v1", model_id="mock-model")
        c2 = LLMCache(tmp_path, prompt_version="v2", model_id="mock-model")
        assert c1.make_cell_hash("s", "a", "v") != c2.make_cell_hash("s", "a", "v")

    def test_hash_changes_with_model_id(self, tmp_path: Path) -> None:
        c1 = LLMCache(tmp_path, prompt_version="v1", model_id="m1")
        c2 = LLMCache(tmp_path, prompt_version="v1", model_id="m2")
        assert c1.make_cell_hash("s", "a", "v") != c2.make_cell_hash("s", "a", "v")

    def test_call_or_cache_hits(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path, prompt_version="v1", model_id="mock-model")
        call_count = [0]

        def api() -> dict:
            call_count[0] += 1
            return {"paraphrase": "hello"}

        cache.call_or_cache("s", "a", "v", api)
        assert call_count[0] == 1
        # Second call hits cache.
        cache.call_or_cache("s", "a", "v", api)
        assert call_count[0] == 1

    def test_call_or_cache_strict_raises_on_miss(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path, prompt_version="v1", model_id="mock-model")
        with pytest.raises(LLMCacheMiss):
            cache.call_or_cache("s", "a", "v", api_fn=None, strict=True)

    def test_disk_persistence(self, tmp_path: Path) -> None:
        cache1 = LLMCache(tmp_path, prompt_version="v1", model_id="mock-model")
        cache1.call_or_cache("s", "a", "v", lambda: {"paraphrase": "hi"})

        cache2 = LLMCache(tmp_path, prompt_version="v1", model_id="mock-model")
        # New cache instance reads from disk.
        payload = cache2.call_or_cache("s", "a", "v", api_fn=None, strict=True)
        assert payload["result"]["paraphrase"] == "hi"


# ---- llm_paraphrase operator test ------------------------------------------


class TestLlmParaphraseOperator:
    def test_cache_hit(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path, prompt_version="v1", model_id="mock-model")
        # Pre-populate.
        h = cache.make_cell_hash("forbes", "Company", "Apple Inc.")
        cache.put(
            h,
            {
                "source": "forbes",
                "attribute": "Company",
                "original_value": "Apple Inc.",
                "prompt_version": "v1",
                "model_id": "mock-model",
                "result": {"paraphrase": "Apple"},
            },
        )

        result = llm_paraphrase(
            "Apple Inc.",
            source="forbes",
            attribute="Company",
            attribute_class="primary",
            cache=cache,
            prompt_template="prompt",
            api_client=None,  # Should not be called.
            strict_cache=False,
        )
        assert result is not None
        new_val, params = result
        assert new_val == "Apple"
        assert params["transform_fn"] == "llm_paraphrase_short"
        assert params["contamination_check"]["ngram_overlap_passed"] is True

    def test_contamination_fail_returns_none(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path, prompt_version="v1", model_id="mock-model")
        # 9-token original; paraphrase overlaps 9 tokens.
        value = "the quick brown fox jumps over the lazy dog"
        h = cache.make_cell_hash("src", "col", value)
        cache.put(
            h,
            {
                "result": {
                    "paraphrase": ("the quick brown fox jumps over the lazy dog today")
                }
            },
        )
        result = llm_paraphrase(
            value,
            source="src",
            attribute="col",
            attribute_class="primary",
            cache=cache,
            prompt_template="p",
            ngram_n=8,
        )
        assert result is None

    def test_committee_rejection_returns_none(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path, prompt_version="v1", model_id="mock-model")
        h = cache.make_cell_hash("src", "col", "Apple Inc.")
        cache.put(h, {"result": {"paraphrase": "Apple Corp."}})

        result = llm_paraphrase(
            "Apple Inc.",
            source="src",
            attribute="col",
            attribute_class="primary",
            cache=cache,
            prompt_template="p",
            committee_fn=lambda *args: False,
        )
        assert result is None


# ---- paraphrase_value_for_knob_04 export test ------------------------------


class TestParaphraseForKnob04:
    def test_deterministic(self) -> None:
        config = {
            "abbreviation_table": {"Incorporated": "Inc."},
            "stopword_list": ["the", "of"],
            "key_token_skiplist_global": [],
        }
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1 = paraphrase_value_for_knob_04(
            "companies", "primary", "Apple Incorporated", config, rng1
        )
        r2 = paraphrase_value_for_knob_04(
            "companies", "primary", "Apple Incorporated", config, rng2
        )
        assert r1 == r2

    def test_passthrough_on_no_op(self) -> None:
        config = {
            "abbreviation_table": {},
            "stopword_list": [],
            "key_token_skiplist_global": [],
        }
        rng = np.random.default_rng(42)
        new_val, params = paraphrase_value_for_knob_04(
            "companies", "primary", "Single", config, rng
        )
        assert new_val == "Single"
        assert params["transform_fn"] == "passthrough"

    def test_always_returns_tuple(self) -> None:
        config = {
            "abbreviation_table": {"Incorporated": "Inc."},
            "stopword_list": [],
            "key_token_skiplist_global": [],
        }
        for i in range(20):
            rng = np.random.default_rng(i)
            result = paraphrase_value_for_knob_04(
                "companies", "primary", "Alpha Beta Gamma Incorporated", config, rng
            )
            assert isinstance(result, tuple)
            assert isinstance(result[0], str)
            assert "transform_fn" in result[1]


# ---- Dispatcher integration tests ------------------------------------------


class TestApplyKnob01Easy:
    def test_easy_normalize_down(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        """At easy, forbes Country gets normalized to canonical dbpedia form."""
        paraphrased, prov_df, _, _ = apply_knob_01(
            domain="companies",
            level="easy",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        # All forbes Country should now equal "United States" (dbpedia canonical).
        countries = paraphrased["forbes"]["Country"].tolist()
        assert all(c == "United States" for c in countries), countries
        # Provenance should log normalize_to_canonical rows.
        assert (prov_df["transform_fn"] == "normalize_to_canonical").any()

    def test_easy_no_random_paraphrase(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        """At easy, rates are 0 — no random operators fire."""
        _, prov_df, _, _ = apply_knob_01(
            domain="companies",
            level="easy",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        # Only normalize_to_canonical rows allowed.
        assert set(prov_df["transform_fn"].unique()) <= {"normalize_to_canonical"}


class TestApplyKnob01Medium:
    def test_medium_fires_operators(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        _, prov_df, _, _ = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        assert len(prov_df) > 0
        assert set(prov_df["transform_fn"]).issubset(VALID_TRANSFORM_FNS)

    def test_medium_token_overlap(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        """At medium, token overlap between original and paraphrased >= 50%."""
        _, prov_df, _, _ = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        for _, row in prov_df.iterrows():
            orig_tokens = set(row["original_value"].lower().split())
            new_tokens = set(row["new_value"].lower().split())
            if not orig_tokens:
                continue
            overlap = len(orig_tokens & new_tokens) / max(len(orig_tokens), 1)
            # Allow abbreviation to fully change the string; only check EDA ops.
            if row["transform_fn"] in ("eda_random_swap", "eda_random_delete"):
                assert overlap >= 0.5, (
                    f"{row['transform_fn']}: {row['original_value']} -> "
                    f"{row['new_value']}, overlap={overlap}"
                )

    def test_determinism(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        out1 = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        out2 = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        for src in out1[0]:
            pd.testing.assert_frame_equal(out1[0][src], out2[0][src])
        pd.testing.assert_frame_equal(out1[1], out2[1])


class TestAnchorSurvivorFloor:
    def test_floor_holds_at_hard(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        """At every level, every multi-source entity keeps 1 clean primary."""
        small_config = {**small_config}
        small_config["paraphrase_rate_primary"] = {
            "easy": 0.0,
            "medium": 1.0,
            "hard": 1.0,
        }
        _, prov_df, _, _ = apply_knob_01(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        primary_cols = {
            "dbpedia": "name",
            "forbes": "Company",
            "fullcontact": "name",
        }
        paraphrased_ids: dict[str, set[str]] = {src: set() for src in primary_cols}
        for _, row in prov_df.iterrows():
            src = row["source"]
            attr = row["attribute"]
            if primary_cols.get(src) == attr:
                paraphrased_ids[src].add(row["entity_id"])

        for group_id, members in small_entity_groups.items():
            if len(members) <= 1:
                continue
            clean = sum(
                1 for src, rid in members if rid not in paraphrased_ids.get(src, set())
            )
            assert clean >= 1, (
                f"Entity group {group_id} lost all clean primaries: "
                f"members={members}, paraphrased={paraphrased_ids}"
            )


class TestCollisionIndex:
    def test_skips_touched_cells(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
        tmp_path: Path,
    ) -> None:
        """Cells with prior provenance are skipped."""
        prov_dir = tmp_path / "output" / "provenance"
        prov_dir.mkdir(parents=True)

        fake = ProvenanceLog(knob=5, level="medium")
        fake.append(
            entity_id="db_0",
            source="dbpedia",
            attribute="name",
            original_value="X",
            new_value="Y",
            transform_fn="reformat_number",
            transform_params={},
        )
        fake.flush(prov_dir / "knob_05_format_unit.csv")

        idx = CollisionIndex(prov_dir)

        small_config = {**small_config}
        small_config["paraphrase_rate_primary"] = {
            "easy": 0.0,
            "medium": 1.0,
            "hard": 1.0,
        }

        _, _, skipped, _ = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            collision_index=idx,
            seed=42,
        )
        collision = skipped[skipped["reason"] == "cell_collision_with_prior_knob"]
        found = any(
            (r["entity_id"] == "db_0" and r["attribute"] == "name")
            for _, r in collision.iterrows()
        )
        assert found, "db_0 name should be skipped due to K5 collision"

    def test_k4_fabricated_also_skipped(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
        tmp_path: Path,
    ) -> None:
        """K4-fabricated cells are also skipped by K1 (double-augmentation guard)."""
        prov_dir = tmp_path / "output" / "provenance"
        prov_dir.mkdir(parents=True)

        fake = ProvenanceLog(knob=4, level="medium")
        fake.append(
            entity_id="db_0",
            source="dbpedia",
            attribute="name",
            original_value="",
            new_value="Fabricated Corp",
            transform_fn="fabricate_coverage",
            transform_params={"k4_fabricated": True},
        )
        fake.flush(prov_dir / "knob_04_coverage.csv")

        idx = CollisionIndex(prov_dir)

        small_config = {**small_config}
        small_config["paraphrase_rate_primary"] = {
            "easy": 0.0,
            "medium": 1.0,
            "hard": 1.0,
        }

        _, _, skipped, _ = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            collision_index=idx,
            seed=42,
        )
        reasons = set(skipped["reason"])
        assert "cell_collision_with_k4_fabricated" in reasons


class TestLLMCachingAtHard:
    def test_hard_uses_cache_no_reruns(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
        tmp_path: Path,
    ) -> None:
        """Hard level invokes the API client exactly once per unique cell and
        never again on rerun."""
        cache_dir = tmp_path / "cache"
        cache = LLMCache(cache_dir, prompt_version="v1", model_id="mock-model")

        call_count = [0]

        def mock_client(prompt: str, value: str) -> str:
            call_count[0] += 1
            return f"paraphrased: {value[:10]}"

        # First run: API called.
        _, prov_1, _, _ = apply_knob_01(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            llm_cache=cache,
            llm_client=mock_client,
            seed=42,
        )
        first_calls = call_count[0]
        assert first_calls > 0

        # Second run with a fresh cache instance pointing to the same dir:
        # API must NOT be called again.
        cache2 = LLMCache(cache_dir, prompt_version="v1", model_id="mock-model")
        call_count[0] = 0

        _, prov_2, _, _ = apply_knob_01(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            llm_cache=cache2,
            llm_client=mock_client,
            strict_cache=True,  # Enforce no-regen.
            seed=42,
        )
        assert (
            call_count[0] == 0
        ), f"LLM regenerated {call_count[0]} cells on rerun; cache broken"

    def test_hard_determinism(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
        tmp_path: Path,
    ) -> None:
        """Same cache + same seed = identical hard output."""
        cache = LLMCache(tmp_path / "cache", prompt_version="v1", model_id="mock-model")

        def mock_client(prompt: str, value: str) -> str:
            return f"paraphrased: {value[:10]}"

        out1 = apply_knob_01(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            llm_cache=cache,
            llm_client=mock_client,
            seed=42,
        )
        out2 = apply_knob_01(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            llm_cache=cache,
            llm_client=mock_client,
            seed=42,
        )
        for src in out1[0]:
            pd.testing.assert_frame_equal(out1[0][src], out2[0][src])
        pd.testing.assert_frame_equal(
            out1[1].reset_index(drop=True), out2[1].reset_index(drop=True)
        )


class TestMonotoneCounts:
    def test_hard_ge_medium_ge_easy(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
        tmp_path: Path,
    ) -> None:
        """Total paraphrased cell count: hard >= medium >= easy."""
        _, prov_e, _, _ = apply_knob_01(
            domain="companies",
            level="easy",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        _, prov_m, _, _ = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        cache = LLMCache(tmp_path / "cache", prompt_version="v1", model_id="mock-model")
        _, prov_h, _, _ = apply_knob_01(
            domain="companies",
            level="hard",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            llm_cache=cache,
            llm_client=lambda p, v: f"paraphrased: {v[:5]}",
            seed=42,
        )

        # Exclude normalize_to_canonical rows (easy-only) from comparison.
        n_easy = len(prov_e[prov_e["transform_fn"] != "normalize_to_canonical"])
        n_medium = len(prov_m)
        n_hard = len(prov_h)

        assert n_easy == 0
        assert n_hard >= n_medium >= n_easy


class TestProvenanceSchema:
    def test_schema(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        _, prov_df, _, _ = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        assert list(prov_df.columns) == PROVENANCE_COLUMNS
        for _, row in prov_df.iterrows():
            assert row["knob"] == 1
            assert row["level"] == "medium"
            assert row["transform_fn"] in VALID_TRANSFORM_FNS
            params = json.loads(row["transform_params"])
            assert isinstance(params, dict)


class TestSkippedSchema:
    def test_schema(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        _, _, skipped_df, _ = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        assert list(skipped_df.columns) == SKIPPED_COLUMNS


class TestLoadConfig:
    def test_load_companies_config(self) -> None:
        """Real companies config loads and has required keys."""
        config = load_knob_01_config("companies")
        assert config["domain"] == "companies"
        assert "paraphrase_rate_primary" in config
        assert "operator_mix" in config
        assert "abbreviation_table" in config

    def test_companies_rates_monotone(self) -> None:
        config = load_knob_01_config("companies")
        for key in (
            "paraphrase_rate_primary",
            "paraphrase_rate_key",
            "paraphrase_rate_secondary",
            "paraphrase_rate_categorical",
        ):
            rates = config[key]
            assert (
                rates["easy"] <= rates["medium"] <= rates["hard"]
            ), f"Non-monotone {key}: {rates}"

    def test_companies_operator_mix_non_shrinking(self) -> None:
        config = load_knob_01_config("companies")
        mix = config["operator_mix"]
        easy_ops = set(mix["easy"].keys())
        medium_ops = set(mix["medium"].keys())
        hard_ops = set(mix["hard"].keys())
        # Easy uses normalize_to_canonical; it is the one allowed asymmetry.
        # Medium and hard must be non-shrinking.
        assert medium_ops.issubset(hard_ops)


class TestAttrsPreserved:
    def test_attrs_preserved(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        paraphrased, _, _, _ = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        for src, df in paraphrased.items():
            assert df.attrs.get("dataset_name") == src


class TestRealisedAudit:
    """Cover the K1 realised-CSV writer (plan_revision.md R-1 / G9 / step 4f).

    Two layers of coverage:

    1. Pure helpers (``_token_jaccard_drop``, ``build_realised_df``) get
       small synthetic-input tests so corner cases (empty strings, list
       wrappers, zero commits) don't regress silently.
    2. End-to-end ``apply_knob_01`` integration on the existing small
       companies fixtures verifies (a) the 4-tuple shape, (b) column
       order matches ``REALISED_COLUMNS``, (c) attempts >= committed,
       and (d) ``write_outputs`` lands the artifact at
       ``output/baselines/knob_01_realised.csv`` only when realised_df
       is supplied.
    """

    def test_token_jaccard_drop_identical(self) -> None:
        assert _token_jaccard_drop("alpha beta", "alpha beta") == 0.0

    def test_token_jaccard_drop_reorder_is_zero(self) -> None:
        # Token-set Jaccard is order-invariant — random_swap leaves the set
        # unchanged, so the drop is 0 even though the cell value differs.
        # That is the "shallow paraphrase" detection contract: the
        # secondary intensity signal must catch this case where edit
        # distance is non-zero but token jaccard says nothing changed.
        assert _token_jaccard_drop("alpha beta gamma", "gamma alpha beta") == 0.0

    def test_token_jaccard_drop_disjoint(self) -> None:
        assert _token_jaccard_drop("alpha beta", "gamma delta") == 1.0

    def test_token_jaccard_drop_one_empty(self) -> None:
        assert _token_jaccard_drop("alpha beta", "") == 1.0
        assert _token_jaccard_drop("", "alpha beta") == 1.0

    def test_token_jaccard_drop_both_empty(self) -> None:
        assert _token_jaccard_drop("", "") == 0.0

    def test_token_jaccard_drop_partial_overlap(self) -> None:
        # alpha,beta vs alpha,gamma → intersection=1, union=3 → drop=2/3.
        result = _token_jaccard_drop("alpha beta", "alpha gamma")
        assert abs(result - 2 / 3) < 1e-9

    def test_build_realised_df_no_commits(self) -> None:
        prov_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)
        skipped_df = pd.DataFrame(columns=SKIPPED_COLUMNS)
        realised = build_realised_df(
            level="easy",
            provenance_df=prov_df,
            skipped_df=skipped_df,
        )
        assert list(realised.columns) == REALISED_COLUMNS
        assert len(realised) == 1
        row = realised.iloc[0]
        assert row["level"] == "easy"
        assert int(row["paraphrase_attempts"]) == 0
        assert int(row["paraphrase_committed"]) == 0
        assert float(row["mean_edit_distance"]) == 0.0
        assert float(row["mean_token_jaccard_drop"]) == 0.0
        assert int(row["strict_cache_miss_count"]) == 0

    def test_build_realised_df_attempts_equals_committed_plus_skipped(
        self,
    ) -> None:
        prov_df = pd.DataFrame(
            [
                {
                    "knob": 1,
                    "level": "hard",
                    "entity_id": "e1",
                    "source": "src_a",
                    "attribute": "name",
                    "original_value": "Acme Corporation",
                    "new_value": "Acme Corp.",
                    "transform_fn": "abbreviate",
                    "transform_params": "{}",
                },
                {
                    "knob": 1,
                    "level": "hard",
                    "entity_id": "e2",
                    "source": "src_a",
                    "attribute": "name",
                    "original_value": "alpha beta gamma",
                    "new_value": "alpha beta gamma",  # zero-edit committed
                    "transform_fn": "eda_random_swap",
                    "transform_params": "{}",
                },
            ],
            columns=PROVENANCE_COLUMNS,
        )
        skipped_df = pd.DataFrame(
            [
                {
                    "entity_id": "e3",
                    "source": "src_a",
                    "attribute": "name",
                    "original_value": "ignore me",
                    "reason": "strict_cache_miss",
                    "knob": 1,
                    "level": "hard",
                },
                {
                    "entity_id": "e4",
                    "source": "src_a",
                    "attribute": "name",
                    "original_value": "ignore me 2",
                    "reason": "strict_cache_miss",
                    "knob": 1,
                    "level": "hard",
                },
                {
                    "entity_id": "e5",
                    "source": "src_a",
                    "attribute": "name",
                    "original_value": "ignore me 3",
                    "reason": "closeness_floor_violation",
                    "knob": 1,
                    "level": "hard",
                },
            ],
            columns=SKIPPED_COLUMNS,
        )

        realised = build_realised_df(
            level="hard",
            provenance_df=prov_df,
            skipped_df=skipped_df,
        )
        row = realised.iloc[0]
        assert int(row["paraphrase_committed"]) == 2
        # attempts = committed + skipped
        assert int(row["paraphrase_attempts"]) == 5
        assert int(row["strict_cache_miss_count"]) == 2
        # mean_edit_distance averages (1 - Lev) over the two committed
        # rows; row 2 has identical values so contributes 0, row 1 has
        # a meaningful edit. Mean must therefore lie strictly between 0
        # and 1 (non-zero, non-degenerate).
        assert 0.0 < float(row["mean_edit_distance"]) < 1.0
        # Same for token jaccard drop — row 2's identical strings drive
        # toward 0; row 1's abbreviate cuts a token so its drop is > 0.
        # We do not pin a specific value because tokenisation rules can
        # evolve; the structural invariant (non-degenerate average) is
        # what we want to lock in.
        assert 0.0 < float(row["mean_token_jaccard_drop"]) <= 1.0

    def test_apply_knob_01_returns_4_tuple_with_realised_df(
        self,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        result = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        assert len(result) == 4
        paraphrased, prov_df, skipped_df, realised_df = result
        assert isinstance(realised_df, pd.DataFrame)
        assert list(realised_df.columns) == REALISED_COLUMNS
        assert len(realised_df) == 1
        row = realised_df.iloc[0]
        assert row["level"] == "medium"
        # The realised audit invariant: committed cells <= attempts.
        assert int(row["paraphrase_committed"]) <= int(row["paraphrase_attempts"])
        # Committed + skipped == attempts by construction.
        assert int(row["paraphrase_attempts"]) == int(
            row["paraphrase_committed"]
        ) + len(skipped_df)
        # Committed == len(prov_df).
        assert int(row["paraphrase_committed"]) == len(prov_df)

    def test_write_outputs_emits_realised_csv(
        self,
        tmp_path: Path,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        _, prov_df, skipped_df, realised_df = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        write_outputs(prov_df, skipped_df, tmp_path, realised_df=realised_df)

        realised_path = tmp_path / "output" / "baselines" / "knob_01_realised.csv"
        assert realised_path.exists()
        round_tripped = pd.read_csv(realised_path)
        assert list(round_tripped.columns) == REALISED_COLUMNS
        assert len(round_tripped) == 1
        # Round-trip preserves the level + counts; floating-point columns
        # round-trip to within rounding tolerance.
        assert round_tripped.iloc[0]["level"] == "medium"
        assert int(round_tripped.iloc[0]["paraphrase_committed"]) == int(
            realised_df.iloc[0]["paraphrase_committed"]
        )

    def test_write_outputs_skips_realised_when_none(
        self,
        tmp_path: Path,
        small_sources: dict[str, pd.DataFrame],
        small_config: dict[str, Any],
        small_entity_groups: dict[str, list[tuple[str, str]]],
    ) -> None:
        _, prov_df, skipped_df, _ = apply_knob_01(
            domain="companies",
            level="medium",
            sources=small_sources,
            config=small_config,
            entity_groups=small_entity_groups,
            seed=42,
        )
        write_outputs(prov_df, skipped_df, tmp_path, realised_df=None)

        realised_path = tmp_path / "output" / "baselines" / "knob_01_realised.csv"
        assert not realised_path.exists()


# ---- R10-D: v2 prompts + post-filter + <UNCHANGED> sentinel ----------------


class TestNearIdentityHelper:
    """Cover ``_is_near_identity`` corner cases (R10-D post-filter primitive)."""

    def test_identical_strings(self) -> None:
        assert _is_near_identity("Apple Inc.", "Apple Inc.") is True

    def test_casing_only_diff(self) -> None:
        assert _is_near_identity("apple inc.", "Apple Inc.") is True

    def test_whitespace_only_diff(self) -> None:
        assert _is_near_identity("Apple   Inc.", "Apple Inc.") is True

    def test_token_added(self) -> None:
        # A real paraphrase that adds a token is NOT near-identity.
        assert _is_near_identity("Apple Computer Inc.", "Apple Inc.") is False

    def test_token_removed(self) -> None:
        assert _is_near_identity("Apple", "Apple Inc.") is False

    def test_token_replaced(self) -> None:
        assert _is_near_identity("Apple Corp.", "Apple Inc.") is False

    def test_both_empty(self) -> None:
        assert _is_near_identity("", "") is True

    def test_one_empty(self) -> None:
        assert _is_near_identity("", "Apple Inc.") is False
        assert _is_near_identity("Apple Inc.", "") is False


class TestLLMParaphraseV2Behavior:
    """Cover R10-D ``<UNCHANGED>`` and near-identity handling in llm_paraphrase."""

    def _put(
        self,
        cache: LLMCache,
        source: str,
        attribute: str,
        value: str,
        paraphrase: str,
    ) -> None:
        cache.put(
            cache.make_cell_hash(source, attribute, value),
            {"result": {"paraphrase": paraphrase}},
        )

    def test_unchanged_sentinel_returns_value_with_unchanged_transform_fn(
        self, tmp_path: Path
    ) -> None:
        cache = LLMCache(tmp_path, prompt_version="v2", model_id="mock-model")
        self._put(cache, "src", "name", "Apple Inc.", UNCHANGED_SENTINEL)
        result = llm_paraphrase(
            "Apple Inc.",
            source="src",
            attribute="name",
            attribute_class="primary",
            cache=cache,
            prompt_template="prompt",
        )
        assert result is not None
        new_val, params = result
        assert new_val == "Apple Inc."
        assert params["transform_fn"] == "llm_paraphrase_unchanged"

    def test_near_identity_casing_only_returns_near_identity_transform_fn(
        self, tmp_path: Path
    ) -> None:
        cache = LLMCache(tmp_path, prompt_version="v2", model_id="mock-model")
        self._put(cache, "src", "name", "Apple Inc.", "APPLE INC.")
        result = llm_paraphrase(
            "Apple Inc.",
            source="src",
            attribute="name",
            attribute_class="primary",
            cache=cache,
            prompt_template="prompt",
        )
        assert result is not None
        new_val, params = result
        assert new_val == "APPLE INC."
        assert params["transform_fn"] == "llm_paraphrase_near_identity"

    def test_secondary_class_gets_secondary_transform_fn(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path, prompt_version="v2", model_id="mock-model")
        self._put(
            cache,
            "src",
            "description",
            "Released June 1973 on Harvest Records.",
            "Issued by Harvest Records in June 1973.",
        )
        result = llm_paraphrase(
            "Released June 1973 on Harvest Records.",
            source="src",
            attribute="description",
            attribute_class="secondary",
            cache=cache,
            prompt_template="prompt",
            ngram_n=20,  # disable contamination check for this small example
        )
        assert result is not None
        _, params = result
        assert params["transform_fn"] == "llm_paraphrase_secondary"

    def test_verbatim_paraphrase_still_returns_none(self, tmp_path: Path) -> None:
        # The legacy verbatim-copy short-circuit must still fire BEFORE
        # the new sentinel checks. paraphrase==value -> None (existing
        # contract preserved for back-compat).
        cache = LLMCache(tmp_path, prompt_version="v2", model_id="mock-model")
        self._put(cache, "src", "name", "Apple Inc.", "Apple Inc.")
        result = llm_paraphrase(
            "Apple Inc.",
            source="src",
            attribute="name",
            attribute_class="primary",
            cache=cache,
            prompt_template="prompt",
        )
        assert result is None

    def test_primary_class_still_gets_short_transform_fn(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path, prompt_version="v2", model_id="mock-model")
        self._put(cache, "src", "name", "Apple Inc.", "Apple Computer Inc.")
        result = llm_paraphrase(
            "Apple Inc.",
            source="src",
            attribute="name",
            attribute_class="primary",
            cache=cache,
            prompt_template="prompt",
        )
        assert result is not None
        _, params = result
        assert params["transform_fn"] == "llm_paraphrase_short"


class TestRealisedAuditV2Counters:
    """Cover the new R10-D realised columns (unchanged / near_identity)."""

    def test_counters_default_to_zero(self) -> None:
        prov_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)
        skipped_df = pd.DataFrame(columns=SKIPPED_COLUMNS)
        realised = build_realised_df(
            level="hard",
            provenance_df=prov_df,
            skipped_df=skipped_df,
        )
        row = realised.iloc[0]
        assert "llm_unchanged_count" in realised.columns
        assert "llm_near_identity_count" in realised.columns
        assert int(row["llm_unchanged_count"]) == 0
        assert int(row["llm_near_identity_count"]) == 0

    def test_counters_count_skipped_rows_by_reason(self) -> None:
        prov_df = pd.DataFrame(columns=PROVENANCE_COLUMNS)
        skipped_df = pd.DataFrame(
            [
                {
                    "entity_id": "e1",
                    "source": "s",
                    "attribute": "a",
                    "original_value": "v1",
                    "reason": "llm_unchanged_sentinel",
                    "knob": 1,
                    "level": "hard",
                },
                {
                    "entity_id": "e2",
                    "source": "s",
                    "attribute": "a",
                    "original_value": "v2",
                    "reason": "llm_unchanged_sentinel",
                    "knob": 1,
                    "level": "hard",
                },
                {
                    "entity_id": "e3",
                    "source": "s",
                    "attribute": "a",
                    "original_value": "v3",
                    "reason": "llm_near_identity",
                    "knob": 1,
                    "level": "hard",
                },
                {
                    "entity_id": "e4",
                    "source": "s",
                    "attribute": "a",
                    "original_value": "v4",
                    "reason": "strict_cache_miss",
                    "knob": 1,
                    "level": "hard",
                },
            ],
            columns=SKIPPED_COLUMNS,
        )
        realised = build_realised_df(
            level="hard",
            provenance_df=prov_df,
            skipped_df=skipped_df,
        )
        row = realised.iloc[0]
        assert int(row["llm_unchanged_count"]) == 2
        assert int(row["llm_near_identity_count"]) == 1
        assert int(row["strict_cache_miss_count"]) == 1
        # All 4 skipped rows count as attempts.
        assert int(row["paraphrase_attempts"]) == 4


class TestPromptVersionDispatch:
    """Cover R10-D prompt template loading + per-attribute-class dispatch."""

    def test_v2_prompts_exist_on_disk(self) -> None:
        from usecases_synthetic.scripts.apply_knob_01_surface import (
            _load_prompt_template,
        )

        short_v2 = _load_prompt_template("prompt_short_v2.txt")
        cat_v2 = _load_prompt_template("prompt_categorical_v2.txt")
        sec_v2 = _load_prompt_template("prompt_secondary_v2.txt")
        assert UNCHANGED_SENTINEL in short_v2
        assert UNCHANGED_SENTINEL in cat_v2
        assert UNCHANGED_SENTINEL in sec_v2
        # v2 prompts must mention the minimum-divergence rule (token
        # change required beyond casing/punctuation).
        for tmpl in (short_v2, cat_v2, sec_v2):
            assert "token" in tmpl.lower()

    def test_all_four_domain_yamls_pin_v2(self) -> None:
        # The four production K1 YAMLs must all carry ``llm_prompt_version: v2``
        # so the next variant regen uses the new prompts.
        from usecases_synthetic.scripts.apply_knob_01_surface import (
            load_knob_01_config,
        )

        for domain in ("music", "games", "products", "companies"):
            cfg = load_knob_01_config(domain)
            assert (
                cfg.get("llm_prompt_version") == "v2"
            ), f"K1 YAML for {domain} must pin llm_prompt_version: v2 (R10-D)"


class TestBuildOpenAIParaphraseClient:
    """Unit tests for the K1 live paraphrase client (Fix A, 2026-05-30).

    The client is the ``(prompt_template, value) -> paraphrase`` callable
    that :func:`llm_paraphrase` invokes on cache miss. Before the fix the
    joint runner hardcoded ``llm_client=None``, so the v2 paraphrase prompt
    was never called at any level.
    """

    def _build_with_chat(self, monkeypatch: pytest.MonkeyPatch, fake_chat: Any) -> Any:
        from usecases_synthetic.lib import llm_client

        def _fake_build(
            *,
            model: str,
            temperature: float = 0.0,
            max_tokens: int | None = None,
        ) -> Any:
            return fake_chat

        monkeypatch.setattr(llm_client, "build_chat_openai", _fake_build)
        return build_openai_paraphrase_client(model_id="gpt-5.4-mini", max_tokens=2048)

    def test_substitutes_value_placeholder_and_returns_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                captured["prompt"] = prompt

                class _R:
                    content = "Acme Corp."

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())
        out = client("Paraphrase this: {value}", "Acme Corporation")

        assert captured["prompt"] == "Paraphrase this: Acme Corporation"
        assert out == "Acme Corp."

    def test_strips_surrounding_quotes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                class _R:
                    content = '"Acme Corp."'

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())
        assert client("{value}", "Acme Corporation") == "Acme Corp."

    def test_unchanged_sentinel_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                class _R:
                    content = "<UNCHANGED>"

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())
        assert client("{value}", "Sony") == UNCHANGED_SENTINEL

    def test_joins_list_content_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                class _R:
                    content = [{"text": "Acme "}, {"text": "Corp."}]

                return _R()

        client = self._build_with_chat(monkeypatch, FakeChat())
        assert client("{value}", "Acme Corporation") == "Acme Corp."

    def test_returns_empty_on_missing_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A template asking for an unknown placeholder fails formatting and
        degrades to '' (caller falls back to a deterministic operator)
        instead of raising."""

        class FakeChat:
            def invoke(self, prompt: str) -> Any:  # pragma: no cover
                raise AssertionError("invoke must not run when formatting fails")

        client = self._build_with_chat(monkeypatch, FakeChat())
        assert client("needs {unknown_placeholder}", "X") == ""

    def test_returns_empty_on_invoke_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeChat:
            def invoke(self, prompt: str) -> Any:
                raise RuntimeError("network down")

        client = self._build_with_chat(monkeypatch, FakeChat())
        assert client("{value}", "X") == ""

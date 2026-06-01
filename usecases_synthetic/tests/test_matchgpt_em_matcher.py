"""Functional tests for MatchGPTMatcher (C2.4b unit c).

Uses the `llm_callable` + `embedder` injection hooks to stub out
network and GPU dependencies so the tests run deterministically on
CPU without an OpenAI key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.matchgpt_em_matcher import (
    MatchGPTMatcher,
    _MISSING_MARKER,
    _PROMPT_VERSION,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def df_left() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "L1",
                "name": "Acme Corp",
                "country": "USA",
                "city": "New York",
                "industry": "Software",
                "sector": "Tech",
                "founded": 1990,
            },
            {
                "id": "L2",
                "name": "Globex Inc",
                "country": "USA",
                "city": "Los Angeles",
                "industry": "Finance",
                "sector": "Services",
                "founded": 1985,
            },
            {
                "id": "L3",
                "name": "Initech",
                "country": "USA",
                "city": "Austin",
                "industry": np.nan,
                "sector": "Tech",
                "founded": 1999,
            },
        ]
    )


@pytest.fixture()
def df_right() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "R1",
                "name": "Acme Corporation",
                "country": "United States",
                "city": "New York City",
                "industry": "Software",
                "sector": "Technology",
                "founded": 1990,
            },
            {
                "id": "R2",
                "name": "Umbrella LLC",
                "country": "UK",
                "city": "London",
                "industry": "Pharma",
                "sector": "Healthcare",
                "founded": 2001,
            },
            {
                "id": "R3",
                "name": "Initech Systems",
                "country": "USA",
                "city": "Austin",
                "industry": "Software",
                "sector": "Tech",
                "founded": 1999,
            },
        ]
    )


@pytest.fixture()
def candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id1": "L1", "id2": "R1"},
            {"id1": "L2", "id2": "R2"},
            {"id1": "L3", "id2": "R3"},
        ]
    )


@pytest.fixture()
def fields() -> list[str]:
    return ["name", "country", "city", "industry", "sector", "founded"]


def _fake_llm(yes_pairs: set[tuple[str, str]]):
    """Return a stub LLM callable driven by the (L*/R*) tokens in the prompt.

    Parses the prompt's Query block to recover the candidate ids and
    answers Yes / No accordingly.
    """

    calls: list[str] = []

    def _call(prompt: str) -> str:
        calls.append(prompt)
        query_block = prompt.rsplit("Query:", 1)[-1]
        # Pull the first left id + right id that appear in the query
        left_id = None
        right_id = None
        for tok in ("L1", "L2", "L3"):
            if tok in query_block:
                left_id = tok
                break
        for tok in ("R1", "R2", "R3"):
            if tok in query_block:
                right_id = tok
                break
        # Names carry the id letters in our fixtures (Acme / Globex / Initech
        # → L1/L2/L3; Acme Corporation / Umbrella / Initech Systems →
        # R1/R2/R3). Fall back on the first Record A / Record B name line
        # if the id token isn't present in the serialized pair text.
        if left_id is None or right_id is None:
            for line in query_block.splitlines():
                stripped = line.strip()
                if stripped.startswith("name:") and left_id is None:
                    left_id = _name_to_id(stripped.split(":", 1)[1].strip(), "L")
                elif stripped.startswith("name:") and right_id is None:
                    right_id = _name_to_id(stripped.split(":", 1)[1].strip(), "R")
        if (left_id, right_id) in yes_pairs:
            return "Yes"
        return "No"

    _call.calls = calls  # type: ignore[attr-defined]
    return _call


def _name_to_id(name: str, side: str) -> str:
    lookup = {
        "Acme Corp": "L1",
        "Globex Inc": "L2",
        "Initech": "L3",
        "Acme Corporation": "R1",
        "Umbrella LLC": "R2",
        "Initech Systems": "R3",
    }
    return lookup.get(name, side + "?")


def _fake_embedder(dim: int = 4):
    """Deterministic pseudo-embedder derived from a SHA256 of the text."""
    import hashlib

    def _embed(texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = np.frombuffer(h[: dim * 4], dtype=np.uint32).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            out[i] = vec
        return out

    return _embed


# ---------------------------------------------------------------------------
# Shape + contract
# ---------------------------------------------------------------------------


class TestShapeAndContract:
    def test_requires_fields(self) -> None:
        with pytest.raises(ValueError, match="at least one field"):
            MatchGPTMatcher(fields=[])

    def test_rejects_negative_k_shot(self, fields: list[str]) -> None:
        with pytest.raises(ValueError, match="k_shot"):
            MatchGPTMatcher(fields=fields, k_shot=-1)

    def test_returns_correct_columns(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=_fake_llm({("L1", "R1"), ("L3", "R3")}),
        )
        result = matcher.match(df_left, df_right, candidates, id_column="id")
        assert list(result.columns) == ["id1", "id2", "score", "notes"]
        assert set(result["id1"]) == {"L1", "L3"}
        assert set(result["id2"]) == {"R1", "R3"}
        assert (result["score"] == 1.0).all()
        assert (result["notes"] == "matchgpt").all()

    def test_empty_candidates(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=_fake_llm(set()),
        )
        empty = pd.DataFrame(columns=["id1", "id2"])
        result = matcher.match(df_left, df_right, empty, id_column="id")
        assert result.empty
        assert list(result.columns) == ["id1", "id2", "score", "notes"]

    def test_missing_id_columns_raise(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=_fake_llm(set()),
        )
        bad = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError, match="id1"):
            matcher.match(df_left, df_right, bad, id_column="id")

    def test_missing_id_column_in_frames_raises(
        self,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=_fake_llm(set()),
        )
        bad_left = pd.DataFrame({"name": ["X"]})
        with pytest.raises(ValueError, match="left dataset"):
            matcher.match(
                bad_left,
                pd.DataFrame({"id": ["R1"], "name": ["Y"]}),
                pd.DataFrame({"id1": ["X"], "id2": ["R1"]}),
                id_column="id",
            )

    def test_accepts_comparators_and_weights_kwargs(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=_fake_llm({("L1", "R1")}),
        )
        result = matcher.match(
            df_left,
            df_right,
            candidates,
            id_column="id",
            threshold=0.5,
            comparators=["ignored"],
            weights=[1.0],
        )
        assert not result.empty

    def test_threshold_filters_output(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=_fake_llm({("L1", "R1")}),
        )
        # Threshold 1.5 is above any possible score
        result = matcher.match(
            df_left, df_right, candidates, id_column="id", threshold=1.5
        )
        assert result.empty


# ---------------------------------------------------------------------------
# Serialization + prompt assembly
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_missing_marker_for_nan(self, fields: list[str]) -> None:
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=Path("/tmp/ignored"),
            llm_callable=_fake_llm(set()),
        )
        rec = pd.Series(
            {
                "name": "Initech",
                "country": "USA",
                "city": "Austin",
                "industry": np.nan,
                "sector": "Tech",
                "founded": 1999,
            }
        )
        text = matcher._serialize_record(rec, "Record A")
        assert "industry: <missing>" in text
        assert "name: Initech" in text

    def test_missing_marker_for_empty_string(self, fields: list[str]) -> None:
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=Path("/tmp/ignored"),
            llm_callable=_fake_llm(set()),
        )
        rec = pd.Series(
            {
                "name": "  ",
                "country": "USA",
                "city": "Austin",
                "industry": "Tech",
                "sector": "Tech",
                "founded": 1999,
            }
        )
        text = matcher._serialize_record(rec, "Record A")
        assert f"name: {_MISSING_MARKER}" in text

    def test_prompt_contains_both_records_and_system(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        stub = _fake_llm({("L1", "R1")})
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=stub,
        )
        matcher.match(df_left, df_right, candidates, id_column="id")
        prompts = stub.calls  # type: ignore[attr-defined]
        assert prompts
        for p in prompts:
            assert "System:" in p
            assert "Record A:" in p
            assert "Record B:" in p
            assert "Do the two records refer to the same entity?" in p
            assert "Query:" in p


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    @pytest.mark.parametrize(
        "response, expected",
        [
            ("Yes", 1.0),
            ("yes", 1.0),
            ("Yes.", 1.0),
            ("Yes, same entity", 1.0),
            ("YES\n", 1.0),
            ("No", 0.0),
            ("no.", 0.0),
            ("", 0.0),
            ("unknown", 0.0),
            ("maybe", 0.0),
        ],
    )
    def test_parse_response(self, response: str, expected: float) -> None:
        assert MatchGPTMatcher._parse_response(response) == expected


# ---------------------------------------------------------------------------
# Determinism + cache
# ---------------------------------------------------------------------------


class TestDeterminismAndCache:
    def test_cache_key_stable_across_instances(
        self, fields: list[str], tmp_path: Path
    ) -> None:
        m1 = MatchGPTMatcher(fields=fields, cache_dir=tmp_path)
        m2 = MatchGPTMatcher(fields=fields, cache_dir=tmp_path)
        prompt = "System: hello\n\nQuery: world"
        assert m1._cache_key(prompt) == m2._cache_key(prompt)

    def test_cache_writes_and_reads(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        stub1 = _fake_llm({("L1", "R1")})
        matcher = MatchGPTMatcher(fields=fields, cache_dir=tmp_path, llm_callable=stub1)
        matcher.match(df_left, df_right, candidates, id_column="id")
        first_call_count = len(stub1.calls)  # type: ignore[attr-defined]
        assert first_call_count == 3  # one call per candidate

        # Second matcher instance with a failing LLM — all answers must come
        # from the cache from the first run.
        def _boom(prompt: str) -> str:
            raise AssertionError("LLM should not be called on cache hit")

        matcher2 = MatchGPTMatcher(
            fields=fields, cache_dir=tmp_path, llm_callable=_boom
        )
        result2 = matcher2.match(df_left, df_right, candidates, id_column="id")
        assert set(result2["id1"]) == {"L1"}
        assert (result2["score"] == 1.0).all()

    def test_cache_file_payload_structure(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=_fake_llm({("L1", "R1")}),
            chat_model_name="openai/gpt-4o-mini",
        )
        matcher.match(df_left, df_right, candidates, id_column="id")
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 3
        payload = json.loads(cache_files[0].read_text())
        assert payload["prompt_version"] == _PROMPT_VERSION
        assert payload["model_id"] == "openai/gpt-4o-mini"
        assert "prompt" in payload
        assert payload["response"] in {"Yes", "No"}

    def test_repeated_match_is_idempotent(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        stub = _fake_llm({("L1", "R1"), ("L3", "R3")})
        matcher = MatchGPTMatcher(fields=fields, cache_dir=tmp_path, llm_callable=stub)
        r1 = matcher.match(df_left, df_right, candidates, id_column="id")
        r2 = matcher.match(df_left, df_right, candidates, id_column="id")
        pd.testing.assert_frame_equal(
            r1.reset_index(drop=True), r2.reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Sanity — positive pairs score higher than negative pairs
# ---------------------------------------------------------------------------


class TestSanity:
    def test_positive_scored_above_negative(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        # Positive pair (L1, R1) answers Yes; negative (L2, R2) answers No.
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=_fake_llm({("L1", "R1"), ("L3", "R3")}),
        )
        result = matcher.match(
            df_left, df_right, candidates, id_column="id", threshold=0.0
        )
        scores = dict(zip(zip(result["id1"], result["id2"]), result["score"]))
        assert scores.get(("L1", "R1"), 0.0) > scores.get(("L2", "R2"), 0.0)
        assert scores[("L1", "R1")] == 1.0


# ---------------------------------------------------------------------------
# Demonstration retrieval (few-shot)
# ---------------------------------------------------------------------------


class TestDemonstrations:
    def test_zero_shot_when_path_none(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        stub = _fake_llm({("L1", "R1")})
        matcher = MatchGPTMatcher(fields=fields, cache_dir=tmp_path, llm_callable=stub)
        matcher.match(df_left, df_right, candidates, id_column="id")
        # No "Example" blocks in any prompt
        for prompt in stub.calls:  # type: ignore[attr-defined]
            assert "Example 1:" not in prompt

    def test_few_shot_injects_demonstrations(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        # Build a demo pool: (L1,R1) is a positive; (L2,R2) is a negative.
        demo_csv = tmp_path / "demos.csv"
        pd.DataFrame(
            [
                {"id1": "L1", "id2": "R1", "label": "true"},
                {"id1": "L2", "id2": "R2", "label": "false"},
            ]
        ).to_csv(demo_csv, index=False)

        stub = _fake_llm({("L3", "R3")})
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path / "cache",
            llm_callable=stub,
            embedder=_fake_embedder(),
            demonstrations_path=demo_csv,
            k_shot=2,
        )
        matcher.match(df_left, df_right, candidates, id_column="id")
        # Query prompts for L3/R3 must include both demo blocks
        found_example_prompt = False
        for prompt in stub.calls:  # type: ignore[attr-defined]
            if "Initech" in prompt.split("Query:", 1)[-1]:
                assert "Example 1:" in prompt
                assert "Example 2:" in prompt
                # Labels are rendered as Yes/No in demos
                assert "Answer: Yes" in prompt
                assert "Answer: No" in prompt
                found_example_prompt = True
        assert found_example_prompt

    def test_k_shot_zero_disables_demos(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        demo_csv = tmp_path / "demos.csv"
        pd.DataFrame([{"id1": "L1", "id2": "R1", "label": "true"}]).to_csv(
            demo_csv, index=False
        )

        stub = _fake_llm({("L1", "R1")})
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path / "cache",
            llm_callable=stub,
            embedder=_fake_embedder(),
            demonstrations_path=demo_csv,
            k_shot=0,
        )
        matcher.match(df_left, df_right, candidates, id_column="id")
        for prompt in stub.calls:  # type: ignore[attr-defined]
            assert "Example 1:" not in prompt

    def test_demonstrations_path_missing_raises(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=_fake_llm(set()),
            embedder=_fake_embedder(),
            demonstrations_path=tmp_path / "does_not_exist.csv",
            k_shot=1,
        )
        with pytest.raises(FileNotFoundError, match="demonstrations_path"):
            matcher.match(df_left, df_right, candidates, id_column="id")

    def test_demonstrations_csv_missing_columns_raises(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame([{"id1": "L1", "id2": "R1"}]).to_csv(bad_csv, index=False)
        matcher = MatchGPTMatcher(
            fields=fields,
            cache_dir=tmp_path,
            llm_callable=_fake_llm(set()),
            embedder=_fake_embedder(),
            demonstrations_path=bad_csv,
            k_shot=1,
        )
        with pytest.raises(ValueError, match="missing columns"):
            matcher.match(df_left, df_right, candidates, id_column="id")

    @pytest.mark.parametrize(
        "label, expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            (True, True),
            ("false", False),
            ("0", False),
            ("no", False),
            (False, False),
        ],
    )
    def test_label_coercion(self, label: object, expected: bool) -> None:
        assert MatchGPTMatcher._coerce_label(label) is expected

    def test_label_coercion_rejects_nonsense(self) -> None:
        with pytest.raises(ValueError, match="Cannot coerce"):
            MatchGPTMatcher._coerce_label("maybe")


# ---------------------------------------------------------------------------
# Robustness — skipping unknown ids
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_skips_candidate_with_unknown_ids(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        fields: list[str],
        tmp_path: Path,
    ) -> None:
        cands = pd.DataFrame(
            [
                {"id1": "L1", "id2": "R1"},
                {"id1": "L_missing", "id2": "R1"},
                {"id1": "L1", "id2": "R_missing"},
            ]
        )
        stub = _fake_llm({("L1", "R1")})
        matcher = MatchGPTMatcher(fields=fields, cache_dir=tmp_path, llm_callable=stub)
        result = matcher.match(df_left, df_right, cands, id_column="id")
        assert len(stub.calls) == 1  # type: ignore[attr-defined]
        assert list(result["id1"]) == ["L1"]
        assert list(result["id2"]) == ["R1"]

"""Tests for the v2 LLMCanonicalizer (C12, 2026-05-25).

v2 contract: JSON output with ``value`` / ``operation`` / ``confidence``
/ ``reasoning``; operations from
``{vocab_canonicalize, date_normalize, numeric_normalize,
categorical_map, synthesize, abstain}``; synthesis permitted; per-call
operation log appended when ``op_log_path`` is wired.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from usecases_synthetic.lib.llm_normalizer import (
    PROMPT_VERSION_V2,
    VALID_NORM_OPERATIONS_V2,
    LLMCanonicalizer,
    _parse_response,
)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParseResponseV2:
    def test_valid_vocab_canonicalize(self) -> None:
        raw = json.dumps(
            {
                "value": "United States",
                "operation": "vocab_canonicalize",
                "confidence": 0.95,
                "reasoning": "matches example",
            }
        )
        parsed = _parse_response(raw)
        assert parsed is not None
        assert parsed["canonical"] == "United States"
        assert parsed["operation"] == "vocab_canonicalize"
        assert parsed["confidence"] == pytest.approx(0.95)
        assert parsed["reasoning"] == "matches example"

    def test_synthesize_operation_keeps_value(self) -> None:
        raw = json.dumps(
            {
                "value": "2007-08-05",
                "operation": "date_normalize",
                "confidence": 0.9,
                "reasoning": "ISO conversion",
            }
        )
        parsed = _parse_response(raw)
        assert parsed is not None
        assert parsed["canonical"] == "2007-08-05"
        assert parsed["operation"] == "date_normalize"

    def test_abstain_returns_none_canonical(self) -> None:
        raw = json.dumps(
            {
                "value": None,
                "operation": "abstain",
                "confidence": 0.0,
                "reasoning": "unparseable",
            }
        )
        parsed = _parse_response(raw)
        assert parsed is not None
        assert parsed["canonical"] is None
        assert parsed["operation"] == "abstain"

    def test_abstain_with_value_still_yields_none(self) -> None:
        """If operation=abstain, canonical is None regardless of `value`."""
        raw = json.dumps(
            {
                "value": "garbage",
                "operation": "abstain",
                "confidence": 0.0,
                "reasoning": "n/a",
            }
        )
        parsed = _parse_response(raw)
        assert parsed is not None
        assert parsed["canonical"] is None

    def test_null_string_treated_as_none(self) -> None:
        raw = json.dumps(
            {
                "value": "NULL",
                "operation": "vocab_canonicalize",
                "confidence": 0.5,
                "reasoning": "n/a",
            }
        )
        parsed = _parse_response(raw)
        assert parsed is not None
        assert parsed["canonical"] is None

    def test_unknown_operation_rejected(self) -> None:
        raw = json.dumps(
            {
                "value": "X",
                "operation": "fancy_new_op",
                "confidence": 0.9,
                "reasoning": "n/a",
            }
        )
        assert _parse_response(raw) is None

    def test_missing_value_rejected(self) -> None:
        raw = json.dumps(
            {
                "operation": "vocab_canonicalize",
                "confidence": 0.9,
                "reasoning": "n/a",
            }
        )
        assert _parse_response(raw) is None

    def test_fenced_json_parses(self) -> None:
        raw = (
            "```json\n"
            + json.dumps(
                {
                    "value": "Rock",
                    "operation": "vocab_canonicalize",
                    "confidence": 0.8,
                    "reasoning": "exact match",
                }
            )
            + "\n```"
        )
        parsed = _parse_response(raw)
        assert parsed is not None
        assert parsed["canonical"] == "Rock"

    def test_garbage_returns_none(self) -> None:
        assert _parse_response("not json at all") is None

    def test_empty_returns_none(self) -> None:
        assert _parse_response("") is None

    def test_clipped_confidence(self) -> None:
        raw = json.dumps(
            {
                "value": "X",
                "operation": "categorical_map",
                "confidence": 2.5,
                "reasoning": "n/a",
            }
        )
        parsed = _parse_response(raw)
        assert parsed is not None
        assert parsed["confidence"] == 1.0


# ---------------------------------------------------------------------------
# Operations enumeration sanity
# ---------------------------------------------------------------------------


def test_valid_operations_v2_complete() -> None:
    assert VALID_NORM_OPERATIONS_V2 == frozenset(
        {
            "vocab_canonicalize",
            "date_normalize",
            "numeric_normalize",
            "categorical_map",
            "synthesize",
            "abstain",
        }
    )


def test_prompt_version_is_v2() -> None:
    assert PROMPT_VERSION_V2 == "v2"


# ---------------------------------------------------------------------------
# LLMCanonicalizer.normalize end-to-end
# ---------------------------------------------------------------------------


class TestLLMCanonicalizerNormalize:
    def _stub_canonicalizer(
        self, tmp_path: Path, *, response: str, op_log: Path | None = None
    ) -> LLMCanonicalizer:
        canon = LLMCanonicalizer(
            num_examples=3,
            cache_dir=tmp_path / "cache",
            op_log_path=op_log,
        )
        canon._llm_callable = lambda system, user: response  # noqa: SLF001
        return canon

    def test_vocab_canonicalize_round_trip(self, tmp_path: Path) -> None:
        canon = self._stub_canonicalizer(
            tmp_path,
            response=json.dumps(
                {
                    "value": "Rock",
                    "operation": "vocab_canonicalize",
                    "confidence": 0.95,
                    "reasoning": "matches",
                }
            ),
        )
        canon.set_examples({"music": {"genre": ["Rock", "Pop"]}})
        canonical = canon.normalize(
            "rock-and-roll",
            attribute="genre",
            kind="nominal",
            domain="music",
        )
        assert canonical == "Rock"

    def test_abstain_returns_none(self, tmp_path: Path) -> None:
        canon = self._stub_canonicalizer(
            tmp_path,
            response=json.dumps(
                {
                    "value": None,
                    "operation": "abstain",
                    "confidence": 0.0,
                    "reasoning": "n/a",
                }
            ),
        )
        canon.set_examples({"music": {"genre": ["Rock"]}})
        assert (
            canon.normalize(
                "????",
                attribute="genre",
                kind="nominal",
                domain="music",
            )
            is None
        )

    def test_synthesize_returns_value(self, tmp_path: Path) -> None:
        """Synthesis is permitted: canonical need not be in the example list."""
        canon = self._stub_canonicalizer(
            tmp_path,
            response=json.dumps(
                {
                    "value": "2007-08-05",
                    "operation": "date_normalize",
                    "confidence": 0.9,
                    "reasoning": "ISO conversion",
                }
            ),
        )
        canon.set_examples({"music": {"release_date": ["2007-08-05"]}})
        canonical = canon.normalize(
            "Aug 5, 2007",
            attribute="release_date",
            kind="date",
            domain="music",
        )
        assert canonical == "2007-08-05"

    def test_cache_hit_skips_llm(self, tmp_path: Path) -> None:
        called = {"count": 0}

        def stub(system: str, user: str) -> str:
            called["count"] += 1
            return json.dumps(
                {
                    "value": "Rock",
                    "operation": "vocab_canonicalize",
                    "confidence": 0.9,
                    "reasoning": "n/a",
                }
            )

        canon = LLMCanonicalizer(
            num_examples=3,
            cache_dir=tmp_path / "cache",
        )
        canon._llm_callable = stub  # noqa: SLF001
        canon.set_examples({"music": {"genre": ["Rock"]}})

        for _ in range(2):
            canon.normalize(
                "rock",
                attribute="genre",
                kind="nominal",
                domain="music",
            )
        assert called["count"] == 1

    def test_op_log_appended(self, tmp_path: Path) -> None:
        op_log = tmp_path / "llm_only_operations.csv"
        canon = self._stub_canonicalizer(
            tmp_path,
            response=json.dumps(
                {
                    "value": "United States",
                    "operation": "vocab_canonicalize",
                    "confidence": 0.92,
                    "reasoning": "matches example",
                }
            ),
            op_log=op_log,
        )
        canon.set_examples({"companies": {"country": ["United States"]}})
        canon.normalize(
            "US",
            attribute="country",
            kind="codelist",
            domain="companies",
        )
        assert op_log.exists()
        rows = pd.read_csv(op_log)
        assert len(rows) == 1
        assert rows.loc[0, "operation"] == "vocab_canonicalize"
        assert rows.loc[0, "domain"] == "companies"
        assert rows.loc[0, "attribute"] == "country"
        assert rows.loc[0, "canonical_value"] == "United States"
        assert int(rows.loc[0, "cache_hit"]) == 0

    def test_op_log_records_cache_hits(self, tmp_path: Path) -> None:
        op_log = tmp_path / "llm_only_operations.csv"
        canon = self._stub_canonicalizer(
            tmp_path,
            response=json.dumps(
                {
                    "value": "Pop",
                    "operation": "vocab_canonicalize",
                    "confidence": 0.9,
                    "reasoning": "n/a",
                }
            ),
            op_log=op_log,
        )
        canon.set_examples({"music": {"genre": ["Pop"]}})
        # Two calls with identical inputs — first is a miss, second a hit.
        for _ in range(2):
            canon.normalize(
                "pop",
                attribute="genre",
                kind="nominal",
                domain="music",
            )
        rows = pd.read_csv(op_log)
        assert len(rows) == 2
        assert int(rows.loc[0, "cache_hit"]) == 0
        assert int(rows.loc[1, "cache_hit"]) == 1

    def test_empty_value_returns_none(self, tmp_path: Path) -> None:
        canon = self._stub_canonicalizer(
            tmp_path,
            response=json.dumps(
                {
                    "value": "X",
                    "operation": "vocab_canonicalize",
                    "confidence": 0.9,
                    "reasoning": "n/a",
                }
            ),
        )
        canon.set_examples({"music": {"genre": ["Rock"]}})
        # ``_stringify`` filters None / NaN / empty / "nan" before the LLM
        # path; normalize should short-circuit to None.
        assert (
            canon.normalize(
                None,
                attribute="genre",
                kind="nominal",
                domain="music",
            )
            is None
        )
        assert (
            canon.normalize(
                "",
                attribute="genre",
                kind="nominal",
                domain="music",
            )
            is None
        )

    def test_prompt_version_invalidates_cache(self, tmp_path: Path) -> None:
        called = {"count": 0}

        def stub(system: str, user: str) -> str:
            called["count"] += 1
            return json.dumps(
                {
                    "value": "Rock",
                    "operation": "vocab_canonicalize",
                    "confidence": 0.9,
                    "reasoning": "n/a",
                }
            )

        for version in ("v2", "v2-experimental"):
            canon = LLMCanonicalizer(
                num_examples=3,
                cache_dir=tmp_path / "cache",
                prompt_version=version,
            )
            canon._llm_callable = stub  # noqa: SLF001
            canon.set_examples({"music": {"genre": ["Rock"]}})
            canon.normalize(
                "rock",
                attribute="genre",
                kind="nominal",
                domain="music",
            )
        assert called["count"] == 2

    def test_garbage_response_yields_none(self, tmp_path: Path) -> None:
        """A response that fails JSON parse becomes an abstain."""
        canon = self._stub_canonicalizer(
            tmp_path,
            response="this is definitely not JSON",
        )
        canon.set_examples({"music": {"genre": ["Rock"]}})
        assert (
            canon.normalize(
                "rock",
                attribute="genre",
                kind="nominal",
                domain="music",
            )
            is None
        )

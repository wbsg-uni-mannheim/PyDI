"""ComEM (Match / Compare / Select) LLM entity matcher.

Adapts the ComEM compound-strategy recipe from
Wang et al. *"Match, Compare, or Select? An Investigation of Large
Language Models for Entity Matching"* (COLING 2025) to PyDI's
:class:`~PyDI.entitymatching.base.BaseMatcher` contract.  Fills the
fourth slot in the frozen EM matching committee (see
``plans/plan_committee_finalization.md`` §C2.4b and
``knobs/committee_review/em_shortlist.md``).

Pipeline
--------

Stage 1 — **Selecting.**  Candidates are grouped by the left-side
``id_column``.  Each group's query entity is presented together with
all its candidates (numbered ``1 … k``) in a single prompt; the LLM
answers with a comma-separated list of matching candidate indices or
``"None"``.  This is the ComEM "cost-effective filter".

Stage 2 — **Matching.**  Each Stage-1 survivor is confirmed with an
independent binary ``Yes``/``No`` prompt, optionally using a more
capable LLM.  Pairs that survive both stages are emitted with
``score = 1.0``; everything else is dropped.

Design notes
------------

* **Serialization.**  Records are serialized with a pinned
  ``name: value`` layout, one line per field, NaN/None cells rendered
  as ``<missing>`` so the LLM does not read blanks as equality.  The
  layout matches :class:`~usecases_synthetic.lib.matchgpt_em_matcher.MatchGPTMatcher`
  so committee-level diffs come from strategy choice, not surface
  serialization.
* **Stage 1 skipping.**  Groups smaller than ``skip_stage1_below``
  skip the selecting prompt and go straight to Stage 2 — ComEM's own
  heuristic (§"Strategy selection heuristics": *if candidate set is
  small, skip Stage 1, use matching directly*).
* **LLM hygiene.**  ``temperature=0``, pinned ``_PROMPT_VERSION``,
  file-backed response cache under ``cache_dir`` keyed by
  ``sha256(prompt_version | model_id | stage | prompt_text)``.  The
  stage marker prevents cross-stage collisions even if prompt bodies
  overlap.  Committed cache files are the source of truth on rerun,
  matching the Magneto / MatchGPT hygiene pattern.
* **Testability.**  ``llm_stage1_callable`` and ``llm_stage2_callable``
  are injectable ``__init__`` arguments so tests can stub the LLM
  without network.  When only ``llm_stage1_callable`` is supplied and
  ``stage2_model`` is ``None``, Stage 2 reuses the Stage 1 callable —
  the common single-model test + production path.

If this adapter ever needs to graduate to PyDI core, the promotion
path is to move it to ``PyDI/entitymatching/comem_based.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

from PyDI.entitymatching.base import BaseMatcher, CorrespondenceSet

logger = logging.getLogger(__name__)


_PROMPT_VERSION = "comem-v1"
_DEFAULT_MODEL = "openai/gpt-4o-mini"
_MISSING_MARKER = "<missing>"
_STAGE1 = "select"
_STAGE2 = "match"

_SYSTEM_PROMPT_STAGE1 = (
    "You are an expert data integrator. Your task is to decide which "
    "candidates describe the same real-world entity as the query.\n"
    "\n"
    "Rules:\n"
    "- Compare the query against each candidate attribute-by-attribute.\n"
    "- Pay attention to abbreviations, alternate spellings, and "
    "formatting differences (e.g. 'Inc.' vs 'Incorporated').\n"
    "- Treat values marked '<missing>' as unknown, not as mismatches.\n"
    "- Respond with a comma-separated list of candidate numbers that "
    "match the query (e.g. '1,3'), or exactly 'None' if no candidate "
    "matches. Do not include any other text."
)

_SYSTEM_PROMPT_STAGE2 = (
    "You are an expert data integrator. Your task is to decide whether "
    "two records refer to the same real-world entity.\n"
    "\n"
    "Rules:\n"
    "- Compare the records attribute-by-attribute.\n"
    "- Pay attention to abbreviations, alternate spellings, and "
    "formatting differences (e.g. 'Inc.' vs 'Incorporated').\n"
    "- Treat values marked '<missing>' as unknown, not as mismatches.\n"
    "- Respond with a single word: 'Yes' if the records refer to the "
    "same entity, 'No' otherwise. Do not include any other text."
)

LLMCallable = Callable[[str], str]


class ComEMMatcher(BaseMatcher):
    """Compound-strategy LLM entity matcher (ComEM).

    Parameters
    ----------
    fields : list of str
        Canonical field names to include in the serialized records.
        Fields absent from a record (or ``NaN``) are rendered as
        ``<missing>``.
    stage1_model : str, optional
        Model id used for Stage 1 (selecting).  Used for cache-keying
        and — when no ``llm_stage1_callable`` is injected — to build
        the default LiteLLM-based client.  Default
        ``"openai/gpt-4o-mini"``.
    stage2_model : str or None, optional
        Model id used for Stage 2 (binary matching).  ``None`` means
        reuse ``stage1_model`` (the common single-model configuration).
        When set to a different value, Stage 2 uses a separate LLM
        client.  Default ``None``.
    stage1_set_size : int, optional
        Maximum candidates per Stage 1 prompt.  Groups exceeding this
        size are split into sub-batches and the selected indices are
        union-merged.  Default 10.
    skip_stage1_below : int, optional
        Groups of fewer than ``skip_stage1_below`` candidates skip
        Stage 1 entirely and go straight to Stage 2 — matching
        ComEM's "small group → direct matching" heuristic.  Default
        2 (so groups of size 1 always skip).
    temperature : float, optional
        Sampling temperature forwarded to the default chat clients.
        Default 0.0.
    max_tokens_stage1 : int, optional
        Response token cap for Stage 1 (must fit a short CSV of
        indices).  Default 128.
    max_tokens_stage2 : int, optional
        Response token cap for Stage 2 (must fit ``Yes``/``No`` plus
        trailing whitespace).  Default 8.
    cache_dir : str or Path, optional
        Directory for the response cache.  Defaults to
        ``usecases_synthetic/cache/comem_prompts/``.
    seed : int, optional
        Random seed used only for deterministic tie-breaking on
        Stage 1 group ordering.  Default 42.
    llm_stage1_callable : callable, optional
        Test-time hook.  Callable ``(prompt_text) -> str`` for
        Stage 1.  When ``None`` a LiteLLM client is built lazily.
    llm_stage2_callable : callable, optional
        Test-time hook.  Callable ``(prompt_text) -> str`` for
        Stage 2.  When ``None`` and ``stage2_model`` is also ``None``,
        ``llm_stage1_callable`` is reused; otherwise a second
        LiteLLM client is built lazily.

    Notes
    -----
    The committee runner forwards ``comparators`` / ``weights`` kwargs
    to every matcher; both are accepted and ignored here (LLM
    matchers do not use comparator-vector features).
    """

    def __init__(
        self,
        fields: list[str],
        stage1_model: str = _DEFAULT_MODEL,
        stage2_model: Optional[str] = None,
        stage1_set_size: int = 10,
        skip_stage1_below: int = 2,
        temperature: float = 0.0,
        max_tokens_stage1: int = 2048,
        max_tokens_stage2: int = 2048,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        llm_stage1_callable: Optional[LLMCallable] = None,
        llm_stage2_callable: Optional[LLMCallable] = None,
    ) -> None:
        if not fields:
            raise ValueError("ComEMMatcher requires at least one field")
        if stage1_set_size < 1:
            raise ValueError("stage1_set_size must be >= 1")
        if skip_stage1_below < 1:
            raise ValueError("skip_stage1_below must be >= 1")
        self.fields = list(fields)
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model if stage2_model is not None else stage1_model
        self.stage1_set_size = int(stage1_set_size)
        self.skip_stage1_below = int(skip_stage1_below)
        self.temperature = float(temperature)
        self.max_tokens_stage1 = int(max_tokens_stage1)
        self.max_tokens_stage2 = int(max_tokens_stage2)
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[1] / "cache" / "comem_prompts"
        self.cache_dir = Path(cache_dir)
        self.seed = int(seed)

        self._llm_stage1: Optional[LLMCallable] = llm_stage1_callable
        self._llm_stage2: Optional[LLMCallable] = llm_stage2_callable
        # Track whether Stage 2 should fall back to Stage 1 when neither
        # callable nor a distinct stage2_model is configured.
        self._share_stage_callables = (
            llm_stage2_callable is None and stage2_model is None
        )

    # ------------------------------------------------------------------
    # Record serialization
    # ------------------------------------------------------------------

    def _format_value(self, value: Any) -> str:
        if value is None:
            return _MISSING_MARKER
        if isinstance(value, float) and np.isnan(value):
            return _MISSING_MARKER
        text = str(value).strip()
        return text if text else _MISSING_MARKER

    def _serialize_record(self, record: pd.Series, label: str) -> str:
        lines = [f"{label}:"]
        for field in self.fields:
            raw = record[field] if field in record.index else None
            lines.append(f"  {field}: {self._format_value(raw)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_stage1_prompt(
        self,
        query: pd.Series,
        candidates: list[pd.Series],
    ) -> str:
        parts = [f"System:\n{_SYSTEM_PROMPT_STAGE1}"]
        parts.append(self._serialize_record(query, "Query Entity"))
        cand_lines = ["Candidates:"]
        for idx, cand in enumerate(candidates, start=1):
            cand_lines.append(self._serialize_record(cand, f"{idx}"))
        parts.append("\n".join(cand_lines))
        parts.append(
            "Which candidate(s) refer to the same entity as the query? "
            "Respond with a comma-separated list of candidate numbers "
            "(e.g. '1,3') or exactly 'None'."
        )
        return "\n\n".join(parts)

    def _build_stage2_prompt(self, left: pd.Series, right: pd.Series) -> str:
        return (
            f"System:\n{_SYSTEM_PROMPT_STAGE2}\n\n"
            f"{self._serialize_record(left, 'Record A')}\n"
            f"{self._serialize_record(right, 'Record B')}\n\n"
            "Do the two records refer to the same entity? Answer:"
        )

    # ------------------------------------------------------------------
    # LLM invocation + cache
    # ------------------------------------------------------------------

    def _cache_key(self, stage: str, model_id: str, prompt_text: str) -> str:
        payload = "|".join([_PROMPT_VERSION, model_id, stage, prompt_text])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, stage: str, model_id: str, prompt_text: str) -> Optional[str]:
        path = self._cache_path(self._cache_key(stage, model_id, prompt_text))
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return str(payload["response"])

    def _cache_put(
        self, stage: str, model_id: str, prompt_text: str, response: str
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(self._cache_key(stage, model_id, prompt_text))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prompt_version": _PROMPT_VERSION,
                    "model_id": model_id,
                    "stage": stage,
                    "prompt": prompt_text,
                    "response": response,
                },
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

    def _ensure_stage1_callable(self) -> LLMCallable:
        if self._llm_stage1 is not None:
            return self._llm_stage1
        self._llm_stage1 = self._build_chat_callable(
            self.stage1_model, self.max_tokens_stage1
        )
        if self._share_stage_callables:
            self._llm_stage2 = self._llm_stage1
        return self._llm_stage1

    def _ensure_stage2_callable(self) -> LLMCallable:
        if self._llm_stage2 is not None:
            return self._llm_stage2
        if self._share_stage_callables:
            return self._ensure_stage1_callable()
        self._llm_stage2 = self._build_chat_callable(
            self.stage2_model, self.max_tokens_stage2
        )
        return self._llm_stage2

    def _build_chat_callable(self, model_id: str, max_tokens: int) -> LLMCallable:
        from .llm_client import build_chat_openai

        chat = build_chat_openai(
            model=model_id,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

        def _call(prompt_text: str) -> str:
            response = chat.invoke(prompt_text)
            content = getattr(response, "content", response)
            return str(content)

        return _call

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    _INDEX_RE = re.compile(r"\d+")

    @classmethod
    def _parse_stage1_response(
        cls, response_text: str, num_candidates: int
    ) -> set[int]:
        """Parse Stage 1's CSV-of-indices response into a 1-based set.

        Robust to ``"None"``, whitespace, trailing punctuation, and
        out-of-range numbers (silently dropped).  Returns an empty set
        when the LLM produces unparseable text — false negatives are
        preferred to corrupt matches, consistent with the ComEM paper
        (§"Limitations": *Stage 1 errors propagate as false negatives*).
        """
        if not response_text:
            return set()
        stripped = response_text.strip()
        if not stripped:
            return set()
        lowered = stripped.lower()
        if lowered.startswith("none") or lowered == "no":
            return set()
        ints = cls._INDEX_RE.findall(stripped)
        out: set[int] = set()
        for tok in ints:
            try:
                idx = int(tok)
            except ValueError:
                continue
            if 1 <= idx <= num_candidates:
                out.add(idx)
        return out

    @staticmethod
    def _parse_stage2_response(response_text: str) -> float:
        if not response_text:
            return 0.0
        stripped = response_text.strip().lower()
        first_token = stripped.split()[0].strip(".,!?:;\"'") if stripped else ""
        if first_token == "yes":
            return 1.0
        if first_token == "no":
            return 0.0
        if stripped.startswith("yes"):
            return 1.0
        if stripped.startswith("no"):
            return 0.0
        return 0.0

    # ------------------------------------------------------------------
    # BaseMatcher contract
    # ------------------------------------------------------------------

    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Union[pd.DataFrame, Iterable[pd.DataFrame]],
        id_column: str,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> CorrespondenceSet:
        self._validate_inputs(df_left, df_right, id_column)
        kwargs.pop("comparators", None)
        kwargs.pop("weights", None)

        if isinstance(candidates, pd.DataFrame):
            candidate_batches: list[pd.DataFrame] = [candidates]
        else:
            candidate_batches = list(candidates)

        left_by_id = df_left.set_index(id_column, drop=False)
        right_by_id = df_right.set_index(id_column, drop=False)

        merged = self._merge_and_validate_batches(candidate_batches)
        if merged.empty:
            return pd.DataFrame(columns=["id1", "id2", "score", "notes"])

        stage1_survivors = self._run_stage1(merged, left_by_id, right_by_id)

        results = self._run_stage2(stage1_survivors, left_by_id, right_by_id, threshold)

        if not results:
            return pd.DataFrame(columns=["id1", "id2", "score", "notes"])
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_and_validate_batches(
        batches: list[pd.DataFrame],
    ) -> pd.DataFrame:
        non_empty: list[pd.DataFrame] = []
        for batch in batches:
            if batch.empty:
                continue
            if "id1" not in batch.columns or "id2" not in batch.columns:
                raise ValueError(
                    "Candidate DataFrame must have 'id1' and 'id2' columns"
                )
            non_empty.append(batch[["id1", "id2"]])
        if not non_empty:
            return pd.DataFrame(columns=["id1", "id2"])
        merged = pd.concat(non_empty, ignore_index=True)
        # Deterministic iteration order for Stage 1 group batching.
        return merged.drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True)

    def _run_stage1(
        self,
        candidates: pd.DataFrame,
        left_by_id: pd.DataFrame,
        right_by_id: pd.DataFrame,
    ) -> list[tuple[Any, Any]]:
        """Return the list of (id1, id2) pairs surviving Stage 1."""
        survivors: list[tuple[Any, Any]] = []
        # Preserve first-seen order across id1 groups for determinism.
        seen_groups: list[Any] = []
        groups: dict[Any, list[Any]] = {}
        for _, row in candidates.iterrows():
            gid = row["id1"]
            if gid not in groups:
                groups[gid] = []
                seen_groups.append(gid)
            groups[gid].append(row["id2"])

        llm: Optional[LLMCallable] = None

        for gid in seen_groups:
            id2_list = groups[gid]
            if gid not in left_by_id.index:
                continue
            query_rec = self._row_series(left_by_id.loc[gid])

            if len(id2_list) < self.skip_stage1_below:
                # Small group: skip Stage 1 entirely
                for id2 in id2_list:
                    if id2 in right_by_id.index:
                        survivors.append((gid, id2))
                continue

            if llm is None:
                llm = self._ensure_stage1_callable()

            # Split the group into chunks of size stage1_set_size and
            # union the selected indices.
            for chunk_start in range(0, len(id2_list), self.stage1_set_size):
                chunk = id2_list[chunk_start : chunk_start + self.stage1_set_size]
                cand_records: list[pd.Series] = []
                chunk_id2s: list[Any] = []
                for id2 in chunk:
                    if id2 not in right_by_id.index:
                        continue
                    cand_records.append(self._row_series(right_by_id.loc[id2]))
                    chunk_id2s.append(id2)
                if not cand_records:
                    continue

                prompt_text = self._build_stage1_prompt(query_rec, cand_records)
                cached = self._cache_get(_STAGE1, self.stage1_model, prompt_text)
                if cached is not None:
                    response_text = cached
                else:
                    response_text = llm(prompt_text)
                    self._cache_put(
                        _STAGE1, self.stage1_model, prompt_text, response_text
                    )

                selected_idx = self._parse_stage1_response(
                    response_text, len(chunk_id2s)
                )
                for idx in selected_idx:
                    survivors.append((gid, chunk_id2s[idx - 1]))

        return survivors

    def _run_stage2(
        self,
        survivors: list[tuple[Any, Any]],
        left_by_id: pd.DataFrame,
        right_by_id: pd.DataFrame,
        threshold: float,
    ) -> list[dict[str, Any]]:
        if not survivors:
            return []
        llm = self._ensure_stage2_callable()
        results: list[dict[str, Any]] = []
        for id1, id2 in survivors:
            if id1 not in left_by_id.index or id2 not in right_by_id.index:
                continue
            left_rec = self._row_series(left_by_id.loc[id1])
            right_rec = self._row_series(right_by_id.loc[id2])

            prompt_text = self._build_stage2_prompt(left_rec, right_rec)
            cached = self._cache_get(_STAGE2, self.stage2_model, prompt_text)
            if cached is not None:
                response_text = cached
            else:
                response_text = llm(prompt_text)
                self._cache_put(_STAGE2, self.stage2_model, prompt_text, response_text)

            score = self._parse_stage2_response(response_text)
            if score >= threshold:
                results.append(
                    {
                        "id1": id1,
                        "id2": id2,
                        "score": float(score),
                        "notes": "comem",
                    }
                )
        return results

    @staticmethod
    def _row_series(row: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row


__all__ = ["ComEMMatcher"]

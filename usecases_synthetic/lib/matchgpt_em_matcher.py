"""MatchGPT-style LLM entity matcher (synthetic-local infrastructure).

Adapts the MatchGPT (WBSG Mannheim, arXiv:2310.11244) prompt recipe to
PyDI's :class:`~PyDI.entitymatching.base.BaseMatcher` contract.  The
frozen EM matching committee slot `llm_matcher` (see
`plans/plan_committee_finalization.md` C2.4b) calls for MatchGPT-style
structured prompts with optional in-context demonstrations; the
vanilla :class:`PyDI.entitymatching.llm_based.LLMBasedMatcher` ships
a simpler default prompt and is kept untouched per the read-only-PyDI
rule.  This adapter is the upgrade path.

Design notes
------------

* **Serialization.**  Records are serialized as a pinned
  ``Record A`` / ``Record B`` structured-table block, one line per
  field, NaN/None cells rendered as ``<missing>`` so the LLM does not
  misread a blank as equality.
* **Prompt template.**  System message pins the task + response format
  (bare ``Yes``/``No``); human message presents the two records and
  asks the question.  Follows the MatchGPT v4 paper's finding that
  bare yes/no is the most reliable response format for cross-provider
  parsing.
* **Demonstrations (optional).**  When ``demonstrations_path`` is
  provided, the adapter loads an EM-gold CSV (``id1``, ``id2``,
  ``label``), encodes each demonstration pair + every candidate pair
  with a pinned sentence-transformer model, and prepends the ``k_shot``
  cosine-nearest demonstrations as in-context examples.
* **LLM hygiene.**  ``temperature=0``, pinned
  ``_PROMPT_VERSION``, file-backed response cache under ``cache_dir``
  keyed by ``sha256(prompt_version | model_id | prompt_text)``.
  Committed cache files are the source of truth on rerun — same
  pattern as :class:`~usecases_synthetic.lib.magneto_sm_matcher.MagnetoSchemaMatcher`.
* **Testability.**  ``llm_callable`` and ``embedder`` are both
  injectable ``__init__`` arguments so unit tests can stub the LLM +
  sentence-transformer dependencies without network or GPU.

If this adapter ever needs to graduate to PyDI core, the promotion
path is to move it to ``PyDI/entitymatching/matchgpt_based.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from PyDI.entitymatching.base import BaseMatcher, CorrespondenceSet

logger = logging.getLogger(__name__)


_PROMPT_VERSION = "matchgpt-v1"
_DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MISSING_MARKER = "<missing>"

_SYSTEM_PROMPT = (
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
NDArrayF = "np.ndarray[Any, np.dtype[np.floating[Any]]]"
Embedder = Callable[[Sequence[str]], "np.ndarray[Any, np.dtype[np.floating[Any]]]"]


class MatchGPTMatcher(BaseMatcher):
    """LLM entity matcher with MatchGPT-style prompts.

    Parameters
    ----------
    fields : list of str
        Canonical field names to include in the serialized records.
        Fields absent from a record (or ``NaN``) are rendered as
        ``<missing>``.
    chat_model_name : str, optional
        Model id used both for cache-keying and (when no
        ``llm_callable`` is injected) for building the default
        ``ChatOpenAI`` client.  Default ``"openai/gpt-4o-mini"``.
        LiteLLM-style ``provider/model`` or bare ``model`` accepted.
    temperature : float, optional
        Sampling temperature passed to the default chat model.
        Default 0.0 (deterministic).
    max_tokens : int, optional
        Maximum response tokens.  Default 8 — enough for ``Yes``/``No``
        plus a trailing newline.
    demonstrations_path : str or Path, optional
        Path to an EM-gold CSV with ``id1``, ``id2``, ``label``
        columns.  When set, the ``k_shot`` cosine-nearest pairs are
        prepended as in-context examples.  ``None`` = zero-shot.
    k_shot : int, optional
        Number of nearest-neighbour demonstrations per candidate.
        Ignored when ``demonstrations_path`` is ``None``.  Default 3.
    embedding_model : str, optional
        Sentence-transformer model id for demonstration retrieval.
        Default ``"sentence-transformers/all-MiniLM-L6-v2"``.  Ignored
        when ``embedder`` is injected or when no demonstrations are
        loaded.
    cache_dir : str or Path, optional
        Directory for the response cache.  Defaults to
        ``usecases_synthetic/cache/matchgpt_prompts/``.
    seed : int, optional
        Random seed for deterministic tie-breaking in NN retrieval.
        Default 42.
    llm_callable : callable, optional
        Test-time hook.  Callable ``(prompt_text: str) -> str`` that
        returns the LLM response text.  When ``None`` (production
        default), a ``langchain_openai.ChatOpenAI`` client is built
        lazily at match time.
    embedder : callable, optional
        Test-time hook.  Callable
        ``(texts: Sequence[str]) -> np.ndarray`` returning a
        ``(len(texts), d)`` dense matrix of unit-norm embeddings.
        When ``None`` (production default), ``sentence_transformers``
        is loaded lazily.

    Notes
    -----
    The committee runner's ``_run_matcher`` calls this matcher with
    ``id_column``, ``threshold``, and optionally ``comparators`` /
    ``weights`` kwargs.  ``comparators`` and ``weights`` are accepted
    and ignored (learned/LLM matchers don't use comparator vectors).
    """

    def __init__(
        self,
        fields: list[str],
        chat_model_name: str = _DEFAULT_CHAT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        demonstrations_path: Optional[Union[str, Path]] = None,
        k_shot: int = 3,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
        llm_callable: Optional[LLMCallable] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        if not fields:
            raise ValueError("MatchGPTMatcher requires at least one field")
        self.fields = list(fields)
        self.chat_model_name = chat_model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.demonstrations_path = (
            Path(demonstrations_path) if demonstrations_path is not None else None
        )
        self.k_shot = int(k_shot)
        if self.k_shot < 0:
            raise ValueError("k_shot must be non-negative")
        self.embedding_model = embedding_model
        if cache_dir is None:
            cache_dir = (
                Path(__file__).resolve().parents[1] / "cache" / "matchgpt_prompts"
            )
        self.cache_dir = Path(cache_dir)
        self.seed = int(seed)

        self._llm_callable: Optional[LLMCallable] = llm_callable
        self._embedder: Optional[Embedder] = embedder

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

    def _pair_text(self, left: pd.Series, right: pd.Series) -> str:
        return (
            f"{self._serialize_record(left, 'Record A')}\n"
            f"{self._serialize_record(right, 'Record B')}"
        )

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        left: pd.Series,
        right: pd.Series,
        demonstrations: list[tuple[pd.Series, pd.Series, bool]],
    ) -> str:
        parts = [f"System:\n{_SYSTEM_PROMPT}"]
        for i, (demo_left, demo_right, is_match) in enumerate(demonstrations, start=1):
            answer = "Yes" if is_match else "No"
            parts.append(
                f"Example {i}:\n"
                f"{self._pair_text(demo_left, demo_right)}\n"
                f"Do the two records refer to the same entity? Answer: {answer}"
            )
        parts.append(
            "Query:\n"
            f"{self._pair_text(left, right)}\n"
            "Do the two records refer to the same entity? Answer:"
        )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Demonstration retrieval
    # ------------------------------------------------------------------

    def _load_demonstrations(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
    ) -> list[dict[str, Any]]:
        if self.demonstrations_path is None or self.k_shot == 0:
            return []
        if not self.demonstrations_path.exists():
            raise FileNotFoundError(
                f"MatchGPT demonstrations_path not found: {self.demonstrations_path}"
            )
        # EM gold ships headerless (``id1,id2,label`` columns implied)
        # across companies/games/music/products. Some tooling rewrites
        # them with explicit headers; sniff the first line to pick the
        # right reader. The headerless path uses ``read_em_gold_csv``
        # (URL-comma robust); the headerful path stays on
        # ``pd.read_csv`` so schema-validation errors surface clearly.
        from .loaders import read_em_gold_csv

        with open(self.demonstrations_path, encoding="utf-8") as f:
            first_line = f.readline().strip()
        has_header = first_line.lower().startswith("id1,id2")
        if has_header:
            df = pd.read_csv(self.demonstrations_path)
        else:
            df = read_em_gold_csv(self.demonstrations_path)
        required = {"id1", "id2", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"MatchGPT demonstrations CSV missing columns: "
                f"{sorted(missing)} — expected {sorted(required)}"
            )

        left_by_id = df_left.set_index(id_column, drop=False)
        right_by_id = df_right.set_index(id_column, drop=False)

        demos: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            id1, id2 = row["id1"], row["id2"]
            if id1 not in left_by_id.index or id2 not in right_by_id.index:
                continue
            left_rec = left_by_id.loc[id1]
            right_rec = right_by_id.loc[id2]
            if isinstance(left_rec, pd.DataFrame):
                left_rec = left_rec.iloc[0]
            if isinstance(right_rec, pd.DataFrame):
                right_rec = right_rec.iloc[0]
            label = row["label"]
            is_match = self._coerce_label(label)
            demos.append(
                {
                    "id1": id1,
                    "id2": id2,
                    "left": left_rec,
                    "right": right_rec,
                    "is_match": is_match,
                    "text": self._pair_text(left_rec, right_rec),
                }
            )
        return demos

    @staticmethod
    def _coerce_label(label: Any) -> bool:
        if isinstance(label, bool):
            return label
        if isinstance(label, (int, float)) and not isinstance(label, bool):
            return bool(int(label))
        text = str(label).strip().lower()
        if text in {"true", "1", "yes", "match"}:
            return True
        if text in {"false", "0", "no", "nonmatch", "non-match"}:
            return False
        raise ValueError(f"Cannot coerce label {label!r} to bool")

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is not None:
            return self._embedder

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.embedding_model)

        def _embed(
            texts: Sequence[str],
        ) -> "np.ndarray[Any, np.dtype[np.floating[Any]]]":
            vecs = model.encode(
                list(texts),
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.asarray(vecs, dtype=np.float32)

        self._embedder = _embed
        return _embed

    def _select_demonstrations(
        self,
        candidate_text: str,
        candidate_vec: "np.ndarray[Any, np.dtype[np.floating[Any]]]",
        demos: list[dict[str, Any]],
        demo_matrix: "np.ndarray[Any, np.dtype[np.floating[Any]]]",
    ) -> list[tuple[pd.Series, pd.Series, bool]]:
        if not demos or self.k_shot == 0:
            return []
        sims = demo_matrix @ candidate_vec
        k = min(self.k_shot, len(demos))
        idx = np.argsort(-sims, kind="stable")[:k]
        rng = random.Random(hashlib.sha256(candidate_text.encode("utf-8")).hexdigest())
        idx_list = list(idx)
        rng.shuffle(idx_list)
        return [
            (demos[i]["left"], demos[i]["right"], bool(demos[i]["is_match"]))
            for i in idx_list
        ]

    # ------------------------------------------------------------------
    # LLM invocation + cache
    # ------------------------------------------------------------------

    def _cache_key(self, prompt_text: str) -> str:
        payload = "|".join([_PROMPT_VERSION, self.chat_model_name, prompt_text])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, prompt_text: str) -> Optional[str]:
        path = self._cache_path(self._cache_key(prompt_text))
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return str(payload["response"])

    def _cache_put(self, prompt_text: str, response: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(self._cache_key(prompt_text))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prompt_version": _PROMPT_VERSION,
                    "model_id": self.chat_model_name,
                    "prompt": prompt_text,
                    "response": response,
                },
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

    def _ensure_llm_callable(self) -> LLMCallable:
        if self._llm_callable is not None:
            return self._llm_callable

        from .llm_client import build_chat_openai

        chat = build_chat_openai(
            model=self.chat_model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        def _call(prompt_text: str) -> str:
            response = chat.invoke(prompt_text)
            content = getattr(response, "content", response)
            return str(content)

        self._llm_callable = _call
        return _call

    @staticmethod
    def _parse_response(response_text: str) -> float:
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
        # Committee runner forwards comparators/weights — learned/LLM
        # matchers don't use them.
        kwargs.pop("comparators", None)
        kwargs.pop("weights", None)

        if isinstance(candidates, pd.DataFrame):
            candidate_batches: list[pd.DataFrame] = [candidates]
        else:
            candidate_batches = list(candidates)

        left_by_id = df_left.set_index(id_column, drop=False)
        right_by_id = df_right.set_index(id_column, drop=False)

        demos = self._load_demonstrations(df_left, df_right, id_column)
        demo_matrix: Optional["np.ndarray[Any, np.dtype[np.floating[Any]]]"] = None
        if demos:
            embedder = self._ensure_embedder()
            demo_matrix = embedder([d["text"] for d in demos])

        llm = self._ensure_llm_callable()

        results: list[dict[str, Any]] = []
        for batch in candidate_batches:
            if batch.empty:
                continue
            if "id1" not in batch.columns or "id2" not in batch.columns:
                raise ValueError(
                    "Candidate DataFrame must have 'id1' and 'id2' columns"
                )
            for _, row in batch.iterrows():
                id1, id2 = row["id1"], row["id2"]
                if id1 not in left_by_id.index or id2 not in right_by_id.index:
                    continue
                left_rec = left_by_id.loc[id1]
                right_rec = right_by_id.loc[id2]
                if isinstance(left_rec, pd.DataFrame):
                    left_rec = left_rec.iloc[0]
                if isinstance(right_rec, pd.DataFrame):
                    right_rec = right_rec.iloc[0]

                candidate_text = self._pair_text(left_rec, right_rec)
                if demo_matrix is not None and demos:
                    embedder = self._ensure_embedder()
                    candidate_vec = embedder([candidate_text])[0]
                    selected_demos = self._select_demonstrations(
                        candidate_text, candidate_vec, demos, demo_matrix
                    )
                else:
                    selected_demos = []

                prompt_text = self._build_prompt(left_rec, right_rec, selected_demos)

                cached = self._cache_get(prompt_text)
                if cached is not None:
                    response_text = cached
                else:
                    response_text = llm(prompt_text)
                    self._cache_put(prompt_text, response_text)

                score = self._parse_response(response_text)
                if score >= threshold:
                    results.append(
                        {
                            "id1": id1,
                            "id2": id2,
                            "score": float(score),
                            "notes": "matchgpt",
                        }
                    )

        if not results:
            return pd.DataFrame(columns=["id1", "id2", "score", "notes"])
        return pd.DataFrame(results)


__all__ = ["MatchGPTMatcher"]

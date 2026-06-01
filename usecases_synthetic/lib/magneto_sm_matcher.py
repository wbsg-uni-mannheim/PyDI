"""Magneto schema matcher (synthetic-local infrastructure).

Adapts the vendored Magneto framework (see
``usecases_synthetic/third_party/magneto_matcher/``) to PyDI's
:class:`~PyDI.schemamatching.base.BaseSchemaMatcher` contract.

Added by C1.5 of
`plans/plan_committee_finalization.md <../../plans/plan_committee_finalization.md>`_.
Used as the opt-in ``magneto_slm_llm`` member of the SM committee
(see ``config/committees/sm_committee.yaml``). ``enabled_by_default`` is
set to ``False`` so the default ``--with-llm=false`` pipeline runs do
not incur Magneto's per-column LLM calls; the flag flips it on for the
final validation runs alongside ``llm_openai``.

If the adapter ever needs to graduate from synthetic-local to
PyDI-core, the promotion path is: move the file to
``PyDI/schemamatching/magneto_based.py`` and re-export it from
``PyDI.schemamatching``.  The vendored subtree stays where it is.

Design notes
------------

* Two-phase Magneto pipeline: (1) SLM retrieval via
  :class:`sentence_transformers.SentenceTransformer` over column
  sentences; (2) optional LLM rerank via ``litellm``.
* LLM hygiene (``cross_cutting.md §LLM hygiene``):

  - Pinned prompt version (``_PROMPT_VERSION``).
  - ``temperature=0``.
  - File-backed prompt cache under ``cache_dir`` keyed by
    ``sha256(prompt_version | model_id | prompt_text)``.  Committed
    outputs are the sole source of truth on rerun in strict mode.

* Runner compatibility: the SM committee runner
  (``lib/committee_sm.py::_instantiate_matcher``) injects a
  ``chat_model`` kwarg built from ``params.model_name`` for every
  ``signal_type == "llm"`` entry.  This adapter accepts that kwarg and
  derives its litellm model id from it so the YAML can set
  ``model_name: gpt-5.4-mini`` the same way ``llm_openai`` does.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from PyDI.schemamatching.base import (
    BaseSchemaMatcher,
    SchemaMapping,
    get_schema_columns,
)

logger = logging.getLogger(__name__)


_PROMPT_VERSION = "v1"
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"


class MagnetoSchemaMatcher(BaseSchemaMatcher):
    """Schema matcher backed by the vendored Magneto framework.

    Two-phase pipeline:

    1. **SLM retrieval.**  Each source column's
       ``(name + sample values)`` sentence is encoded with a pinned
       sentence-transformer model, cosine-scored against the target
       columns, and the top-``topk`` candidates are kept per source
       column.
    2. **LLM rerank** *(optional, opt-in)*.  Magneto's ``LLMReranker``
       prompts an LLM with the candidate column + its sample values
       and the shortlist; the LLM returns calibrated scores in
       ``[0, 1]``.  Cached on disk by prompt hash so subsequent runs
       do not re-call the LLM.

    Parameters
    ----------
    embedding_model : str, optional
        Sentence-transformer model id or Hugging Face identifier.
        Default ``"sentence-transformers/all-MiniLM-L6-v2"``, chosen
        to match the ``embedding_sbert`` committee member so the SLM
        retrieval step stays cheap.
    encoding_mode : str, optional
        Magneto ``ColumnEncoder`` mode (e.g.
        ``"header_values_verbose"``).  Default
        ``"header_values_verbose"``.
    sampling_mode : str, optional
        Magneto value-sampling strategy.  Default
        ``"priority_sampling"``.
    sampling_size : int, optional
        Number of sample values encoded per column.  Default 10.
    topk : int, optional
        Number of target-column candidates kept per source column
        after SLM retrieval (inputs to the LLM reranker).  Default
        10.
    embedding_threshold : float, optional
        Minimum cosine similarity to keep a candidate during SLM
        retrieval.  Default 0.1.
    use_llm_rerank : bool, optional
        Whether to run the LLM reranker after SLM retrieval.  Default
        ``True``.  Set ``False`` to evaluate the SLM-only variant.
    llm_model : str, optional
        LiteLLM model identifier (e.g.
        ``"openai/gpt-4o-mini"``).  Default
        ``"openai/gpt-4o-mini"``.  If a ``chat_model`` is injected by
        the committee runner, its ``model_name`` takes precedence.
    llm_temperature : float, optional
        LLM sampling temperature.  Default 0.0 (deterministic).
    cache_dir : str or Path, optional
        Directory for the prompt-level cache.  Defaults to
        ``usecases_synthetic/cache/magneto_prompts/``.  Committed
        cache files are the single source of truth on rerun.
    chat_model : Any, optional
        LangChain ``ChatOpenAI`` instance injected by the committee
        runner.  When present, its ``model_name`` attribute overrides
        *llm_model* (with an ``openai/`` prefix added if needed).

    Notes
    -----
    The embedding_threshold default of 0.1 follows Magneto's upstream
    recommendation — the LLM rerank does the heavy filtering, so the
    retrieval threshold should stay permissive.
    """

    def __init__(
        self,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        encoding_mode: str = "header_values_verbose",
        sampling_mode: str = "priority_sampling",
        sampling_size: int = 10,
        topk: int = 10,
        embedding_threshold: float = 0.1,
        use_llm_rerank: bool = True,
        llm_model: str = _DEFAULT_LLM_MODEL,
        llm_temperature: float = 0.0,
        cache_dir: Optional[str | Path] = None,
        chat_model: Any = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.encoding_mode = encoding_mode
        self.sampling_mode = sampling_mode
        self.sampling_size = int(sampling_size)
        self.topk = int(topk)
        self.embedding_threshold = float(embedding_threshold)
        self.use_llm_rerank = bool(use_llm_rerank)
        self.llm_temperature = float(llm_temperature)

        if chat_model is not None:
            model_name = getattr(chat_model, "model_name", None) or getattr(
                chat_model, "model", None
            )
            if model_name is None:
                raise ValueError(
                    "chat_model has neither 'model_name' nor 'model' "
                    "attribute; cannot derive litellm model id."
                )
            llm_model = model_name if "/" in model_name else f"openai/{model_name}"
        self.llm_model = llm_model

        if cache_dir is None:
            cache_dir = (
                Path(__file__).resolve().parents[1] / "cache" / "magneto_prompts"
            )
        self.cache_dir = Path(cache_dir)

        self._magneto: Any | None = None

    # ------------------------------------------------------------------
    # Lazy Magneto loader
    # ------------------------------------------------------------------

    def _ensure_magneto(self) -> Any:
        if self._magneto is not None:
            return self._magneto

        from usecases_synthetic.third_party.magneto_matcher.magneto import (
            Magneto,
        )

        logger.info(
            "Initialising Magneto (embedding_model=%s, llm_model=%s, "
            "use_llm_rerank=%s, topk=%d)",
            self.embedding_model,
            self.llm_model,
            self.use_llm_rerank,
            self.topk,
        )
        magneto = Magneto(
            embedding_model=self.embedding_model,
            encoding_mode=self.encoding_mode,
            sampling_mode=self.sampling_mode,
            sampling_size=self.sampling_size,
            topk=self.topk,
            embedding_threshold=self.embedding_threshold,
            include_strsim_matches=False,
            include_embedding_matches=True,
            include_equal_matches=True,
            use_bp_reranker=True,
            use_gpt_reranker=self.use_llm_rerank,
            llm_model=self.llm_model,
            llm_model_kwargs={"temperature": self.llm_temperature},
        )

        if self.use_llm_rerank:
            magneto.call_llm_reranker = self._wrap_llm_reranker(  # type: ignore[method-assign]
                magneto.call_llm_reranker
            )

        self._magneto = magneto
        return magneto

    # ------------------------------------------------------------------
    # LLM prompt cache
    # ------------------------------------------------------------------

    def _cache_key(self, prompt_text: str) -> str:
        payload = "|".join([_PROMPT_VERSION, self.llm_model, prompt_text])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, prompt_text: str) -> str | None:
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
                    "model_id": self.llm_model,
                    "prompt": prompt_text,
                    "response": response,
                },
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

    def _wrap_llm_reranker(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Patch Magneto's LLM reranker to route through the prompt cache."""

        from usecases_synthetic.third_party.magneto_matcher.magneto.llm_reranker import (
            LLMReranker,
        )

        adapter = self

        def cached_rematch(
            source_table: Any,
            target_table: Any,
            source_values: Any,
            target_values: Any,
            matched_columns: Any,
            score_based: bool = True,
        ) -> Any:
            # Build a cache-backed stand-in for LLMReranker that reuses
            # Magneto's own prompt + parse logic but hits the cache
            # before calling litellm.
            from litellm import completion

            reranker = LLMReranker(
                llm_model=adapter.llm_model,
                temperature=adapter.llm_temperature,
            )
            reranker.llm_attempts = 5

            def _llm_call_cached(cand: str, targets: str) -> str:
                prompt = reranker._get_prompt(cand, targets)
                cached = adapter._cache_get(prompt)
                if cached is not None:
                    return cached
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI trained to perform schema "
                            "matching by providing column similarity "
                            "scores."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                response = completion(
                    model=adapter.llm_model,
                    messages=messages,
                    temperature=adapter.llm_temperature,
                )
                response_text = str(response.choices[0].message.content)
                adapter._cache_put(prompt, response_text)
                return response_text

            reranker._get_matches = _llm_call_cached  # type: ignore[method-assign]
            return reranker.rematch(
                source_table,
                target_table,
                source_values,
                target_values,
                matched_columns,
                score_based=score_based,
            )

        def call_llm_reranker_wrapped(
            source_table: Any,
            target_table: Any,
            matches: Any,
        ) -> dict[str, Any]:
            # Re-implement Magneto.call_llm_reranker so we can inject
            # the cached reranker without subclassing Magneto.
            source_table_df = source_table.get_df()
            target_table_df = target_table.get_df()

            from usecases_synthetic.third_party.magneto_matcher.magneto.utils.utils import (
                get_samples,
            )

            source_values = {
                col: get_samples(source_table_df[col], 10)
                for col in source_table_df.columns
            }
            target_values = {
                col: get_samples(target_table_df[col], 10)
                for col in target_table_df.columns
            }

            matched_columns: dict[str, list[tuple[str, float]]] = {}
            for entry, score in matches.items():
                source_col = entry[0][1]
                target_col = entry[1][1]
                matched_columns.setdefault(source_col, []).append((target_col, score))

            return cached_rematch(
                source_table_df,
                target_table_df,
                source_values,
                target_values,
                matched_columns,
            )

        # Keep ``original`` reachable for debugging, but unused.
        _ = original
        return call_llm_reranker_wrapped

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def match(
        self,
        source_dataset: pd.DataFrame,
        target_dataset: pd.DataFrame,
        preprocess: Optional[Callable[[str], str]] = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> SchemaMapping:
        """Match source columns to target columns via Magneto.

        Parameters
        ----------
        source_dataset : pandas.DataFrame
            Source frame with a meaningful ``attrs['dataset_name']``.
        target_dataset : pandas.DataFrame
            Target frame with a meaningful ``attrs['dataset_name']``.
            May be empty-rowed — Magneto's LLM reranker handles empty
            columns gracefully via its sample-value sampler.
        preprocess : callable, optional
            Ignored — Magneto applies its own cleaning pipeline to
            columns and values.  Accepted for interface parity with
            other ``BaseSchemaMatcher`` implementations.
        threshold : float, optional
            Minimum final score to keep a correspondence.  Default
            0.5.
        **kwargs
            Reserved — accepted for forward compatibility with the
            committee runner.  Passing ``correspondences`` is a no-op
            (Magneto is not duplicate-based).

        Returns
        -------
        SchemaMapping
            DataFrame with columns ``source_dataset``,
            ``source_column``, ``target_dataset``, ``target_column``,
            ``score``, ``notes``.
        """
        _ = preprocess  # intentionally unused
        _ = kwargs

        source_name = source_dataset.attrs.get("dataset_name", "source")
        target_name = target_dataset.attrs.get("dataset_name", "target")

        source_columns = get_schema_columns(source_dataset)
        target_columns = get_schema_columns(target_dataset)

        if not source_columns or not target_columns:
            logger.info(
                "MagnetoSchemaMatcher: empty column list "
                "(source=%d, target=%d) — returning empty mapping",
                len(source_columns),
                len(target_columns),
            )
            return pd.DataFrame(
                columns=[
                    "source_dataset",
                    "source_column",
                    "target_dataset",
                    "target_column",
                    "score",
                    "notes",
                ]
            )

        source_view = source_dataset.loc[:, source_columns].copy()
        target_view = target_dataset.loc[:, target_columns].copy()

        if target_view.empty:
            target_view = pd.DataFrame({col: [""] for col in target_columns})

        magneto = self._ensure_magneto()
        raw_matches = magneto.get_matches(source_view, target_view)

        results: list[dict[str, Any]] = []
        for key, score in raw_matches.items():
            try:
                (_src_table, src_col), (_tgt_table, tgt_col) = key
            except (TypeError, ValueError):
                logger.warning(
                    "MagnetoSchemaMatcher: skipping malformed key %r",
                    key,
                )
                continue
            score_f = float(score)
            if score_f < threshold:
                continue
            results.append(
                {
                    "source_dataset": source_name,
                    "source_column": str(src_col),
                    "target_dataset": target_name,
                    "target_column": str(tgt_col),
                    "score": score_f,
                    "notes": (
                        f"magneto={self.embedding_model},"
                        f"llm_rerank={self.use_llm_rerank},"
                        f"llm_model={self.llm_model}"
                    ),
                }
            )

        return pd.DataFrame(
            results,
            columns=[
                "source_dataset",
                "source_column",
                "target_dataset",
                "target_column",
                "score",
                "notes",
            ],
        )

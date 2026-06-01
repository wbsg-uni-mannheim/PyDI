"""COMA schema matcher (synthetic-local infrastructure).

Adapts Valentine's pure-Python :class:`~valentine.algorithms.ComaPy`
re-implementation of COMA 3.0 CE to PyDI's
:class:`~PyDI.schemamatching.base.BaseSchemaMatcher` contract.

Added by C1.6 of
`plans/plan_committee_finalization.md <../../plans/plan_committee_finalization.md>`_.
Used as the ``coma_hybrid`` member of the SM committee (see
``config/committees/sm_committee.yaml``).  ``enabled_by_default`` is
``True`` because ``ComaPy`` is deterministic and CPU-tractable — the
historical JRE-dependency concern cited in the C1.3 rejection
(`knobs/committee_review/sm_shortlist.md §Exclusion rationale (COMA specifically)
<../../knobs/committee_review/sm_shortlist.md#exclusion-rationale-coma-specifically--user-requested-evaluation>`_)
does not apply to the Python implementation.

Deviation from the plan
-----------------------

:mod:`plans/plan_committee_finalization.md` C1.6 prescribes
``valentine.algorithms.Coma`` (Java CLI via subprocess).  Valentine
shipped a pure-Python reimplementation, ``ComaPy``, and marked the
Java variant as deprecated (removal in Valentine v1.0.0).  We use
``ComaPy`` because:

* no JRE dependency in ``pydi-dev/`` (the original plan's biggest
  integration cost);
* no JVM warm-start amortisation / process-caching logic needed;
* deterministic (same random state across calls — seeded by the
  underlying tokenizer / n-gram rules);
* the two implementations target the same algorithm (COMA 3.0 CE's
  ``COMA_OPT`` / ``COMA_OPT_INST`` strategies), so the committee
  member's *semantic* role — a hybrid-ensemble matcher that combines
  label, value, and structural signals — is preserved.

Promotion path
--------------

If this adapter generalises beyond the synthetic committee, the
promotion route is: move the file to ``PyDI/schemamatching/coma_based.py``
and re-export from ``PyDI.schemamatching``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Optional

import pandas as pd

from PyDI.schemamatching.base import (
    BaseSchemaMatcher,
    SchemaMapping,
    get_schema_columns,
)

logger = logging.getLogger(__name__)


_OUTPUT_COLUMNS: list[str] = [
    "source_dataset",
    "source_column",
    "target_dataset",
    "target_column",
    "score",
    "notes",
]


class ComaSchemaMatcher(BaseSchemaMatcher):
    """Schema matcher backed by Valentine's ``ComaPy``.

    COMA 3.0 CE is a hybrid-ensemble schema matcher: it runs a bank of
    individual matchers (label-based, instance-based, structural) and
    aggregates their similarity cubes into a single confidence score
    per column pair.  ``ComaPy`` is a faithful pure-Python
    reimplementation of that algorithm from the Valentine project.

    This adapter fills the ``hybrid`` axis of the synthetic SM
    committee — the other members (``duplicate_majority``,
    ``embedding_sbert``, ``llm_openai``, ``magneto_slm_llm``) each
    produce a single-signal score, whereas ``coma_hybrid`` is itself
    an ensemble of heterogeneous matchers.  Including it gives the
    committee a second-order aggregation signal that is uncorrelated
    with the embedding + LLM axes.

    Parameters
    ----------
    max_n : int, optional
        Max number of matches per source column to return from the
        underlying ``ComaPy`` run.  ``0`` means "no cap".  Default ``1``
        — aligned with the committee's expectation that each source
        column maps to at most one target column per SM member.
    use_instances : bool, optional
        Pass through to ``ComaPy.use_instances``.  When ``True``, COMA
        uses the column value distribution as an additional matcher.
        Default ``True`` — the synthetic pipeline always has sample
        values on hand, and instance-level signal is a meaningful
        differentiator from the embedding member's value-sampling
        strategy.
    use_schema : bool, optional
        Pass through to ``ComaPy.use_schema``.  Controls whether the
        label-based matchers run.  Default ``True``.
    delta : float, optional
        Per-matcher threshold inside the ``ComaPy`` aggregator.  Default
        ``0.15`` (Valentine default).
    coma_threshold : float, optional
        Per-pair floor applied *inside* ``ComaPy`` before it emits a
        match.  Default ``0.0`` — we rely on the caller's ``threshold``
        kwarg in :meth:`match` to filter the final output instead, so
        the caller sees all candidate scores and can tune the filter.

    Notes
    -----

    * ``ComaPy`` emits a :class:`DeprecationWarning` on construction
      of the upstream ``Coma`` class only — the Python variant is
      the recommended path.  No warnings are raised for ``ComaPy``.
    * Deterministic: the n-gram and TF-IDF weight computations are
      pure functions of the input column names + values.  No random
      state on the Python side; no hidden JVM state.
    * Runtime budget: ~0.5-2s per source-target call on companies-small
      (3 sources × 1 target × 6-12 columns each).  Scales roughly
      quadratically in the number of columns.
    """

    def __init__(
        self,
        max_n: int = 1,
        use_instances: bool = True,
        use_schema: bool = True,
        delta: float = 0.15,
        coma_threshold: float = 0.0,
    ) -> None:
        self.max_n = int(max_n)
        self.use_instances = bool(use_instances)
        self.use_schema = bool(use_schema)
        self.delta = float(delta)
        self.coma_threshold = float(coma_threshold)

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def match(
        self,
        source_dataset: pd.DataFrame,
        target_dataset: pd.DataFrame,
        preprocess: Optional[Callable[[str], str]] = None,
        threshold: float = 0.0,
        **kwargs: Any,
    ) -> SchemaMapping:
        """Run ComaPy on the source × target column pairs.

        Parameters
        ----------
        source_dataset : pandas.DataFrame
            Source frame.  Must set ``attrs['dataset_name']``.
        target_dataset : pandas.DataFrame
            Target frame.  Must set ``attrs['dataset_name']``.  May be
            zero-rows (the committee runner builds the target from the
            variant's target schema with no rows by default) — in that
            case ``use_instances=True`` still works because ComaPy
            degrades to label-only matchers when a side has no values.
        preprocess : callable, optional
            Kept for API compatibility with other schema matchers.
            Not applied — ``ComaPy`` does its own normalisation.  A
            warning is logged if supplied.
        threshold : float, optional
            Minimum aggregated COMA score required to include a
            correspondence in the output.  Applied *after* ``ComaPy``
            emits its candidate set.  Default ``0.0`` (keep everything
            ComaPy returns — the caller typically filters further).

        Returns
        -------
        SchemaMapping
            Frame with columns ``source_dataset``, ``source_column``,
            ``target_dataset``, ``target_column``, ``score`` and
            ``notes``.  Empty frame with the correct columns is returned
            when either side has no schema columns or when ``ComaPy``
            returns no matches above the threshold.
        """
        if preprocess is not None:
            logger.warning(
                "ComaSchemaMatcher.match ignores the preprocess kwarg — "
                "ComaPy performs its own tokenisation and normalisation."
            )

        source_name = source_dataset.attrs.get("dataset_name", "source")
        target_name = target_dataset.attrs.get("dataset_name", "target")

        source_columns = get_schema_columns(source_dataset)
        target_columns = get_schema_columns(target_dataset)

        if not source_columns or not target_columns:
            logger.info(
                "ComaSchemaMatcher: empty column list "
                "(source=%d, target=%d) — returning empty mapping",
                len(source_columns),
                len(target_columns),
            )
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)

        # Project onto the schema columns so PyDI's internal id_column is
        # not seen by ComaPy's matchers.
        source_slice = source_dataset[source_columns]
        target_slice = target_dataset[target_columns]

        # Lazy-import to keep PyDI core free of an unconditional
        # valentine dependency.
        try:
            from valentine import valentine_match
            from valentine.algorithms import ComaPy
        except ImportError as exc:
            raise ImportError(
                "ComaSchemaMatcher requires the 'valentine' package. "
                "Install via `uv pip install -e '.[magneto]' --python "
                "pydi-dev/bin/python` (the magneto extras group pulls "
                "valentine transitively)."
            ) from exc

        matcher = ComaPy(
            max_n=self.max_n,
            use_instances=self.use_instances,
            use_schema=self.use_schema,
            delta=self.delta,
            threshold=self.coma_threshold,
        )

        # valentine_match emits a harmless DeprecationWarning from
        # valentine.metrics — not our concern here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            matches = valentine_match(
                source_slice,
                target_slice,
                matcher,
                df1_name=source_name,
                df2_name=target_name,
            )

        rows: list[dict[str, Any]] = []
        for pair, raw_score in matches.items():
            score = float(raw_score)
            if score < threshold:
                continue
            (s_table, s_col), (t_table, t_col) = pair
            rows.append(
                {
                    "source_dataset": str(s_table),
                    "source_column": str(s_col),
                    "target_dataset": str(t_table),
                    "target_column": str(t_col),
                    "score": score,
                    "notes": (
                        f"coma_py:max_n={self.max_n},"
                        f"use_instances={self.use_instances},"
                        f"use_schema={self.use_schema}"
                    ),
                }
            )

        if not rows:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)

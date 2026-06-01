"""Magellan-style classical ML matcher for the EM matching committee.

Wraps a scikit-learn classifier trained on comparator-vector features
(the Magellan recipe: *features from similarity comparators + classical
supervised learning*) in PyDI's :class:`BaseMatcher` interface.  Training
is lazy — the classifier is built on the first :meth:`match` call from an
EM training gold CSV and cached for subsequent calls on the same
instance.

Design points
-------------

- **Classical axis.** The matching committee's four slots are
  ``ditto_plm`` (supervised PLM), ``llm_matcher`` (MatchGPT-style
  prompts, zero-shot), ``magellan`` (this adapter), and ``comem`` (LLM
  match/compare/select).  Magellan fills the classical-ML slot: CPU-only,
  deterministic under a pinned seed, no LLM budget.
- **Auto-feature-gen (Magellan philosophy).**  When ``comparators`` is
  omitted (or empty), comparators are auto-generated at training time
  via :func:`magellan_auto_features.auto_generate_comparators`: one
  StringComparator per (column × similarity-function × tokenization)
  slot in the bank, plus NumericComparator/DateComparator slots per
  numeric/date column. The RandomForest classifier's
  ``feature_importances_`` then performs implicit feature selection.
  This mirrors py_entitymatching's
  ``get_features_for_matching`` — synth-local because
  ``py_entitymatching`` does not install on Python 3.12 (the
  ``py-stringsimjoin`` transitive dep fails at build time).
- **Per-pair training data.**  The committee runner injects
  ``training_gold_path`` at instantiation time for each source pair
  (see :data:`committee_em._PER_PAIR_TRAIN_INJECTION`).  Pairs with no
  ``_train.csv`` on disk skip the matcher entirely.
- **Lazy training.**  Training happens on the first ``match`` call and
  the fitted classifier is held on the instance.
- **Synthetic-local.**  The adapter lives under ``usecases_synthetic/lib/``
  rather than ``PyDI/entitymatching/`` because its committee-YAML wiring
  (and the pair-specific gold-path convention) is synthetic-pipeline
  infrastructure, not a general PyDI feature.

Example
-------
>>> from usecases_synthetic.lib.magellan_em_matcher import MagellanMatcher
>>> # Auto-feature mode (recommended; YAML omits `comparators`).
>>> matcher = MagellanMatcher(
...     training_gold_path="tests/fixtures/forbes_2_dbpedia_train.csv",
...     numeric_attributes=["founded", "assets", "revenue"],
... )
>>> matcher.match(df_left, df_right, candidates, id_column="id", threshold=0.5)  # doctest: +SKIP
>>>
>>> # Legacy hand-authored mode (kept for tests / explicit recipes).
>>> matcher = MagellanMatcher(
...     comparators=[{"class": "StringComparator", "module": "PyDI.entitymatching.comparators",
...                   "params": {"column": "name", "similarity_function": "jaccard"}}],
...     training_gold_path="tests/fixtures/forbes_2_dbpedia_train.csv",
... )
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from PyDI.entitymatching.base import BaseComparator, BaseMatcher, CorrespondenceSet
from PyDI.entitymatching.feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)


_DEFAULT_CLASSIFIER_CLASS = "sklearn.ensemble.RandomForestClassifier"
"""Default scikit-learn classifier class used when the YAML omits it."""

_DEFAULT_CLASSIFIER_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": 1,
    "class_weight": "balanced",
}
"""Default kwargs for the default classifier.

``class_weight="balanced"`` is the Magellan-practical default: EM gold
files typically have ~50/50 positive/negative balance at the file level
but can skew per-pair after sampling, and the balanced reweighting keeps
the minority class from being swamped.  ``n_jobs=1`` keeps the adapter
deterministic across test runs (parallel trees in scikit-learn can
reorder floating-point accumulations).
"""


def _import_class(dotted_path: str) -> type:
    """Import a class from a ``module.Class`` dotted path.

    Parameters
    ----------
    dotted_path : str
        Import path, e.g. ``"sklearn.ensemble.RandomForestClassifier"``.

    Returns
    -------
    type
        The resolved class.
    """
    module_path, _, cls_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"classifier_class must be a dotted path, got {dotted_path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _build_comparators_from_specs(
    specs: Sequence[dict[str, Any]],
    preprocess_fn: Any | None = None,
) -> list[BaseComparator]:
    """Instantiate comparators from YAML-style spec dicts.

    Mirrors the pattern used by the EM committee runner so the same
    YAML fragment can feed either the rule-based aggregation or this
    adapter's feature pipeline.

    Parameters
    ----------
    specs : sequence of dict
        Each spec has keys ``class``, ``module``, ``params``.
    preprocess_fn : callable, optional
        Shared text-preprocessing function injected into each comparator
        under ``params["preprocess"]`` when the spec does not set its
        own.  Mirrors ``committee_em._build_comparators``.

    Returns
    -------
    list of BaseComparator
        Instantiated comparators, in input order.
    """
    out: list[BaseComparator] = []
    for spec in specs:
        if "class" not in spec or "module" not in spec:
            raise ValueError(
                "Comparator spec must have 'class' and 'module' keys; got "
                f"{sorted(spec.keys())}"
            )
        module = importlib.import_module(spec["module"])
        cls = getattr(module, spec["class"])
        params = dict(spec.get("params", {}))
        if preprocess_fn is not None and "preprocess" not in params:
            params["preprocess"] = preprocess_fn
        out.append(cls(**params))
    return out


def _coerce_labels(series: pd.Series, positive_label: str = "true") -> pd.Series:
    """Map EM-gold label values to 0/1 integers.

    EM gold CSVs label positives as string ``"true"`` / ``"false"`` or
    boolean ``True`` / ``False``.  This helper accepts both and returns a
    pandas Series of ``int64`` values (1 for positives, 0 otherwise).

    Parameters
    ----------
    series : pandas.Series
        Label column.
    positive_label : str, default "true"
        String value that marks a positive pair, case-insensitive.  Used
        only when ``series`` has object dtype.

    Returns
    -------
    pandas.Series
        Integer 0/1 series with the same index as the input.
    """
    if series.dtype == object:
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .eq(positive_label.lower())
            .astype(np.int64)
        )
    return series.astype(bool).astype(np.int64)


class MagellanMatcher(BaseMatcher):
    """Classical supervised-ML matcher on comparator-vector features.

    Lazily trains an sklearn classifier on the EM training gold CSV
    referenced by ``training_gold_path`` and caches the fitted classifier
    on the instance.  Inference uses the same comparators as training to
    keep the feature space aligned.

    Parameters
    ----------
    training_gold_path : str or Path
        Path to the EM gold CSV with columns ``id1``, ``id2``, ``label``.
        The CSV's ids must resolve against ``df_left`` / ``df_right`` at
        match time.  Injected by the committee runner per source pair.
    comparators : sequence of dict, optional
        Hand-authored comparator specs in the same shape used by
        ``em_matching_committee.yaml``: each entry has ``class``,
        ``module``, and ``params``. Instantiated via the PyDI entity-
        matching comparators. **Defaults to ``None`` (auto-feature-gen
        mode)** — when omitted or empty, comparators are generated by
        :func:`magellan_auto_features.auto_generate_comparators` at
        training time.
    attributes : sequence of str, optional
        Explicit attribute list passed through to
        ``auto_generate_comparators``. Defaults to the intersection of
        ``df_left.columns`` and ``df_right.columns`` minus ``id_columns``.
        Only consulted in auto mode.
    id_columns : sequence of str, default ``("id",)``
        Column names to exclude from the auto-generated attribute set.
    numeric_attributes : sequence of str, default ``()``
        Columns to treat as numeric in auto mode (overrides dtype
        inference). Required for numeric columns that ship as object
        dtype after a CSV load.
    date_attributes : sequence of str, default ``()``
        Columns to treat as dates in auto mode (heuristic does not
        auto-detect dates).
    classifier_class : str, default ``"sklearn.ensemble.RandomForestClassifier"``
        Dotted path to the sklearn classifier class.
    classifier_params : dict, optional
        Kwargs passed to the classifier constructor.  Defaults to the
        Magellan-practical settings recorded in
        ``_DEFAULT_CLASSIFIER_PARAMS``.
    positive_label : str, default ``"true"``
        Value in the gold CSV's ``label`` column that denotes a positive
        pair (case-insensitive for string columns).
    preprocess : str, optional
        Name of a preprocess function registered on the committee runner
        (e.g. ``"normalize_text"``) to inject into each string
        comparator.  Currently only ``"normalize_text"`` is supported,
        mirroring the rule-based members.
    use_probabilities : bool, default ``True``
        Whether to use ``predict_proba(X)[:, 1]`` as the match score
        (``True``) or ``predict(X).astype(float)`` (``False``).  Almost
        every downstream caller wants probabilities so the primary
        default is ``True``.
    min_positive_support : int, default ``3``
        Minimum number of positive and negative rows required in the
        training gold before the classifier is fit.  Below this the
        adapter emits a clear error at train time rather than produce a
        degenerate one-class classifier.
    seed : int, default ``42``
        Seed applied to ``numpy`` before training to keep bagging
        / bootstrap choices deterministic in conjunction with the
        classifier's own ``random_state`` kwarg.
    """

    def __init__(
        self,
        training_gold_path: str | Path,
        *,
        comparators: Sequence[dict[str, Any]] | None = None,
        attributes: Sequence[str] | None = None,
        id_columns: Sequence[str] = ("id",),
        numeric_attributes: Sequence[str] = (),
        date_attributes: Sequence[str] = (),
        classifier_class: str = _DEFAULT_CLASSIFIER_CLASS,
        classifier_params: dict[str, Any] | None = None,
        positive_label: str = "true",
        preprocess: str | None = None,
        use_probabilities: bool = True,
        min_positive_support: int = 3,
        seed: int = 42,
    ) -> None:
        self._comparator_specs: list[dict[str, Any]] = (
            [dict(c) for c in comparators] if comparators else []
        )
        self._auto_features: bool = not self._comparator_specs
        self.attributes: list[str] | None = (
            list(attributes) if attributes is not None else None
        )
        self.id_columns: tuple[str, ...] = tuple(id_columns)
        self.numeric_attributes: tuple[str, ...] = tuple(numeric_attributes)
        self.date_attributes: tuple[str, ...] = tuple(date_attributes)
        self.training_gold_path = Path(training_gold_path)
        self.classifier_class = classifier_class
        self.classifier_params: dict[str, Any] = (
            dict(classifier_params)
            if classifier_params is not None
            else dict(_DEFAULT_CLASSIFIER_PARAMS)
        )
        self.positive_label = positive_label
        self.preprocess_name = preprocess
        self.use_probabilities = bool(use_probabilities)
        self.min_positive_support = int(min_positive_support)
        self.seed = int(seed)

        self._classifier: Any | None = None
        self._feature_extractor: FeatureExtractor | None = None
        self._feature_columns: list[str] | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _resolve_preprocess(self) -> Any | None:
        """Return the callable behind ``preprocess_name`` or ``None``.

        Currently only ``"normalize_text"`` is supported (matches
        :func:`committee_em._normalize_text`).  Unknown names raise so
        the caller catches typos in the YAML rather than silently losing
        the preprocessing step.
        """
        if self.preprocess_name is None:
            return None
        if self.preprocess_name == "normalize_text":
            from .committee_em import _normalize_text

            return _normalize_text
        raise ValueError(
            f"Unknown preprocess function: {self.preprocess_name!r}. "
            "Supported: 'normalize_text'."
        )

    def _build_feature_extractor(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
    ) -> FeatureExtractor:
        """Instantiate the comparator list and wrap it in a FeatureExtractor.

        In auto-feature mode (no comparators provided at init time), the
        comparator list is generated from the source frames via
        :func:`magellan_auto_features.auto_generate_comparators`. The
        RandomForest classifier then performs implicit feature selection
        via ``feature_importances_``.

        In legacy mode (comparators provided at init time), the spec list
        is instantiated as authored — back-compat with the original
        explicit-feature recipe.
        """
        preprocess_fn = self._resolve_preprocess()
        if self._auto_features:
            from .magellan_auto_features import auto_generate_comparators

            comparators = auto_generate_comparators(
                df_left,
                df_right,
                attributes=self.attributes,
                id_columns=self.id_columns,
                numeric_attributes=self.numeric_attributes,
                date_attributes=self.date_attributes,
                preprocess_fn=preprocess_fn,
            )
        else:
            comparators = _build_comparators_from_specs(
                self._comparator_specs, preprocess_fn
            )
        return FeatureExtractor(comparators)

    def _train(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
    ) -> None:
        """Train the classifier from the gold CSV and cache it on self.

        Parameters
        ----------
        df_left, df_right : pandas.DataFrame
            Source frames (with ``id_column``) the gold's ids resolve
            against.
        id_column : str
            Id-column name in both source frames.

        Raises
        ------
        FileNotFoundError
            If ``training_gold_path`` does not exist.
        ValueError
            If the gold is empty, lacks required columns, or does not
            meet ``min_positive_support`` for both classes.
        """
        if not self.training_gold_path.exists():
            raise FileNotFoundError(
                f"MagellanMatcher training gold not found: {self.training_gold_path}"
            )

        # EM gold CSVs ship header-less with columns ``id1, id2, label``
        # and URLs that may carry embedded commas. Test fixtures
        # sometimes use ``pandas.to_csv`` which emits a header — peek
        # at the first line to pick the right reader. The headerless
        # path uses :func:`loaders.read_em_gold_csv` (URL-comma robust);
        # the headerful path stays on ``pd.read_csv`` so schema-
        # validation errors (missing ``label`` column) surface clearly.
        from .loaders import read_em_gold_csv

        with open(self.training_gold_path, encoding="utf-8") as f:
            first_line = f.readline().strip()
        has_header = first_line.lower().startswith("id1,id2")
        if has_header:
            gold = pd.read_csv(self.training_gold_path)
        else:
            gold = read_em_gold_csv(self.training_gold_path)
        for col in ("id1", "id2", "label"):
            if col not in gold.columns:
                raise ValueError(
                    f"Training gold {self.training_gold_path} missing column "
                    f"{col!r}; have {list(gold.columns)}"
                )
        if gold.empty:
            raise ValueError(f"Training gold is empty: {self.training_gold_path}")

        labels = _coerce_labels(gold["label"], self.positive_label)
        n_pos = int(labels.sum())
        n_neg = int((labels == 0).sum())
        if n_pos < self.min_positive_support or n_neg < self.min_positive_support:
            raise ValueError(
                f"Training gold has insufficient class support: "
                f"{n_pos} positives, {n_neg} negatives "
                f"(need >= {self.min_positive_support} of each). "
                f"Path: {self.training_gold_path}"
            )

        pairs = gold[["id1", "id2"]].copy()
        extractor = self._build_feature_extractor(df_left, df_right)
        features = extractor.create_features(
            df_left, df_right, pairs, id_column=id_column, labels=labels
        )
        if features.empty:
            raise ValueError(
                "FeatureExtractor returned no rows on the training gold; "
                "check that the gold's ids resolve against the source frames."
            )
        # FeatureExtractor always returns id1/id2 + one column per comparator,
        # plus 'label' when labels is supplied.  Feature columns are everything
        # except id1/id2/label.
        exclude = {"id1", "id2", "label"}
        feature_cols = [c for c in features.columns if c not in exclude]
        if not feature_cols:
            raise ValueError(
                "FeatureExtractor produced no feature columns; check that "
                "the comparator specs are well-formed."
            )
        X = features[feature_cols].fillna(0.0).to_numpy(dtype=float)
        y = features["label"].to_numpy(dtype=np.int64)

        cls = _import_class(self.classifier_class)
        clf = cls(**self.classifier_params)
        np.random.seed(self.seed)
        clf.fit(X, y)

        self._classifier = clf
        self._feature_extractor = extractor
        self._feature_columns = feature_cols
        logger.info(
            "MagellanMatcher trained %s on %d pairs (%d pos / %d neg) "
            "with %d features",
            self.classifier_class,
            len(features),
            n_pos,
            n_neg,
            len(feature_cols),
        )

    def _ensure_trained(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
    ) -> None:
        if self._classifier is None:
            self._train(df_left, df_right, id_column)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _score_pairs(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        pairs: pd.DataFrame,
        id_column: str,
    ) -> np.ndarray:
        """Produce scores for ``pairs`` using the cached classifier."""
        assert self._feature_extractor is not None
        assert self._feature_columns is not None
        assert self._classifier is not None

        features = self._feature_extractor.create_features(
            df_left, df_right, pairs, id_column=id_column, labels=None
        )
        if features.empty:
            return np.zeros(0, dtype=float)

        X = features[self._feature_columns].fillna(0.0).to_numpy(dtype=float)

        if self.use_probabilities and hasattr(self._classifier, "predict_proba"):
            probs = self._classifier.predict_proba(X)
            if probs.shape[1] == 2:
                scores = probs[:, 1]
            else:
                # Single-class classifier (degenerate) — fall through to
                # the predict() path rather than emit a hard zero vector.
                scores = self._classifier.predict(X).astype(float)
        else:
            scores = self._classifier.predict(X).astype(float)

        # FeatureExtractor may drop rows if an id is missing; rebuild a
        # score array aligned to ``pairs`` with 0 for any missing pair.
        kept = features[["id1", "id2"]].copy()
        kept["__score"] = scores
        merged = pairs[["id1", "id2"]].merge(kept, on=["id1", "id2"], how="left")
        return merged["__score"].fillna(0.0).to_numpy(dtype=float)

    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Iterable[pd.DataFrame] | pd.DataFrame,
        id_column: str,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> CorrespondenceSet:
        """Score candidate pairs with the cached classifier.

        Parameters
        ----------
        df_left, df_right : pandas.DataFrame
            Source frames with ``id_column``.  The committee runner's
            ``column_mapping`` rename has already been applied by the
            time this method runs.
        candidates : pandas.DataFrame or iterable of pandas.DataFrame
            Candidate pair batches with ``id1`` / ``id2`` columns.
        id_column : str
            Id-column name in both source frames.
        threshold : float, default 0.5
            Minimum score retained in the output.
        **kwargs
            Accepted for compatibility with the committee runner.  The
            runner may pass ``comparators`` / ``weights`` keys that apply
            only to the rule-based aggregator — this adapter ignores them
            since the comparators are pinned at init time.

        Returns
        -------
        CorrespondenceSet
            DataFrame with columns ``id1``, ``id2``, ``score``,
            ``notes``.
        """
        self._validate_inputs(df_left, df_right, id_column)

        if isinstance(candidates, pd.DataFrame):
            batches: list[pd.DataFrame] = [candidates]
        else:
            batches = list(candidates)
        # Drop empty / missing batches up-front so the trainer never sees
        # an accidental empty-frame call when every blocker batch is
        # empty.
        batches = [b for b in batches if b is not None and not b.empty]
        if not batches:
            return pd.DataFrame(columns=["id1", "id2", "score", "notes"])

        self._ensure_trained(df_left, df_right, id_column)

        out_rows: list[dict[str, Any]] = []
        for batch in batches:
            if "id1" not in batch.columns or "id2" not in batch.columns:
                raise ValueError("candidate batch must have 'id1' and 'id2' columns")
            pairs = batch[["id1", "id2"]].copy()
            scores = self._score_pairs(df_left, df_right, pairs, id_column)
            for (id1, id2), score in zip(zip(pairs["id1"], pairs["id2"]), scores):
                if score >= threshold:
                    out_rows.append(
                        {
                            "id1": id1,
                            "id2": id2,
                            "score": float(score),
                            "notes": "magellan",
                        }
                    )

        if not out_rows:
            return pd.DataFrame(columns=["id1", "id2", "score", "notes"])
        return pd.DataFrame(out_rows, columns=["id1", "id2", "score", "notes"])


__all__ = ["MagellanMatcher"]

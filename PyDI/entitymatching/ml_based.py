"""
Machine learning-based entity matching using scikit-learn integration.

This module provides ML-based entity matching that leverages scikit-learn's
full ecosystem while maintaining compatibility with PyDI's design principles.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseMatcher, CorrespondenceSet
from .feature_extraction import FeatureExtractor, VectorFeatureExtractor


class MLBasedMatcher(BaseMatcher):
    """Machine learning-based entity matcher with scikit-learn integration.

    This matcher uses trained scikit-learn classifiers to predict entity matches
    based on feature vectors extracted from candidate pairs. It provides minimal
    wrapping around scikit-learn while maintaining PyDI compatibility.

    The matcher requires a FeatureExtractor to convert entity pairs into features
    and a trained scikit-learn classifier. This design allows users to leverage
    the full scikit-learn ecosystem (hyperparameter tuning, cross-validation,
    feature selection, etc.) while maintaining PyDI's DataFrame-first approach.

    Parameters
    ----------
    feature_extractor : FeatureExtractor or VectorFeatureExtractor
        Feature extraction component that converts entity pairs to feature vectors.

    Examples
    --------
    >>> from PyDI.entitymatching import MLBasedMatcher, FeatureExtractor
    >>> from PyDI.entitymatching.comparators import StringComparator
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> # Create feature extractor
    >>> extractor = FeatureExtractor([
    ...     StringComparator("title", "jaro_winkler"),
    ...     NumericComparator("year", "absolute_difference")
    ... ])
    >>>
    >>> # Prepare training data
    >>> train_features = extractor.create_features(df_left, df_right, train_pairs, train_labels)
    >>> X = train_features.drop(['label', 'id1', 'id2'], axis=1)
    >>> y = train_features['label']
    >>>
    >>> # Train classifier using full scikit-learn workflow
    >>> clf = RandomForestClassifier(n_estimators=100)
    >>> clf.fit(X, y)
    >>>
    >>> # Use trained classifier with matcher
    >>> matcher = MLBasedMatcher(extractor)
    >>> matches = matcher.match(df_left, df_right, candidates, clf, threshold=0.7)
    """

    def __init__(
        self, feature_extractor: Union[FeatureExtractor, VectorFeatureExtractor]
    ):
        """Initialize ML-based matcher.

        Parameters
        ----------
        feature_extractor : FeatureExtractor or VectorFeatureExtractor
            Feature extraction component.
        """
        if not isinstance(
            feature_extractor, (FeatureExtractor, VectorFeatureExtractor)
        ):
            raise ValueError(
                "feature_extractor must be FeatureExtractor or VectorFeatureExtractor instance"
            )

        self.feature_extractor = feature_extractor

    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Iterable[pd.DataFrame],
        id_column: str,
        trained_classifier: Any,
        threshold: float = 0.5,
        use_probabilities: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> Union[CorrespondenceSet, Tuple[CorrespondenceSet, pd.DataFrame]]:
        """Find entity correspondences using trained ML classifier.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset with _id column and dataset_name in attrs.
        df_right : pandas.DataFrame
            Right dataset with _id column and dataset_name in attrs.
        candidates : Iterable[pandas.DataFrame]
            Candidate pair batches with id1, id2 columns.
        trained_classifier : sklearn classifier
            Trained scikit-learn classifier with predict() method.
            If use_probabilities=True, must also have predict_proba() method.
        threshold : float, optional
            Decision threshold for classification. Default is 0.5.
            If use_probabilities=True, applied to probability of positive class.
            If use_probabilities=False, applied to raw classifier output.
        use_probabilities : bool, optional
            Whether to use predict_proba() for probabilistic scores (default)
            or predict() for binary decisions. Default is False.
        debug : bool, optional
            If True, captures detailed feature extraction and prediction results
            for debugging. Returns tuple of (correspondences, debug_results). Default is False.
        **kwargs
            Additional arguments (ignored).

        Returns
        -------
        CorrespondenceSet or Tuple[CorrespondenceSet, pandas.DataFrame]
            If debug=False: DataFrame with columns id1, id2, score, notes containing
            entity correspondences above the threshold.
            If debug=True: Tuple of (correspondences, debug_results) where
            debug_results contains detailed ML prediction information.

        Raises
        ------
        ValueError
            If classifier doesn't have required methods or inputs are invalid.
        """
        # Setup logger
        logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Log start of entity matching
        logger.info("Starting Entity Matching")
        start_time = time.time()
        
        # Validate inputs
        self._validate_inputs(df_left, df_right, id_column)

        if trained_classifier is None:
            raise ValueError("trained_classifier cannot be None")

        # Check classifier methods
        if not hasattr(trained_classifier, "predict"):
            raise ValueError("trained_classifier must have predict() method")

        if use_probabilities and not hasattr(trained_classifier, "predict_proba"):
            raise ValueError(
                "trained_classifier must have predict_proba() method when use_probabilities=True"
            )

        # Log blocking info (similar to Winter's blocking log)
        logger.info(f"Blocking {len(df_left)} x {len(df_right)} elements")

        # Convert candidates to list to count total pairs
        if hasattr(candidates, '__iter__') and not isinstance(candidates, pd.DataFrame):
            candidate_list = list(candidates)
        else:
            candidate_list = [candidates] if isinstance(candidates, pd.DataFrame) else []
        
        # Count total candidate pairs for reduction ratio calculation
        total_pairs_processed = sum(len(batch) for batch in candidate_list if not batch.empty)
        total_possible_pairs = len(df_left) * len(df_right)
        reduction_ratio = 1 - (total_pairs_processed / total_possible_pairs) if total_possible_pairs > 0 else 0
        
        # Calculate elapsed time for blocking phase
        blocking_time = time.time() - start_time
        blocking_time_str = f"{blocking_time:.3f}"
        
        # Log matching phase info with reduction ratio
        logger.info(f"Matching {len(df_left)} x {len(df_right)} elements after 0:00:{blocking_time_str}; "
                   f"{total_pairs_processed} blocked pairs (reduction ratio: {reduction_ratio})")
        
        results = []
        debug_results = [] if debug else None

        # Process candidate batches
        for batch in candidate_list:
            if batch.empty:
                continue

            if debug:
                batch_results, batch_debug = self._process_batch(
                    batch,
                    df_left,
                    df_right,
                    id_column,
                    trained_classifier,
                    threshold,
                    use_probabilities,
                    debug=True,
                )
                debug_results.extend(batch_debug)
            else:
                batch_results = self._process_batch(
                    batch,
                    df_left,
                    df_right,
                    id_column,
                    trained_classifier,
                    threshold,
                    use_probabilities,
                    debug=False,
                )
            results.extend(batch_results)

        # Calculate total elapsed time
        total_time = time.time() - start_time
        total_time_str = f"{total_time:.3f}"
        
        # Log completion info
        logger.info(f"Entity Matching finished after 0:00:{total_time_str}; found {len(results)} correspondences.")

        # Create correspondence set
        if results:
            corr_df = pd.DataFrame(results)

            # Add metadata
            corr_df.attrs["classifier_type"] = type(trained_classifier).__name__
            corr_df.attrs["threshold"] = threshold
            corr_df.attrs["use_probabilities"] = use_probabilities
            corr_df.attrs["feature_extractor"] = str(self.feature_extractor)

        else:
            corr_df = pd.DataFrame(columns=["id1", "id2", "score", "notes"])
            corr_df.attrs["classifier_type"] = type(trained_classifier).__name__
            corr_df.attrs["threshold"] = threshold

        if debug:
            debug_df = pd.DataFrame(debug_results) if debug_results else pd.DataFrame(columns=[
                "id1", "id2", "feature_names", "feature_values", "prediction_score",
                "classifier_type", "prediction_proba_positive", "prediction_proba_negative"
            ])
            return corr_df, debug_df
        else:
            return corr_df

    def _process_batch(
        self,
        batch: pd.DataFrame,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        id_column: str,
        trained_classifier: Any,
        threshold: float,
        use_probabilities: bool,
        debug: bool = False,
    ) -> Union[list, Tuple[list, list]]:
        """Process a batch of candidate pairs using the trained classifier.

        Parameters
        ----------
        batch : pandas.DataFrame
            Candidate pairs with id1, id2 columns.
        df_left : pandas.DataFrame
            Left dataset.
        df_right : pandas.DataFrame
            Right dataset.
        trained_classifier : sklearn classifier
            Trained classifier.
        threshold : float
            Decision threshold.
        use_probabilities : bool
            Whether to use probabilistic scores.
        debug : bool, optional
            Whether to capture debug information. Default is False.

        Returns
        -------
        list or Tuple[list, list]
            If debug=False: List of correspondence dictionaries.
            If debug=True: Tuple of (correspondence_list, debug_results_list).
        """
        try:
            # Extract features for this batch
            feature_df = self.feature_extractor.create_features(
                df_left, df_right, batch, id_column, labels=None
            )

            if feature_df.empty:
                logging.warning("No features extracted for batch")
                return []

            # Prepare feature matrix (exclude id columns)
            id_columns = ["id1", "id2"]
            feature_columns = [
                col for col in feature_df.columns if col not in id_columns
            ]

            if not feature_columns:
                logging.warning("No feature columns found")
                return []

            X = feature_df[
                feature_columns
            ]  # Keep as DataFrame to preserve column names

            # Get predictions
            if use_probabilities:
                # Use probabilities of positive class
                try:
                    probas = trained_classifier.predict_proba(X)
                    if probas.shape[1] == 2:
                        # Binary classification - use positive class probability
                        scores = probas[:, 1]
                    else:
                        # Multi-class or single-class - use max probability
                        scores = np.max(probas, axis=1)
                except Exception as e:
                    logging.warning(
                        f"Error getting probabilities, falling back to predictions: {e}"
                    )
                    # Fall back to binary predictions
                    predictions = trained_classifier.predict(X)
                    scores = predictions.astype(float)
            else:
                # Use raw predictions
                predictions = trained_classifier.predict(X)
                scores = predictions.astype(float)

            # Filter by threshold and create results
            results = []
            debug_results = [] if debug else None

            for i, score in enumerate(scores):
                row = feature_df.iloc[i]

                # Always capture debug info if requested
                if debug:
                    # Get feature names and values for this pair
                    feature_names = [col for col in feature_df.columns if col not in ["id1", "id2"]]
                    feature_values = {name: row[name] for name in feature_names}

                    # Get probabilities if available
                    proba_positive = None
                    proba_negative = None
                    if use_probabilities and hasattr(trained_classifier, "predict_proba"):
                        try:
                            proba = trained_classifier.predict_proba(X.iloc[[i]])
                            if proba.shape[1] == 2:
                                proba_negative = float(proba[0, 0])
                                proba_positive = float(proba[0, 1])
                        except:
                            pass

                    debug_results.append({
                        "id1": row["id1"],
                        "id2": row["id2"],
                        "feature_names": str(feature_names),
                        "feature_values": str(feature_values),
                        "prediction_score": float(score),
                        "classifier_type": type(trained_classifier).__name__,
                        "prediction_proba_positive": proba_positive,
                        "prediction_proba_negative": proba_negative,
                    })

                # Add to results if above threshold
                if score >= threshold:
                    results.append(
                        {
                            "id1": row["id1"],
                            "id2": row["id2"],
                            "score": float(score),
                            "notes": f"ml_classifier={type(trained_classifier).__name__}",
                        }
                    )

            if debug:
                return results, debug_results
            else:
                return results

        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            if debug:
                return [], []
            else:
                return []

    def predict_pairs(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        pairs: pd.DataFrame,
        trained_classifier: Any,
        use_probabilities: bool = True,
    ) -> pd.DataFrame:
        """Predict match probabilities for specific pairs without threshold filtering.

        This method is useful for getting raw predictions or probabilities for
        evaluation purposes without applying a threshold.

        Parameters
        ----------
        df_left : pandas.DataFrame
            Left dataset.
        df_right : pandas.DataFrame
            Right dataset.
        pairs : pandas.DataFrame
            Pairs to predict with id1, id2 columns.
        trained_classifier : sklearn classifier
            Trained classifier.
        use_probabilities : bool, optional
            Whether to use predict_proba() or predict(). Default is True.

        Returns
        -------
        pandas.DataFrame
            DataFrame with id1, id2, prediction columns.
        """
        # Extract features
        feature_df = self.feature_extractor.create_features(
            df_left, df_right, pairs, labels=None
        )

        if feature_df.empty:
            return pd.DataFrame(columns=["id1", "id2", "prediction"])

        # Prepare feature matrix
        id_columns = ["id1", "id2"]
        feature_columns = [col for col in feature_df.columns if col not in id_columns]
        X = feature_df[feature_columns]  # Keep as DataFrame

        # Get predictions
        if use_probabilities and hasattr(trained_classifier, "predict_proba"):
            probas = trained_classifier.predict_proba(X)
            if probas.shape[1] == 2:
                predictions = probas[:, 1]  # Positive class probability
            else:
                predictions = np.max(probas, axis=1)
        else:
            predictions = trained_classifier.predict(X).astype(float)

        # Create result DataFrame
        result_df = feature_df[["id1", "id2"]].copy()
        result_df["prediction"] = predictions

        return result_df

    def get_feature_importance(
        self,
        trained_classifier: Any,
        feature_names: Optional[list] = None,
    ) -> pd.DataFrame:
        """Get feature importance from trained classifier if available.

        Parameters
        ----------
        trained_classifier : sklearn classifier
            Trained classifier with feature_importances_ attribute.
        feature_names : list, optional
            Names of features. If None, uses extractor's feature names.

        Returns
        -------
        pandas.DataFrame
            DataFrame with feature names and importance scores.

        Raises
        ------
        AttributeError
            If classifier doesn't have feature importance information.
        """
        if not hasattr(trained_classifier, "feature_importances_"):
            raise AttributeError(
                f"{type(trained_classifier).__name__} doesn't provide feature importance"
            )

        importances = trained_classifier.feature_importances_

        if feature_names is None:
            feature_names = self.feature_extractor.get_feature_names()

        # Handle mismatch by using the actual number of importances
        if len(importances) != len(feature_names):
            logging.warning(
                f"Feature name mismatch: {len(importances)} importances vs {len(feature_names)} names"
            )
            # Use generic names if mismatch
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        importance_df = pd.DataFrame(
            {"feature": feature_names[: len(importances)], "importance": importances}
        ).sort_values("importance", ascending=False)

        # Format importance values to 4 decimal places
        importance_df["importance"] = importance_df["importance"].apply(lambda x: f"{x:.4f}")

        return importance_df

    def __repr__(self) -> str:
        return f"MLBasedMatcher(feature_extractor={self.feature_extractor})"

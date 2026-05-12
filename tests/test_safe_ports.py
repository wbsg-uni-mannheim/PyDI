import numpy as np
import pandas as pd

from PyDI.entitymatching import EntityMatchingEvaluator
from PyDI.entitymatching import FeatureExtractor, MLBasedMatcher
from PyDI.entitymatching.blocking import EmbeddingBlocker
from PyDI.schemamatching import LLMBasedSchemaMatcher


class _DummyChatModel:
    def invoke(self, _messages):
        raise AssertionError("This test should not call the LLM")


def test_embedding_blocker_sanitizes_non_finite_embeddings():
    left = pd.DataFrame({"id": ["l1", "l2"], "name": ["Alpha", ""]})
    right = pd.DataFrame({"id": ["r1"], "name": ["Alpha"]})

    blocker = EmbeddingBlocker(
        left,
        right,
        text_cols=["name"],
        id_column="id",
        left_embeddings=np.array([[np.nan, np.inf], [0.0, 0.0]]),
        right_embeddings=np.array([[1.0, 0.0]]),
    )

    left_embeddings, right_embeddings = blocker._ensure_embeddings()

    assert np.isfinite(left_embeddings).all()
    assert np.isfinite(right_embeddings).all()
    assert np.allclose(left_embeddings[1], [0.0, 0.0])


def test_evaluate_blocking_accepts_total_possible_pairs_without_blocker():
    candidates = pd.DataFrame({"id1": ["a", "b"], "id2": ["x", "y"]})
    truth = pd.DataFrame(
        {
            "id1": ["a", "b", "c"],
            "id2": ["x", "z", "w"],
            "label": [1, 0, 1],
        }
    )

    result = EntityMatchingEvaluator.evaluate_blocking(
        candidates,
        truth,
        total_possible_pairs=10,
    )

    assert result["true_positives_found"] == 1
    assert result["total_possible_pairs"] == 10


def test_evaluate_matching_with_labels_after_pair_set_cleanup():
    correspondences = pd.DataFrame(
        {"id1": ["a"], "id2": ["x"], "score": [0.9]}
    )
    truth = pd.DataFrame(
        {
            "id1": ["a", "b"],
            "id2": ["x", "y"],
            "label": [1, 0],
        }
    )

    result = EntityMatchingEvaluator.evaluate_matching(
        correspondences,
        truth,
    )

    assert result["true_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["accuracy"] == 1.0


def test_schema_matcher_samples_sparse_columns(tmp_path):
    matcher = LLMBasedSchemaMatcher(
        _DummyChatModel(),
        num_rows=3,
        out_dir=str(tmp_path),
    )
    df = pd.DataFrame(
        {
            "Attribute_1": [None, None, "US", None],
            "Attribute_2": ["A", "B", "C", "D"],
        }
    )

    sample = matcher._select_sample_rows(df, ["Attribute_1", "Attribute_2"], 2)
    summary = matcher._generate_column_summary(df)

    assert "US" in sample["Attribute_1"].astype(str).tolist()
    assert "Attribute_1" in summary
    assert "US" in summary


def test_ml_matcher_reuses_cached_feature_lookups():
    class SpyFeatureExtractor(FeatureExtractor):
        def __init__(self):
            super().__init__([
                {
                    "name": "same_name",
                    "function": lambda left, right: left["name"] == right["name"],
                }
            ])
            self.left_lookup_ids = []
            self.right_lookup_ids = []

        def create_features(self, *args, **kwargs):
            self.left_lookup_ids.append(id(kwargs.get("left_lookup")))
            self.right_lookup_ids.append(id(kwargs.get("right_lookup")))
            return super().create_features(*args, **kwargs)

    class AlwaysMatchClassifier:
        def predict(self, X):
            return np.ones(len(X))

    left = pd.DataFrame({"id": ["l1", "l2"], "name": ["Acme", "Beta"]})
    right = pd.DataFrame({"id": ["r1", "r2"], "name": ["Acme", "Beta"]})
    candidates = [
        pd.DataFrame({"id1": ["l1"], "id2": ["r1"]}),
        pd.DataFrame({"id1": ["l2"], "id2": ["r2"]}),
    ]

    extractor = SpyFeatureExtractor()
    matcher = MLBasedMatcher(extractor)

    matches = matcher.match(
        left,
        right,
        candidates,
        "id",
        AlwaysMatchClassifier(),
        threshold=0.5,
    )

    assert len(matches) == 2
    assert len(set(extractor.left_lookup_ids)) == 1
    assert len(set(extractor.right_lookup_ids)) == 1

import pandas as pd

from PyDI.fusion import DataFusionEvaluator, DataFusionStrategy


def test_fusion_evaluation_aligns_by_fusion_sources():
    evaluator = DataFusionEvaluator(DataFusionStrategy())

    fused = pd.DataFrame(
        [
            {
                "_id": "left-1+right-1",
                "_fusion_sources": ["left-1", "right-1"],
                "name": "Acme",
            }
        ]
    )
    expected = pd.DataFrame([{"id": "right-1", "name": "Acme"}])

    aligned_fused, aligned_expected = evaluator._align_datasets_two_ids(
        fused,
        "_id",
        expected,
        "id",
    )

    assert len(aligned_fused) == 1
    assert len(aligned_expected) == 1
    assert aligned_fused.iloc[0]["name"] == "Acme"
    assert aligned_expected.iloc[0]["name"] == "Acme"


def test_fusion_evaluation_aligns_by_expected_source_ids_column():
    evaluator = DataFusionEvaluator(DataFusionStrategy())

    fused = pd.DataFrame(
        [
            {
                "_id": "left-1+right-1",
                "_fusion_sources": ["left-1", "right-1"],
                "name": "Acme",
            }
        ]
    )
    expected = pd.DataFrame(
        [{"id": "entity-1", "source_ids": "missing, right-1", "name": "Acme"}]
    )

    aligned_fused, aligned_expected = evaluator._align_datasets_two_ids(
        fused,
        "_id",
        expected,
        "id",
    )

    assert len(aligned_fused) == 1
    assert len(aligned_expected) == 1
    assert aligned_fused.iloc[0]["name"] == "Acme"

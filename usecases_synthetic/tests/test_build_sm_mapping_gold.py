"""R10-C: tests for the baseline SM-gold generator.

Confirms the regenerated ``sm_mapping_gold.csv`` is the K8 ``sm_mapping``
ground truth with the original (un-renamed) source column names — so the
baseline SM committee scores on the same attribute set + target mapping
the variant ``sm_mapping.csv`` is generated from (apples-to-apples).
"""

from __future__ import annotations

import pandas as pd

from usecases_synthetic.lib.domain_config import load_knob_config
from usecases_synthetic.scripts.build_sm_mapping_gold import (
    build_sm_mapping_gold,
    gold_path,
)


def _truth_triples(domain: str) -> set[tuple[str, str, str]]:
    sm = load_knob_config(8, domain)["sm_mapping"]
    return {
        (source, col, target)
        for source, mapping in sm.items()
        for col, target in mapping.items()
    }


class TestBuildSmMappingGold:
    def test_products_full_scope(self) -> None:
        df = build_sm_mapping_gold("products")
        # 4 sources x 20 attrs.
        assert len(df) == 80
        assert list(df.columns) == [
            "source_dataset",
            "source_column",
            "target_dataset",
            "target_column",
            "score",
        ]
        assert (df["score"] == 1.0).all()
        assert (df["target_dataset"] == "products").all()

    def test_products_includes_r1_attrs_with_original_names(self) -> None:
        df = build_sm_mapping_gold("products")
        cols = set(df["source_column"])
        # New R1 attrs are present...
        assert {"title_description", "chipset_name", "write_speed_mb_s"} <= cols
        # ...under their ORIGINAL names, not K8 rename tokens.
        assert "t_desc" not in cols  # abbreviated rename for title_description
        assert "prd_ttl" not in cols  # abbreviated rename for title

    def test_matches_k8_ground_truth(self) -> None:
        df = build_sm_mapping_gold("products")
        triples = {
            (r.source_dataset, r.source_column, r.target_column)
            for r in df.itertuples(index=False)
        }
        assert triples == _truth_triples("products")

    def test_other_domains_already_full_scope(self) -> None:
        """companies/games/music gold on disk already equals the K8 truth
        (no R1 schema upgrade), so the generator is a content no-op there."""
        for domain in ("companies", "games", "music"):
            gen = build_sm_mapping_gold(domain)
            gen_set = {
                (r.source_dataset, r.source_column, r.target_column)
                for r in gen.itertuples(index=False)
            }
            disk = pd.read_csv(gold_path(domain))
            disk_set = {
                (str(r.source_dataset), str(r.source_column), str(r.target_column))
                for r in disk.itertuples(index=False)
            }
            assert gen_set == disk_set, domain

    def test_products_gold_on_disk_refreshed(self) -> None:
        """The committed products gold carries the full 80-row scope."""
        disk = pd.read_csv(gold_path("products"))
        assert len(disk) == 80

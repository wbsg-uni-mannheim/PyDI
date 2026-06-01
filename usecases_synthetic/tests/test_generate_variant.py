"""Tests for the S1 orchestrator and variant packager (Module 10).

Covers:

- :mod:`usecases_synthetic.scripts.package_variant` — directory
  assembly, provenance consolidation, difficulty.yaml writing.
- :mod:`usecases_synthetic.scripts.generate_variant` — per-level
  orchestration, knob invocation order, and cross-level monotonicity
  checks.

The orchestrator test uses monkey-patching to stub the six per-knob
runners so we can verify pipeline plumbing without paying the cost of
the real knob implementations (which have extensive unit tests of
their own). Running the real knobs end-to-end on companies is covered
by the existing per-module tests.

Acceptance criteria (from ``plans/module_10_orchestrator.md``):

1. ``generate_variant.py --domain companies --level easy`` produces a
   valid variant directory.
2. All 3 levels produce complete variants with all expected artifacts.
3. Cross-level monotonicity holds for all 7 checks.
4. ``difficulty.yaml`` contains all knob parameters and seeds.
5. Provenance CSVs consolidated under ``output/provenance/`` with no gaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from usecases_synthetic.lib.provenance import PROVENANCE_COLUMNS, ProvenanceLog
from usecases_synthetic.scripts import generate_variant as gv
from usecases_synthetic.scripts import package_variant as pv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_prov_csv(
    path: Path,
    rows: list[dict[str, Any]],
    knob: int,
    level: str,
) -> None:
    """Write a provenance CSV at ``path`` with ``rows``."""
    log = ProvenanceLog(knob=knob, level=level)
    for r in rows:
        log.append(
            entity_id=r.get("entity_id", ""),
            source=r.get("source", ""),
            attribute=r.get("attribute", ""),
            original_value=r.get("original_value", ""),
            new_value=r.get("new_value", ""),
            transform_fn=r.get("transform_fn", "test"),
            transform_params=r.get("transform_params", {}),
        )
    log.flush(path)


def _make_tiny_sources() -> dict[str, pd.DataFrame]:
    """Return three tiny DataFrames with ``dataset_name`` attrs set."""
    dbpedia = pd.DataFrame(
        {
            "identifier": ["db1", "db2"],
            "name": ["Foo Inc.", "Bar Corp."],
            "countryName": ["United States", "Germany"],
        }
    )
    dbpedia.attrs["dataset_name"] = "dbpedia"
    forbes = pd.DataFrame(
        {
            "Identifier": ["fb1", "fb2"],
            "companyName": ["Foo", "Bar"],
            "countryCode3": ["USA", "GER"],
        }
    )
    forbes.attrs["dataset_name"] = "forbes"
    fullcontact = pd.DataFrame(
        {
            "id": ["fc1", "fc2"],
            "name": ["Foo", "Bar"],
            "country": ["US", "DE"],
        }
    )
    fullcontact.attrs["dataset_name"] = "fullcontact"
    return {"dbpedia": dbpedia, "forbes": forbes, "fullcontact": fullcontact}


# ---------------------------------------------------------------------------
# package_variant tests
# ---------------------------------------------------------------------------


class TestPackageVariant:
    """Directory assembly tests for :func:`package_variant.package_variant`."""

    def _seed_work_dir(self, work_dir: Path) -> None:
        """Populate a work directory with minimal knob artifacts."""
        prov = work_dir / "output" / "provenance"
        base = work_dir / "output" / "baselines"
        sm = work_dir / "input" / "schemamatching"
        em = work_dir / "input" / "entitymatching"
        for d in (prov, base, sm, em):
            d.mkdir(parents=True, exist_ok=True)

        # Standard-schema provenance rows from three knobs.
        _write_prov_csv(
            prov / "knob_02_niche.csv",
            [
                {
                    "entity_id": "e1",
                    "source": "dbpedia",
                    "attribute": "name",
                    "original_value": "Foo Inc.",
                    "new_value": "Foo Inc",
                    "transform_fn": "normalise_label",
                }
            ],
            knob=2,
            level="easy",
        )
        _write_prov_csv(
            prov / "knob_03_attribute_drop.csv",
            [
                {
                    "entity_id": "e1",
                    "source": "dbpedia",
                    "attribute": "countryName",
                    "original_value": "United States",
                    "new_value": "",
                    "transform_fn": "drop_cell",
                }
            ],
            knob=3,
            level="easy",
        )
        _write_prov_csv(
            prov / "knob_08_naming.csv",
            [
                {
                    "entity_id": "",
                    "source": "dbpedia",
                    "attribute": "name",
                    "original_value": "name",
                    "new_value": "name",
                    "transform_fn": "rename_descriptive",
                }
            ],
            knob=8,
            level="easy",
        )

        # Non-standard auxiliary files (should be copied but not merged).
        pd.DataFrame({"entity_id": ["e1"], "density": [0.5]}).to_csv(
            prov / "knob_02_niche_scores.csv", index=False
        )
        pd.DataFrame(
            {
                "check": ["k1_k5_disjoint"],
                "status": ["PASS"],
                "detail": ["0 overlapping cells"],
            }
        ).to_csv(prov / "joint_values_audit.csv", index=False)

        # Baselines.
        pd.DataFrame({"source": ["dbpedia"], "missing_rate": [0.1]}).to_csv(
            base / "knob_03_baseline_missingness.csv", index=False
        )
        (base / "knob_10_gold_hash.txt").write_text("abc123\n")

        # SM mapping (from K8).
        pd.DataFrame(
            {
                "source_dataset": ["dbpedia"],
                "source_column": ["name"],
                "target_dataset": ["companies"],
                "target_column": ["name"],
                "score": [1.0],
            }
        ).to_csv(sm / "sm_mapping.csv", index=False)

        # Regenerated EM per-pair per-split files (from K2, C11): two
        # parallel versions per split (baseline_pruned + corner_filled),
        # named ``<pair>_<split>_<version>.csv`` per generate_variant.
        for split in ("train", "val", "test"):
            for version in ("baseline_pruned", "corner_filled"):
                pd.DataFrame(
                    {"id1": ["db1"], "id2": ["fb1"], "label": ["true"]}
                ).to_csv(em / f"dbpedia_2_forbes_{split}_{version}.csv", index=False)

    def test_package_variant_creates_full_directory(self, tmp_path: Path) -> None:
        """Packaging assembles all expected subdirectories and files."""
        work_dir = tmp_path / "work"
        variant_dir = tmp_path / "variant"
        self._seed_work_dir(work_dir)

        sources = _make_tiny_sources()

        summary = {
            "domain": "companies",
            "level": "easy",
            "master_seed": 42,
            "knobs": {"knob_08": {"sm_mapping_rows": 1}},
        }

        result = pv.package_variant(
            domain="companies",
            level="easy",
            sources=sources,
            work_dir=work_dir,
            variant_dir=variant_dir,
            difficulty_summary=summary,
        )

        # --- Verify directory structure -----------------------------------
        assert (variant_dir / "input" / "data").is_dir()
        assert (variant_dir / "input" / "schemamatching").is_dir()
        assert (variant_dir / "input" / "entitymatching").is_dir()
        assert (variant_dir / "input" / "fusion").is_dir()
        assert (variant_dir / "output" / "provenance").is_dir()
        assert (variant_dir / "output" / "baselines").is_dir()
        assert (variant_dir / "config").is_dir()

        # --- Sources serialised as CSV ------------------------------------
        for src in ("dbpedia", "forbes", "fullcontact"):
            data_path = variant_dir / "input" / "data" / f"{src}.csv"
            assert data_path.exists(), f"Missing {data_path}"
            df = pd.read_csv(data_path)
            assert len(df) == 2

        # --- SM mapping from K8 copied ------------------------------------
        sm_path = variant_dir / "input" / "schemamatching" / "sm_mapping.csv"
        assert sm_path.exists()
        sm = pd.read_csv(sm_path)
        assert "source_column" in sm.columns
        assert len(sm) == 1

        # --- Regenerated EM per-pair per-split files copied (C11) --------
        # R10-F: copy_regenerated_em globs the C11 version-suffixed files
        # (*_baseline_pruned.csv / *_corner_filled.csv), not the legacy
        # *_regenerated.csv suffix.
        em_out = variant_dir / "input" / "entitymatching"
        for split in ("train", "val", "test"):
            for version in ("baseline_pruned", "corner_filled"):
                path = em_out / f"dbpedia_2_forbes_{split}_{version}.csv"
                assert path.exists(), f"Missing {path}"
            # The legacy suffix must NOT be copied anymore.
            legacy = em_out / f"dbpedia_2_forbes_{split}_regenerated.csv"
            assert not legacy.exists(), f"Legacy suffix should not be copied: {legacy}"

        # --- Original EM files copied (from real companies use case) -----
        # The real companies use case ships multi-source EM correspondences
        # (forbes_2_dbpedia_*, forbes_2_fullcontact_*). Package must carry
        # those over so the downstream pipeline has training data.
        em_copied = {
            p.name for p in (variant_dir / "input" / "entitymatching").glob("*.csv")
        }
        assert any(
            "forbes_2_dbpedia" in n for n in em_copied
        ), f"Missing original EM correspondences: {em_copied}"

        # --- Fusion gold copied unchanged ---------------------------------
        assert (variant_dir / "input" / "fusion" / "test_set.xml").exists()

        # --- Provenance consolidated --------------------------------------
        prov_dir = variant_dir / "output" / "provenance"
        assert (prov_dir / "knob_02_niche.csv").exists()
        assert (prov_dir / "knob_02_niche_scores.csv").exists()
        assert (prov_dir / "joint_values_audit.csv").exists()
        assert (prov_dir / "provenance_all.csv").exists()

        merged = pd.read_csv(prov_dir / "provenance_all.csv")
        assert set(merged.columns) == set(PROVENANCE_COLUMNS)
        # Three standard-schema rows (knobs 02, 03, 08).
        assert len(merged) == 3
        assert result["provenance_rows"] == 3

        # --- Baseline files copied ----------------------------------------
        base_out = variant_dir / "output" / "baselines"
        assert (base_out / "knob_03_baseline_missingness.csv").exists()
        assert (base_out / "knob_10_gold_hash.txt").exists()

        # --- difficulty.yaml written --------------------------------------
        yaml_path = variant_dir / "config" / "difficulty.yaml"
        assert yaml_path.exists()
        loaded = yaml.safe_load(yaml_path.read_text())
        assert loaded["domain"] == "companies"
        assert loaded["level"] == "easy"
        assert loaded["master_seed"] == 42
        assert loaded["knobs"]["knob_08"]["sm_mapping_rows"] == 1

    def test_package_variant_invalid_level_raises(self, tmp_path: Path) -> None:
        """Empty level is rejected. Non-canonical labels are accepted
        to support ablation variants (e.g. ``ablation_knob_08``)."""
        with pytest.raises(ValueError, match="Invalid level"):
            pv.package_variant(
                domain="companies",
                level="",
                sources=_make_tiny_sources(),
                work_dir=tmp_path / "work",
                variant_dir=tmp_path / "variant",
                difficulty_summary={},
            )

    def test_write_sources_preserves_row_count(self, tmp_path: Path) -> None:
        """Source serialisation round-trips row counts."""
        sources = _make_tiny_sources()
        paths = pv.write_sources_as_csv(sources, tmp_path / "data")
        assert len(paths) == 3
        for p in paths:
            df = pd.read_csv(p)
            assert len(df) == 2


# ---------------------------------------------------------------------------
# check_monotonicity tests
# ---------------------------------------------------------------------------


class TestCheckMonotonicity:
    """Cross-level monotonicity audit tests."""

    def _seed_variant(
        self,
        variant_dir: Path,
        *,
        k03_drops: list[tuple[str, str, str]],
        k_counts: dict[str, int],
        k4_realised_mean: float | None = None,
        k4_realised_level: str = "easy",
        k02_realised_ratio: float | None = None,
        k02_target_ratio: float | None = None,
        k10_swap_rate: float | None = None,
        k10_reshufflable: int | None = None,
        k10_swap_cells: int | None = None,
        k01_committed: int | None = None,
        k01_attempts: int | None = None,
        k01_mean_edit: float | None = None,
        k01_mean_jaccard_drop: float | None = None,
        k01_strict_cache_miss: int | None = None,
    ) -> None:
        """Seed a variant directory's provenance folder.

        Parameters
        ----------
        variant_dir : Path
            Variant root.
        k03_drops : list of (entity, source, attribute)
            K3 drop cells to write as provenance rows.
        k_counts : dict
            Count of rows to write for each other knob
            (``k02, k04, k05, k06, k08, k10``).
        k4_realised_mean : float or None
            Mean number of sources per entity to encode in the K4
            realised histogram. When provided, a minimal
            ``baselines/knob_04_realized_vs_target.csv`` file is written
            so :func:`check_monotonicity` can compute the realised mean
            for this level. Lower values ⇒ harder fusion.
        k4_realised_level : str
            Level label to encode in the realised histogram. Must match
            the variant's level so ``check_monotonicity`` can find it.
        """
        prov_dir = variant_dir / "output" / "provenance"
        prov_dir.mkdir(parents=True, exist_ok=True)
        baselines_dir = variant_dir / "output" / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)

        if k4_realised_mean is not None:
            # Encode the target mean as a single coverage bin equal to
            # the mean (mean = coverage * fraction with fraction=1.0).
            pd.DataFrame(
                [
                    {
                        "label": f"realised_{k4_realised_level}",
                        "coverage": float(k4_realised_mean),
                        "fraction": 1.0,
                    }
                ]
            ).to_csv(baselines_dir / "knob_04_realized_vs_target.csv", index=False)

        # K2 realised CSV (added by step-1 instrumentation in
        # plan_revision.md). The audit reads ``final_ratio`` (realised
        # corner-case ratio) and ``target_ratio`` (configured target)
        # from this file. Seed defaults that satisfy
        # ``knob_02_configured_monotonicity`` /
        # ``knob_02_realised_monotonicity`` /
        # ``knob_02_realised_vs_configured`` so individual tests can
        # focus on the K3-nesting + count checks unless they override.
        if k02_realised_ratio is not None or k02_target_ratio is not None:
            realised = k02_realised_ratio if k02_realised_ratio is not None else 0.0
            target = k02_target_ratio if k02_target_ratio is not None else realised
            pd.DataFrame(
                [
                    {
                        "level": variant_dir.name,
                        "baseline_ratio": 0.0,
                        "target_ratio": float(target),
                        "final_ratio": float(realised),
                        "operator": "interpolate_paired_drop",
                        "removed": 0,
                        "interpolated": 0,
                    }
                ]
            ).to_csv(baselines_dir / "knob_02_realised.csv", index=False)

        # K1 realised CSV (plan_revision.md R-1 / G9 / step 4f). Audit
        # consumes ``paraphrase_committed`` (rate check) and
        # ``mean_edit_distance`` + ``mean_token_jaccard_drop`` (intensity
        # check). Default values reflect the typical "K1 fires more on
        # harder levels with more aggressive paraphrases" expectation.
        k01_any = any(
            v is not None
            for v in (
                k01_committed,
                k01_attempts,
                k01_mean_edit,
                k01_mean_jaccard_drop,
                k01_strict_cache_miss,
            )
        )
        if k01_any:
            committed = int(k01_committed if k01_committed is not None else 0)
            attempts = int(k01_attempts if k01_attempts is not None else committed)
            mean_edit = float(k01_mean_edit if k01_mean_edit is not None else 0.0)
            mean_jaccard = float(
                k01_mean_jaccard_drop if k01_mean_jaccard_drop is not None else 0.0
            )
            strict_miss = int(
                k01_strict_cache_miss if k01_strict_cache_miss is not None else 0
            )
            pd.DataFrame(
                [
                    {
                        "level": variant_dir.name,
                        "paraphrase_attempts": attempts,
                        "paraphrase_committed": committed,
                        "mean_edit_distance": mean_edit,
                        "mean_token_jaccard_drop": mean_jaccard,
                        "strict_cache_miss_count": strict_miss,
                    }
                ]
            ).to_csv(baselines_dir / "knob_01_realised.csv", index=False)

        # K10 realised CSV (step-1 C3 K10). Audit consumes ``swap_rate``
        # (rate-based, K3-drop-invariant) plus the legacy
        # ``compromised_mask_count`` row. Default to non-decreasing rate.
        if (
            k10_swap_rate is not None
            or k10_reshufflable is not None
            or k10_swap_cells is not None
        ):
            rate = k10_swap_rate if k10_swap_rate is not None else 0.0
            reshuffle = k10_reshufflable if k10_reshufflable is not None else 100
            cells = (
                k10_swap_cells
                if k10_swap_cells is not None
                else int(round(rate * reshuffle))
            )
            pd.DataFrame(
                [
                    {
                        "level": variant_dir.name,
                        "reshufflable_count": reshuffle,
                        "swap_cells": cells,
                        "swap_rate": float(rate),
                        "compromised_mask_count": cells,
                    }
                ]
            ).to_csv(baselines_dir / "knob_10_realised.csv", index=False)

        # K3 provenance with explicit drop cells.
        _write_prov_csv(
            prov_dir / "knob_03_attribute_drop.csv",
            [
                {
                    "entity_id": e,
                    "source": s,
                    "attribute": a,
                    "transform_fn": "drop_cell",
                }
                for e, s, a in k03_drops
            ],
            knob=3,
            level="easy",
        )

        knob_to_file = {
            "k02": ("knob_02_niche.csv", 2),
            "k04": ("knob_04_coverage_skew.csv", 4),
            "k05": ("knob_05_format_unit.csv", 5),
            "k06": ("knob_06_noise.csv", 6),
            "k08": ("knob_08_naming.csv", 8),
            "k10": ("knob_10_reliability.csv", 10),
        }
        # K2 + K6 audits filter by transform_fn (see check_monotonicity); use
        # realistic op names so the count-based checks see the seeded rows.
        # K10 audit also filters to ``reassign_gold_carrier`` for the same
        # reason (real perturbation signal).
        knob_transform_fns = {
            "k02": "remove_entity",
            "k06": "typo_substitute",
            "k10": "reassign_gold_carrier",
        }
        for key, (fname, knob) in knob_to_file.items():
            count = k_counts.get(key, 0)
            transform_fn = knob_transform_fns.get(key, f"test_knob_{knob}")
            _write_prov_csv(
                prov_dir / fname,
                [
                    {
                        "entity_id": f"e_{knob}_{i}",
                        "source": "dbpedia",
                        "attribute": "x",
                        "transform_fn": transform_fn,
                    }
                    for i in range(count)
                ],
                knob=knob,
                level="easy",
            )

    def test_monotonic_pass(self, tmp_path: Path) -> None:
        """Correctly-nested drops + non-decreasing counts pass all checks."""
        dirs = {}
        drops_easy = [("e1", "dbpedia", "a")]
        drops_medium = [("e1", "dbpedia", "a"), ("e2", "forbes", "b")]
        drops_hard = drops_medium + [("e3", "fullcontact", "c")]
        drop_map = {
            "easy": drops_easy,
            "medium": drops_medium,
            "hard": drops_hard,
        }
        count_map = {
            "easy": {"k02": 1, "k04": 1, "k05": 1, "k06": 1, "k08": 1, "k10": 1},
            "medium": {"k02": 2, "k04": 2, "k05": 2, "k06": 2, "k08": 2, "k10": 2},
            "hard": {"k02": 3, "k04": 3, "k05": 3, "k06": 3, "k08": 3, "k10": 3},
        }
        # K4 realised mean sources per entity — non-increasing easy → hard.
        k4_mean_map = {"easy": 2.8, "medium": 2.5, "hard": 2.0}
        # K2 realised + configured ratios: non-decreasing across levels so
        # the K2 configured + realised monotonicity checks pass.
        k02_target_map = {"easy": 0.2, "medium": 0.5, "hard": 0.8}
        k02_realised_map = {"easy": 0.2, "medium": 0.5, "hard": 0.8}
        # K10 realised swap rate: non-decreasing across levels (dispersion
        # increases with difficulty) for the rate-based check.
        k10_rate_map = {"easy": 0.1, "medium": 0.3, "hard": 0.6}
        # K1 realised metrics (plan_revision.md R-1 / G9 / step 4f):
        # both committed count and intensity grow easy → hard for the
        # rate + intensity monotonicity checks.
        k01_committed_map = {"easy": 30, "medium": 60, "hard": 120}
        k01_edit_map = {"easy": 0.2, "medium": 0.4, "hard": 0.6}
        k01_jaccard_map = {"easy": 0.1, "medium": 0.3, "hard": 0.5}
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant(
                vd,
                k03_drops=drop_map[lvl],
                k_counts=count_map[lvl],
                k4_realised_mean=k4_mean_map[lvl],
                k4_realised_level=lvl,
                k02_target_ratio=k02_target_map[lvl],
                k02_realised_ratio=k02_realised_map[lvl],
                k10_swap_rate=k10_rate_map[lvl],
                k10_reshufflable=100,
                k01_committed=k01_committed_map[lvl],
                k01_attempts=k01_committed_map[lvl] + 5,
                k01_mean_edit=k01_edit_map[lvl],
                k01_mean_jaccard_drop=k01_jaccard_map[lvl],
                k01_strict_cache_miss=0,
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)

        # Audit shape after step-1 instrumentation: K3 nesting + 6 count
        # rows + K2 (configured / realised_vs_configured / realised) +
        # K5 distinct_format_families + K8 naming_intensity + K10
        # (configured / realised_vs_configured / realised_rate). The
        # exact set depends on what passes — the contract is "no FAIL".
        fails = audit[audit["status"] == "FAIL"]
        assert fails.empty, f"Unexpected failures: {fails.to_string(index=False)}"

        nesting = audit[audit["check"] == "knob_03_drop_nesting"].iloc[0]
        assert nesting["easy"] == 1
        assert nesting["medium"] == 2
        assert nesting["hard"] == 3

    def test_drop_nesting_violation_fails(self, tmp_path: Path) -> None:
        """A drop cell present at easy but absent at medium fails the nest check."""
        dirs = {}
        drop_map = {
            "easy": [("e1", "dbpedia", "a")],
            "medium": [("e2", "dbpedia", "b")],  # disjoint!
            "hard": [("e1", "dbpedia", "a"), ("e2", "dbpedia", "b")],
        }
        counts = {"k02": 1, "k04": 1, "k05": 1, "k06": 1, "k08": 1, "k10": 1}
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant(vd, k03_drops=drop_map[lvl], k_counts=counts)
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)
        nesting = audit[audit["check"] == "knob_03_drop_nesting"].iloc[0]
        assert nesting["status"] == "FAIL"
        assert "easy⊆medium=False" in nesting["detail"]

    def test_count_violation_fails(self, tmp_path: Path) -> None:
        """Non-monotone K2 realised ratio is flagged.

        The legacy K2 count check (``knob_02_corner_case_count``) was
        retired by step-1 instrumentation. K2 monotonicity now reads
        ``knob_02_realised.csv`` (``final_ratio``). This test seeds a
        non-monotone realised ratio (easy > medium) and confirms the
        ``knob_02_realised_monotonicity`` check fails while everything
        else still passes.
        """
        dirs = {}
        drops = [("e1", "dbpedia", "a")]
        counts = {"k02": 1, "k04": 1, "k05": 1, "k06": 1, "k08": 1, "k10": 1}
        k4_mean_map = {"easy": 2.8, "medium": 2.5, "hard": 2.0}
        # K2 realised: easy > medium (non-monotone). Configured ratio
        # stays monotone so the configured check still passes.
        k02_target_map = {"easy": 0.2, "medium": 0.5, "hard": 0.8}
        k02_realised_map = {"easy": 0.6, "medium": 0.3, "hard": 0.8}
        k10_rate_map = {"easy": 0.1, "medium": 0.3, "hard": 0.6}
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant(
                vd,
                k03_drops=drops,
                k_counts=counts,
                k4_realised_mean=k4_mean_map[lvl],
                k4_realised_level=lvl,
                k02_target_ratio=k02_target_map[lvl],
                k02_realised_ratio=k02_realised_map[lvl],
                k10_swap_rate=k10_rate_map[lvl],
                k10_reshufflable=100,
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)
        k02_realised = audit[audit["check"] == "knob_02_realised_monotonicity"].iloc[0]
        assert k02_realised["status"] == "FAIL"
        # The K2 configured monotonicity row still passes (configured
        # target stays non-decreasing across levels).
        k02_configured = audit[
            audit["check"] == "knob_02_configured_monotonicity"
        ].iloc[0]
        assert k02_configured["status"] == "PASS"

    def test_missing_level_raises(self, tmp_path: Path) -> None:
        """check_monotonicity rejects incomplete variant dir sets."""
        vd = tmp_path / "easy"
        self._seed_variant(vd, k03_drops=[], k_counts={})
        with pytest.raises(ValueError, match="Missing variant dirs"):
            gv.check_monotonicity("companies", {"easy": vd})


class TestK5DistinctFormatFamilies:
    """C3 K5: distinct (transform_fn, target_fmt) families per level."""

    def test_empty_df_returns_zero(self) -> None:
        assert gv._k5_distinct_format_families(pd.DataFrame()) == 0

    def test_missing_params_column_returns_zero(self) -> None:
        df = pd.DataFrame([{"transform_fn": "reformat_date"}])
        assert gv._k5_distinct_format_families(df) == 0

    def test_counts_distinct_pairs_not_rows(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "transform_fn": "reformat_date",
                    "transform_params": '{"target_fmt": "iso"}',
                },
                {
                    "transform_fn": "reformat_date",
                    "transform_params": '{"target_fmt": "iso"}',
                },
                {
                    "transform_fn": "reformat_date",
                    "transform_params": '{"target_fmt": "rfc"}',
                },
                {
                    "transform_fn": "reconvert_unit",
                    "transform_params": '{"target_fmt": "metric"}',
                },
            ]
        )
        # (date, iso), (date, rfc), (unit, metric)  ->  3 families
        assert gv._k5_distinct_format_families(df) == 3

    def test_malformed_json_treated_as_empty_target_fmt(self) -> None:
        df = pd.DataFrame(
            [
                {"transform_fn": "reformat_date", "transform_params": "not-json"},
                {"transform_fn": "reformat_date", "transform_params": ""},
            ]
        )
        # Both rows fold into (reformat_date, "") -> 1 family
        assert gv._k5_distinct_format_families(df) == 1

    def test_reads_operator_target_keys_not_target_fmt(self) -> None:
        """K5 operators write to_format/to_unit/to_locale, never target_fmt.

        Reading only target_fmt collapses every row to (fn, "") and pins
        the family count flat (the music 2/2/2 bug). The helper must read
        the real per-operator token key.
        """
        df = pd.DataFrame(
            [
                {
                    "transform_fn": "reformat_date",
                    "transform_params": (
                        '{"from_format": "iso", "to_format": "eu_dot"}'
                    ),
                },
                {
                    "transform_fn": "reconvert_unit",
                    "transform_params": (
                        '{"from_unit": "seconds", "to_unit": "hh_mm_ss"}'
                    ),
                },
                {
                    "transform_fn": "reconvert_unit",
                    "transform_params": (
                        '{"from_unit": "seconds", "to_unit": "mm_ss"}'
                    ),
                },
                {
                    "transform_fn": "reformat_number",
                    "transform_params": '{"to_locale": "de_DE"}',
                },
            ]
        )
        # (date,eu_dot) (unit,hh_mm_ss) (unit,mm_ss) (number,de_DE) -> 4
        assert gv._k5_distinct_format_families(df) == 4

    def test_same_token_under_different_fns_are_distinct(self) -> None:
        """A shared target token under different operators = distinct families."""
        df = pd.DataFrame(
            [
                {
                    "transform_fn": "append_unit_suffix",
                    "transform_params": '{"to_unit": "GB/s"}',
                },
                {
                    "transform_fn": "reconvert_unit",
                    "transform_params": '{"to_unit": "GB/s"}',
                },
            ]
        )
        # (append_unit_suffix, GB/s) != (reconvert_unit, GB/s) -> 2
        assert gv._k5_distinct_format_families(df) == 2


class TestStatusDowngrades:
    """``_apply_status_downgrades`` keeps the gate honest but not red-forever.

    Advisory proxies downgrade for every domain; documented-weak
    exceptions downgrade only for their listed (domain, check); any
    unlisted FAIL stays FAIL so new regressions still block the gate.
    """

    def _row(self, check: str, status: str = "FAIL") -> list[dict[str, Any]]:
        return [
            {
                "check": check,
                "easy": 1,
                "medium": 2,
                "hard": 3,
                "status": status,
                "detail": "base detail",
            }
        ]

    def test_advisory_check_downgraded_for_any_domain(self) -> None:
        rows = self._row("knob_05_format_prov_rows")
        gv._apply_status_downgrades(rows, "music")
        assert rows[0]["status"] == "WARN"
        assert "[ADVISORY:" in rows[0]["detail"]

        rows2 = self._row("knob_10_realised_monotonicity")
        gv._apply_status_downgrades(rows2, "products")
        assert rows2[0]["status"] == "WARN"
        assert "[ADVISORY:" in rows2[0]["detail"]

    def test_known_weak_music_k2_is_domain_scoped(self) -> None:
        listed = self._row("knob_02_realised_monotonicity")
        gv._apply_status_downgrades(listed, "music")
        assert listed[0]["status"] == "WARN"
        assert "[KNOWN-WEAK EXCEPTION (music)" in listed[0]["detail"]

        # Unlisted domain: the same check still FAILs (new regression).
        other = self._row("knob_02_realised_monotonicity")
        gv._apply_status_downgrades(other, "companies")
        assert other[0]["status"] == "FAIL"

    def test_known_weak_products_k10_rate_is_domain_scoped(self) -> None:
        listed = self._row("knob_10_realised_rate_monotonicity")
        gv._apply_status_downgrades(listed, "products")
        assert listed[0]["status"] == "WARN"

        other = self._row("knob_10_realised_rate_monotonicity")
        gv._apply_status_downgrades(other, "music")
        assert other[0]["status"] == "FAIL"

    def test_known_weak_products_k2_calibration_is_domain_scoped(self) -> None:
        listed = self._row("knob_02_realised_vs_configured")
        gv._apply_status_downgrades(listed, "products")
        assert listed[0]["status"] == "WARN"
        assert "REVISIT in a future variant iteration" in listed[0]["detail"]

        other = self._row("knob_02_realised_vs_configured")
        gv._apply_status_downgrades(other, "music")
        assert other[0]["status"] == "FAIL"

    def test_pass_rows_untouched(self) -> None:
        rows = self._row("knob_05_format_prov_rows", status="PASS")
        gv._apply_status_downgrades(rows, "music")
        assert rows[0]["status"] == "PASS"
        assert rows[0]["detail"] == "base detail"

    def test_unlisted_fail_stays_fail(self) -> None:
        rows = self._row("knob_03_drop_nesting")
        gv._apply_status_downgrades(rows, "music")
        assert rows[0]["status"] == "FAIL"


class TestK10RealisedSwapRate:
    """C3 K10: reading knob_10_realised.csv from a variant dir."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert gv._k10_realised_swap_rate(tmp_path) is None

    def test_empty_csv_returns_none(self, tmp_path: Path) -> None:
        baselines = tmp_path / "output" / "baselines"
        baselines.mkdir(parents=True)
        pd.DataFrame().to_csv(baselines / "knob_10_realised.csv", index=False)
        assert gv._k10_realised_swap_rate(tmp_path) is None

    def test_reads_swap_rate(self, tmp_path: Path) -> None:
        baselines = tmp_path / "output" / "baselines"
        baselines.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "level": "hard",
                    "reshufflable_count": 200,
                    "swap_cells": 90,
                    "swap_rate": 0.45,
                    "compromised_mask_count": 80,
                }
            ]
        ).to_csv(baselines / "knob_10_realised.csv", index=False)
        assert gv._k10_realised_swap_rate(tmp_path) == 0.45


class TestK8NamingIntensity:
    """C3 K8: rung-weighted naming intensity."""

    def test_empty_df_returns_zero(self) -> None:
        assert gv._k8_naming_intensity(pd.DataFrame()) == 0

    def test_descriptive_rows_contribute_zero(self) -> None:
        df = pd.DataFrame([{"transform_fn": "rename_descriptive"} for _ in range(10)])
        assert gv._k8_naming_intensity(df) == 0

    def test_rung_weights_sum(self) -> None:
        df = pd.DataFrame(
            [
                {"transform_fn": "rename_abbreviated"},
                {"transform_fn": "rename_abbreviated"},
                {"transform_fn": "rename_cryptic"},
                {"transform_fn": "rename_anonymize"},
                {"transform_fn": "rename_descriptive"},
            ]
        )
        # 1 + 1 + 2 + 3 + 0 = 7
        assert gv._k8_naming_intensity(df) == 7

    def test_unknown_transform_fn_contributes_zero(self) -> None:
        df = pd.DataFrame([{"transform_fn": "some_other_op"}])
        assert gv._k8_naming_intensity(df) == 0


# ---------------------------------------------------------------------------
# generate_variant tests (with knob runner stubs)
# ---------------------------------------------------------------------------


class _RunnerRecorder:
    """Captures the order in which stub knob runners are invoked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def stub(self, label: str, mutation: dict[str, Any] | None = None):
        """Return a stub knob runner that appends ``label`` on call."""

        def _runner(
            domain: str,
            level: str,
            sources: dict[str, pd.DataFrame],
            work_dir: Path,
            seed: int | None = None,
            **kwargs: Any,
        ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
            self.calls.append(label)
            # Produce one provenance row per stub so packaging can merge
            # it into ``provenance_all.csv``.
            knob_num = int(label[1:])
            prov_name = {
                2: "knob_02_niche.csv",
                4: "knob_04_coverage_skew.csv",
                3: "knob_03_attribute_drop.csv",
                10: "knob_10_reliability.csv",
                8: "knob_08_naming.csv",
            }[knob_num]
            prov_dir = work_dir / "output" / "provenance"
            prov_dir.mkdir(parents=True, exist_ok=True)
            _write_prov_csv(
                prov_dir / prov_name,
                [
                    {
                        "entity_id": f"stub_{knob_num}",
                        "source": "dbpedia",
                        "attribute": "x",
                        "transform_fn": f"stub_{label}",
                    }
                ],
                knob=knob_num,
                level=level,
            )
            # Mutate sources so we can verify propagation.
            new_sources = {
                name: df.assign(**{f"_stub_{label}": 1}) for name, df in sources.items()
            }
            for name, df in new_sources.items():
                df.attrs["dataset_name"] = name
            if label == "k08":
                # K8 must also produce an SM mapping artifact so the
                # packager has something to copy.
                sm_dir = work_dir / "input" / "schemamatching"
                sm_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "source_dataset": ["dbpedia"],
                        "source_column": ["name"],
                        "target_dataset": ["companies"],
                        "target_column": ["name"],
                        "score": [1.0],
                    }
                ).to_csv(sm_dir / "sm_mapping.csv", index=False)
            if label == "k04":
                # The K4 monotonicity check reads the realised coverage
                # histogram from ``baselines/knob_04_realized_vs_target.csv``.
                # Emit a minimal histogram so the orchestrator integration
                # test can exercise that check end-to-end. Same mean
                # across levels so every level passes (non-increasing).
                baselines_dir = work_dir / "output" / "baselines"
                baselines_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    [
                        {
                            "label": f"realised_{level}",
                            "coverage": 2.5,
                            "fraction": 1.0,
                        }
                    ]
                ).to_csv(
                    baselines_dir / "knob_04_realized_vs_target.csv",
                    index=False,
                )
            if label == "k02":
                # K2 audit (step-1 instrumentation) reads
                # ``baselines/knob_02_realised.csv``. Emit identical
                # target+final ratios per level — non-decreasing across
                # easy/medium/hard so the configured + realised
                # monotonicity checks pass under stub runs.
                baselines_dir = work_dir / "output" / "baselines"
                baselines_dir.mkdir(parents=True, exist_ok=True)
                ratio = {"easy": 0.2, "medium": 0.5, "hard": 0.8}.get(level, 0.0)
                pd.DataFrame(
                    [
                        {
                            "level": level,
                            "baseline_ratio": 0.0,
                            "target_ratio": ratio,
                            "final_ratio": ratio,
                            "operator": "interpolate_paired_drop",
                            "removed": 0,
                            "interpolated": 0,
                        }
                    ]
                ).to_csv(baselines_dir / "knob_02_realised.csv", index=False)
            if label == "k10":
                # K10 rate-based audit (step-1 C3 K10) reads
                # ``baselines/knob_10_realised.csv``. Non-decreasing
                # swap_rate so the rate monotonicity check passes.
                baselines_dir = work_dir / "output" / "baselines"
                baselines_dir.mkdir(parents=True, exist_ok=True)
                rate = {"easy": 0.1, "medium": 0.3, "hard": 0.6}.get(level, 0.0)
                pd.DataFrame(
                    [
                        {
                            "level": level,
                            "reshufflable_count": 100,
                            "swap_cells": int(round(rate * 100)),
                            "swap_rate": rate,
                            "compromised_mask_count": int(round(rate * 100)),
                        }
                    ]
                ).to_csv(baselines_dir / "knob_10_realised.csv", index=False)
            params = {"stub_level": level, **(mutation or {})}
            return new_sources, params

        return _runner

    def joint_stub(self):
        """Stub for ``_run_joint`` that records and mutates."""

        def _runner(
            domain: str,
            level: str,
            sources: dict[str, pd.DataFrame],
            work_dir: Path,
            seed: int,
            **kwargs: Any,
        ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
            self.calls.append("joint")
            prov_dir = work_dir / "output" / "provenance"
            prov_dir.mkdir(parents=True, exist_ok=True)

            # Joint stub emits K1's realised CSV alongside K1 provenance
            # (plan_revision.md R-1 / G9 / step 4f). Non-decreasing
            # committed + intensity per level so the K1 monotonicity audit
            # checks pass under stub runs.
            baselines_dir = work_dir / "output" / "baselines"
            baselines_dir.mkdir(parents=True, exist_ok=True)
            committed_per_level = {"easy": 30, "medium": 60, "hard": 120}.get(level, 0)
            edit_per_level = {"easy": 0.2, "medium": 0.4, "hard": 0.6}.get(level, 0.0)
            jaccard_per_level = {"easy": 0.1, "medium": 0.3, "hard": 0.5}.get(
                level, 0.0
            )
            pd.DataFrame(
                [
                    {
                        "level": level,
                        "paraphrase_attempts": committed_per_level + 3,
                        "paraphrase_committed": committed_per_level,
                        "mean_edit_distance": edit_per_level,
                        "mean_token_jaccard_drop": jaccard_per_level,
                        "strict_cache_miss_count": 0,
                    }
                ]
            ).to_csv(baselines_dir / "knob_01_realised.csv", index=False)

            for knob, fname in (
                (1, "knob_01_surface.csv"),
                (5, "knob_05_format_unit.csv"),
                (6, "knob_06_noise.csv"),
            ):
                _write_prov_csv(
                    prov_dir / fname,
                    [
                        {
                            "entity_id": f"joint_{knob}",
                            "source": "forbes",
                            "attribute": "y",
                            "transform_fn": f"stub_knob_{knob}",
                        }
                    ],
                    knob=knob,
                    level=level,
                )
            new_sources = {
                name: df.assign(_stub_joint=1) for name, df in sources.items()
            }
            for name, df in new_sources.items():
                df.attrs["dataset_name"] = name
            params = {
                "knob_01": {"stub_level": level},
                "knob_05": {"stub_level": level},
                "knob_06": {"stub_level": level},
                "collision_audit_pass": True,
            }
            return new_sources, params

        return _runner


class TestGenerateVariantPlumbing:
    """Orchestrator tests using stubbed knob runners."""

    def _patch_runners(
        self,
        monkeypatch: pytest.MonkeyPatch,
        recorder: _RunnerRecorder,
    ) -> None:
        """Replace all per-knob runners with recording stubs."""
        monkeypatch.setattr(gv, "_run_knob_02", recorder.stub("k02"))
        monkeypatch.setattr(gv, "_run_knob_04", recorder.stub("k04"))
        monkeypatch.setattr(gv, "_run_joint", recorder.joint_stub())
        monkeypatch.setattr(gv, "_run_knob_03", recorder.stub("k03"))
        monkeypatch.setattr(gv, "_run_knob_10", recorder.stub("k10"))
        monkeypatch.setattr(gv, "_run_knob_08", recorder.stub("k08"))

    def test_canonical_knob_order_enforced(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Orchestrator invokes knobs in the canonical S1 order."""
        recorder = _RunnerRecorder()
        self._patch_runners(monkeypatch, recorder)

        work = tmp_path / "work"
        variant = tmp_path / "variant"

        gv.generate_variant(
            domain="companies",
            level="easy",
            master_seed=7,
            work_dir=work,
            variant_dir=variant,
            sources_override=_make_tiny_sources(),
        )

        assert recorder.calls == ["k02", "k04", "joint", "k03", "k10", "k08"]

    def test_sources_propagate_through_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Final sources carry the stub markers from every knob stage."""
        recorder = _RunnerRecorder()
        self._patch_runners(monkeypatch, recorder)

        work = tmp_path / "work"
        variant = tmp_path / "variant"

        result = gv.generate_variant(
            domain="companies",
            level="medium",
            master_seed=7,
            work_dir=work,
            variant_dir=variant,
            sources_override=_make_tiny_sources(),
        )

        for name, df in result["final_sources"].items():
            # Every knob stub added a sentinel column.
            for col in (
                "_stub_k02",
                "_stub_k04",
                "_stub_joint",
                "_stub_k03",
                "_stub_k10",
                "_stub_k08",
            ):
                assert col in df.columns, f"{name} missing {col}"

    def test_difficulty_yaml_has_all_knobs_and_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``config/difficulty.yaml`` captures seed + one entry per knob."""
        recorder = _RunnerRecorder()
        self._patch_runners(monkeypatch, recorder)

        work = tmp_path / "work"
        variant = tmp_path / "variant"

        result = gv.generate_variant(
            domain="companies",
            level="hard",
            master_seed=99,
            work_dir=work,
            variant_dir=variant,
            sources_override=_make_tiny_sources(),
        )

        yaml_path = Path(result["difficulty_yaml_path"])
        assert yaml_path.exists()
        loaded = yaml.safe_load(yaml_path.read_text())
        assert loaded["domain"] == "companies"
        assert loaded["level"] == "hard"
        assert loaded["master_seed"] == 99
        for knob_name in (
            "knob_01",
            "knob_02",
            "knob_03",
            "knob_04",
            "knob_05",
            "knob_06",
            "knob_08",
            "knob_10",
        ):
            assert knob_name in loaded["knobs"], f"difficulty.yaml missing {knob_name}"
        assert loaded["knob_order"][0] == "knob_02"
        assert loaded["knob_order"][-1] == "knob_08"

    def test_package_variant_output_structure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The packaged variant directory has the expected layout."""
        recorder = _RunnerRecorder()
        self._patch_runners(monkeypatch, recorder)

        work = tmp_path / "work"
        variant = tmp_path / "variant"

        gv.generate_variant(
            domain="companies",
            level="easy",
            master_seed=1,
            work_dir=work,
            variant_dir=variant,
            sources_override=_make_tiny_sources(),
        )

        # Input data serialised for every source.
        for src in ("dbpedia", "forbes", "fullcontact"):
            assert (variant / "input" / "data" / f"{src}.csv").exists()
        # SM mapping copied from K8.
        assert (variant / "input" / "schemamatching" / "sm_mapping.csv").exists()
        # Fusion gold copied unchanged from the real use case.
        assert (variant / "input" / "fusion" / "test_set.xml").exists()
        # Consolidated provenance with rows from every standard-schema knob.
        merged = pd.read_csv(variant / "output" / "provenance" / "provenance_all.csv")
        knobs_present = set(merged["knob"].astype(int).tolist())
        assert knobs_present == {1, 2, 3, 4, 5, 6, 8, 10}

    def test_all_three_levels_then_monotonicity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Running all three levels end-to-end yields a passing monotonicity audit."""
        recorder = _RunnerRecorder()
        self._patch_runners(monkeypatch, recorder)

        variant_dirs: dict[str, Path] = {}
        for level in ("easy", "medium", "hard"):
            work = tmp_path / "work" / level
            variant = tmp_path / "variant" / level
            gv.generate_variant(
                domain="companies",
                level=level,
                master_seed=5,
                work_dir=work,
                variant_dir=variant,
                sources_override=_make_tiny_sources(),
            )
            variant_dirs[level] = variant

        audit = gv.check_monotonicity("companies", variant_dirs)
        # Stub runners emit one row per level — all checks should pass
        # (drop set is identical across levels, counts are equal).
        fails = audit[audit["status"] == "FAIL"]
        assert fails.empty, f"Unexpected failures: {fails.to_string(index=False)}"


# ---------------------------------------------------------------------------
# K2 non-corner cache: regression for the 2026-05-28 level-gate bug
# ---------------------------------------------------------------------------


class TestK2NonCornerCacheBuiltAtEveryLevel:
    """Regression for the 2026-05-28 dispatch bug.

    Pre-fix, `llm_cache_k2_non_corner` was gated behind
    ``if level_k2 == "hard" or is_aliased:`` so the drop-corner-refill
    cache was only constructed at hard. At easy + medium the cache was
    None, which forced the dispatch in `apply_knob_02_niche` to fall
    through to ``noop_baseline_above_target`` even when
    ``non_corner_refill.enabled`` was true in the domain YAML. Products
    easy + medium silently noop'd K2 on the 2026-05-28 first run.

    The fix builds the cache whenever the domain YAML opts in via
    ``non_corner_refill.enabled``, irrespective of level.
    """

    @pytest.mark.parametrize("level", ["easy", "medium", "hard"])
    def test_non_corner_cache_passed_to_run_knob_02(
        self,
        level: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """`_run_knob_02` must receive a non-None `non_corner_cache`
        kwarg at every level when the domain YAML enables
        non-corner-refill. Companies K2 YAML has ``enabled: true``."""
        recorder = _RunnerRecorder()
        captured: dict[str, Any] = {}

        def k02_capturing_stub(*args: Any, **kwargs: Any) -> Any:
            captured["non_corner_cache"] = kwargs.get("non_corner_cache")
            return recorder.stub("k02")(*args, **kwargs)

        monkeypatch.setattr(gv, "_run_knob_02", k02_capturing_stub)
        monkeypatch.setattr(gv, "_run_knob_04", recorder.stub("k04"))
        monkeypatch.setattr(gv, "_run_joint", recorder.joint_stub())
        monkeypatch.setattr(gv, "_run_knob_03", recorder.stub("k03"))
        monkeypatch.setattr(gv, "_run_knob_10", recorder.stub("k10"))
        monkeypatch.setattr(gv, "_run_knob_08", recorder.stub("k08"))

        gv.generate_variant(
            domain="companies",
            level=level,
            master_seed=11,
            work_dir=tmp_path / "work",
            variant_dir=tmp_path / "variant",
            sources_override=_make_tiny_sources(),
        )

        assert captured["non_corner_cache"] is not None, (
            f"non_corner_cache was None at level={level!r} — the level "
            "gate fix (drop the 'hard or is_aliased' check) regressed."
        )

    @pytest.mark.parametrize("level", ["easy", "medium", "hard"])
    def test_interpolation_cache_passed_to_run_knob_02(
        self,
        level: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """`_run_knob_02` must receive a non-None ``llm_cache``
        (interpolation cache) at EVERY level, not just hard. Pre-fix it was
        gated behind ``level_k2 == 'hard' or is_aliased``, so the
        ``interpolate_paired_drop`` operator no-op'd at medium for
        low-baseline domains (`_run_interpolation` skips on a None cache),
        producing interp=0. Regression guard for the 2026-05-31 K2 cache
        gate fix (the K2 twin of the K1 Fix-B bug)."""
        recorder = _RunnerRecorder()
        captured: dict[str, Any] = {}

        def k02_capturing_stub(*args: Any, **kwargs: Any) -> Any:
            captured["llm_cache"] = kwargs.get("llm_cache")
            return recorder.stub("k02")(*args, **kwargs)

        monkeypatch.setattr(gv, "_run_knob_02", k02_capturing_stub)
        monkeypatch.setattr(gv, "_run_knob_04", recorder.stub("k04"))
        monkeypatch.setattr(gv, "_run_joint", recorder.joint_stub())
        monkeypatch.setattr(gv, "_run_knob_03", recorder.stub("k03"))
        monkeypatch.setattr(gv, "_run_knob_10", recorder.stub("k10"))
        monkeypatch.setattr(gv, "_run_knob_08", recorder.stub("k08"))

        gv.generate_variant(
            domain="companies",
            level=level,
            master_seed=11,
            work_dir=tmp_path / "work",
            variant_dir=tmp_path / "variant",
            sources_override=_make_tiny_sources(),
        )

        assert captured["llm_cache"] is not None, (
            f"K2 interpolation llm_cache was None at level={level!r} — the "
            "hard-only cache gate regressed; interpolate_paired_drop would "
            "no-op (interp=0) at this level."
        )


class TestK1LLMClientWiringAndCache:
    """Regression for the K1 LLM-paraphrase wiring bug (2026-05-30).

    Two coupled defects left K1's ``llm_paraphrase`` operator dead at every
    level: (Fix B) the LLM cache was built only at hard, so medium draws
    skipped as ``llm_cache_missing``; and (Fix A) the client was hardcoded
    ``None`` in ``apply_values_joint``, so even hard misses raised
    ``LLMCacheMiss`` -> ``strict_cache_miss`` identity fallback. The fix
    builds the cache for every level whose operator mix draws
    ``llm_paraphrase`` (config-driven) and wires a live client into K1.
    """

    def test_k1_uses_llm_paraphrase_is_config_driven(self) -> None:
        # companies K1: llm_paraphrase weight 0 @ easy, 1.0 @ medium, 3.0 @ hard.
        assert gv._k1_uses_llm_paraphrase("companies", "easy") is False
        assert gv._k1_uses_llm_paraphrase("companies", "medium") is True
        assert gv._k1_uses_llm_paraphrase("companies", "hard") is True

    @pytest.mark.parametrize(
        "level,expect_client",
        [("easy", False), ("medium", True), ("hard", True)],
    )
    def test_k1_client_wired_to_run_joint(
        self,
        level: str,
        expect_client: bool,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """With an API key set, ``_run_joint`` receives a non-None
        ``api_client_k1`` exactly at the levels that draw the operator
        (medium + hard), and None at easy (operator-free)."""
        recorder = _RunnerRecorder()
        captured: dict[str, Any] = {}
        sentinel = object()

        def fake_build(*, model_id: str) -> Any:
            return sentinel

        monkeypatch.setattr(gv, "build_openai_paraphrase_client", fake_build)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        joint = recorder.joint_stub()

        def joint_capturing_stub(*args: Any, **kwargs: Any) -> Any:
            captured["api_client_k1"] = kwargs.get("api_client_k1")
            return joint(*args, **kwargs)

        monkeypatch.setattr(gv, "_run_knob_02", recorder.stub("k02"))
        monkeypatch.setattr(gv, "_run_knob_04", recorder.stub("k04"))
        monkeypatch.setattr(gv, "_run_joint", joint_capturing_stub)
        monkeypatch.setattr(gv, "_run_knob_03", recorder.stub("k03"))
        monkeypatch.setattr(gv, "_run_knob_10", recorder.stub("k10"))
        monkeypatch.setattr(gv, "_run_knob_08", recorder.stub("k08"))

        gv.generate_variant(
            domain="companies",
            level=level,
            master_seed=11,
            work_dir=tmp_path / "work",
            variant_dir=tmp_path / "variant",
            sources_override=_make_tiny_sources(),
        )

        if expect_client:
            assert (
                captured["api_client_k1"] is sentinel
            ), f"K1 client not wired at level={level!r} — Fix A/B regressed."
        else:
            assert captured["api_client_k1"] is None, (
                "K1 client should be None at easy (no llm_paraphrase op); "
                f"got {captured['api_client_k1']!r}"
            )

    def test_no_client_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Without OPENAI_API_KEY the client stays None even at hard; the
        cache then serves pre-baked replays only and a miss degrades to a
        deterministic operator."""
        recorder = _RunnerRecorder()
        captured: dict[str, Any] = {}
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        def boom(*, model_id: str) -> Any:  # pragma: no cover - must not run
            raise AssertionError("client must not be built without an API key")

        monkeypatch.setattr(gv, "build_openai_paraphrase_client", boom)

        joint = recorder.joint_stub()

        def joint_capturing_stub(*args: Any, **kwargs: Any) -> Any:
            captured["api_client_k1"] = kwargs.get("api_client_k1")
            return joint(*args, **kwargs)

        monkeypatch.setattr(gv, "_run_knob_02", recorder.stub("k02"))
        monkeypatch.setattr(gv, "_run_knob_04", recorder.stub("k04"))
        monkeypatch.setattr(gv, "_run_joint", joint_capturing_stub)
        monkeypatch.setattr(gv, "_run_knob_03", recorder.stub("k03"))
        monkeypatch.setattr(gv, "_run_knob_10", recorder.stub("k10"))
        monkeypatch.setattr(gv, "_run_knob_08", recorder.stub("k08"))

        gv.generate_variant(
            domain="companies",
            level="hard",
            master_seed=11,
            work_dir=tmp_path / "work",
            variant_dir=tmp_path / "variant",
            sources_override=_make_tiny_sources(),
        )

        assert captured["api_client_k1"] is None


# ---------------------------------------------------------------------------
# K1 realised audit (plan_revision.md R-1 / G9 / step 4f)
# ---------------------------------------------------------------------------


class TestK1RealisedMetricsReader:
    """``_k1_realised_metrics`` reads ``knob_01_realised.csv`` per variant dir."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert gv._k1_realised_metrics(tmp_path) is None

    def test_empty_csv_returns_none(self, tmp_path: Path) -> None:
        baselines = tmp_path / "output" / "baselines"
        baselines.mkdir(parents=True)
        pd.DataFrame().to_csv(baselines / "knob_01_realised.csv", index=False)
        assert gv._k1_realised_metrics(tmp_path) is None

    def test_reads_full_row(self, tmp_path: Path) -> None:
        baselines = tmp_path / "output" / "baselines"
        baselines.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "level": "hard",
                    "paraphrase_attempts": 120,
                    "paraphrase_committed": 95,
                    "mean_edit_distance": 0.42,
                    "mean_token_jaccard_drop": 0.31,
                    "strict_cache_miss_count": 5,
                }
            ]
        ).to_csv(baselines / "knob_01_realised.csv", index=False)
        metrics = gv._k1_realised_metrics(tmp_path)
        assert metrics is not None
        assert metrics["paraphrase_attempts"] == 120
        assert metrics["paraphrase_committed"] == 95
        assert metrics["mean_edit_distance"] == 0.42
        assert metrics["mean_token_jaccard_drop"] == 0.31
        assert metrics["strict_cache_miss_count"] == 5

    def test_missing_column_returns_none(self, tmp_path: Path) -> None:
        baselines = tmp_path / "output" / "baselines"
        baselines.mkdir(parents=True)
        # Drop the mean_edit_distance column to simulate a malformed file.
        pd.DataFrame(
            [
                {
                    "level": "hard",
                    "paraphrase_attempts": 10,
                    "paraphrase_committed": 5,
                    "mean_token_jaccard_drop": 0.1,
                    "strict_cache_miss_count": 0,
                }
            ]
        ).to_csv(baselines / "knob_01_realised.csv", index=False)
        assert gv._k1_realised_metrics(tmp_path) is None


class TestK1AuditRowsInCheckMonotonicity:
    """End-to-end coverage for ``knob_01_realised_*_monotonicity`` rows."""

    def _seed_variant_with_k1(
        self,
        variant_dir: Path,
        *,
        committed: int,
        attempts: int | None = None,
        mean_edit: float,
        mean_jaccard_drop: float,
        strict_cache_miss: int = 0,
    ) -> None:
        """Seed a minimal variant dir with K1 + K2 + K10 realised CSVs.

        K2 and K10 must seed monotone defaults so this test isolates
        the K1 audit row from other audit FAILs.
        """
        prov_dir = variant_dir / "output" / "provenance"
        prov_dir.mkdir(parents=True, exist_ok=True)
        baselines_dir = variant_dir / "output" / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)

        # Empty prov files for the count-based checks; K3 has no drops.
        for fname in (
            "knob_02_niche.csv",
            "knob_03_attribute_drop.csv",
            "knob_04_coverage_skew.csv",
            "knob_05_format_unit.csv",
            "knob_06_noise.csv",
            "knob_08_naming.csv",
            "knob_10_reliability.csv",
        ):
            _write_prov_csv(prov_dir / fname, [], knob=0, level=variant_dir.name)

        # K1 realised CSV (the file under test).
        pd.DataFrame(
            [
                {
                    "level": variant_dir.name,
                    "paraphrase_attempts": int(
                        attempts if attempts is not None else committed
                    ),
                    "paraphrase_committed": int(committed),
                    "mean_edit_distance": float(mean_edit),
                    "mean_token_jaccard_drop": float(mean_jaccard_drop),
                    "strict_cache_miss_count": int(strict_cache_miss),
                }
            ]
        ).to_csv(baselines_dir / "knob_01_realised.csv", index=False)

    def test_monotone_k1_passes_both_checks(self, tmp_path: Path) -> None:
        dirs: dict[str, Path] = {}
        committed_map = {"easy": 30, "medium": 60, "hard": 120}
        edit_map = {"easy": 0.1, "medium": 0.3, "hard": 0.5}
        jaccard_map = {"easy": 0.05, "medium": 0.2, "hard": 0.4}
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant_with_k1(
                vd,
                committed=committed_map[lvl],
                mean_edit=edit_map[lvl],
                mean_jaccard_drop=jaccard_map[lvl],
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)

        rate = audit[audit["check"] == "knob_01_realised_rate_monotonicity"].iloc[0]
        assert rate["status"] == "PASS"
        assert rate["easy"] == 30
        assert rate["medium"] == 60
        assert rate["hard"] == 120

        intensity = audit[
            audit["check"] == "knob_01_realised_intensity_monotonicity"
        ].iloc[0]
        assert intensity["status"] == "PASS"
        # All three levels clear the floor, so the comparison spans all of
        # them (behaviour is byte-identical to the pre-floor check here).
        assert "compared=['easy', 'medium', 'hard']" in intensity["detail"]

    def test_flat_committed_fails_rate_check(self, tmp_path: Path) -> None:
        """K1 cache-miss dormancy: committed flat across levels → rate FAIL."""
        dirs: dict[str, Path] = {}
        edit_map = {"easy": 0.1, "medium": 0.3, "hard": 0.5}
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant_with_k1(
                vd,
                committed=10,  # flat across levels
                mean_edit=edit_map[lvl],
                mean_jaccard_drop=0.2,
                strict_cache_miss=200 if lvl == "hard" else 0,
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)

        # Rate check: committed is flat (10/10/10), so non-decreasing
        # holds → PASS (flat is allowed). This test pins the contract:
        # the rate check is non-decreasing, so dormancy (flat across
        # levels) is intentionally accepted as a passing state at this
        # level; the FAIL trigger is strictly decreasing committed.
        # Strict-cache miss count is surfaced in the detail for visual
        # inspection of dormancy regardless.
        rate = audit[audit["check"] == "knob_01_realised_rate_monotonicity"].iloc[0]
        assert rate["status"] == "PASS"
        assert "strict_cache_miss easy=0 medium=0 hard=200" in rate["detail"]

        # Flat committed at 10/10/10 is below K1_INTENSITY_MIN_COMMITTED at
        # every level, so <2 levels qualify and the intensity check FAILs —
        # this is the gate-safe backstop for all-level dormancy that the
        # (weak, non-decreasing) rate check intentionally lets pass.
        intensity = audit[
            audit["check"] == "knob_01_realised_intensity_monotonicity"
        ].iloc[0]
        assert intensity["status"] == "FAIL"
        assert "intensity not assessable" in intensity["detail"]

    def test_decreasing_committed_fails_rate_check(self, tmp_path: Path) -> None:
        dirs: dict[str, Path] = {}
        for lvl, committed in (("easy", 50), ("medium", 25), ("hard", 10)):
            vd = tmp_path / lvl
            self._seed_variant_with_k1(
                vd,
                committed=committed,
                mean_edit=0.3,
                mean_jaccard_drop=0.2,
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)
        rate = audit[audit["check"] == "knob_01_realised_rate_monotonicity"].iloc[0]
        assert rate["status"] == "FAIL"
        assert rate["easy"] == 50
        assert rate["hard"] == 10

    def test_shallow_paraphrase_fails_intensity_check(self, tmp_path: Path) -> None:
        """Rate grows but intensity flat → intensity FAIL."""
        dirs: dict[str, Path] = {}
        committed_map = {"easy": 30, "medium": 60, "hard": 120}
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant_with_k1(
                vd,
                committed=committed_map[lvl],
                mean_edit=0.05,  # flat — every paraphrase is shallow
                mean_jaccard_drop=0.05,
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)
        intensity = audit[
            audit["check"] == "knob_01_realised_intensity_monotonicity"
        ].iloc[0]
        # Flat intensity (0.05 / 0.05 / 0.05) is non-decreasing in the
        # weak sense — the check PASSes. The shallowness shows up as
        # very low absolute values, not as a slope FAIL.
        assert intensity["status"] == "PASS"

    def test_inverted_intensity_fails_check(self, tmp_path: Path) -> None:
        """edit_distance decreases easy → hard → intensity FAIL."""
        dirs: dict[str, Path] = {}
        committed_map = {"easy": 30, "medium": 60, "hard": 120}
        edit_map = {"easy": 0.5, "medium": 0.3, "hard": 0.1}  # inverted
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant_with_k1(
                vd,
                committed=committed_map[lvl],
                mean_edit=edit_map[lvl],
                mean_jaccard_drop=0.2,
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)
        intensity = audit[
            audit["check"] == "knob_01_realised_intensity_monotonicity"
        ].iloc[0]
        assert intensity["status"] == "FAIL"
        assert "edit_ok=False" in intensity["detail"]

    def test_inverted_jaccard_drop_fails_check(self, tmp_path: Path) -> None:
        """token_jaccard_drop decreases → intensity FAIL even if edit grows."""
        dirs: dict[str, Path] = {}
        committed_map = {"easy": 30, "medium": 60, "hard": 120}
        edit_map = {"easy": 0.1, "medium": 0.3, "hard": 0.5}  # monotone up
        jaccard_map = {"easy": 0.5, "medium": 0.3, "hard": 0.1}  # inverted
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant_with_k1(
                vd,
                committed=committed_map[lvl],
                mean_edit=edit_map[lvl],
                mean_jaccard_drop=jaccard_map[lvl],
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)
        intensity = audit[
            audit["check"] == "knob_01_realised_intensity_monotonicity"
        ].iloc[0]
        assert intensity["status"] == "FAIL"
        assert "jaccard_ok=False" in intensity["detail"]

    def test_undersampled_level_excluded_from_intensity(self, tmp_path: Path) -> None:
        """Music regression: easy commits too few cells to measure intensity.

        easy=2 cells with a fluke-high mean (edit 0.9) must not fail the
        check; it is excluded and only the well-sampled medium/hard are
        compared. Mirrors music easy=2/edit≈0.405 vs medium/hard≈0.346.
        """
        dirs: dict[str, Path] = {}
        committed_map = {"easy": 2, "medium": 5400, "hard": 12886}
        edit_map = {"easy": 0.9, "medium": 0.345, "hard": 0.347}
        jaccard_map = {"easy": 0.9, "medium": 0.432, "hard": 0.506}
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant_with_k1(
                vd,
                committed=committed_map[lvl],
                mean_edit=edit_map[lvl],
                mean_jaccard_drop=jaccard_map[lvl],
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)
        intensity = audit[
            audit["check"] == "knob_01_realised_intensity_monotonicity"
        ].iloc[0]
        assert intensity["status"] == "PASS"
        assert "compared=['medium', 'hard']" in intensity["detail"]
        assert "excluded=['easy']" in intensity["detail"]

    def test_fewer_than_two_qualifying_levels_fails(self, tmp_path: Path) -> None:
        """Only one level clears the floor -> intensity FAILs (gate-safe)."""
        dirs: dict[str, Path] = {}
        committed_map = {"easy": 2, "medium": 5, "hard": 120}
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant_with_k1(
                vd,
                committed=committed_map[lvl],
                mean_edit=0.3,
                mean_jaccard_drop=0.3,
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)
        # committed 2<=5<=120 is non-decreasing, so rate PASSes; the
        # dormancy at easy+medium is caught only by the intensity FAIL.
        rate = audit[audit["check"] == "knob_01_realised_rate_monotonicity"].iloc[0]
        assert rate["status"] == "PASS"
        intensity = audit[
            audit["check"] == "knob_01_realised_intensity_monotonicity"
        ].iloc[0]
        assert intensity["status"] == "FAIL"
        assert "intensity not assessable" in intensity["detail"]
        assert "only 1 level(s)" in intensity["detail"]

    def test_floor_boundary_committed_is_inclusive(self, tmp_path: Path) -> None:
        """committed == floor qualifies (the >= boundary is inclusive)."""
        dirs: dict[str, Path] = {}
        floor = gv.K1_INTENSITY_MIN_COMMITTED
        committed_map = {"easy": floor, "medium": floor + 10, "hard": floor + 20}
        edit_map = {"easy": 0.1, "medium": 0.3, "hard": 0.5}
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            self._seed_variant_with_k1(
                vd,
                committed=committed_map[lvl],
                mean_edit=edit_map[lvl],
                mean_jaccard_drop=edit_map[lvl],
            )
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)
        intensity = audit[
            audit["check"] == "knob_01_realised_intensity_monotonicity"
        ].iloc[0]
        assert intensity["status"] == "PASS"
        assert "compared=['easy', 'medium', 'hard']" in intensity["detail"]

    def test_missing_k1_csv_fails_both_with_detail(self, tmp_path: Path) -> None:
        """Variant dirs without knob_01_realised.csv → both K1 audits FAIL."""
        dirs: dict[str, Path] = {}
        for lvl in ("easy", "medium", "hard"):
            vd = tmp_path / lvl
            # Seed everything EXCEPT the K1 realised CSV.
            prov_dir = vd / "output" / "provenance"
            prov_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                "knob_02_niche.csv",
                "knob_03_attribute_drop.csv",
                "knob_04_coverage_skew.csv",
                "knob_05_format_unit.csv",
                "knob_06_noise.csv",
                "knob_08_naming.csv",
                "knob_10_reliability.csv",
            ):
                _write_prov_csv(prov_dir / fname, [], knob=0, level=lvl)
            dirs[lvl] = vd

        audit = gv.check_monotonicity("companies", dirs)
        rate = audit[audit["check"] == "knob_01_realised_rate_monotonicity"].iloc[0]
        intensity = audit[
            audit["check"] == "knob_01_realised_intensity_monotonicity"
        ].iloc[0]
        assert rate["status"] == "FAIL"
        assert intensity["status"] == "FAIL"
        assert "knob_01_realised.csv missing" in rate["detail"]
        assert "knob_01_realised.csv missing" in intensity["detail"]

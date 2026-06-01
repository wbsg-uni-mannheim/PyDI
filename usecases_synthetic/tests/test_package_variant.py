"""Tests for variant packaging helpers.

R10-F (2026-05-29): ``copy_regenerated_em`` previously globbed
``*_regenerated.csv`` — a suffix the C11 regen writer never emits — so
the per-pair per-split ``baseline_pruned`` / ``corner_filled`` EM gold
files silently never reached the packaged variant directory, and every
dual-test surface fell back to the baseline gold. These tests pin the
correct glob.
"""

from __future__ import annotations

from pathlib import Path

from usecases_synthetic.scripts.package_variant import copy_regenerated_em


def _seed_work_dir(work_dir: Path, names: list[str]) -> Path:
    """Create ``work_dir/input/entitymatching`` with the given CSV files."""
    em = work_dir / "input" / "entitymatching"
    em.mkdir(parents=True, exist_ok=True)
    for name in names:
        (em / name).write_text("id1,id2,label\na,b,true\n", encoding="utf-8")
    return em


class TestCopyRegeneratedEm:
    def test_copies_both_versions_for_all_splits(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "work"
        pair = "products_1_2_products_2"
        names = [
            f"{pair}_{split}_{version}.csv"
            for split in ("train", "val", "test")
            for version in ("baseline_pruned", "corner_filled")
        ]
        _seed_work_dir(work_dir, names)
        em_out = tmp_path / "out"
        em_out.mkdir()

        copied = copy_regenerated_em(work_dir, em_out)

        assert set(copied) == set(names)
        for name in names:
            assert (em_out / name).exists()

    def test_ignores_stale_regenerated_suffix_and_original_gold(
        self, tmp_path: Path
    ) -> None:
        work_dir = tmp_path / "work"
        pair = "products_1_2_products_2"
        keep = [f"{pair}_test_baseline_pruned.csv", f"{pair}_test_corner_filled.csv"]
        ignore = [
            f"{pair}_test_regenerated.csv",  # legacy suffix, never emitted now
            f"{pair}_test.csv",  # original baseline gold (copied elsewhere)
            f"{pair}_all.csv",
        ]
        _seed_work_dir(work_dir, keep + ignore)
        em_out = tmp_path / "out"
        em_out.mkdir()

        copied = copy_regenerated_em(work_dir, em_out)

        assert set(copied) == set(keep)
        for name in ignore:
            assert not (em_out / name).exists()

    def test_no_double_copy_when_patterns_would_overlap(self, tmp_path: Path) -> None:
        """Each file is copied at most once even if globs overlapped."""
        work_dir = tmp_path / "work"
        names = ["p_test_baseline_pruned.csv", "p_test_corner_filled.csv"]
        _seed_work_dir(work_dir, names)
        em_out = tmp_path / "out"
        em_out.mkdir()

        copied = copy_regenerated_em(work_dir, em_out)

        assert sorted(copied) == sorted(names)
        assert len(copied) == len(set(copied))

    def test_missing_source_dir_returns_empty(self, tmp_path: Path) -> None:
        copied = copy_regenerated_em(tmp_path / "nonexistent", tmp_path / "out")
        assert copied == []

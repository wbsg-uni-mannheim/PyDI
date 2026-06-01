"""Tests for the joint value-perturbation orchestrator (Module 7).

Acceptance criteria (from ``plans/module_07_joint_values.md``):

1. No ``(entity_id, source, attribute)`` triple appears in more than one
   of K1/K5 provenance outputs.
2. K4-fabricated cells appear in K6 provenance but NOT in K1/K5
   provenance.
3. Total provenance rows = K1 + K5 + K6 (no gaps, no overlaps except
   the K6-on-K4-fabricated exception).
4. Order enforcement: K1 runs before K5 runs before K6 (verified by
   the collision rules — K5 sees K1's provenance, K6 sees both).
5. ``pytest usecases_synthetic/tests/test_joint_values.py -v`` passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from usecases_synthetic.lib.collision_index import CollisionIndex
from usecases_synthetic.lib.provenance import ProvenanceLog
from usecases_synthetic.scripts.apply_values_joint import (
    _audit_collisions,
    apply_values_joint,
)

# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def joint_sources() -> dict[str, pd.DataFrame]:
    """Small companies-schema DataFrames with columns from K1/K5/K6 configs.

    Column names match the K5 ``config/knob_05_format/companies.yaml``
    schema refresh of 2026-05-04 (source-raw names, not canonical):

    - K1-managed: ``name``/``Company``, ``countryName``/``Country``,
      ``cityName``/``locality``, ``industryName``/``Industry``
    - K5-managed: ``established``, ``total_assets_val``, ``annual_income``
      (dbpedia); ``asset_value``, ``sales_figure`` (forbes);
      ``Attribute_6`` (fullcontact founded date)
    - K6-managed: overlaps with both K1 and K5 (primaries, keys, secondaries)
    """
    rng = np.random.default_rng(99)
    n = 20

    base_names = [f"Company_{i}" for i in range(n)]
    countries = rng.choice(
        ["United States", "Germany", "Japan", "China", "Brazil"], size=n
    ).tolist()
    cities = rng.choice(
        ["New York", "Berlin", "Tokyo", "Beijing", "Sao Paulo"], size=n
    ).tolist()
    industries = rng.choice(
        ["Software", "Automotive", "Pharmaceuticals", "Finance"], size=n
    ).tolist()

    dbpedia = pd.DataFrame(
        {
            "id": [f"db_{i}" for i in range(n)],
            "name": base_names,
            "countryName": countries,
            "cityName": cities,
            "industryName": industries,
            "annual_income": [f"{1000 + i * 100}" for i in range(n)],
            "total_assets_val": [f"{5000 + i * 200}" for i in range(n)],
            "established": [f"2020-{(i % 12) + 1:02d}-15" for i in range(n)],
        }
    )
    dbpedia.attrs["dataset_name"] = "dbpedia"

    forbes = pd.DataFrame(
        {
            "id": [f"fb_{i}" for i in range(n)],
            "Company": [name.replace("_", " ") for name in base_names],
            "Country": countries,
            "Sector": industries,
            "Industry": industries,
            "Market Value": [f"{10 + i}" for i in range(n)],
            "sales_figure": [f"{5 + i * 0.5}" for i in range(n)],
            "Profits": [f"{1 + i * 0.1}" for i in range(n)],
            "asset_value": [f"{50 + i * 2}" for i in range(n)],
        }
    )
    forbes.attrs["dataset_name"] = "forbes"

    fullcontact = pd.DataFrame(
        {
            "id": [f"fc_{i}" for i in range(n)],
            "name": base_names,
            "country": countries,
            "locality": cities,
            "onlinesince": [f"201{i % 10}-06-01" for i in range(n)],
            "Attribute_6": [f"{1950 + i}-01-01" for i in range(n)],
        }
    )
    fullcontact.attrs["dataset_name"] = "fullcontact"

    return {"dbpedia": dbpedia, "forbes": forbes, "fullcontact": fullcontact}


def _write_k4_fabricated_provenance(
    prov_dir: Path,
    entries: list[tuple[str, str, str, str]],
) -> None:
    """Seed the provenance directory with a fake K4 fabrication log.

    Parameters
    ----------
    prov_dir : Path
        The ``output/provenance`` directory (will be created).
    entries : list of (entity_id, source, attribute, new_value)
        Cells to mark as K4-fabricated.
    """
    prov_dir.mkdir(parents=True, exist_ok=True)
    fake_prov = ProvenanceLog(knob=4, level="medium")
    for entity_id, source, attribute, new_value in entries:
        fake_prov.append(
            entity_id=entity_id,
            source=source,
            attribute=attribute,
            original_value="",
            new_value=new_value,
            transform_fn="fabricate_coverage",
            transform_params={"k4_fabricated": True},
        )
    fake_prov.flush(prov_dir / "knob_04_coverage.csv")


# ---- Unit tests: _audit_collisions -----------------------------------------


class TestAuditCollisions:
    """Unit tests for ``_audit_collisions``."""

    def _make_prov_df(
        self, rows: list[tuple[str, str, str]], knob: int
    ) -> pd.DataFrame:
        """Build a minimal provenance DataFrame for audit testing."""
        prov = ProvenanceLog(knob=knob, level="medium")
        for entity_id, source, attribute in rows:
            prov.append(
                entity_id=entity_id,
                source=source,
                attribute=attribute,
                original_value="x",
                new_value="y",
                transform_fn=f"test_knob_{knob}",
                transform_params={},
            )
        return pd.DataFrame(
            [r.as_dict() for r in prov._rows],
        )

    def _empty_collision_index(self, tmp_path: Path) -> CollisionIndex:
        prov_dir = tmp_path / "output" / "provenance"
        prov_dir.mkdir(parents=True)
        return CollisionIndex(prov_dir)

    def test_all_disjoint_passes(self, tmp_path: Path) -> None:
        """Distinct cells per knob — audit passes."""
        prov_k1 = self._make_prov_df([("e1", "s", "a")], knob=1)
        prov_k5 = self._make_prov_df([("e1", "s", "b")], knob=5)
        prov_k6 = self._make_prov_df([("e1", "s", "c")], knob=6)

        ci = self._empty_collision_index(tmp_path)
        ci.reload()

        audit = _audit_collisions(prov_k1, prov_k5, prov_k6, ci)
        failures = audit[audit["status"] == "FAIL"]
        assert failures.empty, f"Unexpected failures: {failures.to_dict()}"

    def test_k1_k5_overlap_fails(self, tmp_path: Path) -> None:
        """K1/K5 cell collision flagged as FAIL."""
        prov_k1 = self._make_prov_df([("e1", "s", "a")], knob=1)
        prov_k5 = self._make_prov_df([("e1", "s", "a")], knob=5)
        prov_k6 = self._make_prov_df([], knob=6)

        ci = self._empty_collision_index(tmp_path)
        ci.reload()

        audit = _audit_collisions(prov_k1, prov_k5, prov_k6, ci)
        row = audit[audit["check"] == "k1_k5_disjoint"].iloc[0]
        assert row["status"] == "FAIL"
        assert "1 overlapping" in row["detail"]

    def test_k4_fabricated_in_k1_fails(self, tmp_path: Path) -> None:
        """K4-fabricated cell touched by K1 flagged as FAIL."""
        prov_dir = tmp_path / "output" / "provenance"
        _write_k4_fabricated_provenance(prov_dir, [("e1", "s", "fab_col", "v")])
        ci = CollisionIndex(prov_dir)
        ci.reload()

        prov_k1 = self._make_prov_df([("e1", "s", "fab_col")], knob=1)
        prov_k5 = self._make_prov_df([], knob=5)
        prov_k6 = self._make_prov_df([], knob=6)

        audit = _audit_collisions(prov_k1, prov_k5, prov_k6, ci)
        row = audit[audit["check"] == "k4_fabricated_not_in_k1"].iloc[0]
        assert row["status"] == "FAIL"

    def test_k4_fabricated_in_k6_allowed(self, tmp_path: Path) -> None:
        """K4-fabricated cells in K6 provenance are permitted."""
        prov_dir = tmp_path / "output" / "provenance"
        _write_k4_fabricated_provenance(prov_dir, [("e1", "s", "fab_col", "v")])
        ci = CollisionIndex(prov_dir)
        ci.reload()

        prov_k1 = self._make_prov_df([], knob=1)
        prov_k5 = self._make_prov_df([], knob=5)
        prov_k6 = self._make_prov_df([("e1", "s", "fab_col")], knob=6)

        audit = _audit_collisions(prov_k1, prov_k5, prov_k6, ci)
        failures = audit[audit["status"] == "FAIL"]
        assert failures.empty


# ---- Integration tests: apply_values_joint ---------------------------------


class TestApplyValuesJoint:
    """Integration tests for the joint orchestrator."""

    def test_medium_no_collisions_and_audit_passes(
        self,
        joint_sources: dict[str, pd.DataFrame],
        tmp_path: Path,
    ) -> None:
        """End-to-end medium run: audit passes; K1/K5/K6 provenance is disjoint."""
        output_dir = tmp_path

        result = apply_values_joint(
            domain="companies",
            level="medium",
            sources=joint_sources,
            output_dir=output_dir,
            seed=42,
        )

        audit = result["audit"]
        failures = audit[audit["status"] == "FAIL"]
        assert failures.empty, f"Joint audit failed: {failures.to_string(index=False)}"

        # Verify disjoint K1/K5 cells (acceptance criterion 1).
        prov_k1 = result["provenance_k1"]
        prov_k5 = result["provenance_k5"]
        s1 = {
            (r["entity_id"], r["source"], r["attribute"]) for _, r in prov_k1.iterrows()
        }
        s5 = {
            (r["entity_id"], r["source"], r["attribute"]) for _, r in prov_k5.iterrows()
        }
        assert not (s1 & s5), "K1 and K5 touched overlapping cells"

    def test_api_client_k1_forwarded_to_k1(
        self,
        joint_sources: dict[str, pd.DataFrame],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fix A: ``apply_values_joint`` forwards ``api_client_k1`` to K1's
        ``llm_client`` argument. Pre-fix this was hardcoded ``None``, so the
        live paraphrase client never reached the operator."""
        import usecases_synthetic.scripts.apply_values_joint as avj

        captured: dict[str, Any] = {}

        def sentinel_client(_template: str, _value: str) -> str:
            return "x"

        class _StopAfterK1(Exception):
            pass

        def capturing_apply_knob_01(*args: Any, **kwargs: Any) -> Any:
            captured["llm_client"] = kwargs.get("llm_client")
            raise _StopAfterK1

        monkeypatch.setattr(avj, "apply_knob_01", capturing_apply_knob_01)

        with pytest.raises(_StopAfterK1):
            avj.apply_values_joint(
                domain="companies",
                level="medium",
                sources=joint_sources,
                output_dir=tmp_path,
                seed=42,
                api_client_k1=sentinel_client,
            )

        assert captured["llm_client"] is sentinel_client

    def test_k1_client_defaults_to_none(
        self,
        joint_sources: dict[str, pd.DataFrame],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When no client is supplied, K1 stays in strict-cache-only mode
        (``llm_client=None``) — the documented fallback behaviour."""
        import usecases_synthetic.scripts.apply_values_joint as avj

        captured: dict[str, Any] = {}

        class _StopAfterK1(Exception):
            pass

        def capturing_apply_knob_01(*args: Any, **kwargs: Any) -> Any:
            captured["llm_client"] = kwargs.get("llm_client")
            raise _StopAfterK1

        monkeypatch.setattr(avj, "apply_knob_01", capturing_apply_knob_01)

        with pytest.raises(_StopAfterK1):
            avj.apply_values_joint(
                domain="companies",
                level="medium",
                sources=joint_sources,
                output_dir=tmp_path,
                seed=42,
            )

        assert captured["llm_client"] is None

    def test_k4_fabricated_only_in_k6(
        self,
        joint_sources: dict[str, pd.DataFrame],
        tmp_path: Path,
    ) -> None:
        """Pre-seeded K4-fabricated cells must be ignored by K1/K5 and may be touched by K6."""
        output_dir = tmp_path
        prov_dir = output_dir / "output" / "provenance"

        # Seed K4-fabricated cells on a K1-managed column (dbpedia.name)
        # and a K5-managed column (dbpedia.annual_income, per the
        # 2026-05-04 K5 schema refresh) and a K6-overlap column
        # (forbes.Country).
        _write_k4_fabricated_provenance(
            prov_dir,
            [
                ("db_0", "dbpedia", "name", "Fabricated Co"),
                ("db_1", "dbpedia", "annual_income", "9999"),
                ("fb_2", "forbes", "Country", "Fabricated Land"),
            ],
        )

        result = apply_values_joint(
            domain="companies",
            level="hard",
            sources=joint_sources,
            output_dir=output_dir,
            seed=42,
        )

        prov_k1 = result["provenance_k1"]
        prov_k5 = result["provenance_k5"]
        prov_k6 = result["provenance_k6"]

        def _triples(df: pd.DataFrame) -> set[tuple[str, str, str]]:
            if df.empty:
                return set()
            return {
                (str(r["entity_id"]), str(r["source"]), str(r["attribute"]))
                for _, r in df.iterrows()
            }

        k4_cells = {
            ("db_0", "dbpedia", "name"),
            ("db_1", "dbpedia", "annual_income"),
            ("fb_2", "forbes", "Country"),
        }

        # Acceptance criterion 2: K4-fab cells not in K1 / K5 provenance.
        assert not (_triples(prov_k1) & k4_cells), "K1 touched K4-fabricated cells"
        assert not (_triples(prov_k5) & k4_cells), "K5 touched K4-fabricated cells"

        # Audit must still pass.
        audit = result["audit"]
        failures = audit[audit["status"] == "FAIL"]
        assert failures.empty, f"Joint audit failed: {failures.to_string(index=False)}"

    def test_provenance_files_written(
        self,
        joint_sources: dict[str, pd.DataFrame],
        tmp_path: Path,
    ) -> None:
        """All six per-knob CSVs + audit CSV are written under provenance/."""
        output_dir = tmp_path
        apply_values_joint(
            domain="companies",
            level="medium",
            sources=joint_sources,
            output_dir=output_dir,
            seed=42,
        )

        prov_dir = output_dir / "output" / "provenance"
        expected = [
            "knob_01_surface.csv",
            "knob_01_skipped.csv",
            "knob_05_format_unit.csv",
            "knob_05_skipped.csv",
            "knob_06_noise.csv",
            "knob_06_skipped.csv",
            "joint_values_audit.csv",
        ]
        for name in expected:
            assert (prov_dir / name).exists(), f"Missing output file: {name}"

    def test_determinism_same_seed(
        self,
        joint_sources: dict[str, pd.DataFrame],
        tmp_path: Path,
    ) -> None:
        """Same seed + same input = identical provenance for both runs."""
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        out1.mkdir()
        out2.mkdir()

        # Clone sources to avoid any in-place mutation between runs.
        sources1 = {k: v.copy(deep=True) for k, v in joint_sources.items()}
        for k, v in sources1.items():
            v.attrs = joint_sources[k].attrs.copy()
        sources2 = {k: v.copy(deep=True) for k, v in joint_sources.items()}
        for k, v in sources2.items():
            v.attrs = joint_sources[k].attrs.copy()

        r1 = apply_values_joint(
            domain="companies",
            level="medium",
            sources=sources1,
            output_dir=out1,
            seed=42,
        )
        r2 = apply_values_joint(
            domain="companies",
            level="medium",
            sources=sources2,
            output_dir=out2,
            seed=42,
        )

        for key in ("provenance_k1", "provenance_k5", "provenance_k6"):
            df1 = r1[key].reset_index(drop=True)
            df2 = r2[key].reset_index(drop=True)
            pd.testing.assert_frame_equal(df1, df2)

    def test_k5_defensive_skip_with_k1_cells(
        self,
        joint_sources: dict[str, pd.DataFrame],
        tmp_path: Path,
    ) -> None:
        """If K5 config and K1 config shared a column, K5 must skip K1-touched cells.

        The real companies configs have no overlap between K1 and K5
        columns, so we pre-seed a K1 provenance row on a K5-managed
        column (``annual_income``, per the 2026-05-04 K5 schema
        refresh) to simulate the collision explicitly. Then we invoke
        only K5 via ``apply_knob_05`` with the collision index and
        verify the K5 skipped-log records the collision.
        """
        from usecases_synthetic.scripts.apply_knob_05_format import (
            SKIP_COLLISION_PRIOR,
            apply_knob_05,
            load_knob_05_config,
        )

        prov_dir = tmp_path / "output" / "provenance"
        prov_dir.mkdir(parents=True)

        # Seed a fake K1 provenance row on dbpedia.annual_income.
        fake_k1 = ProvenanceLog(knob=1, level="medium")
        fake_k1.append(
            entity_id="db_0",
            source="dbpedia",
            attribute="annual_income",
            original_value="1000",
            new_value="1000 USD",
            transform_fn="llm_paraphrase",
            transform_params={},
        )
        fake_k1.flush(prov_dir / "knob_01_surface.csv")

        ci = CollisionIndex(prov_dir)
        ci.reload()

        config = load_knob_05_config("companies")
        _, _, skipped_df = apply_knob_05(
            domain="companies",
            level="medium",
            sources=joint_sources,
            config=config,
            collision_index=ci,
            seed=42,
        )

        # db_0.annual_income should appear in skipped with prior-knob reason.
        hits = skipped_df[
            (skipped_df["entity_id"] == "db_0")
            & (skipped_df["attribute"] == "annual_income")
            & (skipped_df["reason"] == SKIP_COLLISION_PRIOR)
        ]
        assert (
            not hits.empty
        ), "K5 should have skipped db_0.annual_income due to K1 collision"


# ---- R10-A: cross-level cell-selection nesting -----------------------------


class TestLevelNesting:
    """R10-A: K1/K6 cell selection nests across levels (easy in medium in hard).

    Cell selection is driven by ``lib.rng.cell_selection_uniform``, keyed on
    cell identity but *not* on the difficulty level. With monotone per-level
    target rates, the set of cells whose uniform falls below the rate is
    cumulative across levels, which fixes the medium > easy (and medium > hard)
    EM non-monotonicity that came from per-level RNG reseeding.

    With ``entity_groups=None`` and ``collision_index=None`` there are no
    protection floors or collision skips, so every cell passing the gate lands
    in either the provenance log (operator succeeded) or the skipped log
    (operator no-op). Their union therefore equals the gate-selected set,
    independent of operator behaviour, and that union is what must nest.
    """

    @staticmethod
    def _frame(n: int = 200) -> dict[str, pd.DataFrame]:
        """Two text sources with stable ids and multi-token cell values."""
        sources: dict[str, pd.DataFrame] = {}
        for src in ("s1", "s2"):
            df = pd.DataFrame(
                {
                    "id": [f"{src}_{i}" for i in range(n)],
                    "text_a": [
                        f"alpha bravo charlie delta {src} {i}" for i in range(n)
                    ],
                    "text_b": [f"echo foxtrot golf hotel {src} {i}" for i in range(n)],
                }
            )
            df.attrs["dataset_name"] = src
            sources[src] = df
        return sources

    @staticmethod
    def _gate_cells(
        prov_df: pd.DataFrame, skipped_df: pd.DataFrame
    ) -> set[tuple[str, str, str]]:
        """Union of provenance + skipped cells == the gate-selected set."""
        cells: set[tuple[str, str, str]] = set()
        for frame in (prov_df, skipped_df):
            if frame is not None and len(frame):
                cells |= set(
                    zip(frame["source"], frame["entity_id"], frame["attribute"])
                )
        return cells

    @staticmethod
    def _k6_config(easy: float, medium: float, hard: float) -> dict[str, Any]:
        return {
            "id_columns": {"s1": "id", "s2": "id"},
            "attribute_classes": {
                "s1": {"text_a": "secondary", "text_b": "secondary"},
                "s2": {"text_a": "secondary", "text_b": "secondary"},
            },
            "noise_rates_per_level": {
                "easy": {"secondary": easy},
                "medium": {"secondary": medium},
                "hard": {"secondary": hard},
            },
            "operator_mix": {
                lvl: {"whitespace_corrupt": 1.0, "case_corrupt": 1.0}
                for lvl in ("easy", "medium", "hard")
            },
        }

    def _k6_gate(self, config: dict[str, Any], level: str) -> set[tuple[str, str, str]]:
        from usecases_synthetic.scripts.apply_knob_06_noise import apply_knob_06

        # A real domain name is required (the closeness context validates
        # it), but with ``entity_groups=None`` no protection floor fires, so
        # the synthetic cells are never skipped on protection grounds.
        _, prov_df, skipped_df = apply_knob_06(
            domain="companies",
            level=level,  # type: ignore[arg-type]
            sources=self._frame(),
            config=config,
            entity_groups=None,
            collision_index=None,
            seed=42,
        )
        return self._gate_cells(prov_df, skipped_df)

    def test_k6_noise_selection_nests(self) -> None:
        config = self._k6_config(easy=0.1, medium=0.3, hard=0.6)
        easy = self._k6_gate(config, "easy")
        medium = self._k6_gate(config, "medium")
        hard = self._k6_gate(config, "hard")
        assert easy <= medium <= hard
        assert 0 < len(easy) < len(medium) < len(hard)

    def test_k6_equal_rates_select_identical_set(self) -> None:
        """Regression guard against per-level resampling.

        Equal medium/hard rates must yield the *identical* selected set
        (level-independent gate), not merely sets of similar size.
        """
        config = self._k6_config(easy=0.2, medium=0.5, hard=0.5)
        assert self._k6_gate(config, "medium") == self._k6_gate(config, "hard")
        assert self._k6_gate(config, "easy") < self._k6_gate(config, "medium")

    def test_k1_paraphrase_selection_nests(self) -> None:
        from usecases_synthetic.scripts.apply_knob_01_surface import apply_knob_01

        zero = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
        config = {
            "id_columns": {"s1": "id", "s2": "id"},
            "attribute_classes": {
                "s1": {"text_a": "secondary", "text_b": "secondary"},
                "s2": {"text_a": "secondary", "text_b": "secondary"},
            },
            "attribute_mapping": {},
            "paraphrase_rate_primary": zero,
            "paraphrase_rate_key": zero,
            "paraphrase_rate_secondary": {"easy": 0.1, "medium": 0.3, "hard": 0.6},
            "paraphrase_rate_categorical": zero,
            "operator_mix": {
                lvl: {"eda_random_swap": 1.0} for lvl in ("easy", "medium", "hard")
            },
            "baseline_above_target_rules": [],
            "abbreviation_table": {},
            "stopword_list": [],
            "key_token_skiplist": {},
            "anchor_survivor_floor": {
                "primary": False,
                "key": False,
                "secondary": False,
                "categorical": False,
            },
        }

        def gate(level: str) -> set[tuple[str, str, str]]:
            _, prov_df, skipped_df, _ = apply_knob_01(
                domain="companies",
                level=level,  # type: ignore[arg-type]
                sources=self._frame(),
                config=config,
                entity_groups=None,
                collision_index=None,
                llm_cache=None,
                seed=42,
            )
            return self._gate_cells(prov_df, skipped_df)

        easy, medium, hard = gate("easy"), gate("medium"), gate("hard")
        assert easy <= medium <= hard
        assert 0 < len(easy) < len(medium) < len(hard)

    def test_k1_and_k6_select_independent_cells(self) -> None:
        """K1 and K6 must not select correlated cells (distinct knob keys)."""
        from usecases_synthetic.lib.rng import cell_selection_uniform

        rate = 0.5
        k1 = {
            (s, e)
            for s in ("s1", "s2")
            for e in (f"{s}_{i}" for i in range(200))
            if cell_selection_uniform("nesttest", s, e, "text_a", knob=1) < rate
        }
        k6 = {
            (s, e)
            for s in ("s1", "s2")
            for e in (f"{s}_{i}" for i in range(200))
            if cell_selection_uniform("nesttest", s, e, "text_a", knob=6) < rate
        }
        # Independent selections overlap only by chance (~rate*rate); they must
        # not be identical, which would indicate a shared (knob-blind) key.
        assert k1 != k6

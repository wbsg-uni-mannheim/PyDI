"""Tests for deterministic RNG factory."""

from __future__ import annotations

import numpy as np

from usecases_synthetic.lib.rng import (
    cell_selection_uniform,
    make_rng,
    spawn_sub_rng,
)


class TestMakeRng:
    """Determinism and divergence tests for make_rng."""

    def test_same_inputs_same_draws(self) -> None:
        """Identical inputs must produce identical first 100 draws."""
        rng1 = make_rng("companies", "hard", knob=6)
        rng2 = make_rng("companies", "hard", knob=6)

        draws1 = [rng1.random() for _ in range(100)]
        draws2 = [rng2.random() for _ in range(100)]
        assert draws1 == draws2

    def test_different_domain_diverges(self) -> None:
        rng1 = make_rng("companies", "hard", knob=6)
        rng2 = make_rng("games", "hard", knob=6)

        draws1 = [rng1.random() for _ in range(100)]
        draws2 = [rng2.random() for _ in range(100)]
        assert draws1 != draws2

    def test_different_variant_diverges(self) -> None:
        rng1 = make_rng("companies", "easy", knob=6)
        rng2 = make_rng("companies", "hard", knob=6)

        draws1 = [rng1.random() for _ in range(100)]
        draws2 = [rng2.random() for _ in range(100)]
        assert draws1 != draws2

    def test_different_knob_diverges(self) -> None:
        rng1 = make_rng("companies", "hard", knob=1)
        rng2 = make_rng("companies", "hard", knob=6)

        draws1 = [rng1.random() for _ in range(100)]
        draws2 = [rng2.random() for _ in range(100)]
        assert draws1 != draws2

    def test_different_master_seed_diverges(self) -> None:
        rng1 = make_rng("companies", "hard", knob=6, master_seed=42)
        rng2 = make_rng("companies", "hard", knob=6, master_seed=99)

        draws1 = [rng1.random() for _ in range(100)]
        draws2 = [rng2.random() for _ in range(100)]
        assert draws1 != draws2

    def test_integer_draws_deterministic(self) -> None:
        rng1 = make_rng("music", "medium", knob=3)
        rng2 = make_rng("music", "medium", knob=3)

        draws1 = rng1.integers(0, 1000, size=50).tolist()
        draws2 = rng2.integers(0, 1000, size=50).tolist()
        assert draws1 == draws2


class TestSpawnSubRng:
    """Tests for sub-RNG spawning."""

    def test_spawn_deterministic(self) -> None:
        rng1 = make_rng("companies", "hard", knob=1)
        rng2 = make_rng("companies", "hard", knob=1)

        sub1 = spawn_sub_rng(rng1, "attr_name")
        sub2 = spawn_sub_rng(rng2, "attr_name")

        draws1 = [sub1.random() for _ in range(50)]
        draws2 = [sub2.random() for _ in range(50)]
        assert draws1 == draws2

    def test_different_labels_diverge(self) -> None:
        rng1 = make_rng("companies", "hard", knob=1)
        rng2 = make_rng("companies", "hard", knob=1)

        sub1 = spawn_sub_rng(rng1, "attr_name")
        sub2 = spawn_sub_rng(rng2, "attr_revenue")

        draws1 = [sub1.random() for _ in range(50)]
        draws2 = [sub2.random() for _ in range(50)]
        assert draws1 != draws2


class TestCellSelectionUniform:
    """R10-A: level-independent per-cell selection uniform (K1/K6 nesting)."""

    def test_deterministic(self) -> None:
        """Identical cell identity -> identical uniform."""
        u1 = cell_selection_uniform("products", "products_1", "e1", "title", knob=1)
        u2 = cell_selection_uniform("products", "products_1", "e1", "title", knob=1)
        assert u1 == u2

    def test_in_unit_range(self) -> None:
        for entity_id in (f"e{i}" for i in range(200)):
            u = cell_selection_uniform(
                "products", "products_1", entity_id, "title", knob=1
            )
            assert 0.0 <= u < 1.0

    def test_no_level_parameter(self) -> None:
        """The signature must not accept ``level`` -- nesting depends on it."""
        import inspect

        params = inspect.signature(cell_selection_uniform).parameters
        assert "level" not in params
        assert "variant" not in params

    def test_knob_diverges(self) -> None:
        """K1 and K6 must draw independent uniforms for the same cell."""
        u_k1 = cell_selection_uniform("products", "products_1", "e1", "title", knob=1)
        u_k6 = cell_selection_uniform("products", "products_1", "e1", "title", knob=6)
        assert u_k1 != u_k6

    def test_identity_components_diverge(self) -> None:
        base = cell_selection_uniform("products", "products_1", "e1", "title", knob=1)
        assert base != cell_selection_uniform(
            "music", "products_1", "e1", "title", knob=1
        )
        assert base != cell_selection_uniform(
            "products", "products_2", "e1", "title", knob=1
        )
        assert base != cell_selection_uniform(
            "products", "products_1", "e2", "title", knob=1
        )
        assert base != cell_selection_uniform(
            "products", "products_1", "e1", "brand", knob=1
        )

    def test_master_seed_diverges(self) -> None:
        u1 = cell_selection_uniform(
            "products", "products_1", "e1", "title", knob=1, master_seed=42
        )
        u2 = cell_selection_uniform(
            "products", "products_1", "e1", "title", knob=1, master_seed=99
        )
        assert u1 != u2

    def test_roughly_uniform_distribution(self) -> None:
        """The hash-derived draws should be ~uniform on [0, 1)."""
        draws = [
            cell_selection_uniform("products", "products_1", f"e{i}", "title", knob=1)
            for i in range(2000)
        ]
        mean = sum(draws) / len(draws)
        assert 0.45 < mean < 0.55
        # Reasonable spread across deciles -- no decile empty.
        deciles = [0] * 10
        for d in draws:
            deciles[min(int(d * 10), 9)] += 1
        assert all(count > 0 for count in deciles)

    def test_selection_sets_nest_for_monotone_rates(self) -> None:
        """Cells selected at lower rates are a subset of those at higher rates.

        This is the load-bearing R10-A invariant: with a single
        level-independent uniform per cell and monotone per-level rates,
        ``easy_cells subset of medium_cells subset of hard_cells``.
        """
        cells = [
            (f"products_{s}", f"e{i}", "title") for s in (1, 2) for i in range(500)
        ]

        def selected(rate: float) -> set[tuple[str, str, str]]:
            return {
                (src, eid, attr)
                for (src, eid, attr) in cells
                if cell_selection_uniform("products", src, eid, attr, knob=1) < rate
            }

        easy = selected(0.04)
        medium = selected(0.12)
        hard = selected(0.30)
        assert easy <= medium <= hard
        # And the rates actually select progressively more cells.
        assert len(easy) < len(medium) < len(hard)

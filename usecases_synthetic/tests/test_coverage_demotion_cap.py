"""R10-E: per-source demotion cap + hash tie-break in K4 row selection.

Without the cap the conflict ranking + alphabetical tie-break let one
source (the EM-anchor source) absorb nearly every demotion, silently
deleting EM-gold edges that are not regenerated post-K4. These tests pin
the cap (a hard per-source ceiling) and the hash tie-break (genuine ties
spread across sources instead of locking to the alphabetically-first).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from usecases_synthetic.lib.coverage_ops import (
    EntityView,
    RemovalConstraints,
    _source_tiebreak,
    select_removal_candidates,
)

_SOURCES = ("aaa", "bbb", "ccc", "ddd")


def _make_sources(n: int = 10) -> dict[str, pd.DataFrame]:
    """Four sources, ``n`` rows each, identical primary values.

    Identical values make every source tie at the top of the conflict
    ranking — the products-style case where one source would otherwise
    be locked in by the alphabetical tie-break.
    """
    return {
        src: pd.DataFrame(
            {"id": [f"{src}_{i}" for i in range(n)], "name": ["same"] * n}
        )
        for src in _SOURCES
    }


def _make_view(n: int = 10) -> EntityView:
    view = EntityView(source_count=len(_SOURCES))
    for i in range(n):
        view.members[f"e{i}"] = {src: (i, f"{src}_{i}") for src in _SOURCES}
    return view


def _constraints(protected: set[tuple[str, str]] | None = None) -> RemovalConstraints:
    return RemovalConstraints(
        fusion_val_test_ids=set(),
        protected_records=protected or set(),
        distractor_entity_ids=set(),
        singleton_cap=0.70,
    )


def _counts_by_source(selected: list[tuple[str, str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for _eid, source, _rid in selected:
        out[source] = out.get(source, 0) + 1
    return out


class TestPerSourceDemotionCap:
    def test_uncapped_concentrates_on_only_eligible_source(self) -> None:
        """With bbb/ccc/ddd protected and no cap, all demotions hit aaa."""
        n = 10
        sources = _make_sources(n)
        view = _make_view(n)
        protected = {
            (src, f"{src}_{i}") for src in ("bbb", "ccc", "ddd") for i in range(n)
        }
        selected = select_removal_candidates(
            view=view,
            demotions={4: n},
            constraints=_constraints(protected),
            pool_pairs=[],
            sources=sources,
            primary_cols={s: "name" for s in _SOURCES},
            rng=np.random.default_rng(0),
            per_source_demotion_cap=1.0,  # no-op default
        )
        counts = _counts_by_source(selected)
        assert counts == {"aaa": n}

    def test_cap_limits_a_dominant_source(self) -> None:
        """cap=0.40 on 10 rows caps aaa at 4 removals; the rest are skipped
        (the other sources are protected, so no spread is possible)."""
        n = 10
        sources = _make_sources(n)
        view = _make_view(n)
        protected = {
            (src, f"{src}_{i}") for src in ("bbb", "ccc", "ddd") for i in range(n)
        }
        selected = select_removal_candidates(
            view=view,
            demotions={4: n},
            constraints=_constraints(protected),
            pool_pairs=[],
            sources=sources,
            primary_cols={s: "name" for s in _SOURCES},
            rng=np.random.default_rng(0),
            per_source_demotion_cap=0.40,
        )
        counts = _counts_by_source(selected)
        # 0.40 * 10 = 4.0 -> the 5th removal from aaa is blocked.
        assert counts.get("aaa", 0) == 4
        assert len(selected) == 4

    def test_cap_lets_remaining_demotions_spread(self) -> None:
        """With every source eligible, cap=0.40 keeps aaa <= 4 and the
        remaining demotions fall through to the other sources."""
        n = 10
        sources = _make_sources(n)
        view = _make_view(n)
        selected = select_removal_candidates(
            view=view,
            demotions={4: n},
            constraints=_constraints(),
            pool_pairs=[],
            sources=sources,
            primary_cols={s: "name" for s in _SOURCES},
            rng=np.random.default_rng(0),
            per_source_demotion_cap=0.40,
        )
        counts = _counts_by_source(selected)
        assert len(selected) == n  # all demotions satisfied
        assert all(c <= 4 for c in counts.values()), counts
        assert max(counts.values()) <= 4


class TestHashTiebreak:
    def test_ties_spread_across_sources(self) -> None:
        """All sources eligible + tied conflict: the hash tie-break must
        spread the per-entity pick across more than one source (the
        alphabetical tie-break would have locked every pick to 'aaa')."""
        n = 12
        sources = _make_sources(n)
        view = _make_view(n)
        selected = select_removal_candidates(
            view=view,
            demotions={4: n},
            constraints=_constraints(),
            pool_pairs=[],
            sources=sources,
            primary_cols={s: "name" for s in _SOURCES},
            rng=np.random.default_rng(0),
            per_source_demotion_cap=1.0,
        )
        chosen_sources = {src for _eid, src, _rid in selected}
        assert len(chosen_sources) >= 2

    def test_tiebreak_is_deterministic_and_source_specific(self) -> None:
        a = _source_tiebreak("e1", "aaa")
        assert a == _source_tiebreak("e1", "aaa")  # deterministic
        assert a != _source_tiebreak("e1", "bbb")  # per-source
        assert a != _source_tiebreak("e2", "aaa")  # per-entity

"""Fold-in regression guard for the 2026 ``papers`` domain.

Papers is the first synthetic domain whose sources ship as JSON-lines
(``*.jsonl`` with no on-disk id), whose EM gold is header-bearing with
0/1 integer labels under a condensed ``dblp_<other>_<split>.csv`` naming,
and whose fusion gold is flat-by-DOI JSON-lines (not the per-attribute
``provenance`` XML the pre-2026 domains use). The loader extensions that
make this work (``loaders._FORMAT_LOADERS["jsonl"]``, header-aware
``read_em_gold_csv``, ``em_gold_candidates``, ``variant_loader.
_load_fusion_file``) are shared with every other domain, so this module
both exercises the papers happy path and pins the guarded behaviour so a
future edit cannot silently regress it.

These tests load the full papers sources (~60k rows x 3) once via a
module-scoped fixture; they do NOT run any variant generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.domain_config import (
    VALID_DOMAINS,
    load_knob_config,
)
from usecases_synthetic.lib.loaders import (
    em_gold_candidates,
    read_em_gold_csv,
)
from usecases_synthetic.lib.variant_loader import (
    VariantBundle,
    _load_fusion_file,
    load_variant,
)

CONFIG_DIR: Path = Path(__file__).resolve().parents[1] / "config"
PAPERS_SOURCES = ("dblp", "crossref", "open_alex")
PAPERS_PAIRS = (("dblp", "crossref"), ("dblp", "open_alex"))


# ---------------------------------------------------------------------------
# Loader-unit behaviour (fast; no domain data)
# ---------------------------------------------------------------------------


def test_papers_registered() -> None:
    assert "papers" in VALID_DOMAINS


def test_read_em_gold_csv_detects_header_and_keeps_int_labels(
    tmp_path: Path,
) -> None:
    """Papers-style header-bearing EM gold parses to id1/id2/label with an
    integer label dtype (so the downstream ``astype(bool)`` positive
    extraction works)."""
    p = tmp_path / "dblp_crossref_test.csv"
    p.write_text(
        "id_dblp,id_crossref,label\n"
        "dblp-00001,crossref-00009,1\n"
        "dblp-00002,crossref-00010,0\n",
        encoding="utf-8",
    )
    df = read_em_gold_csv(p)
    assert list(df.columns) == ["id1", "id2", "label"]
    assert df["id1"].tolist() == ["dblp-00001", "dblp-00002"]
    assert pd.api.types.is_integer_dtype(df["label"])
    assert df["label"].astype(bool).tolist() == [True, False]


def test_read_em_gold_csv_headerless_unchanged(tmp_path: Path) -> None:
    """A header-less companies-style file (URL pairs + True/False string
    labels) is parsed exactly as before — the header auto-detection must
    not misfire on real data rows, and string labels stay strings."""
    p = tmp_path / "forbes_2_dbpedia_test.csv"
    p.write_text(
        "http://www.forbes.com/companies/apple/,"
        "http://dbpedia.org/resource/Apple_Inc.,True\n"
        "http://www.forbes.com/companies/x/,http://dbpedia.org/resource/Y,False\n",
        encoding="utf-8",
    )
    df = read_em_gold_csv(p)
    assert list(df.columns) == ["id1", "id2", "label"]
    assert df["id1"].tolist() == [
        "http://www.forbes.com/companies/apple/",
        "http://www.forbes.com/companies/x/",
    ]
    assert df["id2"].tolist() == [
        "http://dbpedia.org/resource/Apple_Inc.",
        "http://dbpedia.org/resource/Y",
    ]
    # header-less path keeps the raw string labels (no header row ingested)
    assert df["label"].tolist() == ["True", "False"]


def test_em_gold_candidates_canonical_first_then_condensed() -> None:
    """Canonical ``_2_`` forms come first (existing domains unaffected);
    the condensed ``dblp_openalex`` papers form is offered as a fallback,
    incl. the ``open_alex`` -> ``openalex`` token."""
    cands = em_gold_candidates(Path("/x"), ("dblp", "open_alex"), "test")
    names = [p.name for p, _ in cands]
    assert names[0] == "dblp_2_open_alex_test.csv"
    assert "dblp_openalex_test.csv" in names
    # the resolved papers file is forward-direction (no swap)
    swap = {p.name: s for p, s in cands}
    assert swap["dblp_openalex_test.csv"] is False


def test_load_fusion_file_dispatches_jsonl() -> None:
    fusion_dir = REPO_ROOT / "usecases" / "papers" / "input" / "fusion"
    df = _load_fusion_file(fusion_dir / "fusion_test.jsonl", "fusion_test_set")
    assert len(df) == 100
    assert "doi" in df.columns


# ---------------------------------------------------------------------------
# Bundle integrity (module-scoped: load once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def papers_bundle() -> VariantBundle:
    return load_variant("papers", "baseline")


def test_sources_load_with_dash_ids(papers_bundle: VariantBundle) -> None:
    assert set(papers_bundle.sources) == set(PAPERS_SOURCES)
    for name in PAPERS_SOURCES:
        df = papers_bundle.sources[name]
        assert "id" in df.columns
        assert len(df) > 50_000
        first = str(df["id"].iloc[0])
        assert first.startswith(f"{name}-"), first


def test_em_gold_int_labels_and_pairs(papers_bundle: VariantBundle) -> None:
    assert set(papers_bundle.em_gold) == set(PAPERS_PAIRS)
    for pair in PAPERS_PAIRS:
        g = papers_bundle.em_gold[pair]
        assert list(g.columns) == ["id1", "id2", "label"]
        assert pd.api.types.is_integer_dtype(g["label"])
        # both classes present
        assert set(g["label"].unique()) == {0, 1}


def test_em_gold_ids_resolve_and_positives_share_doi(
    papers_bundle: VariantBundle,
) -> None:
    """The minted source ids align with the EM gold ids (add_index order
    matches the gold-generation order): every gold id resolves to a source
    row, and positive pairs point at the same paper (shared doi)."""
    dblp = papers_bundle.sources["dblp"].set_index("id")
    for pair in PAPERS_PAIRS:
        right = papers_bundle.sources[pair[1]].set_index("id")
        g = papers_bundle.em_gold[pair]
        assert (~g["id1"].isin(dblp.index)).sum() == 0
        assert (~g["id2"].isin(right.index)).sum() == 0
        pos = g[g["label"].astype(bool)].head(500)
        share = (
            dblp.loc[pos["id1"], "doi"].values == right.loc[pos["id2"], "doi"].values
        ).mean()
        assert share > 0.95, f"{pair} positives doi-match only {share:.3f}"


def test_em_splits_present(papers_bundle: VariantBundle) -> None:
    for pair in PAPERS_PAIRS:
        splits = papers_bundle.em_splits[pair]
        assert {"train", "val", "test"}.issubset(splits)


def test_fusion_gold_jsonl_joined_on_doi(papers_bundle: VariantBundle) -> None:
    assert len(papers_bundle.fusion_gold) == 100
    assert "doi" in papers_bundle.fusion_gold.columns
    assert papers_bundle.fusion_validation is not None
    assert len(papers_bundle.fusion_validation) == 100


# ---------------------------------------------------------------------------
# Per-knob configs
# ---------------------------------------------------------------------------

PAPERS_KNOBS = (1, 2, 3, 4, 5, 6, 8, 10)
_RUNGS = {"descriptive", "abbreviated", "cryptic", "anonymized"}


@pytest.mark.parametrize("knob", PAPERS_KNOBS)
def test_knob_config_loads(knob: int) -> None:
    cfg = load_knob_config(knob, "papers")
    assert isinstance(cfg, dict) and cfg
    if knob != 8:
        assert cfg.get("id_columns") == {
            "dblp": "id",
            "crossref": "id",
            "open_alex": "id",
        }


@pytest.mark.parametrize("knob", PAPERS_KNOBS)
def test_knob_config_level_triples_monotone(knob: int) -> None:
    """Every ``{easy, medium, hard}`` numeric triple in a papers knob config
    is monotone (seeded from music, which is monotone by construction)."""
    cfg = load_knob_config(knob, "papers")
    errs: list[str] = []

    def walk(node: object, path: str) -> None:
        if not isinstance(node, dict):
            return
        levels = ("easy", "medium", "hard")
        if all(k in node for k in levels) and all(
            isinstance(node[k], (int, float)) for k in levels
        ):
            e, m, h = (node[k] for k in levels)
            if not (e <= m <= h or e >= m >= h):
                errs.append(f"{path}: {e},{m},{h}")
        else:
            for k, v in node.items():
                walk(v, f"{path}.{k}")

    walk(cfg, f"knob_{knob:02d}")
    assert not errs, f"non-monotone level triples: {errs}"


def test_knob_08_rename_table_invariants() -> None:
    cfg = load_knob_config(8, "papers")
    rt = cfg["rename_table"]
    for src in PAPERS_SOURCES:
        cols = rt[src]
        assert "id" not in cols, f"{src}: rename_table must omit id"
        assert len(cols) == 14, f"{src}: expected 14 renamed columns"
        for col, rungs in cols.items():
            assert set(rungs) == _RUNGS, f"{src}.{col} rungs {set(rungs)}"
            assert len(set(rungs.values())) == 4, f"{src}.{col} rung collision"
        # renamed headers must be unique per tier (no column collisions)
        for tier in _RUNGS:
            names = [rungs[tier] for rungs in cols.values()]
            assert len(names) == len(set(names)), f"{src} tier {tier} dup headers"

"""Tests for the reusable domain downsampler.

Builds a tiny synthetic two-source domain in a ``tmp_path``-rooted fake
use-cases tree, runs :func:`downsample_domain`, and asserts the
gold-preservation, buffer-sizing, and file-generation contract.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
import xml.etree.ElementTree as ET

from usecases_synthetic.scripts.downsample_domain import (
    _classify_id,
    _collect_em_ids,
    _collect_fusion_ids,
    downsample_domain,
)


def _write_tiny_domain(
    *,
    usecases_dir: Path,
    config_dir: Path,
    domain: str = "tiny",
) -> None:
    """Create a two-source fake domain with gold files on disk."""
    data_dir = usecases_dir / domain / "input" / "data"
    em_dir = usecases_dir / domain / "input" / "entitymatching"
    fusion_dir = usecases_dir / domain / "input" / "fusion"
    sm_dir = usecases_dir / domain / "input" / "schemamatching"
    for d in (data_dir, em_dir, fusion_dir, sm_dir, config_dir / "domains"):
        d.mkdir(parents=True, exist_ok=True)

    left_records = [{"identifier": f"left:{i}", "name": f"L{i}"} for i in range(100)]
    with open(data_dir / "left.json", "w", encoding="utf-8") as f:
        json.dump(left_records, f)

    with open(data_dir / "right.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Name", "Identifier"])
        for i in range(80):
            w.writerow([f"R{i}", f"right:{i}"])

    gold_pairs = [(f"left:{i}", f"right:{i}") for i in range(5)]
    with open(em_dir / "left_2_right_all.csv", "w", encoding="utf-8") as f:
        for a, b in gold_pairs:
            f.write(f"{a},{b},true\n")

    root = ET.Element("items")
    for i in range(3):
        item = ET.SubElement(root, "item")
        ET.SubElement(item, "id").text = f"left:{i}"
        n = ET.SubElement(item, "name")
        n.text = f"L{i}-fused"
        n.set("provenance", f"left:{i}+right:{i}")
    ET.ElementTree(root).write(fusion_dir / "test_set.xml", encoding="utf-8")

    (sm_dir / "sm_mapping_gold.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    domain_yaml = {
        "domain": domain,
        "master_seed": 7,
        "sources": [
            {
                "name": "left",
                "file": "left.json",
                "format": "json",
                "id_prefix": "left:",
            },
            {
                "name": "right",
                "file": "right.csv",
                "format": "csv",
                "id_prefix": "right:",
                "reader_kwargs": {"delimiter": "\t"},
            },
        ],
        "source_pairs": [["left", "right"]],
        "attribute_classes": {"name": "primary"},
    }
    with open(config_dir / "domains" / f"{domain}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(domain_yaml, f, sort_keys=False)


@pytest.fixture
def tiny_env(tmp_path, monkeypatch):
    """Patch the module-level dirs so tests use a sandboxed layout."""
    usecases_dir = tmp_path / "usecases"
    config_dir = tmp_path / "config"
    pools_dir = tmp_path / "pools"
    usecases_dir.mkdir()
    config_dir.mkdir()
    pools_dir.mkdir()

    import usecases_synthetic.lib.domain_config as dc
    import usecases_synthetic.lib.loaders as loaders
    import usecases_synthetic.scripts.downsample_domain as ds

    monkeypatch.setattr(dc, "USECASES_DIR", usecases_dir)
    monkeypatch.setattr(dc, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(dc, "POOLS_DIR", pools_dir)
    monkeypatch.setattr(loaders, "USECASES_DIR", usecases_dir)
    monkeypatch.setattr(ds, "USECASES_DIR", usecases_dir)
    monkeypatch.setattr(ds, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(ds, "POOLS_DIR", pools_dir)
    monkeypatch.setattr(dc, "VALID_DOMAINS", ["tiny", "tiny-small"])

    _write_tiny_domain(usecases_dir=usecases_dir, config_dir=config_dir)
    return usecases_dir, config_dir, pools_dir


def test_classify_id_picks_longest_prefix():
    prefixes = {"a": "left:", "b": "left:sub:"}
    assert _classify_id("left:sub:1", prefixes) == "b"
    assert _classify_id("left:1", prefixes) == "a"
    assert _classify_id("other:1", prefixes) is None


def test_em_and_fusion_ids_collected(tiny_env):
    usecases_dir, _, _ = tiny_env
    em = _collect_em_ids(
        usecases_dir / "tiny" / "input" / "entitymatching",
        {"left": "left:", "right": "right:"},
    )
    fu = _collect_fusion_ids(
        usecases_dir / "tiny" / "input" / "fusion",
        {"left": "left:", "right": "right:"},
    )
    assert em["left"] == {f"left:{i}" for i in range(5)}
    assert em["right"] == {f"right:{i}" for i in range(5)}
    assert fu["left"] >= {f"left:{i}" for i in range(3)}
    assert fu["right"] >= {f"right:{i}" for i in range(3)}


def test_downsample_preserves_gold_and_writes_yaml(tiny_env):
    usecases_dir, config_dir, _ = tiny_env
    report = downsample_domain(
        source_domain="tiny",
        target_domain="tiny-small",
        usecases_dir=usecases_dir,
        config_dir=config_dir,
        gold_multiplier=2.0,
        min_rows_per_source=0,
        seed=0,
    )

    target_dir = usecases_dir / "tiny-small" / "input"

    with open(target_dir / "data" / "left.json", encoding="utf-8") as f:
        left_out = json.load(f)
    left_ids = {r["identifier"] for r in left_out}
    assert {f"left:{i}" for i in range(5)} <= left_ids

    with open(target_dir / "data" / "right.csv", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    right_ids = {row[1] for row in rows[1:]}
    assert {f"right:{i}" for i in range(5)} <= right_ids

    for src in report.sources:
        assert src.output_rows >= src.protected_rows
        assert (
            src.output_rows <= int(src.protected_rows * 2.0)
            or src.output_rows == src.protected_rows
        )

    assert (target_dir / "entitymatching" / "left_2_right_all.csv").exists()
    assert (target_dir / "fusion" / "test_set.xml").exists()
    assert (target_dir / "schemamatching" / "sm_mapping_gold.csv").exists()

    yaml_path = config_dir / "domains" / "tiny-small.yaml"
    assert yaml_path.exists()
    with open(yaml_path, encoding="utf-8") as f:
        out_yaml = yaml.safe_load(f)
    assert out_yaml["domain"] == "tiny-small"
    assert out_yaml["knob_config_alias"] == "tiny"
    assert out_yaml["master_seed"] == 7
    assert [s["name"] for s in out_yaml["sources"]] == ["left", "right"]


def test_min_rows_floor_applied(tiny_env):
    usecases_dir, config_dir, _ = tiny_env
    report = downsample_domain(
        source_domain="tiny",
        target_domain="tiny-small",
        usecases_dir=usecases_dir,
        config_dir=config_dir,
        gold_multiplier=1.0,
        min_rows_per_source=20,
        seed=0,
    )
    for src in report.sources:
        assert src.output_rows >= 20


def test_gold_files_copied_verbatim(tiny_env):
    usecases_dir, config_dir, _ = tiny_env
    downsample_domain(
        source_domain="tiny",
        target_domain="tiny-small",
        usecases_dir=usecases_dir,
        config_dir=config_dir,
    )
    src_em = (
        usecases_dir / "tiny" / "input" / "entitymatching" / "left_2_right_all.csv"
    ).read_text()
    dst_em = (
        usecases_dir
        / "tiny-small"
        / "input"
        / "entitymatching"
        / "left_2_right_all.csv"
    ).read_text()
    assert src_em == dst_em


def test_pool_filtered_to_retained_ids(tiny_env):
    usecases_dir, config_dir, pools_dir = tiny_env
    src_pool = pools_dir / "tiny"
    src_pool.mkdir(parents=True)
    (src_pool / "pooled_positives.csv").write_text(
        "id1,id2,source_1,source_2,pool_agreement\n"
        "left:0,right:0,left,right,1\n"
        "left:1,right:1,left,right,1\n"
        "left:500,right:500,left,right,1\n",
        encoding="utf-8",
    )
    report = downsample_domain(
        source_domain="tiny",
        target_domain="tiny-small",
        usecases_dir=usecases_dir,
        config_dir=config_dir,
        gold_multiplier=1.0,
        min_rows_per_source=0,
        seed=0,
    )
    dst_pool = pools_dir / "tiny-small" / "pooled_positives.csv"
    assert dst_pool.exists()
    content = dst_pool.read_text()
    assert "left:0,right:0" in content
    assert "left:1,right:1" in content
    assert "left:500" not in content
    assert report.pool_original_rows == 3
    assert report.pool_output_rows == 2

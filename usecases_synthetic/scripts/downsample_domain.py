"""Downsample a PyDI use-case domain while preserving all gold-referenced rows.

Produces a reduced clone of ``usecases/<source_domain>/input/`` at
``usecases/<target_domain>/input/`` plus a matching domain YAML at
``usecases_synthetic/config/domains/<target_domain>.yaml``. The target
domain reuses the source domain's per-knob configs via
``knob_config_alias`` so ablation and validation runs inherit the same
knob settings without duplication.

Protection policy
-----------------
Every row whose primary ID appears in any of

* EM gold CSVs under ``input/entitymatching/`` (all splits, including
  ``*_all.csv``), either column
* Fusion gold XMLs under ``input/fusion/`` (``<id>`` text or
  ``provenance=``-split source IDs)

is retained. Additional non-gold rows are sampled deterministically
(seeded) up to a per-source budget.

Format handling
---------------
Source files are rewritten in place:

* JSON arrays of dicts — keyed by ``id_field`` (default tries common
  identifier fields).
* CSV/TSV — identified by ``id_column``; the original header and
  delimiter are preserved.
* XML — ``<company>`` (or configurable root child) elements are kept or
  dropped based on the text content of their ``<id>`` child.

Gold files are copied verbatim — they already reference the protected
IDs by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usecases_synthetic.lib.domain_config import (  # noqa: E402
    CONFIG_DIR,
    POOLS_DIR,
    SourceSpec,
    USECASES_DIR,
    load_domain_config,
)
from usecases_synthetic.lib.loaders import read_em_gold_csv  # noqa: E402


@dataclass
class SourceDownsampleReport:
    """Per-source row accounting for a downsampling run."""

    name: str
    original_rows: int
    protected_rows: int
    sampled_extra_rows: int
    output_rows: int


@dataclass
class DownsampleReport:
    """Full report for a domain downsampling run."""

    source_domain: str
    target_domain: str
    seed: int
    sources: list[SourceDownsampleReport]
    em_files_copied: list[str]
    fusion_files_copied: list[str]
    sm_files_copied: list[str]
    pool_original_rows: int = 0
    pool_output_rows: int = 0


def _classify_id(
    raw_id: str,
    source_prefixes: dict[str, str],
) -> str | None:
    """Return the source name whose ``id_prefix`` matches ``raw_id``.

    Returns ``None`` if no prefix matches. Prefixes are evaluated
    longest-first so overlapping prefixes resolve deterministically.
    """
    candidates = sorted(source_prefixes.items(), key=lambda kv: -len(kv[1]))
    for name, prefix in candidates:
        if prefix and raw_id.startswith(prefix):
            return name
    return None


def _collect_em_ids(
    em_dir: Path,
    source_prefixes: dict[str, str],
) -> dict[str, set[str]]:
    """Scan EM gold CSVs and bucket ids by source."""
    protected: dict[str, set[str]] = {name: set() for name in source_prefixes}
    if not em_dir.exists():
        return protected
    for csv_path in sorted(em_dir.glob("*.csv")):
        df = read_em_gold_csv(csv_path)
        for col in ("id1", "id2"):
            for raw_id in df[col].astype(str):
                source = _classify_id(raw_id, source_prefixes)
                if source is not None:
                    protected[source].add(raw_id)
    return protected


def _collect_fusion_ids(
    fusion_dir: Path,
    source_prefixes: dict[str, str],
) -> dict[str, set[str]]:
    """Scan fusion gold XMLs and bucket ids by source.

    Both the top-level ``<id>`` text and every ``provenance`` attribute
    (split on ``+``) contribute.
    """
    protected: dict[str, set[str]] = {name: set() for name in source_prefixes}
    if not fusion_dir.exists():
        return protected
    for xml_path in sorted(fusion_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for entity in root:
            id_el = entity.find("id")
            if id_el is not None and id_el.text:
                source = _classify_id(id_el.text.strip(), source_prefixes)
                if source is not None:
                    protected[source].add(id_el.text.strip())
            for child in entity.iter():
                prov = child.attrib.get("provenance")
                if not prov:
                    continue
                for token in prov.split("+"):
                    token = token.strip()
                    if not token:
                        continue
                    source = _classify_id(token, source_prefixes)
                    if source is not None:
                        protected[source].add(token)
    return protected


def _sample_extras(
    all_ids: list[str],
    protected_ids: set[str],
    target_total: int,
    rng: np.random.Generator,
) -> set[str]:
    """Pick additional non-protected ids up to ``target_total``."""
    if target_total <= len(protected_ids):
        return set()
    remaining = [i for i in all_ids if i not in protected_ids]
    need = target_total - len(protected_ids)
    if need >= len(remaining):
        return set(remaining)
    picks = rng.choice(len(remaining), size=need, replace=False)
    return {remaining[int(i)] for i in picks}


def _json_id_field(records: list[dict[str, Any]]) -> str:
    """Guess the primary id field for a JSON source."""
    if not records:
        raise ValueError("Empty JSON source — cannot infer id field.")
    for candidate in ("identifier", "id", "uri", "url"):
        if candidate in records[0]:
            return candidate
    raise ValueError(
        f"Could not infer JSON id field from keys: {list(records[0])[:10]}"
    )


def _detect_csv_id_column(header: list[str], *, override: str | None = None) -> str:
    """Pick the column holding the primary id in a CSV/TSV source.

    When ``override`` is supplied (typically from a domain YAML
    ``id_column`` field), use it directly after verifying it is in the
    header. Otherwise fall back to the legacy candidate list, which
    covers the pre-2026-05-04 sources whose id columns followed a
    handful of conventional names.
    """
    if override is not None:
        if override in header:
            return override
        raise ValueError(
            f"Configured id_column={override!r} not present in CSV header: " f"{header}"
        )
    for candidate in (
        "Identifier",
        "identifier",
        "id",
        "ID",
        "URI",
        "URL",
        "rel_id",
    ):
        if candidate in header:
            return candidate
    raise ValueError(f"Could not infer CSV id column from header: {header}")


_XML_ID_ELEMENT_CANDIDATES = ("id", "rel_id", "identifier")


def _xml_local_name(tag: str) -> str:
    """Return ``"foo"`` for ``"{http://...}foo"`` or ``"foo"``."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _xml_id_text(el: ET.Element) -> str:
    """Return the element's primary id text by trying common child-element
    names. musicbrainz uses ``<rel_id>`` (the synthetic prefix-bearing id),
    while the XML attribute ``id="..."`` carries an unrelated UUID and is
    not used as the entity primary key. Namespace prefixes are stripped
    before matching so namespaced sources (musicbrainz mmd-2.0) work
    without a per-source override.
    """
    candidates = {c.lower() for c in _XML_ID_ELEMENT_CANDIDATES}
    for child in el:
        if _xml_local_name(child.tag).lower() in candidates and child.text:
            return child.text.strip()
    return ""


def _synthesised_inject_ids(source_name: str, n_rows: int) -> list[str]:
    """Return the inject_id stream for an N-row source.

    Mirrors the loader's convention at
    ``usecases_synthetic/lib/loaders.py:_ensure_id_column``: when a source
    spec sets ``inject_id: true``, the loader synthesises
    ``f"{source_name}_{1-based-row-index}"`` per row. The downsample
    script needs the same enumeration to know which on-disk rows survive
    the gold-protection + extras sampling.
    """
    return [f"{source_name}_{i + 1}" for i in range(n_rows)]


def _filter_json_source(
    src_path: Path,
    dst_path: Path,
    keep_ids: set[str],
) -> tuple[int, int]:
    """Filter a JSON-array source. Returns (original_count, kept_count)."""
    with open(src_path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected JSON array at {src_path}, got {type(records)}")
    id_field = _json_id_field(records)
    filtered = [r for r in records if str(r.get(id_field, "")) in keep_ids]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=4)
    return len(records), len(filtered)


def _filter_csv_source(
    src_path: Path,
    dst_path: Path,
    keep_ids: set[str],
    reader_kwargs: dict[str, Any],
    *,
    id_column: str | None = None,
) -> tuple[int, int]:
    """Filter a CSV/TSV source. Returns (original_count, kept_count).

    Preserves the original delimiter and header. ``reader_kwargs`` may
    contain ``delimiter`` to match the source format. ``id_column``
    overrides the heuristic id-column detection — passed through from
    the source's ``id_column`` field in the domain YAML.
    """
    delimiter = reader_kwargs.get("delimiter", ",")
    with open(src_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV source: {src_path}")
    header, data = rows[0], rows[1:]
    id_col = _detect_csv_id_column(header, override=id_column)
    id_idx = header.index(id_col)
    kept = [row for row in data if len(row) > id_idx and row[id_idx] in keep_ids]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(header)
        writer.writerows(kept)
    return len(data), len(kept)


def _filter_xml_source(
    src_path: Path,
    dst_path: Path,
    keep_ids: set[str],
) -> tuple[int, int]:
    """Filter an XML source. Returns (original_count, kept_count).

    Each top-level child of the root is kept if its primary id (looked up
    via :func:`_xml_id_text`) is in ``keep_ids``. The lookup tries
    ``<id>``, ``<rel_id>``, ``<identifier>`` in order so namespaced
    sources like musicbrainz (which carry the synthetic id under
    ``<rel_id>``, not the unrelated UUID attribute on ``<release>``) work
    without a per-source override.
    """
    tree = ET.parse(src_path)
    root = tree.getroot()
    entities = list(root)
    original = len(entities)
    for el in entities:
        if _xml_id_text(el) not in keep_ids:
            root.remove(el)
    kept = len(list(root))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_path, encoding="utf-8", xml_declaration=False)
    return original, kept


def _filter_inject_id_source(
    src_path: Path,
    dst_path: Path,
    keep_ids: set[str],
    source_name: str,
    source_format: str,
    reader_kwargs: dict[str, Any],
) -> tuple[int, int]:
    """Filter a source whose loader synthesises IDs at runtime (``inject_id``).

    The on-disk source has no id column; the loader synthesises
    ``f"{source_name}_{1-based-row-index}"`` per row. Naively filtering
    + re-writing the file would re-number the surviving rows from 1
    again, breaking the EM/fusion gold (which references the *original*
    ids — e.g. games' EM gold cites ``dbpedia_52062``, the 52062nd row
    of the original file). We solve this by **materialising the
    original ids as an explicit ``id`` column** in the downsampled
    output: the loader's inject_id path is a no-op when an ``id``
    column is already present, so the kept rows keep their original
    ids and the gold-protection chain holds.
    """
    if source_format == "csv":
        delimiter = reader_kwargs.get("delimiter", ",")
        with open(src_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty CSV source: {src_path}")
        header, data = rows[0], rows[1:]
        # Prepend an "id" column with the original 1-based row id.
        new_header = ["id"] + list(header)
        kept_rows: list[list[str]] = []
        for i, row in enumerate(data):
            rid = f"{source_name}_{i + 1}"
            if rid in keep_ids:
                kept_rows.append([rid, *row])
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(new_header)
            writer.writerows(kept_rows)
        return len(data), len(kept_rows)
    if source_format == "json":
        with open(src_path, encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array at {src_path}, got {type(records)}")
        kept_records: list[dict[str, Any]] = []
        for i, r in enumerate(records):
            rid = f"{source_name}_{i + 1}"
            if rid in keep_ids:
                # Preserve original key order with id first.
                kept_records.append({"id": rid, **r})
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(kept_records, f, ensure_ascii=False, indent=4)
        return len(records), len(kept_records)
    if source_format == "xml":
        tree = ET.parse(src_path)
        root = tree.getroot()
        entities = list(root)
        original = len(entities)
        kept_xml = 0
        for i, el in enumerate(entities):
            rid = f"{source_name}_{i + 1}"
            if rid not in keep_ids:
                root.remove(el)
            else:
                # Insert <id>{rid}</id> as the first child so the loader's
                # XML aggregator surfaces it as the ``id`` column.
                id_el = ET.Element("id")
                id_el.text = rid
                el.insert(0, id_el)
                kept_xml += 1
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(dst_path, encoding="utf-8", xml_declaration=False)
        return original, kept_xml
    raise ValueError(f"Unsupported inject_id format: {source_format!r}")


def _count_rows(
    src_path: Path,
    source_format: str,
    reader_kwargs: dict[str, Any],
) -> int:
    """Return the number of records in *src_path* without parsing ids."""
    if source_format == "csv":
        delimiter = reader_kwargs.get("delimiter", ",")
        with open(src_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
        if not rows:
            return 0
        return len(rows) - 1  # subtract header
    if source_format == "json":
        with open(src_path, encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array at {src_path}, got {type(records)}")
        return len(records)
    if source_format == "xml":
        tree = ET.parse(src_path)
        return len(list(tree.getroot()))
    raise ValueError(f"Unsupported source format: {source_format!r}")


def _count_source_ids(
    src_path: Path,
    source_format: str,
    reader_kwargs: dict[str, Any],
    *,
    inject_id: bool = False,
    source_name: str = "",
    id_column: str | None = None,
) -> list[str]:
    """Return the full list of ids present in a source file.

    When *inject_id* is True the on-disk file has no id column; instead
    the loader synthesises ``f"{source_name}_{1-based-row-index}"`` per
    row, so we count rows and synthesise the same id stream.

    ``id_column`` overrides CSV id-column detection (from the source's
    ``id_column`` field in the domain YAML).
    """
    if inject_id:
        if not source_name:
            raise ValueError("inject_id requires source_name")
        n_rows = _count_rows(src_path, source_format, reader_kwargs)
        return _synthesised_inject_ids(source_name, n_rows)
    if source_format == "json":
        with open(src_path, encoding="utf-8") as f:
            records = json.load(f)
        id_field = id_column if id_column else _json_id_field(records)
        return [str(r.get(id_field, "")) for r in records]
    if source_format == "csv":
        delimiter = reader_kwargs.get("delimiter", ",")
        with open(src_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
        header, data = rows[0], rows[1:]
        id_col = _detect_csv_id_column(header, override=id_column)
        id_idx = header.index(id_col)
        return [row[id_idx] for row in data if len(row) > id_idx]
    if source_format == "xml":
        tree = ET.parse(src_path)
        ids: list[str] = []
        for el in tree.getroot():
            text = _xml_id_text(el)
            if text:
                ids.append(text)
        return ids
    raise ValueError(f"Unsupported source format: {source_format!r}")


def downsample_domain(
    source_domain: str,
    target_domain: str,
    *,
    usecases_dir: Path = USECASES_DIR,
    config_dir: Path = CONFIG_DIR,
    gold_multiplier: float = 1.5,
    min_rows_per_source: int = 50,
    max_rows_per_source: int | None = None,
    seed: int = 0,
) -> DownsampleReport:
    """Produce a downsampled clone of ``source_domain`` at ``target_domain``.

    Parameters
    ----------
    source_domain : str
        Domain to downsample (must have a registered domain YAML).
    target_domain : str
        Name of the new domain. Must not already exist on disk.
    usecases_dir : Path
        Root of the use-cases directory (override in tests).
    config_dir : Path
        Root of the synthetic config directory (override in tests).
    gold_multiplier : float
        Target source size = ``gold_multiplier * |protected_ids|``.
        Values ``<= 1.0`` keep only the protected rows. Default ``1.5``.
    min_rows_per_source : int
        Floor on target rows per source (useful when gold coverage is
        tiny). Default ``50``.
    max_rows_per_source : int or None
        Optional cap on target rows per source.
    seed : int
        RNG seed for the extras sampler.

    Returns
    -------
    DownsampleReport
        Full accounting of the downsampling run.
    """
    config = load_domain_config(source_domain)
    source_prefixes = {s.name: s.id_prefix for s in config.sources}

    source_in = usecases_dir / source_domain / "input"
    target_in = usecases_dir / target_domain / "input"

    em_gold = _collect_em_ids(source_in / "entitymatching", source_prefixes)
    fusion_gold = _collect_fusion_ids(source_in / "fusion", source_prefixes)
    protected: dict[str, set[str]] = {
        name: em_gold.get(name, set()) | fusion_gold.get(name, set())
        for name in source_prefixes
    }

    rng = np.random.default_rng(seed)
    reports: list[SourceDownsampleReport] = []

    for spec in config.sources:
        src_path = source_in / "data" / spec.file
        dst_path = target_in / "data" / spec.file
        all_ids = _count_source_ids(
            src_path,
            spec.format,
            spec.reader_kwargs,
            inject_id=spec.inject_id,
            source_name=spec.name,
            id_column=spec.id_column,
        )

        keep = set(protected[spec.name])
        keep &= set(all_ids)
        target_total = max(min_rows_per_source, int(len(keep) * gold_multiplier))
        if max_rows_per_source is not None:
            target_total = min(target_total, max_rows_per_source)
        extras = _sample_extras(all_ids, keep, target_total, rng)
        keep |= extras

        if spec.inject_id:
            original, kept = _filter_inject_id_source(
                src_path,
                dst_path,
                keep,
                spec.name,
                spec.format,
                spec.reader_kwargs,
            )
        elif spec.format == "json":
            original, kept = _filter_json_source(src_path, dst_path, keep)
        elif spec.format == "csv":
            original, kept = _filter_csv_source(
                src_path,
                dst_path,
                keep,
                spec.reader_kwargs,
                id_column=spec.id_column,
            )
        elif spec.format == "xml":
            original, kept = _filter_xml_source(src_path, dst_path, keep)
        else:
            raise ValueError(f"Unsupported format: {spec.format!r}")

        reports.append(
            SourceDownsampleReport(
                name=spec.name,
                original_rows=original,
                protected_rows=len(protected[spec.name] & set(all_ids)),
                sampled_extra_rows=len(extras),
                output_rows=kept,
            )
        )

    em_copied = _copy_tree(source_in / "entitymatching", target_in / "entitymatching")
    fusion_copied = _copy_tree(source_in / "fusion", target_in / "fusion")
    sm_copied = _copy_tree(source_in / "schemamatching", target_in / "schemamatching")

    retained_ids = {
        spec.name: set(
            _count_source_ids(
                target_in / "data" / spec.file,
                spec.format,
                spec.reader_kwargs,
                inject_id=spec.inject_id,
                source_name=spec.name,
                id_column=spec.id_column,
            )
        )
        for spec in config.sources
    }
    pool_original, pool_kept = _filter_pool(
        source_domain=source_domain,
        target_domain=target_domain,
        retained_ids_by_source=retained_ids,
        pools_dir=POOLS_DIR,
    )

    # Propagate the parent's ``data_root`` (if set) so the downsampled
    # domain reads + writes from the same root. Stored relative to
    # REPO_ROOT to keep the YAML repo-portable.
    data_root_rel: str | None = None
    if config.data_root is not None:
        try:
            data_root_rel = str(config.data_root.relative_to(REPO_ROOT))
        except ValueError:
            data_root_rel = str(config.data_root)

    _write_target_domain_yaml(
        target_domain=target_domain,
        source_domain=source_domain,
        sources=config.sources,
        attribute_classes=config.attribute_classes,
        source_pairs=config.source_pairs,
        master_seed=config.master_seed,
        config_dir=config_dir,
        fusion_files=dict(config.fusion_files) if config.fusion_files else None,
        data_root=data_root_rel,
    )

    return DownsampleReport(
        source_domain=source_domain,
        target_domain=target_domain,
        seed=seed,
        sources=reports,
        em_files_copied=em_copied,
        fusion_files_copied=fusion_copied,
        sm_files_copied=sm_copied,
        pool_original_rows=pool_original,
        pool_output_rows=pool_kept,
    )


def _filter_pool(
    *,
    source_domain: str,
    target_domain: str,
    retained_ids_by_source: dict[str, set[str]],
    pools_dir: Path,
) -> tuple[int, int]:
    """Filter the source pool CSV to rows whose ids survive downsampling.

    Returns ``(original_row_count, kept_row_count)``. Returns ``(0, 0)``
    when the source pool is absent.
    """
    src_pool = pools_dir / source_domain / "pooled_positives.csv"
    if not src_pool.exists():
        return (0, 0)
    with open(src_pool, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    retained_all = set().union(*retained_ids_by_source.values())
    kept = [r for r in rows if r["id1"] in retained_all and r["id2"] in retained_all]

    dst_dir = pools_dir / target_domain
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_dir / "pooled_positives.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return len(rows), len(kept)


def _copy_tree(src_dir: Path, dst_dir: Path) -> list[str]:
    """Copy every file from ``src_dir`` to ``dst_dir``. Returns filenames."""
    if not src_dir.exists():
        return []
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(src_dir.iterdir()):
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)
            copied.append(path.name)
    return copied


def _write_target_domain_yaml(
    *,
    target_domain: str,
    source_domain: str,
    sources: list[SourceSpec],
    attribute_classes: dict[str, str],
    source_pairs: list[tuple[str, str]],
    master_seed: int,
    config_dir: Path,
    fusion_files: dict[str, str] | None = None,
    data_root: str | None = None,
) -> Path:
    """Write the target domain's YAML with ``knob_config_alias`` set.

    Propagates ``fusion_files`` and ``data_root`` from the source domain
    when set so the target uses the same fusion gold filename pair (e.g.
    music's ``validation_set_final.xml`` / ``test_set_final.xml``) and
    the same on-disk root (e.g. products' synthetic-side data_root).
    """
    payload: dict[str, Any] = {
        "domain": target_domain,
        "master_seed": master_seed,
        "knob_config_alias": source_domain,
        "sources": [
            {
                "name": s.name,
                "file": s.file,
                "format": s.format,
                "id_prefix": s.id_prefix,
                **({"reader_kwargs": s.reader_kwargs} if s.reader_kwargs else {}),
                **({"inject_id": True} if s.inject_id else {}),
                **({"id_column": s.id_column} if s.id_column else {}),
            }
            for s in sources
        ],
        "source_pairs": [list(pair) for pair in source_pairs],
        "attribute_classes": dict(attribute_classes),
    }
    if fusion_files:
        payload["fusion_files"] = dict(fusion_files)
    if data_root:
        payload["data_root"] = data_root
    out_path = config_dir / "domains" / f"{target_domain}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out_path


def _format_report(report: DownsampleReport) -> str:
    """Human-readable summary for CLI output."""
    lines = [
        f"Downsampled {report.source_domain} -> {report.target_domain}"
        f" (seed={report.seed})",
        "",
        f"{'source':<16} {'orig':>8} {'protected':>10} {'extras':>8} {'output':>8}",
    ]
    for s in report.sources:
        lines.append(
            f"{s.name:<16} {s.original_rows:>8} {s.protected_rows:>10} "
            f"{s.sampled_extra_rows:>8} {s.output_rows:>8}"
        )
    lines.append("")
    lines.append(f"EM gold files copied: {len(report.em_files_copied)}")
    lines.append(f"Fusion gold files copied: {len(report.fusion_files_copied)}")
    lines.append(f"Schema-matching files copied: {len(report.sm_files_copied)}")
    lines.append(f"Pool: {report.pool_original_rows} -> {report.pool_output_rows} rows")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument(
        "--gold-multiplier",
        type=float,
        default=1.5,
        help="Target source size as a multiple of the protected-id count.",
    )
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from usecases_synthetic.lib.domain_config import data_root_for_domain

    report = downsample_domain(
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        usecases_dir=data_root_for_domain(args.source_domain) or USECASES_DIR,
        gold_multiplier=args.gold_multiplier,
        min_rows_per_source=args.min_rows,
        max_rows_per_source=args.max_rows,
        seed=args.seed,
    )
    print(_format_report(report))


if __name__ == "__main__":
    main()

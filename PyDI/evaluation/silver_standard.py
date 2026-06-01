"""
Silver-standard data contract for end-to-end pipeline evaluation.

The :class:`SilverStandard` bundle is the reference dataset against which
the pipeline's fused output is compared. Two concrete loaders ship with
PyDI:

* :func:`load_workflow_silver` — parses the hand-authored fusion gold
  XML files (``input/fusion/validation_set.xml`` +
  ``input/fusion/test_set.xml``) shipped with each non-synthetic use
  case under ``usecases/<domain>/``.
* :func:`load_synthetic_silver` — reads the silver CSV produced by
  ``usecases_synthetic/lib/fusion_silver_standard.py`` (each domain's
  workflow-notebook fusion stack applied to the pooled clusters).

Both return the same :class:`SilverStandard` shape; the runner does not
care which loader produced it.
"""

from __future__ import annotations

import ast
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default source-prefix maps for the synthetic loader. Lifted from
# ``usecases_synthetic/lib/fusion_silver_standard.py`` so callers don't
# have to pass them for the in-tree domains.
# ---------------------------------------------------------------------------

_DEFAULT_PREFIX_MAPS: Dict[str, Dict[str, str]] = {
    "music": {
        "mbrainz_": "musicbrainz",
        "discogs_": "discogs",
        "lastFM_": "lastfm",
    },
    "games": {
        "metacritic_": "metacritic",
        "sales_": "sales",
        "dbpedia_": "dbpedia",
    },
    "companies": {
        "dbpedia_": "dbpedia",
        "fullcontact_": "fullcontact",
        "forbes_": "forbes",
    },
    "products": {
        "alternate_": "alternate",
        "buy_": "buy",
        "ebay_": "ebay",
        "newegg_": "newegg",
    },
}


@dataclass(frozen=True)
class SilverStandard:
    """Reference dataset for end-to-end pipeline evaluation.

    Attributes
    ----------
    fused : pandas.DataFrame
        Per-cluster fused values. Columns: ``cluster_id`` plus the
        fused attribute set. One row per cluster.
    membership : pandas.DataFrame
        Long-form ``(record_id, source, cluster_id)`` used for the
        alignment-based clustering metrics. ``record_id`` is the source
        record identifier (string); ``source`` is the dataset name;
        ``cluster_id`` matches a value in ``fused["cluster_id"]``.
    cell_provenance : pandas.DataFrame or None
        Long-form ``(cluster_id, attribute, source_ids)`` where
        ``source_ids`` is a list of source record identifiers that won
        the fused cell. ``None`` when the silver doesn't carry per-cell
        provenance (e.g. the synthetic CSV format, which only records
        cluster-level membership). The source-attribution and
        synthesis-rate metrics skip with a panel warning when this is
        ``None``.
    """

    fused: pd.DataFrame
    membership: pd.DataFrame
    cell_provenance: Optional[pd.DataFrame]


# ---------------------------------------------------------------------------
# Workflow XML loader (non-synthetic use cases)
# ---------------------------------------------------------------------------


def _infer_source_from_id(
    record_id: str, prefix_map: Mapping[str, str]
) -> Optional[str]:
    """Map a record id like ``mbrainz_974`` to its source name via prefix."""
    for prefix, source in prefix_map.items():
        if record_id.startswith(prefix):
            return source
    return None


def _parse_provenance_token(token: str) -> list[str]:
    """Split a composite provenance string like ``"A+B"`` into ``["A", "B"]``."""
    token = token.strip()
    if not token:
        return []
    return [part.strip() for part in token.split("+") if part.strip()]


def _coerce_xml_value(text: Optional[str]) -> Any:
    """Decode a text payload from a fusion-gold XML element.

    The XML serialization stores list values via Python ``repr``
    (``"['a', 'b']"``); scalars are stored as raw strings. We try
    ``ast.literal_eval`` first and fall back to the trimmed string.
    """
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return stripped


def _load_workflow_xml(
    xml_path: Path,
    *,
    record_tag: str,
    id_tag: str,
    prefix_map: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    """Parse one fusion-gold XML file into (fused_rows, membership_rows, cell_provenance_rows)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fused_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, str]] = []
    cell_provenance_rows: list[dict[str, Any]] = []

    for record_elem in root.findall(record_tag):
        id_elem = record_elem.find(id_tag)
        if id_elem is None or id_elem.text is None:
            logger.warning(
                "Skipping %s record without %s element in %s",
                record_tag,
                id_tag,
                xml_path,
            )
            continue
        cluster_id = id_elem.text.strip()

        fused_row: dict[str, Any] = {"cluster_id": cluster_id}
        record_member_ids: set[str] = set()

        for attr_elem in record_elem:
            if attr_elem.tag == id_tag:
                continue
            attribute = attr_elem.tag
            fused_row[attribute] = _coerce_xml_value(attr_elem.text)

            provenance_attr = attr_elem.attrib.get("provenance", "").strip()
            if not provenance_attr:
                continue

            source_ids = _parse_provenance_token(provenance_attr)
            cell_provenance_rows.append(
                {
                    "cluster_id": cluster_id,
                    "attribute": attribute,
                    "source_ids": source_ids,
                }
            )
            record_member_ids.update(source_ids)

        record_member_ids.add(cluster_id)

        for member_id in sorted(record_member_ids):
            source = _infer_source_from_id(member_id, prefix_map) or "unknown"
            membership_rows.append(
                {
                    "record_id": member_id,
                    "source": source,
                    "cluster_id": cluster_id,
                }
            )

        fused_rows.append(fused_row)

    return fused_rows, membership_rows, cell_provenance_rows


def load_workflow_silver(
    usecase_dir: Union[str, Path],
    *,
    record_tag: Optional[str] = None,
    id_tag: str = "id",
    prefix_map: Optional[Mapping[str, str]] = None,
    domain: Optional[str] = None,
    include_validation: bool = False,
) -> SilverStandard:
    """Fusion test silver (test_set.xml only) from the workflow XML files.

    Parses ``input/fusion/test_set.xml`` under ``usecase_dir`` by
    default. Each ``<record_tag>`` element becomes one cluster; its
    ``<id>`` text is the canonical cluster id. Per-attribute elements
    carry a ``provenance="source_id"`` (or composite ``"A+B"`` for
    union/synthesis) attribute that names the source record(s) the
    value came from — used for the per-cell provenance table.

    Parameters
    ----------
    usecase_dir : str or pathlib.Path
        Use-case root, e.g. ``usecases/music``. The loader looks under
        ``input/fusion/test_set.xml`` (and ``validation_set.xml`` when
        ``include_validation=True``).
    record_tag : str, optional
        Outer record element tag. If ``None``, derived from the XML
        root: the first child element's tag is used.
    id_tag : str, default ``"id"``
        Child element holding the canonical cluster id.
    prefix_map : mapping, optional
        Maps source-id prefix → source name. Used to attribute each
        member id to a source dataset. Defaults to the in-tree map
        for *domain* if provided, else inferred from
        ``usecase_dir.name``.
    domain : str, optional
        Domain name used to look up :data:`_DEFAULT_PREFIX_MAPS` when
        ``prefix_map`` is ``None``. Defaults to ``usecase_dir.name``.
    include_validation : bool, default ``False``
        When ``True``, also load ``validation_set.xml`` and concatenate
        with the test set (legacy behaviour). The default is to load
        the test set only — the fusion *test* set is the gold reference
        for end-to-end pipeline evaluation; the validation set is for
        tuning and should not bleed into reported metrics.

    Returns
    -------
    SilverStandard
        Fusion test silver. ``fused`` carries one row per cluster;
        ``membership`` is long-form; ``cell_provenance`` is populated
        from the XML ``provenance`` attributes.
    """
    usecase_dir = Path(usecase_dir)
    fusion_dir = usecase_dir / "input" / "fusion"

    test_path = fusion_dir / "test_set.xml"
    if include_validation:
        val_path = fusion_dir / "validation_set.xml"
        paths = [p for p in (val_path, test_path) if p.exists()]
        if not paths:
            raise FileNotFoundError(
                f"No validation_set.xml or test_set.xml under {fusion_dir}"
            )
    else:
        if not test_path.exists():
            raise FileNotFoundError(f"No test_set.xml under {fusion_dir}")
        paths = [test_path]

    resolved_domain = domain or usecase_dir.name
    resolved_prefix_map: Mapping[str, str] = (
        prefix_map
        if prefix_map is not None
        else _DEFAULT_PREFIX_MAPS.get(resolved_domain, {})
    )
    if not resolved_prefix_map:
        logger.warning(
            "No source prefix map for domain '%s'; membership rows will use 'unknown' for source",
            resolved_domain,
        )

    if record_tag is None:
        tree = ET.parse(paths[0])
        root = tree.getroot()
        first_child = next(iter(root), None)
        if first_child is None:
            raise ValueError(f"Empty fusion XML: {paths[0]}")
        record_tag = first_child.tag

    fused_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, str]] = []
    cell_provenance_rows: list[dict[str, Any]] = []

    for path in paths:
        f_rows, m_rows, p_rows = _load_workflow_xml(
            path, record_tag=record_tag, id_tag=id_tag, prefix_map=resolved_prefix_map
        )
        fused_rows.extend(f_rows)
        membership_rows.extend(m_rows)
        cell_provenance_rows.extend(p_rows)

    fused = pd.DataFrame(fused_rows)
    membership = pd.DataFrame(
        membership_rows, columns=["record_id", "source", "cluster_id"]
    ).drop_duplicates(ignore_index=True)
    cell_provenance: Optional[pd.DataFrame]
    if cell_provenance_rows:
        cell_provenance = pd.DataFrame(
            cell_provenance_rows, columns=["cluster_id", "attribute", "source_ids"]
        )
    else:
        cell_provenance = None

    logger.info(
        "load_workflow_silver(%s): %d clusters, %d membership rows, %d provenance cells",
        usecase_dir.name,
        len(fused),
        len(membership),
        0 if cell_provenance is None else len(cell_provenance),
    )

    return SilverStandard(
        fused=fused, membership=membership, cell_provenance=cell_provenance
    )


# ---------------------------------------------------------------------------
# Synthetic loader
# ---------------------------------------------------------------------------


def load_synthetic_silver(
    domain: str,
    *,
    baselines_dir: Union[str, Path, None] = None,
    prefix_map: Optional[Mapping[str, str]] = None,
) -> SilverStandard:
    """Load the synthetic silver standard produced by ``fusion_silver_standard``.

    Reads the long-format CSV at
    ``<baselines_dir>/<domain>/fusion_silver_standard.csv`` and pivots
    to one row per cluster. Cluster membership is rebuilt from the
    ``source_ids`` column (comma-joined member id list); per-cell
    provenance is not present in the current artifact format and
    therefore ``cell_provenance`` is returned as ``None``.

    Parameters
    ----------
    domain : str
        Domain name (``music``, ``games``, ``companies``, ...).
    baselines_dir : str or pathlib.Path, optional
        Root directory containing per-domain silver artifacts. Defaults
        to ``usecases_synthetic/baselines`` relative to the current
        working directory.
    prefix_map : mapping, optional
        Maps source-id prefix → source name. Defaults to
        :data:`_DEFAULT_PREFIX_MAPS` for *domain*.

    Returns
    -------
    SilverStandard
        ``fused`` is one row per cluster; ``membership`` is long-form;
        ``cell_provenance`` is ``None`` (see notes).

    Notes
    -----
    The current synthetic silver CSV only carries cluster-level
    member ids in the ``source_ids`` column — there is no per-cell
    winning-source column. The source-attribution (§3.7.2) and
    synthesis-rate (§3.7.7) metrics in the panel skip with a warning
    when ``cell_provenance`` is ``None``. Extending the silver builder
    to emit per-cell provenance is a separate workstream.
    """
    if baselines_dir is None:
        baselines_dir = Path("usecases_synthetic") / "baselines"
    baselines_dir = Path(baselines_dir)

    csv_path = baselines_dir / domain / "fusion_silver_standard.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Synthetic silver CSV not found: {csv_path}")

    long_silver = pd.read_csv(csv_path)
    expected_cols = {"cluster_id", "attribute", "fused_value", "source_ids"}
    missing = expected_cols - set(long_silver.columns)
    if missing:
        raise ValueError(
            f"Synthetic silver CSV {csv_path} missing columns: {sorted(missing)}"
        )

    long_silver["fused_value"] = long_silver["fused_value"].map(_coerce_csv_value)

    fused = long_silver.pivot_table(
        index="cluster_id",
        columns="attribute",
        values="fused_value",
        aggfunc="first",
    ).reset_index()
    fused.columns.name = None

    resolved_prefix_map: Mapping[str, str] = (
        prefix_map if prefix_map is not None else _DEFAULT_PREFIX_MAPS.get(domain, {})
    )

    membership_rows: list[dict[str, str]] = []
    seen_member_per_cluster: dict[str, set[str]] = {}
    for _, row in long_silver[["cluster_id", "source_ids"]].iterrows():
        cluster_id = str(row["cluster_id"])
        source_ids_raw = row["source_ids"]
        if pd.isna(source_ids_raw):
            continue
        member_ids = [
            mid.strip() for mid in str(source_ids_raw).split(",") if mid.strip()
        ]
        bucket = seen_member_per_cluster.setdefault(cluster_id, set())
        for member_id in member_ids:
            if member_id in bucket:
                continue
            bucket.add(member_id)
            source = _infer_source_from_id(member_id, resolved_prefix_map) or "unknown"
            membership_rows.append(
                {
                    "record_id": member_id,
                    "source": source,
                    "cluster_id": cluster_id,
                }
            )

    membership = pd.DataFrame(
        membership_rows, columns=["record_id", "source", "cluster_id"]
    ).drop_duplicates(ignore_index=True)

    logger.info(
        "load_synthetic_silver(%s): %d clusters, %d membership rows (per-cell provenance unavailable)",
        domain,
        len(fused),
        len(membership),
    )

    return SilverStandard(fused=fused, membership=membership, cell_provenance=None)


def _coerce_csv_value(value: Any) -> Any:
    """Best-effort restore of list values that were CSV-stringified.

    The synthetic silver CSV stores list-typed cells as Python-repr
    strings (e.g. ``"['Track A', 'Track B']"``). Numerical / string /
    NaN values pass through unchanged.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return value
    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        return value
    return value

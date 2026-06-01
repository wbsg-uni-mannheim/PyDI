#!/usr/bin/env python3
"""Bridge PyDI EM gold CSVs to Ditto WDC-style json.gz pair files.

Reads the PyDI ``usecases/<domain>/input/entitymatching/<pair>_<split>.csv``
gold file, joins each ``id1`` / ``id2`` against the corresponding source
DataFrame, applies the per-source ``attribute_mapping`` from the knob 02 YAML
to normalise columns onto the canonical schema, and emits a Ditto-compatible
WDC ``json.gz`` file with ``{field}_left`` / ``{field}_right`` columns.

This is the one-way bridge between PyDI's source-record EM format and Ditto's
pair-text format. Used by the D5 smoke test and the D8 production retrain.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usecases_synthetic.lib.column_mapping import apply_column_mapping
from usecases_synthetic.lib.committee_paths import (
    canonical_committee_domain,
    resolve_committee_path,
)
from usecases_synthetic.lib.domain_config import (
    SYNTHETIC_DIR,
    USECASES_DIR,
    data_root_for_domain,
)
from usecases_synthetic.lib.domain_value_norm import get_value_normalizer
from usecases_synthetic.lib.loaders import load_domain_sources, read_em_gold_csv
from usecases_synthetic.lib.sc_block_train import DOMAIN_TEXT_COLS

KNOB02_CONFIG_DIR = SYNTHETIC_DIR / "config" / "knob_02_niche"
COMMITTEE_CONFIG_DIR = SYNTHETIC_DIR / "config" / "committees"


def _load_knob02_config(domain: str) -> dict[str, Any]:
    """Load the knob 02 YAML for ``domain``.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).

    Returns
    -------
    dict
        Parsed YAML payload.
    """
    path = KNOB02_CONFIG_DIR / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing knob 02 config for domain {domain!r}: expected {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_pair(pair: str) -> tuple[str, str]:
    """Split a ``<src1>_2_<src2>`` pair string into its two source names."""
    if "_2_" not in pair:
        raise ValueError(
            f"Invalid --pair {pair!r}: expected '<src1>_2_<src2>' (e.g. 'forbes_2_dbpedia')"
        )
    src1, src2 = pair.split("_2_", 1)
    if not src1 or not src2:
        raise ValueError(f"Invalid --pair {pair!r}: empty source name")
    return src1, src2


def _normalize_label(value: Any) -> int:
    """Normalise EM gold label values to ``0`` / ``1``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        iv = int(value)
        if iv in {0, 1}:
            return iv
        raise ValueError(f"unexpected numeric label: {value!r}")
    if isinstance(value, bool):
        return int(value)
    s = str(value).strip().upper()
    if s in {"1", "TRUE", "T", "YES", "Y"}:
        return 1
    if s in {"0", "FALSE", "F", "NO", "N"}:
        return 0
    raise ValueError(f"unexpected label value: {value!r}")


def _canonical_record(
    row: pd.Series,
    source_name: str,
    attribute_mapping: dict[str, dict[str, str]],
    canonical_schema: list[str],
    value_normalize: dict[str, Callable[[Any], Any]] | None = None,
) -> dict[str, str]:
    """Project a source row onto the canonical schema.

    Returns a dict keyed by canonical field name. Missing columns yield
    an empty string so downstream serialization skips them cleanly.

    Parameters
    ----------
    value_normalize : dict mapping canonical field name to callable, optional
        When provided, the callable is applied to the raw value of that
        field before ``str()`` coercion (per
        ``lib/domain_value_norm.get_value_normalizer``). Fields not in
        the mapping pass through untransformed. Used by the Ditto A/B
        retrain (``--normalize`` flag on ``_prep_<domain>.py``).
    """
    mapping = attribute_mapping.get(source_name, {})
    inverse: dict[str, str] = {}
    for source_col, canonical in mapping.items():
        inverse.setdefault(canonical, source_col)

    record: dict[str, str] = {}
    for field in canonical_schema:
        source_col_opt: str | None = inverse.get(field)
        if source_col_opt is None or source_col_opt not in row.index:
            record[field] = ""
            continue
        raw = row[source_col_opt]
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            record[field] = ""
        else:
            if value_normalize is not None and field in value_normalize:
                raw = value_normalize[field](raw)
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    record[field] = ""
                    continue
            record[field] = str(raw)
    return record


def _build_index(
    df: pd.DataFrame,
    id_column: str,
    source_name: str,
) -> dict[str, pd.Series]:
    """Index a source DataFrame by its id column for O(1) lookup."""
    if id_column not in df.columns:
        raise KeyError(
            f"source {source_name!r} is missing id column {id_column!r}; "
            f"available columns: {sorted(df.columns)[:10]}"
        )
    dup_mask = df[id_column].duplicated()
    if dup_mask.any():
        dup_ids = df.loc[dup_mask, id_column].head(3).tolist()
        print(
            f"[warn] source {source_name!r} has {int(dup_mask.sum())} duplicate ids "
            f"in {id_column!r} (e.g. {dup_ids}); keeping first occurrence",
            file=sys.stderr,
        )
    index: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        key = str(row[id_column])
        if key not in index:
            index[key] = row
    return index


def build_ditto_pair_records_from_gold(
    gold: pd.DataFrame,
    domain: str,
    src1: str,
    src2: str,
    *,
    normalize: bool = False,
    sources: dict[str, pd.DataFrame] | None = None,
) -> list[dict[str, Any]]:
    """Build Ditto pair records from an in-memory EM gold DataFrame.

    Useful for callers (e.g. the D5 tiny-prep script) that need to pass a
    sampled subset of a gold CSV rather than a full ``<pair>_<split>`` file.
    ``gold`` must have columns ``id1``, ``id2``, ``label``.

    Parameters
    ----------
    normalize : bool, default False
        When ``True``, looks up a per-domain value normaliser via
        :func:`usecases_synthetic.lib.domain_value_norm.get_value_normalizer`
        and applies it to each canonical field before serialisation.
        Currently configured for ``games`` only (platform alias + title
        cleanup); domains without a configured normaliser raise
        ``ValueError`` so the caller catches missing specs rather than
        silently shipping raw values.
    sources : dict of str to DataFrame, optional
        Source DataFrames to join the gold against, keyed by source name,
        carrying the *baseline* (pre-K8) column names that the knob-02
        ``attribute_mapping`` expects. Defaults to
        :func:`load_domain_sources` (the baseline sources). R10-G passes
        the K8-reversed *variant* sources here so the variant Ditto
        checkpoint trains on the perturbed values under the same
        canonical schema.
    """
    cfg = _load_knob02_config(domain)
    id_columns: dict[str, str] = cfg.get("id_columns", {})
    attribute_mapping: dict[str, dict[str, str]] = cfg.get("attribute_mapping", {})
    canonical_schema: list[str] = list(cfg.get("canonical_schema", []))
    if not canonical_schema:
        raise ValueError(f"knob 02 config for {domain!r} has no canonical_schema")
    for src in (src1, src2):
        if src not in id_columns:
            raise KeyError(
                f"id_columns missing {src!r} in knob 02 config for {domain!r}"
            )

    value_normalize: dict[str, Callable[[Any], Any]] | None = None
    if normalize:
        value_normalize = get_value_normalizer(domain)
        if value_normalize is None:
            raise ValueError(
                f"normalize=True but no value normaliser configured for "
                f"domain {domain!r}; add an entry to "
                f"usecases_synthetic.lib.domain_value_norm._DOMAIN_NORMALIZERS"
            )

    if sources is None:
        sources = load_domain_sources(domain)
    index1 = _build_index(sources[src1], id_columns[src1], src1)
    index2 = _build_index(sources[src2], id_columns[src2], src2)

    records: list[dict[str, Any]] = []
    missing1 = 0
    missing2 = 0
    for _, row in gold.iterrows():
        id1 = str(row["id1"])
        id2 = str(row["id2"])
        row1 = index1.get(id1)
        row2 = index2.get(id2)
        if row1 is None:
            missing1 += 1
            continue
        if row2 is None:
            missing2 += 1
            continue
        rec1 = _canonical_record(
            row1, src1, attribute_mapping, canonical_schema, value_normalize
        )
        rec2 = _canonical_record(
            row2, src2, attribute_mapping, canonical_schema, value_normalize
        )
        out: dict[str, Any] = {
            "id_left": id1,
            "id_right": id2,
            "pair_id": f"{id1}__{id2}",
            "label": _normalize_label(row["label"]),
            "is_hard_negative": 0,
        }
        for field in canonical_schema:
            out[f"{field}_left"] = rec1[field]
            out[f"{field}_right"] = rec2[field]
        records.append(out)

    if missing1 or missing2:
        print(
            f"[warn] dropped {missing1} rows with unknown {src1} id "
            f"and {missing2} rows with unknown {src2} id "
            f"out of {len(gold)} gold pairs",
            file=sys.stderr,
        )
    return records


# ---------------------------------------------------------------------------
# Committee-scope (wide) WDC record builder — R10-I.
#
# The committee EM runner serializes the *wide* per-domain schema
# (``ditto_plm.fields`` == ``sc_block.text_cols`` == ``DOMAIN_TEXT_COLS``,
# guard-enforced by tests/test_em_field_scope_consistency.py). At inference
# it column-maps each source onto the canonical field names, then the
# DittoMatcher emits ``COL <field> VAL <value>`` over those fields, dropping
# empties. The legacy ``build_ditto_pair_records_from_gold`` above projects
# onto the *narrow* knob-02 ``canonical_schema`` via ``attribute_mapping``,
# so the baseline + variant Ditto checkpoints used to train on a different
# (narrower) surface than wide inference serializes. The functions below
# build records on the same column-mapped + wide-field surface the committee
# runner uses, so training and inference serialize byte-identically.
# ---------------------------------------------------------------------------


def committee_ditto_fields(domain: str) -> list[str]:
    """Return the Ditto serialization field scope for ``domain``.

    This is ``DOMAIN_TEXT_COLS[domain]`` — the single source of truth that
    ``tests/test_em_field_scope_consistency.py`` asserts equals the matching
    committee's ``ditto_plm.fields`` and the blocking committee's
    ``sc_block.text_cols`` — **minus any field whose name collides with a
    Ditto WDC reserved metadata key** (``RESERVED_SERIALIZATION_FIELDS``,
    e.g. music's ``label`` record-label attribute). Ditto's WDC format uses
    bare ``label`` / ``pair_id`` / ``id`` keys for the pair's metadata, so an
    attribute named ``label`` cannot be a serialization field there — both
    ``wdc_to_pair_examples`` (training) and :class:`DittoMatcher` (inference)
    drop it, so train and inference stay byte-identical. Other committee
    members (Magellan / LLM / ComEM) keep the full list. See the music
    matching committee YAML header note.
    """
    # Lazy import: the reserved-field set lives in the torch-laden Ditto data
    # module; this config helper stays importable without torch.
    from usecases_synthetic.third_party.ditto_modern.data import (
        RESERVED_SERIALIZATION_FIELDS,
    )

    cdomain = canonical_committee_domain(domain)
    try:
        raw = DOMAIN_TEXT_COLS[cdomain]
    except KeyError as exc:
        raise KeyError(
            f"No DOMAIN_TEXT_COLS entry for domain {domain!r} "
            f"(canonical {cdomain!r}); known: {sorted(DOMAIN_TEXT_COLS)}"
        ) from exc
    return [f for f in raw if f not in RESERVED_SERIALIZATION_FIELDS]


def committee_column_mapping(domain: str) -> dict[str, dict[str, str]]:
    """Read the per-domain EM matching committee ``column_mapping`` block.

    ``{source: {orig_col: canonical_col}}`` — identical to the blocking
    committee's block (the committee runner enforces identity at load), so
    column-mapping a source here yields exactly the canonical column names
    the DittoMatcher reads at inference. Baseline callers pass the result
    straight through; variant callers first translate it through K8 renames
    via ``VariantBundle.resolve_column_mapping``.
    """
    path = resolve_committee_path(
        "em_matching_committee", domain, committee_dir=COMMITTEE_CONFIG_DIR
    )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    static = raw.get("column_mapping") or {}
    return {src: dict(mapping or {}) for src, mapping in static.items()}


def write_committee_fields_sidecar(output_dir: Path, domain: str) -> Path:
    """Write the canonical wide field scope to ``<output_dir>/fields.txt``.

    Records the exact comma-joined ``--fields`` the ``json.gz`` files in
    ``output_dir`` were built for (R10-I train-fields wiring). A downstream
    Ditto train run should pass ``train.py --domain <domain>`` (which reads
    the same ``DOMAIN_TEXT_COLS`` source of truth) so it can never train on
    a narrower field set than wide inference serializes; this sidecar is the
    audit trail for what scope the data carries.
    """
    fields = committee_ditto_fields(domain)
    path = Path(output_dir) / "fields.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(fields) + "\n", encoding="utf-8")
    return path


def _committee_record(
    row: pd.Series,
    fields: Sequence[str],
    value_normalize: dict[str, Callable[[Any], Any]] | None = None,
) -> dict[str, str]:
    """Project an already-column-mapped source row onto ``fields``.

    The row's columns have been renamed onto the canonical committee schema
    by :func:`apply_column_mapping`, so each ``field`` is read directly.
    Missing columns / NA values yield an empty string — dropped by Ditto's
    serializer, matching the committee runner's inference serialization.
    """
    record: dict[str, str] = {}
    for field in fields:
        if field not in row.index:
            record[field] = ""
            continue
        raw = row[field]
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            record[field] = ""
            continue
        if value_normalize is not None and field in value_normalize:
            raw = value_normalize[field](raw)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                record[field] = ""
                continue
        record[field] = str(raw)
    return record


def build_ditto_pair_records_committee_scope(
    gold: pd.DataFrame,
    domain: str,
    src1: str,
    src2: str,
    *,
    sources: dict[str, pd.DataFrame],
    fields: Sequence[str] | None = None,
    column_mapping: dict[str, dict[str, str]] | None = None,
    id_column: str = "id",
    normalize: bool = False,
) -> list[dict[str, Any]]:
    """Build wide, inference-consistent Ditto pair records from EM gold.

    Mirrors the committee EM runner's inference path so the trained Ditto
    checkpoint serializes the same surface it scores against:

    1. Column-map each source onto the canonical committee schema
       (``apply_column_mapping``), exactly as the runner does before calling
       ``DittoMatcher.match``.
    2. Join each gold pair's ``id1`` / ``id2`` against the mapped sources by
       ``id_column`` (the loader renames every source's primary id to
       ``id``).
    3. Emit ``{field}_left`` / ``{field}_right`` over ``fields`` for every
       field in the wide committee scope; missing values are empty strings
       (Ditto's serializer drops them).

    Parameters
    ----------
    sources : dict of str to DataFrame
        Source DataFrames keyed by source name. Baseline callers pass
        ``load_domain_sources(domain)``; variant callers pass the K8-renamed
        ``VariantBundle.sources``.
    fields : sequence of str, optional
        Canonical fields to serialize. Defaults to
        :func:`committee_ditto_fields` (``DOMAIN_TEXT_COLS[domain]``).
    column_mapping : dict, optional
        ``{source: {orig_col: canonical_col}}``. Defaults to
        :func:`committee_column_mapping`. Variant callers pass a mapping
        already translated through K8 renames
        (``bundle.resolve_column_mapping(committee_column_mapping(domain))``).
    id_column : str, default ``"id"``
        Join key present on every source after the loader's id rename.
    normalize : bool, default False
        Apply the per-domain value normaliser
        (:func:`usecases_synthetic.lib.domain_value_norm.get_value_normalizer`)
        before serialisation. Raises if ``True`` and no normaliser is
        configured for the domain.
    """
    cdomain = canonical_committee_domain(domain)
    field_list = list(fields) if fields is not None else committee_ditto_fields(cdomain)
    mapping = (
        column_mapping
        if column_mapping is not None
        else committee_column_mapping(cdomain)
    )

    value_normalize: dict[str, Callable[[Any], Any]] | None = None
    if normalize:
        value_normalize = get_value_normalizer(cdomain)
        if value_normalize is None:
            raise ValueError(
                f"normalize=True but no value normaliser configured for "
                f"domain {domain!r}; add an entry to "
                f"usecases_synthetic.lib.domain_value_norm._DOMAIN_NORMALIZERS"
            )

    mapped: dict[str, pd.DataFrame] = {}
    for src in (src1, src2):
        if src not in sources:
            raise KeyError(f"sources missing {src!r}; available: {sorted(sources)}")
        df = sources[src]
        src_map = mapping.get(src, {})
        mapped[src] = apply_column_mapping(df, src_map) if src_map else df.copy()

    index1 = _build_index(mapped[src1], id_column, src1)
    index2 = _build_index(mapped[src2], id_column, src2)

    records: list[dict[str, Any]] = []
    missing1 = 0
    missing2 = 0
    for _, gold_row in gold.iterrows():
        id1 = str(gold_row["id1"])
        id2 = str(gold_row["id2"])
        row1 = index1.get(id1)
        row2 = index2.get(id2)
        if row1 is None:
            missing1 += 1
            continue
        if row2 is None:
            missing2 += 1
            continue
        rec1 = _committee_record(row1, field_list, value_normalize)
        rec2 = _committee_record(row2, field_list, value_normalize)
        out: dict[str, Any] = {
            "id_left": id1,
            "id_right": id2,
            "pair_id": f"{id1}__{id2}",
            "label": _normalize_label(gold_row["label"]),
            "is_hard_negative": 0,
        }
        for field in field_list:
            out[f"{field}_left"] = rec1[field]
            out[f"{field}_right"] = rec2[field]
        records.append(out)

    if missing1 or missing2:
        print(
            f"[warn] dropped {missing1} rows with unknown {src1} id "
            f"and {missing2} rows with unknown {src2} id "
            f"out of {len(gold)} gold pairs",
            file=sys.stderr,
        )
    return records


def build_ditto_pair_records(
    domain: str,
    pair: str,
    split: str,
    *,
    normalize: bool = False,
) -> list[dict[str, Any]]:
    """Build Ditto-compatible pair records from a PyDI EM gold split.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    pair : str
        Source pair in ``<src1>_2_<src2>`` form (matches the EM gold filename).
    split : str
        One of ``"train"``, ``"val"``, ``"test"``, ``"all"``.
    normalize : bool, default False
        Forwarded to :func:`build_ditto_pair_records_from_gold`.

    Returns
    -------
    list of dict
        Records ready to be serialised to WDC json.gz.
    """
    src1, src2 = _parse_pair(pair)
    root = data_root_for_domain(domain) or USECASES_DIR
    gold_path = root / domain / "input" / "entitymatching" / f"{pair}_{split}.csv"
    if not gold_path.exists():
        raise FileNotFoundError(f"EM gold CSV not found: {gold_path}")
    gold = read_em_gold_csv(gold_path)
    return build_ditto_pair_records_from_gold(
        gold, domain, src1, src2, normalize=normalize
    )


def write_json_gz(records: list[dict[str, Any]], output: Path) -> None:
    """Write Ditto pair records to a gzip-compressed JSON-lines file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PyDI EM gold split into a Ditto WDC json.gz file."
    )
    parser.add_argument("--domain", required=True, help="Domain name, e.g. companies")
    parser.add_argument(
        "--pair",
        required=True,
        help="Source pair in <src1>_2_<src2> form, e.g. forbes_2_dbpedia",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test", "all"],
        help="Gold split to convert.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .json.gz path.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Apply the per-domain value normaliser from "
            "usecases_synthetic.lib.domain_value_norm before serialising. "
            "Currently configured for 'games' only (platform + title)."
        ),
    )
    args = parser.parse_args()

    records = build_ditto_pair_records(
        args.domain, args.pair, args.split, normalize=args.normalize
    )
    write_json_gz(records, args.output)
    pos = sum(1 for r in records if r["label"] == 1)
    neg = len(records) - pos
    norm_tag = " (normalized)" if args.normalize else ""
    print(
        f"wrote {len(records)} pairs ({pos} pos, {neg} neg){norm_tag} -> {args.output}"
    )


if __name__ == "__main__":
    main()

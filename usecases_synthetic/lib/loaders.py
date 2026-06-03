"""Thin data-loading wrappers that preserve DataFrame.attrs["dataset_name"].

Uses ``PyDI.io`` loaders under the hood. Provides a convenience function
to load all sources for a domain at once.

Three synthetic-side post-load adjustments wrap the underlying PyDI
readers:

1. **XML namespace stripping.** PyDI's XML reader retains the qualified
   element names from the source document, so a column declared as
   ``<rel_id>`` inside an ``xmlns="..."`` block surfaces as
   ``"{http://...}rel_id"``. Knob YAMLs need to reference the
   user-visible column name (``rel_id``), so the wrapper strips any
   leading ``{namespace}`` segment from every column name on XML loads.
2. **Explicit ``id`` rename.** Source files whose native id column is
   not literally named ``id`` opt in via ``id_column: <colname>`` in
   their ``config/domains/<d>.yaml`` source spec. The loader renames
   that column to ``id`` immediately after read, so downstream
   committee/runner code can keep its ``id_column="id"`` assumption.
   Used by the 2026-05-04 refreshed CSVs (e.g. companies/dbpedia
   ``entity_uri``, games/metacritic ``mc_id``).
3. **Optional ``id`` injection.** Source files that ship without an
   explicit ID column opt in via ``inject_id: true`` in their domain
   spec. When set, the loader ensures the returned DataFrame has an
   ``id`` column, either by preserving an existing one (e.g. one just
   produced by the rename above) or by injecting
   ``f"{source_name}_{1-based-row-index}"``. Used to materialise the
   per-row IDs that the pre-2026-05-04 XML/JSON sources required.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from PyDI.io import load_csv, load_json, load_xml

from .domain_config import (
    DomainConfig,
    USECASES_DIR,
    data_root_for_domain,
    load_domain_config,
)

_XML_NAMESPACE_RE = re.compile(r"\{[^}]+\}")


def _em_gold_has_header(csv_path: Path) -> bool:
    """Return ``True`` when the EM gold CSV's first line is a header row.

    The canonical synthetic EM gold (companies / games / music /
    products) is header-less: the first line is already a data row whose
    third field is a label value (``True`` / ``False``). The 2026 papers
    domain instead ships header-bearing EM gold whose third column is the
    literal token ``label`` (header ``id_dblp,id_crossref,label`` with
    integer 0/1 labels). Detection keys on that literal token so a real
    label value can never be mistaken for a header.
    """
    with open(csv_path, encoding="utf-8") as f:
        first = f.readline().rstrip("\r\n")
    if not first:
        return False
    parts = first.rsplit(",", 2)
    return len(parts) == 3 and parts[2].strip().lower() == "label"


def read_em_gold_csv(csv_path: Path) -> pd.DataFrame:
    """Read an EM gold correspondence CSV robustly.

    The canonical synthetic EM gold files are header-less CSVs with
    columns ``id1, id2, label`` but both ``id1`` and ``id2`` are URLs
    that may contain unquoted commas (e.g. ``Workday,_Inc.`` in DBpedia
    IRIs). The default ``pandas`` C tokenizer trips on such lines. This
    helper parses each line manually by right-splitting on the last two
    commas so the label and the second URL are isolated before any
    embedded commas in the first URL.

    Header-bearing EM gold (the 2026 papers domain:
    ``id_dblp,id_crossref,label`` with integer 0/1 labels and
    comma-free ``<source>-NNNNN`` ids) is auto-detected and read with
    pandas; the first three columns become ``id1``, ``id2``, ``label``
    positionally and the integer label dtype is preserved so the
    downstream ``astype(bool)`` positive-pair extraction works. The
    header-less branch is byte-for-byte unchanged for every existing
    domain.

    Parameters
    ----------
    csv_path : Path
        Path to the EM gold CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``id1``, ``id2``, ``label``.
    """
    if _em_gold_has_header(csv_path):
        df = pd.read_csv(csv_path)
        out = df.iloc[:, :3].copy()
        out.columns = ["id1", "id2", "label"]
        return out.reset_index(drop=True)

    rows: list[tuple[str, str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            # Right-split on the last two commas so embedded commas in
            # the first URL are preserved.
            parts = line.rsplit(",", 2)
            if len(parts) != 3:
                continue
            rows.append((parts[0], parts[1], parts[2]))
    return pd.DataFrame(rows, columns=["id1", "id2", "label"])


def _source_filename_tokens(src: str) -> list[str]:
    """Return filename tokens to try for a source name.

    Most domains use the source name verbatim in EM gold filenames. The
    2026 papers domain condenses ``open_alex`` to ``openalex`` in its EM
    gold filenames + id columns, so the underscore-stripped variant is
    offered as a fallback token.
    """
    tokens = [src]
    condensed = src.replace("_", "")
    if condensed != src:
        tokens.append(condensed)
    return tokens


def em_gold_candidates(
    em_dir: Path,
    pair: tuple[str, str],
    split: str,
) -> list[tuple[Path, bool]]:
    """Return ordered ``(path, needs_swap)`` candidates for one EM split.

    The canonical synthetic naming ``<src1>_2_<src2>_<split>.csv`` (and
    its reverse) is offered first, so every existing domain resolves to
    exactly the file it always did. The condensed ``<src1>_<src2>``
    forms (with the underscore-stripped token variants) follow as a
    fallback for the 2026 papers domain, whose EM gold ships as
    ``dblp_crossref_<split>.csv`` / ``dblp_openalex_<split>.csv`` with no
    ``_2_`` separator. ``needs_swap`` is ``True`` for reverse-direction
    files so the caller can swap ``id1``/``id2`` to match the declared
    pair direction.

    Parameters
    ----------
    em_dir : Path
        Entity-matching directory.
    pair : tuple of str
        ``(src1, src2)`` source pair in the domain's canonical order.
    split : str
        Split name (``"train"`` / ``"val"`` / ``"test"`` / ``"all"``).

    Returns
    -------
    list of tuple
        ``(path, needs_swap)`` in priority order, de-duplicated.
    """
    src1, src2 = pair
    candidates: list[tuple[Path, bool]] = [
        (em_dir / f"{src1}_2_{src2}_{split}.csv", False),
        (em_dir / f"{src2}_2_{src1}_{split}.csv", True),
    ]
    seen = {p for p, _ in candidates}
    for token1 in _source_filename_tokens(src1):
        for token2 in _source_filename_tokens(src2):
            forward = em_dir / f"{token1}_{token2}_{split}.csv"
            if forward not in seen:
                candidates.append((forward, False))
                seen.add(forward)
            reverse = em_dir / f"{token2}_{token1}_{split}.csv"
            if reverse not in seen:
                candidates.append((reverse, True))
                seen.add(reverse)
    return candidates


def read_em_gold_pair(path: Path, needs_swap: bool) -> pd.DataFrame:
    """Read an EM gold CSV and orient it to the declared pair direction.

    Wraps :func:`read_em_gold_csv` and swaps ``id1``/``id2`` when the
    file was found under the reverse source ordering, so the returned
    frame always carries ``id1`` belonging to the pair's ``src1``.
    """
    df = read_em_gold_csv(path)
    if not needs_swap:
        return df
    return pd.DataFrame(
        {
            "id1": df["id2"].values,
            "id2": df["id1"].values,
            "label": df["label"].values,
        }
    )


_FORMAT_LOADERS = {
    "xml": load_xml,
    "csv": load_csv,
    "json": load_json,
    # JSON-lines (one record per line). Dispatches to the same
    # ``load_json`` reader; the domain source spec opts in via
    # ``reader_kwargs: {lines: true, add_index: true, index_column_name:
    # id, id_prefix: <source>}`` so PyDI mints the ``<source>-NNNNN``
    # dash ids that the papers EM + fusion gold reference. Added for the
    # 2026 papers domain whose sources ship as ``.jsonl`` with no
    # on-disk id column.
    "jsonl": load_json,
}


def _strip_xml_namespaces(df: pd.DataFrame) -> pd.DataFrame:
    """Strip every ``{namespace}`` segment from every column name.

    PyDI's XML reader retains the qualified element name (e.g.
    ``"{http://musicbrainz.org/ns/mmd-2.0#}rel_id"``); the aggregator
    for nested elements concatenates parent/child paths with an
    underscore, embedding the namespace mid-string (e.g.
    ``"medium-list_{ns}medium_{ns}position"``). Knob YAMLs reference
    the local names, so every brace-delimited segment is removed.
    """
    new_cols = {c: _XML_NAMESPACE_RE.sub("", c) for c in df.columns}
    if any(old != new for old, new in new_cols.items()):
        df = df.rename(columns=new_cols)
    return df


def _ensure_id_column(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Ensure ``df`` has an ``id`` column.

    Preserves an existing ``id`` column. If absent, injects
    ``f"{source_name}_{1-based row index}"`` as the first column. The
    1-based convention matches the EM gold IDs already on disk for
    games / movies / products.
    """
    if "id" in df.columns:
        return df
    df = df.copy()
    series = pd.Series(
        [f"{source_name}_{i + 1}" for i in range(len(df))],
        index=df.index,
        dtype="string",
    )
    df.insert(0, "id", series)
    return df


def load_source(
    domain: str,
    source_name: str,
    source_file: str,
    source_format: str,
    reader_kwargs: dict | None = None,
    *,
    inject_id: bool = False,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Load a single source dataset with provenance metadata.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    source_name : str
        Source label (e.g. ``"dbpedia"``).
    source_file : str
        Filename relative to ``usecases/<domain>/input/data/``.
    source_format : str
        File format: ``"xml"``, ``"csv"``, or ``"json"``.
    reader_kwargs : dict or None
        Extra keyword arguments forwarded to the underlying PyDI loader
        (e.g. ``{"delimiter": "\\t"}`` for tab-separated ``.csv``
        files).
    inject_id : bool
        When ``True``, ensure the returned DataFrame has an ``id``
        column. Existing ``id`` columns are preserved; if absent,
        ``id`` is injected as ``f"{source_name}_{1-based row index}"``.
    id_column : str or None
        Name of the on-disk column carrying the entity id when it is
        not literally ``id``. The loader renames this column to ``id``
        immediately after read (and after XML-namespace stripping) so
        downstream code can keep its ``id_column="id"`` assumption.
        Raises ``ValueError`` if the named column is absent.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame with ``attrs["dataset_name"]`` set.
    """
    root = data_root_for_domain(domain) or USECASES_DIR
    data_dir = root / domain / "input" / "data"
    path = data_dir / source_file

    loader = _FORMAT_LOADERS.get(source_format)
    if loader is None:
        raise ValueError(
            f"Unsupported format: {source_format!r}. "
            f"Supported: {list(_FORMAT_LOADERS)}"
        )

    kwargs: dict = {"name": source_name}
    if source_format == "xml":
        kwargs["nested_handling"] = "aggregate"
    if reader_kwargs:
        kwargs.update(reader_kwargs)

    df = loader(path, **kwargs)
    if source_format == "xml":
        df = _strip_xml_namespaces(df)
    if id_column is not None:
        if id_column not in df.columns:
            raise ValueError(
                f"id_column={id_column!r} not present in {source_name} "
                f"({source_file}); columns: {list(df.columns)}"
            )
        if id_column != "id":
            if "id" in df.columns:
                raise ValueError(
                    f"Cannot rename {id_column!r} -> 'id' for {source_name}: "
                    f"a column named 'id' already exists ({source_file})."
                )
            df = df.rename(columns={id_column: "id"})
    if inject_id:
        df = _ensure_id_column(df, source_name)
    df = normalize_loaded_source(df, domain=domain, source_name=source_name)
    df.attrs["dataset_name"] = source_name
    return df


def normalize_loaded_source(
    df: pd.DataFrame,
    *,
    domain: str,
    source_name: str,
) -> pd.DataFrame:
    """Apply post-load normalization shared by baseline + augmented loaders.

    Runs the per-domain post-CSV-load transforms that are safe across
    *every* stage — only the discogs duration sentinel coalesce here
    today. List-aware transforms (e.g. music ``tracks`` literal parse)
    used to live here but were moved into the fusion runner
    ([committee_fusion.parse_source_list_columns][]) because SM /
    EM / Norm matchers (duplicate_based, magneto, coma_hybrid,
    magellan) do not tolerate list-valued cells — they crash on
    ``pd.isna(list)`` or attempt to hash unhashable values. The fusion
    engine + its list-aware strategies are the only consumers that
    need actual Python lists, so the parse is deferred until just
    before fusion engine invocation.

    Idempotent — re-running on already-normalized data leaves it
    unchanged.

    Called by both ``load_source`` (baseline path) and
    ``variant_loader._load_augmented_sources`` (variant path).

    Resolves ``knob_config_alias`` so aliased domains (e.g.
    ``music-small`` → ``music``) get the same data transforms as their
    parent — the source CSVs are byte-identical post-downsample, so the
    same per-domain shape fixups apply.
    """
    from .domain_config import _resolve_knob_config_alias

    canonical_domain = _resolve_knob_config_alias(domain) or domain

    if (
        canonical_domain == "music"
        and source_name == "discogs"
        and "duration" in df.columns
    ):
        # Discogs ships ``0`` as a missing-duration sentinel. Coalesce to NaN
        # at load time so K3 missingness measurement and K5/K6 numeric paths
        # see it as missing rather than as a present value of 0.
        mask = df["duration"].astype(str).str.strip().isin({"0", "0.0"})
        df.loc[mask, "duration"] = pd.NA
    # Games ``genres`` (comma-separated) and music ``tracks`` (Python
    # list literal) used to be parsed here. They were moved into the
    # fusion runner (``committee_fusion._parse_source_list_columns``)
    # so SM / EM / Norm see plain string cells. See the docstring
    # above for the rationale.
    return df


def split_comma_list(value: Any) -> Any:
    """Split a comma-separated string into a list of trimmed tokens.

    Returns the input unchanged when it is already a list/tuple or when
    parsing yields no tokens. ``None`` / NaN passes through. Single
    tokens (no comma) come back as ``[value]`` so downstream list-vs-list
    eval works uniformly across sources that ship single- vs multi-token
    values.
    """
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, float) and pd.isna(value):
        return value
    text = str(value).strip()
    if not text:
        return value
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if not tokens:
        return value
    return tokens


def parse_list_literal(value: Any) -> Any:
    """Best-effort parse of a stringified list literal into an actual list.

    Returns the input unchanged when it is already a list/tuple or when
    parsing fails. ``None`` / NaN passes through.

    Exposed at module top level so :mod:`committee_fusion` can re-parse
    list-typed source columns just before fusion-engine invocation; the
    loader itself emits the raw string so SM / EM / Norm matchers
    (which do not tolerate list cells) keep working.
    """
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, float) and pd.isna(value):
        return value
    text = str(value).strip()
    if not text:
        return value
    try:
        import ast

        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except (ValueError, SyntaxError):
        pass
    return value


def load_domain_sources(domain: str) -> dict[str, pd.DataFrame]:
    """Load all source datasets for a domain.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).

    Returns
    -------
    dict of str to DataFrame
        Source DataFrames keyed by source name.
    """
    config = load_domain_config(domain)
    sources: dict[str, pd.DataFrame] = {}
    for spec in config.sources:
        sources[spec.name] = load_source(
            domain=domain,
            source_name=spec.name,
            source_file=spec.file,
            source_format=spec.format,
            reader_kwargs=spec.reader_kwargs,
            inject_id=spec.inject_id,
            id_column=spec.id_column,
        )
    return sources

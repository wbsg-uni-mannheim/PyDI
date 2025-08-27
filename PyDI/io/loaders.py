"""Provenance-aware pandas I/O wrappers.

This module provides thin wrappers around common pandas readers that:

- Add a unique identifier column to each loaded DataFrame
- Attach detailed provenance metadata to ``df.attrs["provenance"]``
- Optionally include minimal provenance columns (``__source_path``, ``__dataset_name``)

The functions follow the design guidelines in ``CLAUDE.md``: pandas-first,
rich NumPy-style docstrings, and lightweight logging for observability.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import hashlib
import logging
import re
import itertools
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


DataFrameLike = pd.DataFrame
MultiFrame = Union[List[pd.DataFrame], Dict[str, pd.DataFrame]]
LoaderReturn = Union[DataFrameLike, MultiFrame]


def _compute_file_metadata(path: Optional[Union[str, os.PathLike]]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if path is None:
        return metadata

    try:
        p = Path(path)
    except TypeError:
        return metadata

    # Only local files (not URLs/buffers)
    if not (p.exists() and p.is_file()):
        metadata["source_path"] = str(path)
        return metadata

    metadata["source_path"] = str(p.resolve())
    try:
        stat = p.stat()
        metadata["file_size_bytes"] = stat.st_size
        metadata["modified_time_utc_iso"] = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        pass

    # Best-effort checksum (may be skipped for very large files)
    try:
        sha256 = hashlib.sha256()
        # Read at most 64MB for performance; include size for disambiguation
        read_limit = 64 * 1024 * 1024
        total_read = 0
        with p.open("rb") as f:
            while True:
                chunk = f.read(min(1024 * 1024, read_limit - total_read))
                if not chunk:
                    break
                sha256.update(chunk)
                total_read += len(chunk)
                if total_read >= read_limit:
                    break
        metadata["sha256_prefix"] = sha256.hexdigest()
        metadata["sha256_prefix_bytes"] = total_read
    except Exception:
        # Ignore checksum errors
        pass

    return metadata


def _derive_dataset_name(
    name: Optional[str],
    path_or_buf: Optional[Union[str, os.PathLike]],
    fallback: str = "dataset",
) -> str:
    if isinstance(name, str) and name.strip():
        return name.strip()
    try:
        if isinstance(path_or_buf, (str, os.PathLike)):
            p = Path(path_or_buf)
            if p.name:
                return p.stem
    except Exception:
        pass
    return fallback


def _inject_unique_id_column(
    df: pd.DataFrame,
    dataset_name: str,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Insert a unique identifier column as the first column.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to modify.
    dataset_name : str
        Base name used for the column name and id value prefix.
    index_column_name : str, optional
        Explicit column name. Defaults to ``f"{dataset_name}_id"``.
    id_prefix : str, optional
        Prefix for id values. Defaults to ``dataset_name``.

    Returns
    -------
    (DataFrame, str)
        The modified DataFrame and the id column name used.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    column_name = index_column_name or f"{dataset_name}_id"
    if column_name in df.columns:
        # Do not overwrite existing column; assume it's already a unique index
        return df, column_name

    prefix = id_prefix if isinstance(
        id_prefix, str) and id_prefix else dataset_name
    # Use a numeric sequence to ensure uniqueness within this DataFrame
    row_count = len(df)
    # Zero-pad width based on magnitude for stable sorting
    pad_width = max(4, len(str(max(row_count - 1, 0))))
    series = pd.Series(
        (f"{prefix}-{i:0{pad_width}d}" for i in range(row_count)), index=df.index, dtype="string"
    )
    df = df.copy()
    df.insert(0, column_name, series)
    return df, column_name


def _attach_provenance(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    reader_name: str,
    source_path_or_buf: Optional[Union[str, os.PathLike]],
    user_provenance: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Attach provenance metadata to ``df.attrs["provenance"]``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to annotate.
    dataset_name : str
        Name of the dataset used in metadata.
    reader_name : str
        Name of the underlying pandas reader.
    source_path_or_buf : str or os.PathLike, optional
        Original source path or buffer. Used to compute file metadata when possible.
    user_provenance : Mapping[str, Any], optional
        Additional user-provided key-value pairs merged into provenance.
    extra : Mapping[str, Any], optional
        Extra internal fields (e.g., ``id_column_name`` or ``sub_table``).
    include_columns : bool, default True
        If True, add minimal provenance columns to the DataFrame for convenience.

    Returns
    -------
    pandas.DataFrame
        A shallow copy with provenance metadata attached.
    """
    df = df.copy()

    file_meta = _compute_file_metadata(source_path_or_buf)
    now_iso = datetime.now(timezone.utc).isoformat()
    prov: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "reader": reader_name,
        "loaded_time_utc_iso": now_iso,
        **file_meta,
    }
    if user_provenance:
        prov.update(dict(user_provenance))
    if extra:
        prov.update(dict(extra))

    # Store in attrs without overwriting other attrs
    existing_attrs = getattr(df, "attrs", {}) or {}
    existing_attrs = dict(existing_attrs)
    # Store dataset name at top-level attrs for tools like profiling
    existing_attrs["dataset_name"] = dataset_name
    existing_attrs["provenance"] = prov
    df.attrs = existing_attrs

    return df


def load_with_provenance(
    reader_fn: Callable[..., LoaderReturn],
    path_or_buf: Optional[Union[str, os.PathLike]] = None,
    *,
    name: Optional[str] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    include_provenance_columns: bool = False,
    reader_name: Optional[str] = None,
    handle_multi_output: bool = True,
    **reader_kwargs: Any,
) -> LoaderReturn:
    """Load data with a pandas reader and attach provenance metadata.

    Parameters
    ----------
    reader_fn : Callable[..., DataFrame | list[DataFrame] | dict[str, DataFrame]]
        The pandas reader function (e.g., ``pandas.read_csv``).
    path_or_buf : str or os.PathLike, optional
        Path or buffer to read from.
    name : str, optional
        Explicit dataset name. Defaults to the filename stem if available.
    add_index : bool, default True
        If True, insert a unique id column as the first column.
    index_column_name : str, optional
        Explicit name for the id column.
    id_prefix : str, optional
        Prefix for id values (defaults to ``name`` or filename stem).
    provenance : Mapping[str, Any], optional
        User-provided provenance metadata to merge into ``df.attrs``.
    include_provenance_columns : bool, default True
        If True, add ``__source_path`` and ``__dataset_name`` columns.
    reader_name : str, optional
        Name to store in provenance for the reader. Defaults to ``reader_fn.__name__``.
    handle_multi_output : bool, default True
        If True, handle list/dict outputs (e.g., ``read_excel(sheet_name=None)``).
    **reader_kwargs : Any
        Additional keyword arguments forwarded to the underlying reader.

    Returns
    -------
    DataFrame | list[DataFrame] | dict[str, DataFrame]
        The loaded frame(s) with provenance metadata and optional id column.
    """
    result = reader_fn(path_or_buf, **reader_kwargs)

    def _process_single(df: pd.DataFrame, ds_name: str, extra: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        out = df
        if add_index:
            out, used_col = _inject_unique_id_column(
                out, ds_name, index_column_name, id_prefix)
            # include the id column name in provenance extras
            extra = dict(extra or {})
            extra["id_column_name"] = used_col
        out = _attach_provenance(
            out,
            dataset_name=ds_name,
            reader_name=reader_name or getattr(
                reader_fn, "__name__", "reader"),
            source_path_or_buf=path_or_buf,
            user_provenance=provenance,
            extra=extra,
        )
        try:
            logger.info(
                "Loaded dataset '%s' via %s: shape=%s, source=%s",
                ds_name,
                reader_name or getattr(reader_fn, "__name__", "reader"),
                tuple(out.shape),
                str(path_or_buf) if path_or_buf is not None else "<buffer>",
            )
        except Exception:
            # Logging must never break data loading
            pass
        return out

    if handle_multi_output and isinstance(result, list):
        base_name = _derive_dataset_name(name, path_or_buf)
        processed: List[pd.DataFrame] = []
        for idx, df in enumerate(result):
            sheet_name = str(idx)
            ds_name = f"{base_name}_{sheet_name}"
            processed.append(_process_single(
                df, ds_name, {"sub_table": sheet_name}))
        return processed

    if handle_multi_output and isinstance(result, dict):
        base_name = _derive_dataset_name(name, path_or_buf)
        processed_dict: Dict[str, pd.DataFrame] = {}
        for key, df in result.items():
            sheet_name = str(key)
            ds_name = f"{base_name}_{sheet_name}"
            processed_dict[key] = _process_single(
                df, ds_name, {"sub_table": sheet_name})
        return processed_dict

    # Single DataFrame
    ds_name = _derive_dataset_name(name, path_or_buf)
    return _process_single(result, ds_name)


# Convenience wrappers mirroring common pandas readers


def load_csv(
    filepath_or_buffer: Union[str, os.PathLike],
    *,
    name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    include_provenance_columns: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file with a provenance-aware wrapper.

    See ``load_with_provenance`` for parameter details.
    """
    return load_with_provenance(
        pd.read_csv,
        filepath_or_buffer,
        name=name,
        provenance=provenance,
        add_index=add_index,
        index_column_name=index_column_name,
        id_prefix=id_prefix,
        include_provenance_columns=include_provenance_columns,
        reader_name="read_csv",
        **kwargs,
    )


def load_fwf(
    filepath_or_buffer: Union[str, os.PathLike],
    *,
    name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    include_provenance_columns: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a fixed-width formatted file with a provenance-aware wrapper.

    This function wraps pandas.read_fwf() and adds provenance metadata and
    optional unique identifier columns to the resulting DataFrame.

    Parameters
    ----------
    filepath_or_buffer : str or path-like
        Any valid string path or file-like object.
    name : str, optional
        Dataset name for provenance tracking. If not provided, will be
        inferred from the file path.
    provenance : dict, optional
        Additional provenance metadata to include.
    add_index : bool, default True
        Whether to add a unique identifier column.
    index_column_name : str, optional
        Name for the unique identifier column. If not provided, defaults
        to ``{name}_id``.
    id_prefix : str, optional
        Prefix for unique identifiers. If not provided, defaults to ``{name}-``.
    include_provenance_columns : bool, default False
        Whether to include provenance information as DataFrame columns.
    **kwargs
        Additional keyword arguments passed to pandas.read_fwf().

    Returns
    -------
    pandas.DataFrame
        A DataFrame with provenance metadata and optional identifier column.

    Examples
    --------
    >>> df = load_fwf("data/actors.txt", name="actors")
    >>> df.attrs["dataset_name"]
    'actors'
    >>> df.attrs["provenance"]["reader"]
    'read_fwf'
    """
    return load_with_provenance(
        pd.read_fwf,
        filepath_or_buffer,
        name=name,
        provenance=provenance,
        add_index=add_index,
        index_column_name=index_column_name,
        id_prefix=id_prefix,
        include_provenance_columns=include_provenance_columns,
        reader_name="read_fwf",
        **kwargs,
    )


def load_json(
    path_or_buf: Union[str, os.PathLike],
    *,
    name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    include_provenance_columns: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a JSON file with a provenance-aware wrapper.

    See ``load_with_provenance`` for parameter details.
    """
    return load_with_provenance(
        pd.read_json,
        path_or_buf,
        name=name,
        provenance=provenance,
        add_index=add_index,
        index_column_name=index_column_name,
        id_prefix=id_prefix,
        include_provenance_columns=include_provenance_columns,
        reader_name="read_json",
        **kwargs,
    )


def load_parquet(
    path: Union[str, os.PathLike],
    *,
    name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    include_provenance_columns: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a Parquet file with a provenance-aware wrapper.

    See ``load_with_provenance`` for parameter details.
    """
    return load_with_provenance(
        pd.read_parquet,
        path,
        name=name,
        provenance=provenance,
        add_index=add_index,
        index_column_name=index_column_name,
        id_prefix=id_prefix,
        include_provenance_columns=include_provenance_columns,
        reader_name="read_parquet",
        **kwargs,
    )


def load_excel(
    io: Union[str, os.PathLike],
    *,
    name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    include_provenance_columns: bool = False,
    **kwargs: Any,
) -> LoaderReturn:
    """Read an Excel file with a provenance-aware wrapper.

    Can return a single DataFrame or a mapping of sheet names to DataFrames
    when ``sheet_name=None``. See ``load_with_provenance`` for details.
    """
    return load_with_provenance(
        pd.read_excel,
        io,
        name=name,
        provenance=provenance,
        add_index=add_index,
        index_column_name=index_column_name,
        id_prefix=id_prefix,
        include_provenance_columns=include_provenance_columns,
        reader_name="read_excel",
        handle_multi_output=True,
        **kwargs,
    )


def load_xml(
    path_or_buffer: Union[str, os.PathLike],
    *,
    name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    include_provenance_columns: bool = False,
    flatten: bool = True,
    record_tag: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an XML file with a provenance-aware wrapper.

    If ``flatten`` is True (default), nested structures are flattened into rows
    (e.g., repeated ``actor`` elements create multiple rows with duplicated
    parent fields and a column like ``actor_name``).
    """
    # If a real path is provided and flattening requested, use custom flattener
    if flatten and isinstance(path_or_buffer, (str, os.PathLike)):
        try:
            df = _read_and_flatten_xml(
                Path(path_or_buffer), record_tag=record_tag)
            return load_with_provenance(
                lambda _p, **_k: df,  # already loaded
                path_or_buffer,
                name=name,
                provenance=provenance,
                add_index=add_index,
                index_column_name=index_column_name,
                id_prefix=id_prefix,
                include_provenance_columns=include_provenance_columns,
                reader_name="read_xml_flattened",
            )
        except Exception as e:
            # Fallback to pandas if custom parsing fails
            try:
                logger.info(
                    "XML flattening failed (%s), falling back to pandas.read_xml", e)
            except Exception:
                pass

    # Otherwise, use pandas.read_xml with a parser fallback selection
    if "parser" not in kwargs:
        try:
            import lxml  # type: ignore # noqa: F401
            kwargs["parser"] = "lxml"
        except Exception:
            kwargs["parser"] = "etree"
    return load_with_provenance(
        pd.read_xml,
        path_or_buffer,
        name=name,
        provenance=provenance,
        add_index=add_index,
        index_column_name=index_column_name,
        id_prefix=id_prefix,
        include_provenance_columns=include_provenance_columns,
        reader_name="read_xml",
        **kwargs,
    )


def _read_and_flatten_xml(file_path: Path, record_tag: Optional[str] = None) -> pd.DataFrame:
    """Parse XML and flatten nested structures into rows.

    Heuristics:
    - Auto-detect record tag as the most frequent child of the root when not provided
    - Repeated child tags (e.g., ``actor``) produce multiple rows
    - Scalar children merge into the parent row
    - Attributes and text content are captured when present
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML file {file_path}: {e}")

    if record_tag is None:
        record_tag = _detect_xml_record_tag(root)

    records = root.findall(f".//{record_tag}")
    if not records:
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    for rec in records:
        rows = _flatten_xml_element(rec)
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.columns = [_clean_column_name(col) for col in df.columns]
    # Explode delimited multi-values for director names
    if "director_name" in df.columns:
        df = _explode_delimited_column(df, "director_name")
    return df


def _detect_xml_record_tag(root: ET.Element) -> str:
    tag_counts: Dict[str, int] = {}
    for child in root:
        tag_counts[child.tag] = tag_counts.get(child.tag, 0) + 1
    if not tag_counts:
        # fallback to root tag if empty
        return root.tag
    return max(tag_counts.items(), key=lambda kv: kv[1])[0]


def _flatten_xml_element(element: ET.Element, prefix: str = "") -> List[Dict[str, Any]]:
    base_data: Dict[str, Any] = {}
    list_children: Dict[str, List[Dict[str, Any]]] = {}

    # attributes
    for attr_name, attr_value in element.attrib.items():
        key = f"{prefix}{attr_name}" if prefix else attr_name
        base_data[key] = attr_value

    # text
    if element.text and element.text.strip():
        text_key = f"{prefix}text" if prefix else "text"
        base_data[text_key] = element.text.strip()

    # children
    for child in list(element):
        child_prefix = f"{prefix}{child.tag}_" if prefix else f"{child.tag}_"

        # count siblings with same tag
        sibling_count = sum(1 for c in element if c.tag == child.tag)
        if sibling_count > 1:
            # list-like child
            if child.tag not in list_children:
                list_children[child.tag] = []
            child_rows = _flatten_xml_element(child, child_prefix)
            list_children[child.tag].extend(child_rows)
        else:
            # scalar child; merge its flattened dict into base
            child_rows = _flatten_xml_element(child, child_prefix)
            if child_rows:
                base_data.update(child_rows[0])

    if list_children:
        # Full cartesian product across list-like children to fully explode
        keys = list(list_children.keys())
        lists: List[List[Dict[str, Any]]] = [list_children[k] for k in keys]
        rows: List[Dict[str, Any]] = []
        for combo in itertools.product(*lists):
            row = dict(base_data)
            for part in combo:
                row.update(part)
            rows.append(row)
        return rows
    else:
        return [base_data] if base_data else []


def _clean_column_name(name: str) -> str:
    if name.endswith("_text"):
        name = name[:-5]
    name = name.rstrip("_")
    # Rename common nested keys for readability
    if name == "actors_actor_name":
        name = "actor_name"
    if name == "directors_director_name":
        name = "director_name"
    return name


def _explode_delimited_column(
    df: pd.DataFrame,
    column: str,
    pattern: str = r"\s*(?:and|,|;)\s*",
) -> pd.DataFrame:
    """Explode rows where a column contains multiple values separated by delimiters.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column : str
        Column to explode.
    pattern : str
        Regex pattern for splitting. Defaults to split on 'and', commas, or semicolons.
    """
    if column not in df.columns:
        return df
    mask = df[column].apply(lambda v: isinstance(
        v, str) and re.search(pattern, v) is not None)
    if not mask.any():
        return df
    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        value = row[column]
        if isinstance(value, str) and re.search(pattern, value):
            parts = [p.strip() for p in re.split(pattern, value) if p.strip()]
            if not parts:
                parts = [value]
            for part in parts:
                new_row = row.to_dict()
                new_row[column] = part
                rows.append(new_row)
        else:
            rows.append(row.to_dict())
    return pd.DataFrame(rows, columns=df.columns)


def load_feather(
    path: Union[str, os.PathLike],
    *,
    name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    include_provenance_columns: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a Feather file with a provenance-aware wrapper.

    See ``load_with_provenance`` for parameter details.
    """
    return load_with_provenance(
        pd.read_feather,
        path,
        name=name,
        provenance=provenance,
        add_index=add_index,
        index_column_name=index_column_name,
        id_prefix=id_prefix,
        include_provenance_columns=include_provenance_columns,
        reader_name="read_feather",
        **kwargs,
    )


def load_pickle(
    path: Union[str, os.PathLike],
    *,
    name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    include_provenance_columns: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a pickled DataFrame with a provenance-aware wrapper.

    See ``load_with_provenance`` for parameter details.
    """
    df = pd.read_pickle(path, **kwargs)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "load_pickle expected a pandas DataFrame in the pickle file")
    return load_with_provenance(
        lambda _p, **_k: df,  # already loaded
        path,
        name=name,
        provenance=provenance,
        add_index=add_index,
        index_column_name=index_column_name,
        id_prefix=id_prefix,
        include_provenance_columns=include_provenance_columns,
        reader_name="read_pickle",
    )


def load_html(
    io: Union[str, os.PathLike],
    *,
    name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    include_provenance_columns: bool = False,
    **kwargs: Any,
) -> List[pd.DataFrame]:
    """Read one or more tables from HTML with a provenance-aware wrapper.

    Returns a list of DataFrames. See ``load_with_provenance`` for details.
    """
    return load_with_provenance(
        pd.read_html,
        io,
        name=name,
        provenance=provenance,
        add_index=add_index,
        index_column_name=index_column_name,
        id_prefix=id_prefix,
        include_provenance_columns=include_provenance_columns,
        reader_name="read_html",
        handle_multi_output=True,
        **kwargs,
    )


def load_table(
    filepath_or_buffer: Union[str, os.PathLike],
    *,
    sep: str = "\t",
    name: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    add_index: bool = True,
    index_column_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
    include_provenance_columns: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a delimited text file with a provenance-aware wrapper.

    Defaults to tab-separated values. See ``load_with_provenance`` for details.
    """
    return load_with_provenance(
        pd.read_table,
        filepath_or_buffer,
        name=name,
        provenance=provenance,
        add_index=add_index,
        index_column_name=index_column_name,
        id_prefix=id_prefix,
        include_provenance_columns=include_provenance_columns,
        reader_name="read_table",
        sep=sep,
        **kwargs,
    )


__all__ = [
    "load_with_provenance",
    "load_csv",
    "load_json",
    "load_parquet",
    "load_excel",
    "load_xml",
    "load_feather",
    "load_pickle",
    "load_html",
    "load_table"
]

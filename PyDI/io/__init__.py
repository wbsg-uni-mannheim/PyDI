"""Provenance-aware I/O helpers for PyDI.

This package exposes convenience wrappers around pandas readers that attach
provenance metadata and add a unique id column.
"""

from .loaders import (
    load_with_provenance,
    load_csv,
    load_fwf,
    load_json,
    load_parquet,
    load_excel,
    load_xml,
    load_feather,
    load_pickle,
    load_html,
    load_table
)

__all__ = [
    "load_with_provenance",
    "load_csv",
    "load_fwf",
    "load_json",
    "load_parquet",
    "load_excel",
    "load_xml",
    "load_feather",
    "load_pickle",
    "load_html",
    "load_table"
]

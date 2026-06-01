"""Static rate and format table loaders for Knob 05.

Loads the immutable YAML tables under
``usecases_synthetic/config/knob_05_format/_tables/`` and exposes typed
accessor functions. No external calls — all data is on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_TABLES_DIR: Path = (
    Path(__file__).resolve().parents[1] / "config" / "knob_05_format" / "_tables"
)

# Module-level cache (loaded once per process).
_date_formats: dict[str, Any] | None = None
_number_locales: dict[str, Any] | None = None
_fx_rates: dict[str, Any] | None = None
_unit_factors: dict[str, Any] | None = None


def _load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML file from the tables directory."""
    path = _TABLES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- Date formats -----------------------------------------------------------


def _ensure_date_formats() -> dict[str, Any]:
    global _date_formats
    if _date_formats is None:
        _date_formats = _load_yaml("date_formats.yaml")
    return _date_formats


def get_date_format(format_id: str) -> dict[str, Any]:
    """Return the spec for a single date format.

    Parameters
    ----------
    format_id : str
        Key from ``date_formats.yaml`` (e.g. ``"iso"``, ``"us_slash"``).

    Returns
    -------
    dict
        Keys: ``pattern`` (strftime), ``parser_hint``, ``hard_only``,
        ``locale_ambiguous_deny``.
    """
    data = _ensure_date_formats()
    if format_id not in data["formats"]:
        raise KeyError(f"Unknown date format: {format_id!r}")
    return data["formats"][format_id]


def get_all_date_formats() -> dict[str, dict[str, Any]]:
    """Return the full date format table."""
    return _ensure_date_formats()["formats"]


def is_denied_date_format(format_id: str) -> bool:
    """Check if a date format is on the locale-ambiguous deny list."""
    fmt = get_date_format(format_id)
    return bool(fmt.get("locale_ambiguous_deny", False))


# ---- Number locales ---------------------------------------------------------


def _ensure_number_locales() -> dict[str, Any]:
    global _number_locales
    if _number_locales is None:
        _number_locales = _load_yaml("number_locales.yaml")
    return _number_locales


def get_locale_config(locale_id: str) -> dict[str, Any]:
    """Return separator config for a locale.

    Parameters
    ----------
    locale_id : str
        Key from ``number_locales.yaml`` (e.g. ``"en_US"``).

    Returns
    -------
    dict
        Keys: ``decimal_sep``, ``thousands_sep``, ``grouping``.
    """
    data = _ensure_number_locales()
    if locale_id not in data["locales"]:
        raise KeyError(f"Unknown number locale: {locale_id!r}")
    return data["locales"][locale_id]


def get_suffix_scales() -> dict[str, float]:
    """Return the suffix scale table (K, M, B, T)."""
    data = _ensure_number_locales()
    return data.get("suffix_scales", {})


# ---- FX rates ---------------------------------------------------------------


def _ensure_fx_rates() -> dict[str, Any]:
    global _fx_rates
    if _fx_rates is None:
        _fx_rates = _load_yaml("fx_rates.yaml")
    return _fx_rates


def get_fx_rate(from_ccy: str, to_ccy: str) -> float:
    """Return the conversion rate from *from_ccy* to *to_ccy*.

    Parameters
    ----------
    from_ccy : str
        Source currency ISO code (e.g. ``"USD"``).
    to_ccy : str
        Target currency ISO code (e.g. ``"EUR"``).

    Returns
    -------
    float
        Multiplier: ``value_in_to = value_in_from * rate``.
    """
    data = _ensure_fx_rates()
    rates = data["rates"]
    if from_ccy not in rates:
        raise KeyError(f"Unknown currency: {from_ccy!r}")
    if to_ccy not in rates:
        raise KeyError(f"Unknown currency: {to_ccy!r}")
    # rates are all relative to USD: 1 USD = X units
    # from_ccy -> USD -> to_ccy
    usd_per_from = 1.0 / rates[from_ccy]
    return usd_per_from * rates[to_ccy]


def get_fx_rate_date() -> str:
    """Return the rate date string from the FX table."""
    return _ensure_fx_rates()["rate_date"]


# ---- Unit factors -----------------------------------------------------------


def _ensure_unit_factors() -> dict[str, Any]:
    global _unit_factors
    if _unit_factors is None:
        _unit_factors = _load_yaml("unit_factors.yaml")
    return _unit_factors


def get_unit_factor(group: str, from_unit: str, to_unit: str) -> float:
    """Return the conversion factor between two units in the same group.

    Parameters
    ----------
    group : str
        Unit group (e.g. ``"duration"``, ``"magnitude"``).
    from_unit : str
        Source unit.
    to_unit : str
        Target unit.

    Returns
    -------
    float
        Multiplier: ``value_in_to = value_in_from * factor``.
    """
    data = _ensure_unit_factors()
    if group not in data:
        raise KeyError(f"Unknown unit group: {group!r}")
    units = data[group]["units"]
    if from_unit not in units:
        raise KeyError(f"Unknown unit {from_unit!r} in group {group!r}")
    if to_unit not in units:
        raise KeyError(f"Unknown unit {to_unit!r} in group {group!r}")
    # Convert through base unit: from_unit -> base -> to_unit
    # factor_to_base[from] * value_from = value_base
    # value_to = value_base / factor_to_base[to]
    return units[from_unit] / units[to_unit]


def get_unit_group(group: str) -> dict[str, Any]:
    """Return the full unit group spec."""
    data = _ensure_unit_factors()
    if group not in data:
        raise KeyError(f"Unknown unit group: {group!r}")
    return data[group]


def reset_caches() -> None:
    """Clear all module-level caches (for testing)."""
    global _date_formats, _number_locales, _fx_rates, _unit_factors
    _date_formats = None
    _number_locales = None
    _fx_rates = None
    _unit_factors = None

"""Rule-based normalizer members for the Normalization committee.

Each member is a small wrapper class around a PyDI normalization
primitive. The committee runner calls ``member.normalize(value,
attribute, kind)`` per (cell, member); members return the normalized
string or ``None`` when the input is unparseable / out-of-vocabulary.

Five rule-based members are provided:

- :class:`TextCleanNormalizer`  — case-fold + whitespace + unicode.
- :class:`DateIsoNormalizer`    — date / year parser.
- :class:`NumberLocaleNormalizer` — numeric parser (Babel-backed).
- :class:`CountryIsoNormalizer` — pycountry-backed code-list lookup.
- :class:`TaxonomyLookupNormalizer` — exact-match against a taxonomy
  CSV column (no LLM).

The LLM member lives separately in :mod:`llm_normalizer` so this module
has no LangChain / OpenAI dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from PyDI.normalization.integrations.country import normalize_country
from PyDI.normalization.taxonomy import TaxonomyLoader
from PyDI.normalization.text import TextNormalizer
from PyDI.normalization.types import DateNormalizer, NumericParser

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
USECASES_DIR = REPO_ROOT / "usecases"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseNormalizer(Protocol):
    """Per-value normalization member.

    Members are stateful instances configured at construction time. The
    runner calls :meth:`normalize` once per (source-cell, attribute).
    """

    name: str

    def normalize(
        self,
        value: Any,
        *,
        attribute: str,
        kind: str,
        domain: str,
    ) -> str | None:
        """Return the normalized value, or ``None`` to abstain.

        Parameters
        ----------
        value : Any
            Raw cell value from the source DataFrame. May be any
            scalar type pandas surfaces (``str``, ``int``, ``float``,
            ``pd.Timestamp``, ``NaN``).
        attribute : str
            Canonical attribute name (e.g. ``"country"``).
        kind : str
            Tolerance kind for *attribute* (``"continuous"``, ``"year"``,
            ``"date"``, ``"nominal"``, ``"long_string"``, ``"free_text"``,
            ``"list"``).
        domain : str
            Domain name. Some members (taxonomy lookup) need the domain
            to resolve their per-domain taxonomy CSV.

        Returns
        -------
        str or None
            Normalized value as a string, or ``None`` when the
            normalizer cannot produce a canonical form for this input.
        """
        ...


def _stringify(value: Any) -> str | None:
    """Coerce *value* to a non-empty stripped string, else ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    return s


# ---------------------------------------------------------------------------
# TextCleanNormalizer
# ---------------------------------------------------------------------------


class TextCleanNormalizer:
    """Case-fold + whitespace + unicode normalizer (PyDI ``TextNormalizer``).

    Best for ``long_string`` / ``free_text`` / ``nominal`` attributes
    where the canonical form differs from the source only in casing,
    whitespace, or unicode form. ``list`` kinds are handled element-wise
    after splitting on the same delimiters used by
    :func:`protection._split_list_tokens`.
    """

    def __init__(
        self,
        name: str = "text_clean",
        *,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        normalize_unicode: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = False,
    ) -> None:
        self.name = name
        self._normalizer = TextNormalizer(
            lowercase=lowercase,
            strip_whitespace=strip_whitespace,
            remove_html=remove_html,
            remove_punctuation=remove_punctuation,
            normalize_unicode=normalize_unicode,
            fix_encoding=True,
        )

    def normalize(
        self,
        value: Any,
        *,
        attribute: str,
        kind: str,
        domain: str,
    ) -> str | None:
        s = _stringify(value)
        if s is None:
            return None
        if kind == "list":
            from .protection import _split_list_tokens  # avoid circular at import

            tokens = _split_list_tokens(s)
            cleaned = [self._normalizer.clean_text(t) for t in tokens]
            cleaned = [t for t in cleaned if t]
            if not cleaned:
                return None
            return ", ".join(sorted(cleaned))
        cleaned = self._normalizer.clean_text(s)
        return cleaned if cleaned else None


# ---------------------------------------------------------------------------
# DateIsoNormalizer
# ---------------------------------------------------------------------------


class DateIsoNormalizer:
    """Parse arbitrary date strings to ISO 8601 (PyDI ``DateNormalizer``).

    For ``year`` kinds the output is the 4-digit year only. For ``date``
    the full ``%Y-%m-%d``. Returns ``None`` on unparseable input.
    """

    def __init__(
        self,
        name: str = "date_iso",
        *,
        date_format: str = "%Y-%m-%d",
        year_only_format: str = "%Y",
        handle_timezone: bool = True,
    ) -> None:
        self.name = name
        self._date_format = date_format
        self._year_only_format = year_only_format
        self._date_normalizer = DateNormalizer(
            target_format=date_format,
            handle_timezone=handle_timezone,
        )
        self._year_normalizer = DateNormalizer(
            target_format=year_only_format,
            handle_timezone=False,
        )

    def normalize(
        self,
        value: Any,
        *,
        attribute: str,
        kind: str,
        domain: str,
    ) -> str | None:
        s = _stringify(value)
        if s is None:
            return None
        if kind not in ("date", "year"):
            return None
        if kind == "year":
            # Try year-only fast path first (handles bare "2005" + variants).
            try:
                year = int(s[:4])
                if 1500 <= year <= 2200:
                    return f"{year:04d}"
            except ValueError:
                pass
            out = self._year_normalizer.normalize_date(s)
            return out
        return self._date_normalizer.normalize_date(s)


# ---------------------------------------------------------------------------
# NumberLocaleNormalizer
# ---------------------------------------------------------------------------


class NumberLocaleNormalizer:
    """Parse numeric strings to a canonical decimal form (PyDI ``NumericParser``).

    Locale-aware via Babel (handles ``1,5`` vs ``1.5``); strips currency
    symbols, thousand separators, and percentage signs. Returns the
    parsed number as a string (``"148700000000"`` or ``"3.14"``); the
    closeness gate applies relative tolerance per Pending #5.
    """

    def __init__(
        self,
        name: str = "number_locale",
        *,
        babel_candidate_locales: list[str] | None = None,
        handle_currency: bool = True,
        handle_percentages: bool = True,
        decimal_separator: str = ".",
        thousands_separator: str = ",",
    ) -> None:
        self.name = name
        self._parser = NumericParser(
            handle_currency=handle_currency,
            handle_percentages=handle_percentages,
            decimal_separator=decimal_separator,
            thousands_separator=thousands_separator,
            babel_candidate_locales=babel_candidate_locales or ["en_US", "de_DE"],
        )

    def normalize(
        self,
        value: Any,
        *,
        attribute: str,
        kind: str,
        domain: str,
    ) -> str | None:
        s = _stringify(value)
        if s is None:
            return None
        if kind != "continuous":
            return None
        parsed = self._parser.parse_numeric(s)
        if parsed is None:
            return None
        # Format back to a canonical string. Integer-like floats render
        # without trailing ".0" so closeness comparisons against integer
        # fusion targets succeed.
        if isinstance(parsed, float) and parsed.is_integer():
            return str(int(parsed))
        return str(parsed)


# ---------------------------------------------------------------------------
# CountryIsoNormalizer
# ---------------------------------------------------------------------------


class CountryIsoNormalizer:
    """Map country names / codes to a canonical ISO form (pycountry).

    Wraps :func:`PyDI.normalization.integrations.country.normalize_country`.
    Output format is configurable (alpha-2 / alpha-3 / name); the choice
    must match what the fusion val/test reference values ship.
    """

    def __init__(
        self,
        name: str = "country_iso",
        *,
        output_format: str = "name",
    ) -> None:
        self.name = name
        if output_format not in {
            "alpha_2",
            "alpha_3",
            "numeric",
            "name",
            "official_name",
        }:
            raise ValueError(
                f"Unknown output_format {output_format!r} for CountryIsoNormalizer."
            )
        self._output_format = output_format

    def normalize(
        self,
        value: Any,
        *,
        attribute: str,
        kind: str,
        domain: str,
    ) -> str | None:
        s = _stringify(value)
        if s is None:
            return None
        if kind != "nominal":
            return None
        return normalize_country(s, output_format=self._output_format)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TaxonomyLookupNormalizer
# ---------------------------------------------------------------------------


class TaxonomyLookupNormalizer:
    """Exact-match a value against a taxonomy CSV column (no LLM).

    For each (domain, canonical attribute) the YAML maps a taxonomy CSV
    + column-of-canonical-values. The normalizer loads the taxonomy
    once per (path, column) and looks the value up case-insensitively.
    Returns the canonical taxonomy spelling (preserves the CSV's
    casing) on hit, ``None`` on miss.

    Used for attributes whose canonical vocabulary is enumerable —
    industry / platform / genre — alongside a fuzzier LLM member that
    handles open-vocabulary spellings the taxonomy doesn't cover.
    """

    def __init__(
        self,
        name: str = "taxonomy_lookup",
        *,
        taxonomies: dict[str, dict[str, dict[str, str]]] | None = None,
        case_insensitive: bool = True,
    ) -> None:
        """Initialize with per-(domain, attribute) taxonomy bindings.

        Parameters
        ----------
        name : str
        taxonomies : dict
            Nested dict ``{domain: {attribute: {"path": <str>, "column":
            <str>}}}``. Paths are relative to ``usecases/``.
        case_insensitive : bool
            When ``True`` (default), the lookup folds case before
            comparing.
        """
        self.name = name
        self._taxonomies = taxonomies or {}
        self._case_insensitive = case_insensitive
        self._loader = TaxonomyLoader()
        self._cache: dict[tuple[str, str], dict[str, str]] = {}

    def _lookup_table(self, domain: str, attribute: str) -> dict[str, str] | None:
        binding = self._taxonomies.get(domain, {}).get(attribute)
        if not binding:
            # Alias-aware lookup: ``music-small`` etc. inherit the source
            # domain's taxonomies via ``knob_config_alias``. Without this
            # fall-through every taxonomy_lookup cell scores 0.0 on the
            # alias (S.6a finding on music-small, plan_s1_final.md F8).
            from .domain_config import _resolve_knob_config_alias

            alias = _resolve_knob_config_alias(domain)
            if alias:
                binding = self._taxonomies.get(alias, {}).get(attribute)
            if not binding:
                return None
        path = binding.get("path")
        # Accept either a single column string or a list of columns; the
        # union of values across columns becomes the lookup vocabulary.
        columns_field = binding.get("columns") or binding.get("column")
        if not path or not columns_field:
            return None
        columns: list[str] = (
            [columns_field] if isinstance(columns_field, str) else list(columns_field)
        )
        cache_key = (str(path), "|".join(columns))
        if cache_key in self._cache:
            return self._cache[cache_key]
        all_values: list[str] = []
        for col in columns:
            try:
                vals = self._loader.load(str(path), column=col, base_path=USECASES_DIR)
            except Exception:
                logger.exception(
                    "Failed to load taxonomy %s::%s for %s.%s",
                    path,
                    col,
                    domain,
                    attribute,
                )
                continue
            all_values.extend(vals)
        table = {(v.lower() if self._case_insensitive else v): v for v in all_values}
        self._cache[cache_key] = table
        return table

    def normalize(
        self,
        value: Any,
        *,
        attribute: str,
        kind: str,
        domain: str,
    ) -> str | None:
        s = _stringify(value)
        if s is None:
            return None
        table = self._lookup_table(domain, attribute)
        if not table:
            return None
        if kind == "list":
            from .protection import _split_list_tokens

            tokens = _split_list_tokens(s)
            mapped: list[str] = []
            for tok in tokens:
                key = tok.lower() if self._case_insensitive else tok
                hit = table.get(key)
                if hit is not None:
                    mapped.append(hit)
            if not mapped:
                return None
            return ", ".join(sorted(set(mapped)))
        key = s.lower() if self._case_insensitive else s
        return table.get(key)


__all__ = [
    "BaseNormalizer",
    "TextCleanNormalizer",
    "DateIsoNormalizer",
    "NumberLocaleNormalizer",
    "CountryIsoNormalizer",
    "TaxonomyLookupNormalizer",
]

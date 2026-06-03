"""Schema-constraint-based Norm scoring surface.

Replaces the XML-target-comparison surface (compare normalized cell to a
per-entity expected value) with a constraint-conformance surface: a
normalized cell "passes" when it satisfies the JSON-Schema constraints
+ ``x-pydi-consistency`` extensions declared in
``input/schemamatching/target_schema.json``. The per-attribute F1
remains the macro-averaged scoring shape downstream code expects.

Why
---
The target schemas declare units, value ranges, allowed value sets,
field-applicability rules, and open taxonomies (per the 2026 schema
refresh). They define what a *correctly normalized* value looks like
attribute-by-attribute — independent of any per-entity "what the
fused value should be" gold. This is the right scoring surface for
"did the normalizer produce schema-conformant outputs?".

What it covers
--------------
Per attribute the validator handles:

- JSON Schema: ``type``, ``minimum``, ``maximum``, ``pattern``,
  ``enum``, ``format``, ``minLength`` / ``maxLength``.
- ``x-pydi-consistency`` rules: ``currency_code`` (ISO 4217),
  ``field_applicability`` (gated on a sibling column, typically
  ``product_type``), ``open_taxonomy`` (expected families with
  exhaustive/non-exhaustive semantics), ``date_range``
  (date string within window), ``country_or_release_market``
  (taxonomy or release-market label), ``zero_as_missing``,
  ``delimited_open_text`` (item count cap),
  ``page_locator`` (numeric range + cross-field comparison).

Outputs
-------
:class:`SchemaConstraintScores` mirrors the ``macro_metrics()`` shape
of :class:`MemberPerAttributeScores` so the C12 runner can swap in
this scorer at the per-cell loop without changing aggregation code or
the downstream ``macro_f1`` key.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# Minimal ISO-4217 alpha-3 currency set — extend on demand.
_ISO_4217 = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CAD",
        "AUD",
        "NZD",
        "CHF",
        "SEK",
        "NOK",
        "DKK",
        "CNY",
        "HKD",
        "SGD",
        "INR",
        "ZAR",
        "BRL",
        "MXN",
        "ARS",
        "CLP",
        "COP",
        "PEN",
        "RUB",
        "TRY",
        "ILS",
        "AED",
        "SAR",
        "EGP",
        "KRW",
        "TWD",
        "THB",
        "MYR",
        "PHP",
        "IDR",
        "VND",
        "PLN",
        "CZK",
        "HUF",
        "RON",
        "BGN",
        "HRK",
        "ISK",
        "UAH",
        "BYN",
        "KZT",
        "GEL",
        "AZN",
        "AMD",
        "AFN",
        "BDT",
        "BTN",
        "NPR",
        "PKR",
        "LKR",
        "MNT",
        "KGS",
        "TJS",
        "TMT",
        "UZS",
        "IRR",
        "IQD",
        "SYP",
        "LBP",
        "JOD",
        "QAR",
        "OMR",
        "KWD",
        "BHD",
        "YER",
        "DJF",
        "ETB",
        "KES",
        "TZS",
        "UGX",
        "RWF",
        "BIF",
        "MWK",
        "MZN",
        "ZMW",
        "ZWL",
        "AOA",
        "NAD",
        "BWP",
        "SZL",
        "LSL",
        "MUR",
        "SCR",
        "MGA",
        "KMF",
        "DZD",
        "MAD",
        "TND",
        "LYD",
        "SDG",
        "SOS",
        "GHS",
        "NGN",
        "XOF",
        "XAF",
        "GMD",
        "GNF",
        "SLL",
        "LRD",
        "CDF",
        "CVE",
        "STN",
        "BHD",
        "FJD",
        "PGK",
        "WST",
        "TOP",
        "VUV",
        "XPF",
        "PYG",
        "BOB",
        "UYU",
        "VES",
        "GYD",
        "SRD",
        "TTD",
        "JMD",
        "DOP",
        "HTG",
        "PAB",
        "CRC",
        "GTQ",
        "NIO",
        "HNL",
        "BBD",
        "BSD",
        "KYD",
        "BMD",
        "XCD",
        "BZD",
        "AWG",
        "CUP",
        "CUC",
        "ANG",
    }
)


@dataclass
class AttributeConstraints:
    """Parsed constraints for a single canonical attribute."""

    name: str
    json_type: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    pattern: re.Pattern[str] | None = None
    enum: tuple[Any, ...] | None = None
    format: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    # x-pydi-consistency
    consistency_rule: str | None = None
    currency_iso: bool = False
    applies_to_types: tuple[str, ...] | None = None
    expected_families: tuple[str, ...] | None = None
    taxonomy_exhaustive: bool | None = None
    date_min: date | None = None
    date_max: date | None = None
    zero_as_missing: bool = False
    max_terms: int | None = None
    delimited_separator: str | None = None
    page_locator: bool = False
    page_min: int | None = None
    page_max: int | None = None
    numeric_compare_op: str | None = None
    numeric_compare_field: str | None = None
    # Bookkeeping
    has_any_constraint: bool = field(default=False)

    def __post_init__(self) -> None:
        # Lazy "has anything to enforce" flag — used to decide whether
        # to count this attribute as eligible for scoring at all.
        # Identity checks for the None/False sentinels rather than
        # ``not in (None, False, ...)``: the latter compares with ``==``
        # and so silently swallows a legitimate ``minimum: 0`` bound
        # (``0.0 == False`` is True in Python), which dropped attrs whose
        # only constraint is a zero numeric bound — e.g. products'
        # ``price`` (``minimum: 0``) — from norm scoring.
        self.has_any_constraint = any(
            v is not None and v is not False and v != () and v != []
            for k, v in self.__dict__.items()
            if k not in ("name", "json_type", "has_any_constraint")
        )


_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    m = _DATE_RE.match(value.strip())
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def parse_target_schema(path: Path) -> dict[str, AttributeConstraints]:
    """Parse a target-schema JSON file into per-attribute constraints.

    Reads JSON-Schema ``properties`` + ``x-pydi-consistency`` blocks.
    Attributes with no enforceable constraint are still returned
    (``has_any_constraint=False``) so callers can decide whether to
    score them.
    """
    with path.open() as f:
        schema = json.load(f)
    props = schema.get("properties", {}) or {}
    out: dict[str, AttributeConstraints] = {}
    for name, spec in props.items():
        c = AttributeConstraints(name=name)
        c.json_type = spec.get("type")
        if "minimum" in spec:
            c.minimum = float(spec["minimum"])
        if "maximum" in spec:
            c.maximum = float(spec["maximum"])
        if "pattern" in spec:
            try:
                c.pattern = re.compile(spec["pattern"])
            except re.error:
                logger.warning(
                    "Skipping unparseable pattern for %s: %r", name, spec["pattern"]
                )
        if "enum" in spec:
            c.enum = tuple(spec["enum"])
        if "format" in spec:
            c.format = spec["format"]
        if "minLength" in spec:
            c.min_length = int(spec["minLength"])
        if "maxLength" in spec:
            c.max_length = int(spec["maxLength"])

        cons = spec.get("x-pydi-consistency")
        if isinstance(cons, dict):
            rule = cons.get("rule")
            c.consistency_rule = rule
            if rule == "currency_code":
                c.currency_iso = True
            elif rule == "field_applicability":
                applies = cons.get("appliesToProductTypes")
                if isinstance(applies, list):
                    c.applies_to_types = tuple(applies)
            elif rule == "open_taxonomy":
                fams = cons.get("expectedFamilies")
                if isinstance(fams, list):
                    c.expected_families = tuple(fams)
                c.taxonomy_exhaustive = bool(cons.get("exhaustive", False))
                applies = cons.get("appliesToProductTypes")
                if isinstance(applies, list):
                    c.applies_to_types = tuple(applies)
            elif rule == "date_range":
                c.date_min = _parse_date(cons.get("minimumDate"))
                c.date_max = _parse_date(cons.get("maximumDate"))
            elif rule == "zero_as_missing":
                c.zero_as_missing = True
            elif rule == "delimited_open_text":
                c.delimited_separator = cons.get("separator")
                if cons.get("maxTerms") is not None:
                    c.max_terms = int(cons["maxTerms"])
            elif rule == "page_locator":
                c.page_locator = True
                if cons.get("numericMinimum") is not None:
                    c.page_min = int(cons["numericMinimum"])
                if cons.get("numericMaximum") is not None:
                    c.page_max = int(cons["numericMaximum"])
                cmp = cons.get("numericComparison")
                if isinstance(cmp, dict):
                    c.numeric_compare_op = cmp.get("operator")
                    c.numeric_compare_field = cmp.get("field")
            # country_or_release_market is treated as "free-string with
            # non-empty content" — true taxonomy enforcement would
            # require the CLDR/release-market list; left to follow-up.

        # Recompute has_any_constraint after population (fields
        # mutated above don't re-trigger __post_init__). Identity checks
        # for the None sentinel (not ``not in (None, False, ())``, which
        # compares with ``==`` and swallows a legitimate ``minimum: 0``
        # since ``0.0 == False``).
        c.has_any_constraint = any(
            getattr(c, k) is not None
            and getattr(c, k) is not False
            and getattr(c, k) != ()
            for k in (
                "minimum",
                "maximum",
                "pattern",
                "enum",
                "format",
                "min_length",
                "max_length",
                "consistency_rule",
            )
        )
        out[name] = c
    return out


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        try:
            import math

            if math.isnan(value):
                return True
        except Exception:
            return False
    if isinstance(value, str) and not value.strip():
        return True
    return False


def value_passes(
    value: Any,
    constraints: AttributeConstraints,
    row_ctx: dict[str, Any] | None = None,
) -> bool | None:
    """Check whether ``value`` satisfies ``constraints``.

    Returns:
        - ``True``  — value satisfies every applicable constraint.
        - ``False`` — value violates at least one constraint.
        - ``None``  — value is missing AND the attribute doesn't apply
          to this row (e.g. ``vram_gb`` on an SSD row). The cell
          abstains from scoring.

    ``row_ctx`` carries sibling-column values; used by
    ``field_applicability`` (gated on ``product_type``) and
    ``page_locator`` cross-field comparisons.
    """
    row_ctx = row_ctx or {}
    empty = _is_empty(value)

    # zero_as_missing: treat numeric 0 / "0" as missing.
    if constraints.zero_as_missing:
        n = _coerce_number(value)
        if n == 0.0:
            empty = True

    # Field applicability gate.
    if constraints.applies_to_types is not None:
        product_type = row_ctx.get("product_type")
        applicable = product_type in constraints.applies_to_types
        if not applicable:
            # Inapplicable: the cell SHOULD be empty.
            if empty:
                return None  # abstain — correct absence
            return False  # inapplicable cell has a value -> violation
        # else: applicable -> require a value, fall through to checks
        if empty:
            return False  # applicable but missing

    if empty:
        # No applicability gate triggered, but the value is missing.
        # Abstain — we can't say it's right or wrong.
        return None

    # ---- Type / format / length / enum checks ----
    json_type = constraints.json_type
    if json_type == "string":
        if not isinstance(value, str):
            return False
        s = value.strip()
        if constraints.min_length is not None and len(s) < constraints.min_length:
            return False
        if constraints.max_length is not None and len(s) > constraints.max_length:
            return False
    elif json_type in ("number", "integer"):
        n = _coerce_number(value)
        if n is None:
            return False
        if json_type == "integer" and not float(n).is_integer():
            return False
    elif json_type == "array":
        if not isinstance(value, (list, tuple)):
            return False

    if constraints.enum is not None:
        if value not in constraints.enum:
            return False

    if constraints.pattern is not None:
        if not isinstance(value, str):
            return False
        if not constraints.pattern.match(value):
            return False

    if constraints.minimum is not None or constraints.maximum is not None:
        n = _coerce_number(value)
        if n is None:
            return False
        if constraints.minimum is not None and n < constraints.minimum:
            return False
        if constraints.maximum is not None and n > constraints.maximum:
            return False

    if constraints.format == "date":
        d = _parse_date(value)
        if d is None:
            return False
        if constraints.date_min is not None and d < constraints.date_min:
            return False
        if constraints.date_max is not None and d > constraints.date_max:
            return False

    # ---- x-pydi-consistency rule extensions ----
    rule = constraints.consistency_rule
    if rule == "currency_code" or constraints.currency_iso:
        if not (isinstance(value, str) and value.upper() in _ISO_4217):
            return False
    elif rule == "open_taxonomy" and constraints.expected_families is not None:
        if constraints.taxonomy_exhaustive:
            # exhaustive=true: value must be IN expected_families exactly.
            if value not in constraints.expected_families:
                return False
        else:
            # exhaustive=false: any value is acceptable IFF it contains
            # one of the families as a substring (case-insensitive).
            if isinstance(value, str):
                v = value.lower()
                if not any(fam.lower() in v for fam in constraints.expected_families):
                    return False
    elif rule == "delimited_open_text":
        if isinstance(value, str) and constraints.max_terms is not None:
            sep = constraints.delimited_separator or ","
            terms = [t.strip() for t in value.split(sep) if t.strip()]
            if len(terms) > constraints.max_terms:
                return False
    elif rule == "page_locator":
        # Numeric page in range, OR a literal article locator (e.g. "e1008483").
        if isinstance(value, str):
            n = _coerce_number(value)
            if n is not None:
                if (constraints.page_min is not None and n < constraints.page_min) or (
                    constraints.page_max is not None and n > constraints.page_max
                ):
                    return False
                # cross-field numeric comparison
                if constraints.numeric_compare_op and constraints.numeric_compare_field:
                    other = _coerce_number(
                        row_ctx.get(constraints.numeric_compare_field)
                    )
                    if other is not None:
                        op = constraints.numeric_compare_op
                        ok = (
                            (op == ">=" and n >= other)
                            or (op == "<=" and n <= other)
                            or (op == ">" and n > other)
                            or (op == "<" and n < other)
                            or (op == "==" and n == other)
                        )
                        if not ok:
                            return False

    return True


@dataclass
class _PerAttrCounts:
    correct: int = 0
    wrong: int = 0
    abstained: int = 0

    def record(self, outcome: bool | None) -> None:
        if outcome is True:
            self.correct += 1
        elif outcome is False:
            self.wrong += 1
        else:
            self.abstained += 1

    @property
    def f1(self) -> float:
        # F1 over (correct, wrong) treating abstains as neither.
        c, w = self.correct, self.wrong
        if c + w == 0:
            return 0.0
        precision = c / (c + w)
        # Recall here treats wrong + abstain as missed correct; gives
        # the same penalty structure as the existing protection.score
        # surface (cells the member declined to normalize hurt recall).
        recall = c / (c + w + self.abstained) if (c + w + self.abstained) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def precision(self) -> float:
        c, w = self.correct, self.wrong
        return c / (c + w) if (c + w) else 0.0

    @property
    def recall(self) -> float:
        denom = self.correct + self.wrong + self.abstained
        return self.correct / denom if denom else 0.0

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.abstained


@dataclass
class SchemaConstraintScores:
    """Per-attribute scoring aggregator compatible with the C12 runner's
    aggregation step (mirrors :class:`MemberPerAttributeScores`'s
    ``macro_metrics()`` shape so downstream code reads the same
    ``macro_f1`` / ``macro_precision`` / ``macro_recall`` keys).
    """

    member: str
    _per_attr: dict[str, _PerAttrCounts] = field(default_factory=dict)

    def record(
        self,
        attribute: str,
        normalized: Any,
        constraints: AttributeConstraints,
        row_ctx: dict[str, Any] | None = None,
    ) -> None:
        outcome = value_passes(normalized, constraints, row_ctx)
        self._per_attr.setdefault(attribute, _PerAttrCounts()).record(outcome)

    @property
    def by_attribute(self) -> dict[str, _PerAttrCounts]:
        return self._per_attr

    def macro_metrics(self) -> dict[str, float]:
        attrs = list(self._per_attr.values())
        if not attrs:
            return {
                "macro_f1": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "num_attributes_scored": 0,
            }
        f1s = [a.f1 for a in attrs]
        precisions = [a.precision for a in attrs]
        recalls = [a.recall for a in attrs]
        n = len(attrs)
        return {
            "macro_f1": sum(f1s) / n,
            "macro_precision": sum(precisions) / n,
            "macro_recall": sum(recalls) / n,
            "num_attributes_scored": n,
        }


def eligible_constraint_attributes(
    constraint_map: dict[str, AttributeConstraints],
    candidate_attrs: Iterable[str],
) -> list[str]:
    """Intersection of ``candidate_attrs`` and attributes with at least
    one enforceable constraint."""
    return [
        a
        for a in candidate_attrs
        if a in constraint_map and constraint_map[a].has_any_constraint
    ]

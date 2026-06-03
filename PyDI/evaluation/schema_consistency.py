"""
Schema-driven consistency metrics for fused PyDI outputs.

Consistency is evaluated without a silver reference: each filled cell in the
fused output is checked against the target schema's native JSON Schema
constraints plus PyDI-specific ``x-pydi-*`` extensions. The dataset-level score
is weighted by filled cells, not by columns, so sparse columns do not distort
the aggregate.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SchemaInput = Union[str, Path, Mapping[str, Any]]


@dataclass
class _ConsistencyContext:
    schema_path: Optional[Path] = None
    taxonomy_base_path: Optional[Path] = None
    taxonomy_cache: MutableMapping[str, set[str]] = field(default_factory=dict)


@dataclass
class _CellResult:
    evaluated: bool
    consistent: bool
    failures: list[str] = field(default_factory=list)


def evaluate_schema_consistency(
    fused_df: pd.DataFrame,
    target_schema: SchemaInput,
    *,
    schema_path: Optional[Union[str, Path]] = None,
    taxonomy_base_path: Optional[Union[str, Path]] = None,
    exclude_identifier_columns: bool = True,
) -> Dict[str, Any]:
    """Evaluate fused output consistency against a target schema.

    Parameters
    ----------
    fused_df : DataFrame
        Final pipeline output. Only columns declared under
        ``target_schema["properties"]`` are evaluated.
    target_schema : mapping or path
        JSON Schema object, or path to a JSON target schema.
    schema_path : path, optional
        Schema location when ``target_schema`` is already loaded as a dict.
        Used to resolve relative taxonomy CSV paths.
    taxonomy_base_path : path, optional
        Explicit root for relative ``x-pydi-taxonomy`` paths. If omitted, the
        resolver tries the schema directory and its ancestors.
    exclude_identifier_columns : bool, default True
        Exclude target identifier columns such as ``id`` and ``*_id`` from the
        consistency aggregate. They are still reported under ``per_column`` as
        skipped.

    Returns
    -------
    dict
        ``{consistency_score, n_evaluated, n_consistent, n_inconsistent,
        per_column, columns_missing_from_dataframe, extra_columns_ignored}``.
        ``consistency_score`` is ``None`` when there are no filled values to
        evaluate. Empty columns are present in ``per_column`` but excluded from
        the dataset score.
    """
    schema, loaded_schema_path = _load_schema(target_schema, schema_path)
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError("Target schema must define a mapping under 'properties'.")

    context = _ConsistencyContext(
        schema_path=loaded_schema_path,
        taxonomy_base_path=(
            Path(taxonomy_base_path).resolve()
            if taxonomy_base_path is not None
            else None
        ),
    )

    schema_columns = list(properties.keys())
    present_columns = [c for c in schema_columns if c in fused_df.columns]
    missing_columns = [c for c in schema_columns if c not in fused_df.columns]
    extra_columns = [c for c in fused_df.columns if c not in properties]

    per_column: Dict[str, Dict[str, Any]] = {}
    totals = {
        "n_evaluated": 0,
        "n_consistent": 0,
        "n_inconsistent": 0,
    }
    records = fused_df.to_dict("records")

    for column in present_columns:
        field_schema = properties[column]
        if not isinstance(field_schema, Mapping):
            continue
        if exclude_identifier_columns and _is_identifier_column(column):
            per_column[column] = _skipped_column_stats("identifier_column")
            continue
        if _custom_rule(field_schema) == "not_evaluated":
            per_column[column] = _skipped_column_stats("not_evaluated")
            continue

        column_stats = _evaluate_column(records, column, field_schema, context)
        per_column[column] = column_stats
        if column_stats["n_evaluated"] > 0:
            for key in totals:
                totals[key] += int(column_stats[key])

    consistency_score: Optional[float]
    if totals["n_evaluated"] == 0:
        consistency_score = None
    else:
        consistency_score = totals["n_consistent"] / totals["n_evaluated"]

    return {
        "consistency_score": consistency_score,
        **totals,
        "per_column": per_column,
        "columns_missing_from_dataframe": missing_columns,
        "extra_columns_ignored": extra_columns,
        "n_empty_columns_excluded": sum(
            1
            for stats in per_column.values()
            if stats.get("excluded_from_dataset_score") and not stats.get("skipped")
        ),
    }


def evaluate_column_schema_consistency(
    series: pd.Series,
    field_schema: Mapping[str, Any],
    *,
    column: Optional[str] = None,
    row_context: Optional[pd.DataFrame] = None,
    schema_path: Optional[Union[str, Path]] = None,
    taxonomy_base_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Evaluate one column against one field schema.

    ``row_context`` may be supplied for cross-field extensions such as page
    comparisons or product-type applicability. It must align with ``series`` by
    index.
    """
    context = _ConsistencyContext(
        schema_path=Path(schema_path).resolve() if schema_path is not None else None,
        taxonomy_base_path=(
            Path(taxonomy_base_path).resolve()
            if taxonomy_base_path is not None
            else None
        ),
    )
    column_name = column or str(series.name)
    if row_context is None:
        records = [{column_name: value} for value in series]
    else:
        records = row_context.to_dict("records")
        values = series.to_list()
        for record, value in zip(records, values):
            record[column_name] = value
    return _evaluate_column(records, column_name, field_schema, context)


def schema_consistency_per_column_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Convert ``evaluate_schema_consistency`` output to a per-column table."""
    rows = []
    for column, stats in result.get("per_column", {}).items():
        rows.append(
            {
                "column": column,
                "consistency_score": stats.get("consistency_score"),
                "n_evaluated": stats.get("n_evaluated", 0),
                "n_consistent": stats.get("n_consistent", 0),
                "n_inconsistent": stats.get("n_inconsistent", 0),
                "excluded_from_dataset_score": stats.get(
                    "excluded_from_dataset_score", False
                ),
                "failure_counts": json.dumps(
                    stats.get("failure_counts", {}), sort_keys=True
                ),
                "diagnostics": json.dumps(stats.get("diagnostics", {}), sort_keys=True),
                "skipped": stats.get("skipped", False),
                "skip_reason": stats.get("skip_reason"),
            }
        )
    return pd.DataFrame(rows)


def write_metric_report(
    metric: str,
    result: Mapping[str, Any],
    output_path: Union[str, Path],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a single metric report as JSON in the standard envelope.

    The envelope is ``{"metric", "metadata", "result"}`` — the same shape
    long used for ``consistency.json``. Every per-metric report file
    (``consistency.json``, ``coverage.json``, ``correctness.json``, ...) and
    the panel's per-metric emitter route through here so all metric files
    share one structure and one serialization (indented UTF-8 JSON with a
    trailing newline).

    Parameters
    ----------
    metric : str
        Metric/dimension name written under the ``"metric"`` key
        (e.g. ``"consistency"``, ``"coverage"``).
    result : mapping
        The metric result dict, stored verbatim under ``"result"``.
    output_path : str or Path
        Destination file. Parent directories are created.
    metadata : mapping, optional
        Caller-supplied provenance (use case, schema path, run id, ...)
        stored under ``"metadata"``.

    Returns
    -------
    Path
        The written path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "metric": metric,
        "metadata": dict(metadata or {}),
        "result": dict(result),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
        f.write("\n")
    return output_path


def write_schema_consistency_report(
    result: Mapping[str, Any],
    output_path: Union[str, Path],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a schema-consistency report as JSON.

    Thin wrapper over :func:`write_metric_report` with ``metric="consistency"``.
    The written payload keeps the metric result intact and adds optional
    caller-supplied metadata such as use case name, schema path, or fusion
    output path.
    """
    return write_metric_report("consistency", result, output_path, metadata=metadata)


def _load_schema(
    target_schema: SchemaInput,
    schema_path: Optional[Union[str, Path]],
) -> tuple[Mapping[str, Any], Optional[Path]]:
    if isinstance(target_schema, Mapping):
        loaded_path = Path(schema_path).resolve() if schema_path is not None else None
        return target_schema, loaded_path

    loaded_path = Path(target_schema).resolve()
    with loaded_path.open("r", encoding="utf-8") as f:
        return json.load(f), loaded_path


def _evaluate_column(
    records: Sequence[Mapping[str, Any]],
    column: str,
    field_schema: Mapping[str, Any],
    context: _ConsistencyContext,
) -> Dict[str, Any]:
    n_evaluated = 0
    n_consistent = 0
    failure_counts: Dict[str, int] = {}

    for row in records:
        result = _evaluate_cell(row.get(column), field_schema, row, column, context)
        if not result.evaluated:
            continue
        n_evaluated += 1
        if result.consistent:
            n_consistent += 1
        else:
            for failure in result.failures:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1

    n_inconsistent = n_evaluated - n_consistent
    score = n_consistent / n_evaluated if n_evaluated else None
    stats: Dict[str, Any] = {
        "consistency_score": score,
        "n_evaluated": n_evaluated,
        "n_consistent": n_consistent,
        "n_inconsistent": n_inconsistent,
        "failure_counts": dict(sorted(failure_counts.items())),
        "excluded_from_dataset_score": n_evaluated == 0,
    }
    diagnostics = _column_diagnostics(records, column, field_schema, context)
    if diagnostics:
        stats["diagnostics"] = diagnostics
    return stats


def _skipped_column_stats(reason: str) -> Dict[str, Any]:
    return {
        "consistency_score": None,
        "n_evaluated": 0,
        "n_consistent": 0,
        "n_inconsistent": 0,
        "failure_counts": {},
        "excluded_from_dataset_score": True,
        "skipped": True,
        "skip_reason": reason,
    }


def _evaluate_cell(
    value: Any,
    field_schema: Mapping[str, Any],
    row: Mapping[str, Any],
    column: str,
    context: _ConsistencyContext,
) -> _CellResult:
    if _zero_as_missing(value, field_schema) or _is_missing(value):
        return _CellResult(evaluated=False, consistent=True)

    failures = _check_native_constraints(value, field_schema)
    failures.extend(_check_taxonomy(value, field_schema, context))
    failures.extend(
        _check_custom_constraints(value, field_schema, row, column, context)
    )
    return _CellResult(
        evaluated=True,
        consistent=len(failures) == 0,
        failures=sorted(set(failures)),
    )


def _column_diagnostics(
    records: Sequence[Mapping[str, Any]],
    column: str,
    field_schema: Mapping[str, Any],
    context: _ConsistencyContext,
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}

    taxonomy = _taxonomy_membership_diagnostics(
        records,
        column,
        field_schema,
        context,
    )
    if taxonomy:
        diagnostics["taxonomy"] = taxonomy

    open_taxonomy = _open_taxonomy_diagnostics(records, column, field_schema)
    if open_taxonomy:
        diagnostics["open_taxonomy"] = open_taxonomy

    return diagnostics


def _check_native_constraints(value: Any, schema: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    schema_type = schema.get("type")

    normalized = value
    if schema_type is not None:
        type_ok, normalized = _coerce_schema_type(value, schema_type)
        if not type_ok:
            return ["type"]

    if "enum" in schema and not _matches_enum(normalized, schema["enum"]):
        failures.append("enum")

    if schema_type == "array":
        items = normalized
        if "minItems" in schema and len(items) < int(schema["minItems"]):
            failures.append("minItems")
        if "maxItems" in schema and len(items) > int(schema["maxItems"]):
            failures.append("maxItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            item_failures: set[str] = set()
            for item in items:
                for failure in _check_native_constraints(item, item_schema):
                    item_failures.add(f"items.{failure}")
            failures.extend(sorted(item_failures))
        return failures

    if _is_numeric_schema_type(schema_type):
        numeric_value = _as_decimal(normalized)
        if numeric_value is None:
            failures.append("type")
        else:
            failures.extend(_check_numeric_bounds(numeric_value, schema))

    if schema_type == "string":
        text = normalized
        if "minLength" in schema and len(text) < int(schema["minLength"]):
            failures.append("minLength")
        if "maxLength" in schema and len(text) > int(schema["maxLength"]):
            failures.append("maxLength")
        if "pattern" in schema:
            try:
                if re.search(str(schema["pattern"]), text) is None:
                    failures.append("pattern")
            except re.error:
                failures.append("pattern")
        if "format" in schema:
            format_failure = _check_format(text, str(schema["format"]))
            if format_failure is not None:
                failures.append(format_failure)

    return failures


def _coerce_schema_type(value: Any, schema_type: Any) -> tuple[bool, Any]:
    if isinstance(schema_type, SequenceABC) and not isinstance(schema_type, str):
        for candidate in schema_type:
            ok, normalized = _coerce_schema_type(value, candidate)
            if ok:
                return True, normalized
        return False, value

    if schema_type == "string":
        return isinstance(value, str), value
    if schema_type == "integer":
        numeric_value = _as_decimal(value)
        if numeric_value is None:
            return False, value
        return numeric_value == numeric_value.to_integral_value(), numeric_value
    if schema_type == "number":
        numeric_value = _as_decimal(value)
        return numeric_value is not None, numeric_value
    if schema_type == "array":
        values = _coerce_array(value)
        return values is not None, values
    if schema_type == "boolean":
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
            return True, value.strip().casefold() == "true"
        return False, value
    if schema_type == "null":
        return _is_missing(value), None

    return True, value


def _is_numeric_schema_type(schema_type: Any) -> bool:
    if schema_type in {"integer", "number"}:
        return True
    if isinstance(schema_type, SequenceABC) and not isinstance(schema_type, str):
        return any(t in {"integer", "number"} for t in schema_type)
    return False


def _check_numeric_bounds(value: Decimal, schema: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if "minimum" in schema and value < Decimal(str(schema["minimum"])):
        failures.append("minimum")
    if "maximum" in schema and value > Decimal(str(schema["maximum"])):
        failures.append("maximum")
    if "exclusiveMinimum" in schema and value <= Decimal(
        str(schema["exclusiveMinimum"])
    ):
        failures.append("exclusiveMinimum")
    if "exclusiveMaximum" in schema and value >= Decimal(
        str(schema["exclusiveMaximum"])
    ):
        failures.append("exclusiveMaximum")
    return failures


def _check_format(value: str, fmt: str) -> Optional[str]:
    if fmt == "date":
        try:
            date.fromisoformat(value)
        except ValueError:
            return "format.date"
    elif fmt == "date-time":
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return "format.date-time"
    elif fmt in {"uri", "uri-reference"}:
        if re.search(r"^(https?://)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value) is None:
            return f"format.{fmt}"
    return None


def _check_taxonomy(
    value: Any,
    field_schema: Mapping[str, Any],
    context: _ConsistencyContext,
) -> list[str]:
    taxonomy_path = field_schema.get("x-pydi-taxonomy")
    if taxonomy_path is None:
        return []

    exhaustive = bool(field_schema.get("x-pydi-taxonomy-exhaustive", True))
    if not exhaustive:
        return []

    allowed_values = _load_taxonomy_values(field_schema, context)
    values = _values_for_membership_check(value)
    if not values:
        return []

    allowed_folded = {v.casefold() for v in allowed_values}
    for item in values:
        text = str(item).strip()
        if text not in allowed_values and text.casefold() not in allowed_folded:
            return ["taxonomy"]
    return []


def _taxonomy_membership_diagnostics(
    records: Sequence[Mapping[str, Any]],
    column: str,
    field_schema: Mapping[str, Any],
    context: _ConsistencyContext,
) -> Dict[str, Any]:
    if field_schema.get("x-pydi-taxonomy") is None:
        return {}

    allowed_values = _load_taxonomy_values(field_schema, context)
    allowed_folded = {v.casefold() for v in allowed_values}
    n_values_checked = 0
    n_known_values = 0

    for row in records:
        value = row.get(column)
        if _zero_as_missing(value, field_schema) or _is_missing(value):
            continue
        for item in _values_for_membership_check(value):
            if _is_missing(item):
                continue
            n_values_checked += 1
            text = str(item).strip()
            if text in allowed_values or text.casefold() in allowed_folded:
                n_known_values += 1

    if n_values_checked == 0:
        return {}

    exhaustive = bool(field_schema.get("x-pydi-taxonomy-exhaustive", True))
    return {
        "exhaustive": exhaustive,
        "membership_enforced": exhaustive,
        "n_values_checked": n_values_checked,
        "n_known_values": n_known_values,
        "n_unknown_values": n_values_checked - n_known_values,
        "known_value_rate": n_known_values / n_values_checked,
    }


def _load_taxonomy_values(
    field_schema: Mapping[str, Any],
    context: _ConsistencyContext,
) -> set[str]:
    taxonomy_path = str(field_schema["x-pydi-taxonomy"])
    taxonomy_column = str(field_schema.get("x-pydi-taxonomy-column", "")).strip()
    alias_columns = tuple(field_schema.get("x-pydi-taxonomy-alias-columns", []))
    aliases = field_schema.get("x-pydi-taxonomy-aliases", {})
    cache_key = json.dumps(
        {
            "path": taxonomy_path,
            "column": taxonomy_column,
            "alias_columns": alias_columns,
            "aliases": aliases,
        },
        sort_keys=True,
    )
    if cache_key in context.taxonomy_cache:
        return context.taxonomy_cache[cache_key]

    resolved = _resolve_relative_path(taxonomy_path, context)
    values: set[str] = set()
    try:
        taxonomy = pd.read_csv(resolved, keep_default_na=False)
    except FileNotFoundError:
        logger.warning("Taxonomy file not found: %s", resolved)
        context.taxonomy_cache[cache_key] = values
        return values

    columns = [taxonomy_column, *alias_columns]
    for column in columns:
        if column and column in taxonomy.columns:
            for raw in taxonomy[column].tolist():
                text = str(raw).strip()
                if text:
                    values.add(text)

    if isinstance(aliases, Mapping):
        for alias, canonical in aliases.items():
            if alias is not None and str(alias).strip():
                values.add(str(alias).strip())
            if canonical is not None and str(canonical).strip():
                values.add(str(canonical).strip())

    context.taxonomy_cache[cache_key] = values
    return values


def _resolve_relative_path(path: str, context: _ConsistencyContext) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    roots: list[Path] = []
    if context.taxonomy_base_path is not None:
        roots.append(context.taxonomy_base_path)
    if context.schema_path is not None:
        roots.extend([context.schema_path.parent, *context.schema_path.parents])

    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    if roots:
        return (roots[0] / candidate).resolve()
    return candidate


def _check_custom_constraints(
    value: Any,
    field_schema: Mapping[str, Any],
    row: Mapping[str, Any],
    column: str,
    context: _ConsistencyContext,
) -> list[str]:
    del context
    custom = field_schema.get("x-pydi-consistency")
    if not isinstance(custom, Mapping):
        return []

    failures: list[str] = []
    failures.extend(_check_field_applicability(value, custom, row))

    rule = custom.get("rule")
    if rule == "date_range":
        failures.extend(_check_date_range(value, custom))
    elif rule == "country_or_release_market":
        if bool(custom.get("exhaustive", False)):
            failures.extend(_check_country_or_market(value))
    elif rule == "delimited_open_text":
        failures.extend(_check_delimited_open_text(value, custom))
    elif rule == "page_locator":
        failures.extend(_check_page_locator(value, custom, row, column))
    elif rule == "currency_code":
        if not _is_iso_currency_code(str(value).strip()):
            failures.append("currency_code")
    elif rule == "field_applicability":
        pass
    elif rule == "open_taxonomy":
        failures.extend(_check_open_taxonomy(value, custom))
    elif rule == "zero_as_missing":
        pass
    elif rule == "not_evaluated":
        pass
    elif rule:
        logger.warning("Unsupported x-pydi-consistency rule: %s", rule)

    return failures


def _check_field_applicability(
    value: Any,
    custom: Mapping[str, Any],
    row: Mapping[str, Any],
) -> list[str]:
    applies_to = custom.get("appliesToProductTypes")
    if not applies_to or _is_missing(value):
        return []

    product_type = row.get("product_type")
    if _is_missing(product_type):
        return []

    allowed = {str(item).strip() for item in applies_to}
    if str(product_type).strip() not in allowed:
        return ["field_applicability"]
    return []


def _check_date_range(value: Any, custom: Mapping[str, Any]) -> list[str]:
    parsed = _parse_date(value)
    if parsed is None:
        return ["date_range"]

    min_date = _parse_date(custom.get("minimumDate"))
    max_date = _parse_date(custom.get("maximumDate"))
    if min_date is not None and parsed < min_date:
        return ["date_range"]
    if max_date is not None and parsed > max_date:
        return ["date_range"]
    return []


def _check_country_or_market(value: Any) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    try:
        import pycountry
    except ImportError:
        return []

    try:
        pycountry.countries.lookup(text)
        return []
    except LookupError:
        return ["country_or_release_market"]


def _check_delimited_open_text(value: Any, custom: Mapping[str, Any]) -> list[str]:
    separator = str(custom.get("separator", ","))
    max_terms = custom.get("maxTerms")
    if max_terms is None:
        return []
    terms = [term.strip() for term in str(value).split(separator) if term.strip()]
    if len(terms) > int(max_terms):
        return ["delimited_open_text.maxTerms"]
    return []


def _check_page_locator(
    value: Any,
    custom: Mapping[str, Any],
    row: Mapping[str, Any],
    column: str,
) -> list[str]:
    del column
    numeric = _page_locator_number(value)
    if numeric is None:
        return []

    minimum = custom.get("numericMinimum")
    maximum = custom.get("numericMaximum")
    if minimum is not None and numeric < int(minimum):
        return ["page_locator.numeric_range"]
    if maximum is not None and numeric > int(maximum):
        return ["page_locator.numeric_range"]

    comparison = custom.get("numericComparison")
    if isinstance(comparison, Mapping):
        other_field = comparison.get("field")
        other_numeric = _page_locator_number(row.get(str(other_field)))
        if other_numeric is not None:
            operator = comparison.get("operator")
            if not _compare_numbers(numeric, other_numeric, str(operator)):
                return ["page_locator.numeric_comparison"]
    return []


def _check_open_taxonomy(value: Any, custom: Mapping[str, Any]) -> list[str]:
    if not bool(custom.get("exhaustive", False)):
        return []

    families = custom.get("expectedFamilies") or []
    if not families:
        return []

    values = _values_for_membership_check(value)
    for item in values:
        text = str(item).casefold()
        if not any(str(family).casefold() in text for family in families):
            return ["open_taxonomy"]
    return []


def _open_taxonomy_diagnostics(
    records: Sequence[Mapping[str, Any]],
    column: str,
    field_schema: Mapping[str, Any],
) -> Dict[str, Any]:
    custom = field_schema.get("x-pydi-consistency")
    if not isinstance(custom, Mapping) or custom.get("rule") != "open_taxonomy":
        return {}

    families = custom.get("expectedFamilies") or []
    if not families:
        return {}

    n_values_checked = 0
    n_expected_family_matches = 0
    for row in records:
        value = row.get(column)
        if _zero_as_missing(value, field_schema) or _is_missing(value):
            continue
        for item in _values_for_membership_check(value):
            if _is_missing(item):
                continue
            n_values_checked += 1
            text = str(item).casefold()
            if any(str(family).casefold() in text for family in families):
                n_expected_family_matches += 1

    if n_values_checked == 0:
        return {}

    exhaustive = bool(custom.get("exhaustive", False))
    return {
        "exhaustive": exhaustive,
        "membership_enforced": exhaustive,
        "n_values_checked": n_values_checked,
        "n_expected_family_matches": n_expected_family_matches,
        "n_outside_expected_families": (n_values_checked - n_expected_family_matches),
        "expected_family_match_rate": (n_expected_family_matches / n_values_checked),
    }


def _custom_rule(field_schema: Mapping[str, Any]) -> Optional[str]:
    custom = field_schema.get("x-pydi-consistency")
    if isinstance(custom, Mapping):
        rule = custom.get("rule")
        return str(rule) if rule is not None else None
    return None


def _zero_as_missing(value: Any, field_schema: Mapping[str, Any]) -> bool:
    if _custom_rule(field_schema) != "zero_as_missing":
        return False
    numeric_value = _as_decimal(value)
    return numeric_value == Decimal("0") if numeric_value is not None else False


def _values_for_membership_check(value: Any) -> list[Any]:
    values = _coerce_array(value)
    if values is not None:
        return values
    return [value]


def _is_identifier_column(column: str) -> bool:
    normalized = column.strip().casefold()
    return (
        normalized in {"id", "identifier"}
        or normalized.endswith("_id")
        or normalized.startswith("id_")
    )


def _coerce_array(value: Any) -> Optional[list[Any]]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                    continue
                if isinstance(parsed, (list, tuple, set)):
                    return list(parsed)
    return None


def _as_decimal(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, (int, np.integer)):
        return Decimal(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = Decimal(text)
        except InvalidOperation:
            return None
        return parsed if parsed.is_finite() else None
    return None


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _page_locator_number(value: Any) -> Optional[int]:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    return int(text) if re.fullmatch(r"\d+", text) else None


def _compare_numbers(left: int, right: int, operator: str) -> bool:
    if operator == ">=":
        return left >= right
    if operator == ">":
        return left > right
    if operator == "<=":
        return left <= right
    if operator == "<":
        return left < right
    if operator in {"=", "=="}:
        return left == right
    return True


def _matches_enum(value: Any, enum_values: Any) -> bool:
    if not isinstance(enum_values, SequenceABC) or isinstance(enum_values, str):
        return False
    for allowed in enum_values:
        if value == allowed:
            return True
        if str(value) == str(allowed):
            return True
    return False


def _is_iso_currency_code(value: str) -> bool:
    try:
        import pycountry
    except ImportError:
        return bool(re.fullmatch(r"[A-Z]{3}", value))

    try:
        pycountry.currencies.lookup(value)
        return True
    except LookupError:
        return False


def _is_missing(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

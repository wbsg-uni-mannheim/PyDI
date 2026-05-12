"""
JSON Schema integration for PyDI normalization.

This module provides functions to read JSON Schema files and generate:
- NormalizationSpec: for data transformation
- Validation configuration: for use with validators

These are kept separate to maintain a clean separation of concerns.

Examples:
    >>> from PyDI.normalization import load_normalization_spec, load_validation_spec

    >>> # Load normalization spec from JSON schema
    >>> norm_spec = load_normalization_spec("schema.json")

    >>> # Load validation spec from JSON schema
    >>> val_spec = load_validation_spec("schema.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .spec import ColumnSpec, NormalizationSpec


# === TYPE MAPPINGS ===

TYPE_MAPPING: dict[str, str] = {
    "string": "string",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
}
"""Map JSON Schema types to PyDI output_type values."""

FORMAT_MAPPING: dict[str, dict[str, Any]] = {
    # Date/time formats
    "date": {"output_type": "datetime"},
    "date-time": {"output_type": "datetime"},
    "time": {"output_type": "datetime"},
    # Email
    "email": {"normalize_email": True, "output_type": "string"},
    "idn-email": {"normalize_email": True, "output_type": "string"},
    # URI/URL - just type conversion
    "uri": {"output_type": "string"},
    "uri-reference": {"output_type": "string"},
    # PyDI custom formats - Country
    "country-alpha2": {"country_format": "alpha_2", "output_type": "string"},
    "country-alpha3": {"country_format": "alpha_3", "output_type": "string"},
    "country-name": {"country_format": "name", "output_type": "string"},
    "country-numeric": {"country_format": "numeric", "output_type": "string"},
    # PyDI custom formats - Currency
    "currency-alpha3": {"currency_format": "alpha_3", "output_type": "string"},
    "currency-name": {"currency_format": "name", "output_type": "string"},
    # PyDI custom formats - Phone
    "phone-e164": {"phone_format": "e164", "output_type": "string"},
    "phone-international": {"phone_format": "international", "output_type": "string"},
    "phone-national": {"phone_format": "national", "output_type": "string"},
    # PyDI custom formats - Other
    "percentage": {"convert_percentage": "to_decimal", "output_type": "float"},
    "stdnum": {"stdnum_format": True, "output_type": "string"},
}
"""Map JSON Schema format values to ColumnSpec fields."""


# === SCHEMA LOADING ===


def load_schema(source: dict | str | Path) -> dict:
    """
    Load a JSON Schema from various sources.

    Args:
        source: JSON Schema as dict, JSON string, or path to file

    Returns:
        Parsed JSON Schema as dict

    Raises:
        FileNotFoundError: If file path doesn't exist
        json.JSONDecodeError: If JSON is invalid

    Examples:
        >>> schema = load_schema("schema.json")
        >>> schema = load_schema({"type": "object", "properties": {}})
        >>> schema = load_schema('{"type": "object"}')
    """
    if isinstance(source, dict):
        return source

    if isinstance(source, Path) or (
        isinstance(source, str)
        and (source.endswith(".json") or "/" in source or "\\" in source)
    ):
        path = Path(source)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Assume JSON string
    return json.loads(source)


def _resolve_ref(schema: dict, ref: str) -> dict:
    """Resolve a local $ref pointer within a schema."""
    if not ref.startswith("#/"):
        raise ValueError(f"Only local references supported. Got: {ref}")

    parts = ref[2:].split("/")
    result = schema
    for part in parts:
        if part not in result:
            raise ValueError(f"Reference '{ref}' not found in schema")
        result = result[part]
    return result


def _get_properties(schema: dict, property_path: str | None = None) -> dict[str, dict]:
    """Extract properties from a schema, optionally navigating to a nested path."""
    target = schema

    if property_path:
        for part in property_path.split("."):
            if part not in target:
                raise KeyError(f"Path '{property_path}' not found at '{part}'")
            target = target[part]

    if "$ref" in target:
        target = _resolve_ref(schema, target["$ref"])

    if "properties" not in target:
        raise ValueError(f"Schema has no 'properties' field")

    return target["properties"]


# === NORMALIZATION SPEC ===


def load_normalization_spec(
    source: dict | str | Path,
    *,
    property_path: str | None = None,
) -> NormalizationSpec:
    """
    Load a NormalizationSpec from a JSON Schema.

    Extracts transformation settings from the schema (type conversion,
    format handling). Validation constraints are ignored - use
    load_validation_spec() for those.

    Args:
        source: JSON Schema as dict, JSON string, or file path
        property_path: Dot-notation path to navigate to nested definitions
                      (e.g., "definitions.movie")

    Returns:
        NormalizationSpec configured for data transformation

    Examples:
        >>> spec = load_normalization_spec("target_schema.json")
        >>> spec = load_normalization_spec(schema_dict, property_path="definitions.movie")
    """
    root_schema = load_schema(source)
    properties = _get_properties(root_schema, property_path)

    spec = NormalizationSpec()

    for prop_name, prop_schema in properties.items():
        # Skip nested objects and arrays
        prop_type = prop_schema.get("type")
        if prop_type in ("object", "array"):
            continue
        if isinstance(prop_type, list) and any(t in ("object", "array") for t in prop_type):
            continue

        # Build ColumnSpec from property
        col_spec = _property_to_column_spec(prop_schema, root_schema)

        # Only add if there's something to transform
        if col_spec != ColumnSpec():
            spec.columns[prop_name] = col_spec

    return spec


def _property_to_column_spec(prop_schema: dict, root_schema: dict) -> ColumnSpec:
    """Convert a JSON Schema property to a ColumnSpec (transformation only)."""
    if "$ref" in prop_schema:
        prop_schema = _resolve_ref(root_schema, prop_schema["$ref"])

    spec_kwargs: dict[str, Any] = {}

    # Type mapping
    json_type = prop_schema.get("type")
    if isinstance(json_type, list):
        json_type = next((t for t in json_type if t != "null"), None)

    if json_type in TYPE_MAPPING:
        spec_kwargs["output_type"] = TYPE_MAPPING[json_type]

    # Format mapping (contains transformation settings)
    json_format = prop_schema.get("format")
    if json_format in FORMAT_MAPPING:
        spec_kwargs.update(FORMAT_MAPPING[json_format])

    # PyDI extensions for transformation
    if "x-pydi-target-unit" in prop_schema:
        spec_kwargs["target_unit"] = prop_schema["x-pydi-target-unit"]
    if "x-pydi-expand-scale" in prop_schema:
        spec_kwargs["expand_scale_modifiers"] = prop_schema["x-pydi-expand-scale"]
    if "x-pydi-convert-percentage" in prop_schema:
        spec_kwargs["convert_percentage"] = prop_schema["x-pydi-convert-percentage"]
    if "x-pydi-on-failure" in prop_schema:
        spec_kwargs["on_failure"] = prop_schema["x-pydi-on-failure"]
    if "x-pydi-phone-region" in prop_schema:
        spec_kwargs["phone_default_region"] = prop_schema["x-pydi-phone-region"]
    if "x-pydi-case" in prop_schema:
        spec_kwargs["case"] = prop_schema["x-pydi-case"]
    if "x-pydi-strip-whitespace" in prop_schema:
        spec_kwargs["strip_whitespace"] = prop_schema["x-pydi-strip-whitespace"]
    if "x-pydi-taxonomy" in prop_schema:
        spec_kwargs["taxonomy_path"] = prop_schema["x-pydi-taxonomy"]
    if "x-pydi-taxonomy-column" in prop_schema:
        spec_kwargs["taxonomy_column"] = prop_schema["x-pydi-taxonomy-column"]
    if "x-pydi-taxonomy-mapping" in prop_schema:
        spec_kwargs["taxonomy_mapping_path"] = prop_schema[
            "x-pydi-taxonomy-mapping"
        ]

    return ColumnSpec(**spec_kwargs)


# === VALIDATION SPEC ===


def load_validation_spec(
    source: dict | str | Path,
    *,
    property_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Load validation constraints from a JSON Schema.

    Extracts validation rules (min/max, pattern, enum, etc.) that can be
    used with PyDI validators. Transformation settings are ignored - use
    load_normalization_spec() for those.

    Args:
        source: JSON Schema as dict, JSON string, or file path
        property_path: Dot-notation path to navigate to nested definitions

    Returns:
        Dict mapping column names to validation constraints:
        {
            "column_name": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "nullable": True,
                ...
            }
        }

    Examples:
        >>> val_spec = load_validation_spec("target_schema.json")
        >>> # Use with validators
        >>> from PyDI.normalization import RangeValidator
        >>> ranges = {col: {"min": v.get("minimum"), "max": v.get("maximum")}
        ...           for col, v in val_spec.items() if "minimum" in v or "maximum" in v}
        >>> validator = RangeValidator(ranges=ranges)
    """
    root_schema = load_schema(source)
    properties = _get_properties(root_schema, property_path)

    validation_spec: dict[str, dict[str, Any]] = {}

    for prop_name, prop_schema in properties.items():
        # Skip nested objects and arrays
        prop_type = prop_schema.get("type")
        if prop_type in ("object", "array"):
            continue
        if isinstance(prop_type, list) and any(t in ("object", "array") for t in prop_type):
            continue

        constraints = _property_to_validation(prop_schema, root_schema)
        if constraints:
            validation_spec[prop_name] = constraints

    return validation_spec


def _property_to_validation(prop_schema: dict, root_schema: dict) -> dict[str, Any]:
    """Extract validation constraints from a JSON Schema property."""
    if "$ref" in prop_schema:
        prop_schema = _resolve_ref(root_schema, prop_schema["$ref"])

    constraints: dict[str, Any] = {}

    # Type info
    json_type = prop_schema.get("type")
    if isinstance(json_type, list):
        constraints["nullable"] = "null" in json_type
        json_type = next((t for t in json_type if t != "null"), None)
    else:
        constraints["nullable"] = False

    if json_type:
        constraints["type"] = json_type

    # Numeric constraints
    if "minimum" in prop_schema:
        constraints["minimum"] = prop_schema["minimum"]
    if "maximum" in prop_schema:
        constraints["maximum"] = prop_schema["maximum"]
    if "exclusiveMinimum" in prop_schema:
        constraints["exclusive_minimum"] = prop_schema["exclusiveMinimum"]
    if "exclusiveMaximum" in prop_schema:
        constraints["exclusive_maximum"] = prop_schema["exclusiveMaximum"]

    # String constraints
    if "minLength" in prop_schema:
        constraints["min_length"] = prop_schema["minLength"]
    if "maxLength" in prop_schema:
        constraints["max_length"] = prop_schema["maxLength"]
    if "pattern" in prop_schema:
        constraints["pattern"] = prop_schema["pattern"]

    # Enum
    if "enum" in prop_schema:
        constraints["enum"] = prop_schema["enum"]

    # Required (if present in parent schema)
    # Note: This would need to be handled at the object level

    return constraints


# === PUBLIC API ===

__all__ = [
    "load_schema",
    "load_normalization_spec",
    "load_validation_spec",
    "TYPE_MAPPING",
    "FORMAT_MAPPING",
]

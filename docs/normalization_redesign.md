# PyDI Normalization Module Redesign

## Overview

This document outlines the redesign of the PyDI normalization module. The goal is to create a cleaner, more modular system that:

1. **Profiles DataFrames** - Generates reports about columns, detected types, and possible values
2. **Allows user specification** - Users can specify what values should look like in each column
3. **Performs conversions** - Automatically converts values to the specified format

The redesign integrates well-maintained external libraries for specific normalization tasks rather than re-implementing everything from scratch.

---

## Core Workflow

```
DataFrame → Profile/Report → User Specification → Transformation → Normalized DataFrame
```

### 1. Profiling Phase
- Analyze each column to detect:
  - Data type (numeric, text, date, etc.)
  - Unit patterns (e.g., "5 km", "10 MEO", "$500k")
  - Scale modifiers (thousands, millions, MEO, etc.)
  - Standard number formats (ISBN, IBAN, VAT, phone numbers)
  - Country/currency references
  - Common patterns and anomalies

### 2. Specification Phase
- Present findings to user in a structured report
- User can specify target formats per column:
  - "Convert all lengths to meters"
  - "Normalize phone numbers to E.164"
  - "Expand scale modifiers (MEO → multiply by 1,000,000)"

### 3. Transformation Phase
- Apply transformations based on specifications
- Return normalized DataFrame with optional metadata

---

## External Library Integration

### Required Libraries

All libraries are **hard dependencies** - they must be installed.

| Library | Purpose | PyPI | Notes |
|---------|---------|------|-------|
| **Pint** | Physical unit conversions | `pint` | Length, weight, volume, temperature, energy, etc. |
| **pycountry** | Country/currency/language codes | `pycountry` | ISO 3166 countries, ISO 4217 currencies, ISO 639 languages |
| **python-stdnum** | Standard number formats | `python-stdnum` | ISBN, IBAN, VAT numbers, IMEI, 200+ formats |
| **phonenumbers** | Phone number parsing/formatting | `phonenumbers` | Google's libphonenumber port |
| **email-validator** | Email validation/normalization | `email-validator` | Syntax + deliverability checks |

### Future Considerations (Not Yet Implemented)

| Library | Purpose | Notes |
|---------|---------|-------|
| **pypostal** | Address parsing/normalization | Requires libpostal C library - recommended for address normalization when needed |
| **postal-address** | Simpler address handling | Pure Python alternative, less powerful than pypostal |
| **google-i18n-address** | International address formatting | Google's i18n data |
| **forex-python** | Currency exchange rates | Real-time rates (API-based) - for actual currency conversion |
| **CurrencyConverter** | Historical currency rates | Embedded ECB data - offline currency conversion |
| **cleanco** | Company name normalization | Strip legal suffixes (Ltd, GmbH, Inc) - decided not to include as adds little value |

---

## Scale Modifiers

### Supported Patterns

These are abbreviations/words that modify numeric values:

| Pattern | Multiplier | Examples |
|---------|------------|----------|
| hundred(s) | 100 | "5 hundred" |
| thousand(s), k, K | 1,000 | "5k", "5 thousand" |
| million(s), m, M, mil | 1,000,000 | "5M", "5 million" |
| billion(s), b, B, bil | 1,000,000,000 | "5B", "5 billion" |
| trillion(s), t, T, tril | 1,000,000,000,000 | "5T" |

### Currency-Specific Scale Modifiers

| Pattern | Meaning | Example |
|---------|---------|---------|
| kEUR, KEUR | thousands of euros | "500 kEUR" → 500,000 EUR |
| MEUR, mEUR | millions of euros | "5 MEUR" → 5,000,000 EUR |
| MEO | millions of euros (Portuguese) | "5 MEO" → 5,000,000 EUR |
| kUSD, KUSD | thousands of USD | "100 kUSD" → 100,000 USD |
| MUSD, mUSD | millions of USD | "2.5 MUSD" → 2,500,000 USD |

**Note:** Single-letter modifiers (k, m, b) can conflict with units (km = kilometers, not thousands of meters). Context and word boundaries are important.

---

## Module Structure (Implemented)

```
PyDI/normalization/
├── __init__.py           # Public API exports
├── profile.py            # DataFrame profiling and reporting
├── integrations/         # External library wrappers
│   ├── __init__.py
│   ├── pint_units.py     # Pint wrapper for physical units
│   ├── stdnum.py         # python-stdnum wrapper
│   ├── phone.py          # phonenumbers wrapper
│   ├── country.py        # pycountry wrapper
│   └── email.py          # email-validator wrapper
├── scale.py              # Scale modifier handling (MEO, thousands, etc.)
├── units.py              # REFACTORED: Now uses Pint under the hood
├── text.py               # Text normalization (unchanged)
├── types.py              # Type conversion
├── columns.py            # Column analysis
├── datasets.py           # Dataset-level normalization
├── values.py             # Value-level ops
├── validators.py         # Validation utilities
└── detectors.py          # Detection utilities
```

### Changes Made

- **units.py** → Refactored to use Pint under the hood
  - Removed custom UnitRegistry, UnitConverter, UnitDetector classes
  - Now delegates to `integrations/pint_units.py` for all unit operations
  - Scale modifiers moved to `scale.py`
  - Temperature handling now uses Pint (handles offsets correctly)

---

## API Design

### Profiling API

```python
from PyDI.normalization import profile_dataframe, ColumnProfile

# Generate profile for entire DataFrame
report = profile_dataframe(df)

# Access column-level information
for col_name, col_profile in report.columns.items():
    print(f"{col_name}:")
    print(f"  Detected type: {col_profile.detected_type}")
    print(f"  Unit category: {col_profile.unit_category}")
    print(f"  Scale modifiers found: {col_profile.scale_modifiers}")
    print(f"  Sample values: {col_profile.sample_values}")
    print(f"  Suggested normalization: {col_profile.suggestions}")

# Export to dict/JSON
report_dict = report.to_dict()
report_json = report.to_json()
```

### Specification API

```python
from PyDI.normalization import NormalizationSpec

# Create specification from profile suggestions
spec = NormalizationSpec.from_profile(report)

# Or manually specify
spec = NormalizationSpec()
spec.set_column("revenue",
    normalize_scale=True,      # Expand MEO, thousands, etc.
    target_unit="EUR",         # Convert to euros
    output_type="float"        # Numeric output
)
spec.set_column("phone",
    format="E164"              # International phone format
)
spec.set_column("company_name",
    strip_legal_terms=True,    # Remove Ltd, GmbH, etc.
    case="title"               # Title case
)
spec.set_column("country",
    format="alpha_2"           # ISO 3166-1 alpha-2 (e.g., "DE", "US")
)
```

### Transformation API

```python
from PyDI.normalization import normalize_dataframe

# Apply specification
normalized_df = normalize_dataframe(df, spec)

# Or one-liner with auto-detection
normalized_df = normalize_dataframe(df, auto=True)
```

---

## Pint Integration Details

### Why Pint?

1. **Comprehensive unit database** - 100s of units built-in
2. **Correct conversions** - Handles edge cases (temperature offsets, compound units)
3. **String parsing** - `"5 kilometers"` works out of the box
4. **Pandas integration** - Native support for DataFrames
5. **Extensible** - Easy to add custom units
6. **Well-maintained** - Active development, good documentation

### What Pint Handles

- Length (m, km, mi, ft, in, etc.)
- Mass/Weight (kg, g, lb, oz, etc.)
- Volume (l, ml, gal, etc.)
- Temperature (°C, °F, K) - with correct offset handling
- Time (s, min, h, d, etc.)
- Speed (m/s, km/h, mph, etc.)
- Energy (J, kWh, cal, BTU, etc.)
- Power (W, kW, hp, etc.)
- Pressure (Pa, bar, psi, atm, etc.)
- Area (m², ha, acre, etc.)
- Information (B, KB, MB, GB, etc.)
- And many more...

### What We Handle Separately

- **Scale modifiers** (MEO, thousands, kEUR) - Not physical units
- **Currency codes** - Use pycountry for ISO 4217 codes (no actual conversion)
- **Standard numbers** (ISBN, IBAN, VAT) - Use python-stdnum
- **Phone numbers** - Use phonenumbers
- **Country codes** - Use pycountry for ISO 3166 codes

### Pint Wrapper Design

```python
# integrations/pint_units.py

import pint

# Create a unit registry (singleton)
_ureg = pint.UnitRegistry()

def parse_quantity(text: str) -> tuple[float, str] | None:
    """Parse a quantity string like '5 km' or '100 miles'."""
    try:
        q = _ureg.parse_expression(text)
        return (q.magnitude, str(q.units))
    except:
        return None

def convert(value: float, from_unit: str, to_unit: str) -> float | None:
    """Convert value between units."""
    try:
        q = value * _ureg(from_unit)
        return q.to(to_unit).magnitude
    except:
        return None

def normalize_to_base(value: float, unit: str) -> tuple[float, str]:
    """Convert to SI base unit."""
    q = value * _ureg(unit)
    base = q.to_base_units()
    return (base.magnitude, str(base.units))
```

---

## Scale Modifier Design

```python
# scale.py

import re
from dataclasses import dataclass

@dataclass
class ScaleModifier:
    pattern: re.Pattern
    multiplier: float
    name: str

# Generic scale words
GENERIC_SCALES = [
    ScaleModifier(re.compile(r'\bhundreds?\b', re.I), 100, 'hundred'),
    ScaleModifier(re.compile(r'\bthousands?\b', re.I), 1_000, 'thousand'),
    ScaleModifier(re.compile(r'\bmillions?\b', re.I), 1_000_000, 'million'),
    ScaleModifier(re.compile(r'\bbillions?\b', re.I), 1_000_000_000, 'billion'),
    ScaleModifier(re.compile(r'\btrillions?\b', re.I), 1_000_000_000_000, 'trillion'),
]

# Currency-specific scales (higher priority)
CURRENCY_SCALES = [
    # Euros
    ScaleModifier(re.compile(r'\bMEO\b'), 1_000_000, 'MEO'),  # Portuguese
    ScaleModifier(re.compile(r'\bMEUR\b', re.I), 1_000_000, 'MEUR'),
    ScaleModifier(re.compile(r'\bkEUR\b', re.I), 1_000, 'kEUR'),
    # USD
    ScaleModifier(re.compile(r'\bMUSD\b', re.I), 1_000_000, 'MUSD'),
    ScaleModifier(re.compile(r'\bkUSD\b', re.I), 1_000, 'kUSD'),
    # Generic k/m/b (careful with unit conflicts)
    ScaleModifier(re.compile(r'(?<!\w)[kK](?=\s|$|\d)'), 1_000, 'k'),
    ScaleModifier(re.compile(r'(?<!\w)[mM](?=\s|$|\d)'), 1_000_000, 'm'),
    ScaleModifier(re.compile(r'(?<!\w)[bB](?=\s|$|\d)'), 1_000_000_000, 'b'),
]

def detect_scale_modifier(text: str) -> ScaleModifier | None:
    """Detect scale modifier in text, currency-specific first."""
    for modifier in CURRENCY_SCALES + GENERIC_SCALES:
        if modifier.pattern.search(text):
            return modifier
    return None

def expand_scale(text: str, value: float) -> tuple[float, str]:
    """Expand scale modifier and return (scaled_value, cleaned_text)."""
    modifier = detect_scale_modifier(text)
    if modifier:
        # Remove the modifier from text
        cleaned = modifier.pattern.sub('', text).strip()
        return (value * modifier.multiplier, cleaned)
    return (value, text)
```

---

## pycountry Integration (Country Codes)

pycountry handles ISO standards for countries, currencies, and languages - useful for normalizing country references in datasets (e.g., the Companies dataset).

```python
# integrations/country.py

import pycountry

def normalize_country(value: str, output_format: str = "alpha_2") -> str | None:
    """
    Normalize country name/code to standard format.

    Args:
        value: Country name, alpha-2, alpha-3, or numeric code
        output_format: One of "alpha_2", "alpha_3", "numeric", "name", "official_name"

    Examples:
        normalize_country("Germany") → "DE"
        normalize_country("DEU") → "DE"
        normalize_country("276") → "DE"
        normalize_country("deutschland", "name") → "Germany"
    """
    try:
        # Try exact lookup first
        country = pycountry.countries.get(alpha_2=value.upper())
        if not country:
            country = pycountry.countries.get(alpha_3=value.upper())
        if not country:
            country = pycountry.countries.get(numeric=value)
        if not country:
            country = pycountry.countries.get(name=value)
        if not country:
            # Fuzzy search as fallback
            results = pycountry.countries.search_fuzzy(value)
            country = results[0] if results else None

        if country:
            return getattr(country, output_format, country.alpha_2)
    except:
        pass
    return None

def normalize_currency(value: str, output_format: str = "alpha_3") -> str | None:
    """Normalize currency code/name to ISO 4217."""
    try:
        currency = pycountry.currencies.get(alpha_3=value.upper())
        if not currency:
            currency = pycountry.currencies.get(name=value)
        if currency:
            return getattr(currency, output_format, currency.alpha_3)
    except:
        pass
    return None
```

---

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Currency conversion | Code normalization only | No actual EUR→USD conversion; just normalize "euro"→"EUR" |
| Address handling | Not implemented yet | Recommend pypostal when needed (requires C library) |
| Dependencies | Hard requirements | All libraries must be installed |
| Profile output | Dataclass + `.to_dict()/.to_json()` | Programmatic access + serialization |
| Backwards compatibility | Clean break | Module is alpha; old API will be removed |

---

## Implementation Plan

### Phase 1: Foundation
1. Add new dependencies to `pyproject.toml`
2. Create `integrations/` folder with wrappers
3. Create `scale.py` for scale modifier handling
4. Write tests for integrations

### Phase 2: Profiling
1. Create `profile.py` with column analysis
2. Implement detection for:
   - Physical units (via Pint)
   - Scale modifiers
   - Standard numbers (via stdnum)
   - Phone numbers
   - Country codes
   - Company names
3. Create report format with `.to_dict()` and `.to_json()`

### Phase 3: Specification & Transform
1. Create `spec.py` for user specifications
2. Create `transform.py` for orchestration
3. Integrate with existing `datasets.py`

### Phase 4: Cleanup
1. Remove old `units.py` code
2. Simplify `types.py` and `values.py`
3. Update `__init__.py` exports
4. Update documentation

---

## Dependencies Added

```toml
# pyproject.toml additions (already added)

dependencies = [
    # ... existing ...
    "Pint>=0.23,<1.0",
    "pycountry>=24.0,<25.0",
    "python-stdnum>=1.20,<2.0",
    "phonenumbers>=8.13,<9.0",
    "email-validator>=2.0,<3.0",
]
```

---

## Test Coverage

Currently no tests exist for the normalization module. Tests will be added in:
- `tests/normalization/test_integrations.py` - Integration wrapper tests
- `tests/normalization/test_scale.py` - Scale modifier tests
- `tests/normalization/test_profile.py` - Profiling tests
- `tests/normalization/test_transform.py` - Transformation tests

---

## Sources

- [Pint Documentation](https://pint.readthedocs.io/)
- [pycountry on PyPI](https://pypi.org/project/pycountry/)
- [python-stdnum Formats](https://arthurdejong.org/python-stdnum/formats)
- [phonenumbers on PyPI](https://pypi.org/project/phonenumbers/)
- [email-validator on GitHub](https://github.com/JoshData/python-email-validator)

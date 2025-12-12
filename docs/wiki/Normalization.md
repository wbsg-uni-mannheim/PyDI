# Value Normalization

Value Normalization transforms messy, inconsistent data into clean and standardized formats. The input is a DataFrame with columns containing values in varying formats (e.g., "5 MEO", "Germany", "DEU", "+1-555-123-4567"). The output is a DataFrame with normalized values in consistent formats (e.g., `5000000`, "DE", "+15551234567").

PyDI's normalization module provides `profile_dataframe` for auto-detecting column types and suggesting transformations, `NormalizationSpec` for defining how columns should be normalized, and `transform_dataframe` for applying the specified transformations. The module integrates external libraries for specific normalization tasks (Pint for units, pycountry for country codes, python-stdnum for standard numbers, etc.).

## Requirements

- Input must be a pandas `DataFrame`
- Optional: Install additional libraries for specific normalizations:
  - `pint` for unit conversions
  - `pycountry` for country/currency codes
  - `python-stdnum` for ISBN, IBAN, VAT validation
  - `phonenumbers` for phone number parsing
  - `email-validator` for email validation

## Profiling

The profiling step analyzes DataFrame columns to detect types, units, scale modifiers, and other patterns. Use `profile_dataframe()` to generate a profile that suggests normalizations.

```python
from PyDI.normalization import profile_dataframe

profile = profile_dataframe(df)
print(profile.summary())
```

Output:
```
DataFrame Profile: 1000 rows, 5 columns
============================================================

revenue:
  Type: scaled_number
  Samples: ['5.2 MEO', '2.5 MEUR', '750 kEUR']
  Scale modifiers: {'modifiers_detected': {'MEO': 45, 'MEUR': 30, 'kEUR': 25}, 'coverage': 0.8}
  Suggestion: Consider expanding scale modifiers (e.g., '5 MEO' → 5000000)

country:
  Type: country
  Samples: ['Germany', 'DEU', 'US']
  Country formats: {'alpha_2': 40, 'alpha_3': 35, 'name': 25}
  Suggestion: Consider normalizing to ISO 3166 alpha-2 codes

vat_number:
  Type: stdnum
  Samples: ['DE123456789', 'GB123456789', 'FR12345678901']
  Standard numbers: {'types_detected': {'de.vat': 40, 'gb.vat': 30, 'fr.tva': 30}}
  Suggestion: Consider validating and formatting standard numbers
```

### Supported Data Types

The profiler detects the following column types:
- `numeric`, `string`, `date`, `boolean` - Basic types
- `unit_quantity` - Values with units (e.g., "5 km", "100 kg")
- `scaled_number` - Values with scale modifiers (e.g., "5 MEO", "2.5 million")
- `percentage` - Percentage values (e.g., "50%", "0.5")
- `country`, `currency` - Country/currency codes or names
- `phone`, `email`, `url` - Contact information
- `stdnum` - Standard numbers (ISBN, IBAN, VAT, etc.)
- `coordinate` - Geographic coordinates

## Normalization Specification

The `NormalizationSpec` defines how columns should be normalized. You can create a spec manually or generate a spec from a profile if you want to translate heterogenious data into this target profile.

### Manual Specification

```python
from PyDI.normalization import NormalizationSpec

spec = NormalizationSpec()
spec.set_column("revenue", expand_scale_modifiers=True, output_type="float")
spec.set_column("country", country_format="alpha_2")
spec.set_column("currency", currency_format="alpha_3")
spec.set_column("phone", phone_format="e164", phone_default_region="US")
spec.set_column("email", normalize_email=True)
spec.set_column("vat_number", stdnum_format=True, on_failure="null")
spec.set_column("percentage", convert_percentage="to_decimal", output_type="float")
```

### Generate Specification from Profile

```python
from PyDI.normalization import profile_dataframe, NormalizationSpec

profile = profile_dataframe(df)
spec = NormalizationSpec.from_profile(profile)
```

### Column Spec Options

Each column can be configured with the following options:

| Option | Values | Description |
|--------|--------|-------------|
| `output_type` | `"string"`, `"float"`, `"int"`, `"bool"`, `"datetime"`, `"keep"` | Output data type |
| `on_failure` | `"keep"`, `"null"`, `"raise"` | How to handle transformation failures |
| `target_unit` | Unit string (e.g., `"m"`, `"kg"`) | Convert quantities to this unit |
| `expand_scale_modifiers` | `True`/`False` | Expand MEO, million, etc. |
| `convert_percentage` | `"to_decimal"`, `"to_percent"`, `"keep"` | Convert percentage format |
| `country_format` | `"alpha_2"`, `"alpha_3"`, `"numeric"`, `"name"`, `"keep"` | Output country format |
| `currency_format` | `"alpha_3"`, `"name"`, `"keep"` | Output currency format |
| `phone_format` | `"e164"`, `"international"`, `"national"`, `"keep"` | Output phone format |
| `phone_default_region` | Region code (e.g., `"US"`) | Default region for parsing |
| `normalize_email` | `True`/`False` | Normalize email addresses |
| `stdnum_format` | `True`/`False` | Format standard numbers |
| `case` | `"lower"`, `"upper"`, `"title"`, `"keep"` | Text case transformation |
| `strip_whitespace` | `True`/`False` | Strip leading/trailing whitespace |

### Handling Transformation Failures

By default, values that cannot be transformed are kept unchanged. Use `on_failure` to change this behavior:

```python
# Keep original value on failure (default)
spec.set_column("country", country_format="alpha_2", on_failure="keep")

# Set to None/NaN on failure
spec.set_column("vat_number", stdnum_format=True, on_failure="null")

# Raise an error on failure
spec.set_column("email", normalize_email=True, on_failure="raise")
```

## Data Transformation

The `transform_dataframe` function applies the spec to a DataFrame and returns detailed results.

```python
from PyDI.normalization import transform_dataframe

result = transform_dataframe(df, spec)

# Get the normalized DataFrame
normalized_df = result.dataframe

# Check transformation statistics
print(f"Total transformed: {result.total_transformed}")
print(f"Total failed: {result.total_failed}")

# Per-column details
for col_name, col_result in result.columns.items():
    print(f"{col_name}: {col_result.values_transformed} transformed, {col_result.values_failed} failed")
    if col_result.errors:
        for error in col_result.errors[:3]:
            print(f"  - {error}")
```

### Quick Normalization

For simple use cases, use `normalize_dataframe` directly:

```python
from PyDI.normalization import normalize_dataframe

# With manual spec
normalized_df = normalize_dataframe(df, spec)

# With auto-detection
normalized_df = normalize_dataframe(df, auto=True)
```

## Unit Conversions

The module uses Pint for unit of measurement conversions (length, weight, temperature, etc.).

```python
from PyDI.normalization import normalize_quantity, convert_units

# Parse and normalize quantities
normalize_quantity("5 km", target_unit="m")        # (5000.0, 'meter')
normalize_quantity("100 fahrenheit", target_unit="celsius")  # (37.77..., 'degree_Celsius')
normalize_quantity("10 MEO")                        # (10000000.0, 'dimensionless')

# Direct unit conversion
convert_units(100, "fahrenheit", "celsius")         # 37.77...
convert_units(5, "km", "miles")                     # 3.10...
```

### Supported Scale Modifiers

The following scale modifiers are expanded automatically:
- `MEO`, `MEUR` → ×1,000,000 (million euros)
- `kEUR`, `KEUR` → ×1,000 (thousand euros)
- `million`, `mio`, `mn` → ×1,000,000
- `billion`, `bn` → ×1,000,000,000
- `thousand`, `k` → ×1,000

```python
from PyDI.normalization import parse_scaled_number

parse_scaled_number("5.2 MEO")      # (5200000.0, ScaleResult(...))
parse_scaled_number("2.5 million")  # (2500000.0, ScaleResult(...))
```

## Country & Currency Codes

The module uses pycountry for ISO code normalization.

```python
from PyDI.normalization import normalize_country, normalize_currency

# Country normalization
normalize_country("Germany")           # 'DE'
normalize_country("DEU")               # 'DE'
normalize_country("276")               # 'DE' (numeric code)
normalize_country("deutschland")       # 'DE' (fuzzy match)
normalize_country("DE", output_format="name")  # 'Germany'

# Currency normalization
normalize_currency("EUR")              # 'EUR'
normalize_currency("Euro")             # 'EUR'
normalize_currency("978")              # 'EUR' (numeric code)
```

Note: The module relies on pycountry's built-in lookup capabilities. Native language names (e.g., "Nederland", "Brasil") may not resolve if pycountry doesn't support them.

## Standard Numbers (ISBN, IBAN, VAT)

The module uses python-stdnum for standard number validation and formatting.

```python
from PyDI.normalization import validate_stdnum, format_stdnum, detect_stdnum_type

# Detect type automatically
detect_stdnum_type("978-0-13-468599-1")  # 'isbn'
detect_stdnum_type("DE89370400440532013000")  # 'iban'
detect_stdnum_type("DE123456789")  # 'de.vat'

# Validate and format
validate_stdnum("978-0-13-468599-1", "isbn")  # True
format_stdnum("9780134685991", "isbn")  # '978-0-13-468599-1'
```

Note: VAT numbers with invalid checksums will fail validation. Use `on_failure="null"` to mark invalid numbers as None.

## Phone Numbers

The module uses phonenumbers for phone number parsing and formatting.

```python
from PyDI.normalization import parse_phone, format_phone

# Parse phone numbers
info = parse_phone("+1-555-123-4567")
print(info.country_code)  # 1
print(info.national_number)  # 5551234567

# Format phone numbers
format_phone("+15551234567", "e164")          # '+15551234567'
format_phone("+15551234567", "international")  # '+1 555-123-4567'
format_phone("+15551234567", "national")       # '(555) 123-4567'

# With default region for local numbers
format_phone("555-123-4567", "e164", default_region="US")  # '+15551234567'
```

## Email Normalization

The module uses email-validator for email validation and normalization.

```python
from PyDI.normalization import validate_email, normalize_email

# Validate
validate_email("user@example.com")  # True
validate_email("invalid-email")     # False

# Normalize (lowercase, remove dots from gmail, etc.)
normalize_email("User.Name@Gmail.com")  # 'username@gmail.com'
```

## Percentage Conversion

Convert between percentage formats:

```python
spec = NormalizationSpec()

# Convert '50%' to 0.5
spec.set_column("rate", convert_percentage="to_decimal", output_type="float")

# Convert 0.5 to 50
spec.set_column("score", convert_percentage="to_percent", output_type="float")
```

| Input | `to_decimal` | `to_percent` |
|-------|--------------|--------------|
| `'50%'` | `0.5` | `50.0` |
| `50` | `0.5` | `50.0` |
| `0.5` | `0.5` | `50.0` |

## Text Normalization

For text cleaning tasks, use the `TextNormalizer` and `HeaderNormalizer` classes:

```python
from PyDI.normalization import TextNormalizer, HeaderNormalizer

# Text normalization
text_norm = TextNormalizer(
    fix_encoding=True,      # Fix mojibake and encoding issues
    normalize_unicode=True, # NFKC normalization
    remove_html=True,       # Strip HTML tags
    collapse_whitespace=True,
)
text_norm.normalize("Caf\xc3\xa9  &amp; Tea")  # 'Café & Tea'

# Header normalization
header_norm = HeaderNormalizer(
    case="lower",
    separator="_",
    strip_punctuation=True,
)
header_norm.normalize("Product Name (USD)")  # 'product_name_usd'
```

## End-to-End Example

```python
from PyDI.normalization import (
    profile_dataframe,
    NormalizationSpec,
    transform_dataframe,
)

# 1) Profile the DataFrame to understand its structure
profile = profile_dataframe(df)
print(profile.summary())

# 2) Create normalization spec (auto or manual)
spec = NormalizationSpec.from_profile(profile)

# Or manually configure specific columns
spec.set_column("revenue", expand_scale_modifiers=True, output_type="float")
spec.set_column("country", country_format="alpha_2")
spec.set_column("vat_number", stdnum_format=True, on_failure="null")
spec.set_column("email", normalize_email=True)
spec.set_column("margin", convert_percentage="to_decimal", output_type="float")

# 3) Transform the DataFrame
result = transform_dataframe(df, spec)

# 4) Review results
print(f"Transformed: {result.total_transformed}, Failed: {result.total_failed}")

for col_name, col_result in result.columns.items():
    if col_result.values_failed > 0:
        print(f"\n{col_name}: {col_result.values_failed} failures")
        for error in col_result.errors:
            print(f"  - {error}")

# 5) Use the normalized DataFrame
normalized_df = result.dataframe
```

## Validators

The module provides validators for data quality checks:

```python
from PyDI.normalization import (
    validate_emails,
    validate_ranges,
    validate_completeness,
    DataQualityChecker,
)

# Validate email column
results = validate_emails(df["email"])
print(f"Valid: {results.valid_count}, Invalid: {results.invalid_count}")

# Validate numeric ranges
results = validate_ranges(df["age"], min_value=0, max_value=120)

# Check completeness
results = validate_completeness(df, required_columns=["name", "email", "country"])

# Full quality check
checker = DataQualityChecker(df)
report = checker.run_all_checks()
print(report.summary())
```

## Tutorials

- [Value Normalization Tutorial](../tutorial/normalization/value_normalization/value_normalization_tutorial.ipynb) - Full workflow: profiling, specs, transformations with a messy company dataset
- [Schema Matching Tutorial](../tutorial/normalization/schema_matching/schema_matching_tutorial.ipynb) - demonstrates how value normalization is used as part of a overall pipeline which matches different datasets to a JSON target schema and translates the datasets into this target schema afterwards.

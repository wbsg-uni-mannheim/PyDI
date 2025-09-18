# Normalization

Detect, clean, and normalize values, headers, text, and units. Validate and infer types. Normalization reduces noise early to improve matching quality and fusion consistency.

Modules
- `PyDI.normalization.text` — text/header cleaning, tokenization, web table cleanup
- `PyDI.normalization.values` — scalar normalizers + null handling
- `PyDI.normalization.units` — quantity parsing, unit detection/conversion
- `PyDI.normalization.detectors` — nulls/outliers/type detection/duplicates
- `PyDI.normalization.columns` — advanced column type inference
- `PyDI.normalization.validators` — schema/data validation helpers

Text and Header Normalization
- Remove punctuation/HTML/accents; standardize case; collapse whitespace.
- Header normalization expands camelCase, splits alphanumeric tokens, and standardizes separators.

Value Normalization
- Parse numbers, booleans, dates, lists, URLs, emails to canonical forms.
- Null handling replaces known null tokens (e.g., "N/A", "-", "unknown").

Units and Quantities
- Extract units from text and headers; detect categories (e.g., temperature, frequency).
- Convert values across units (e.g., `MB`→`GB`) and normalize to a preferred unit.

Type and Quality Detection
- Detect per‑column types; flag outliers; summarize duplicates and missingness.

Examples
```python
from PyDI.normalization.text import WebTableNormalizer
from PyDI.normalization.values import normalize_numeric
from PyDI.normalization.units import normalize_units

df_clean = WebTableNormalizer().normalize_dataframe(df)
price = normalize_numeric("$1,299.00")          # 1299.0
val = normalize_units("2.5 GHz")                # (2.5, "GHz")
```

Best Practices
- Normalize before schema matching; consistent tokens help label‑based methods.
- Use detectors to quantify data quality and guide cleaning priorities.

Artifacts
- Written by calling code as needed (CSV/JSON after normalization).

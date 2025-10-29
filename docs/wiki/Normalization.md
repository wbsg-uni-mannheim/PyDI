# Value Normalization

The module provides functionality for normalizing attribute values, converting between different units of measurement, cleaning headers and text, and for detecting basic quality issues (nulls, types, outliers, duplicates). 

## Capabilities 
The module `PyDI.normalization` offers the following value normalization functionality:
- Text & headers: clean punctuation/HTML/accents, standardize case/whitespace, normalize header tokens.
- Values: parse numbers, booleans, dates, lists, URLs, emails; handle common null markers.
- Units: extract and convert quantities (e.g., MB↔GB, temperature, frequency) to a preferred unit.
- Detection: null rate summaries, simple outlier flags, column/type inference, duplicate checks.
- Validation: lightweight helpers to check ranges and formats before downstream use.
  
The functionality of the module can be extended with custom functions (e.g., domain‑specific parsers or validators) and incorporated into your pipeline before schema/entity matching.

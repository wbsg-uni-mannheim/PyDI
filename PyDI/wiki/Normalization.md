# Normalization

Normalization prepares datasets for matching and fusion by cleaning headers and text, standardizing values and units, and detecting basic quality issues (nulls, types, outliers, duplicates). It aims to reduce noise while keeping transformations straightforward and reproducible.

Capabilities (module `PyDI.normalization`)
- Text & headers: clean punctuation/HTML/accents, standardize case/whitespace, normalize header tokens.
- Values: parse numbers, booleans, dates, lists, URLs, emails; handle common null markers.
- Units: extract and convert quantities (e.g., MB↔GB, temperature, frequency) to a preferred unit.
- Detection: null rate summaries, simple outlier flags, column/type inference, duplicate checks.
- Validation: lightweight helpers to check ranges and formats before downstream use.
- Normalization can be extended with custom functions (e.g., domain‑specific parsers or validators) and incorporated into your pipeline before schema/entity matching.


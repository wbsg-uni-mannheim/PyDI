Welcome to the PyDI Wiki
========================

PyDI (Python Data Integration) is a lightweight, pandas‑first framework for end‑to‑end data integration inspired by WInte.r. It focuses on simple, composable components with stable, type‑hinted APIs, rich docstrings, and strong observability for reproducible runs.

Principles
- Simplicity over inheritance; small classes and functions
- DataFrames first with native metadata in `df.attrs`/`Series.attrs`
- Deterministic behavior and stable signatures for easy tool use
- Observability by default: each step emits artifacts to `output/`
- Modular adoption: use a single component or the whole pipeline

End‑to‑End Flow
1. Load data with provenance (IO)
2. Profile datasets (Profiling)
3. Match schemas (Schema Matching)
4. Translate/align data (Data Translation)
5. Generate candidate pairs (Blocking)
6. Match entities (Entity Matching)
7. Fuse data into a consolidated dataset (Data Fusion)
8. Evaluate and report (Evaluation)

Contents (PyDI Modules)
- [IO](IO) — loaders with provenance and ID injection
- [Profiling](Profiling) — ydata‑profiling and Sweetviz wrappers
- [Schema Matching](SchemaMatching) — label/instance/duplicate‑based matchers + evaluation
- [Data Translation](DataTranslation) — apply schema mappings to align data
- [Information Extraction](InformationExtraction) — regex/code/LLM extractors + evaluation
- [Normalization](Normalization) — text, values, units, types, detectors, validators
- [Blocking](Blocking) — streaming candidate generation (standard/sorted‑neighbourhood/embedding)
- [Entity Matching](EntityMatching) — rule‑based, ML‑based, LLM‑based matchers + evaluation
- [Data Fusion](DataFusion) — conflict resolution, provenance, strategy, reporting, evaluation
- [Utils](Utils) — comparators, similarity registry, helpers
- [Tutorial](Tutorial) — movie data integration walk‑through

Notes
- All components are DataFrame‑first; metadata lives in `DataFrame.attrs`/`Series.attrs`.
- Each step writes artifacts (CSV/JSON/HTML) to `output/` by default.
- See `docs/high_level_design.md` for design goals and APIs.

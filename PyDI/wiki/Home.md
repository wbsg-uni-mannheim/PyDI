Welcome to the PyDI Wiki
========================

PyDI (Python Data Integration) is an end‑to‑end data integration framework for loading, profiling, matching, and fusing heterogeneous datasets. It combines traditional methods (rule‑based similarity, blocking, voting) with modern approaches (machine learning, deep‑learning embeddings, and LLM‑based extraction/matching), making it a practical testbed for evaluation and experimentation. Each component writes human‑readable artifacts and logs to `output/`, so results are understandable, shortcomings are visible, and improvements can be targeted with evidence.

Principles
- Clear, composable components that can be used independently or as a pipeline
- Reproducible runs with provenance and artifact outputs for inspection

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
- [IO](IO) — load data, set IDs, record provenance
- [Profiling](Profiling) — dataset profiles and comparisons (HTML)
- [Information Extraction](InformationExtraction) — regex/code/LLM extraction + evaluation
- [Normalization](Normalization) — clean headers/text, standardize values and units
- [Schema Matching](SchemaMatching) — label/instance/duplicate‑based matching + evaluation
- [Data Translation](DataTranslation) — apply schema mappings to align data
- [Blocking](Blocking) — candidate generation (standard/sorted‑neighbourhood/token/embedding)
- [Entity Matching](EntityMatching) — rule‑based, ML‑based, and LLM‑based matchers + evaluation
- [Data Fusion](DataFusion) — conflict resolution rules and evaluation/reporting
- [Utils](Utils) — comparators and similarity registry
- [Tutorial](Tutorial) — end‑to‑end example (movies)

Reading Guide
The sections below provide a concise, high‑level overview of PyDI’s functionality. For more detailed and interactive exploration, see the example scripts and the tutorial notebook in `PyDI/examples` and `PyDI/tutorial`.

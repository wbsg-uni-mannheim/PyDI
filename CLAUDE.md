# Project Overview

This project contains **PyDI**, a Python‑based data integration framework inspired by the Java WInte.r framework. PyDI is designed to be lightweight, pandas‑first, and agentic. It provides modules for data profiling, schema matching, blocking and entity matching, data translation, information extraction, normalization and validation, and data fusion. It also demonstrates how to write modular, reproducible data integration pipelines without requiring a complex orchestrator.

## Repository structure

* `PyDI/` – The main package containing all library code. The package is subdivided into logical subpackages:
  * `profiling/` – wrappers around ydata‑profiling and Sweetviz.
  * `schemamatching/` – schema matching algorithms and evaluation utilities.
  * `entitymatching/` – blocking and entity matching strategies.
  * `datatranslation/` – data translation components for schema mapping.
  * `informationextraction/` – information extraction components for feature engineering.
  * `normalization/` – normalization and validation components.
  * `fusion/` – data fusion and conflict resolution rules.
  * `utils/` – reusable utilities such as comparators and logging.
  * `examples/` – example scripts to run end‑to‑end pipelines.
  * `tests/` – placeholder for unit tests.
  * `docs/` – documentation, including a copy of the high‑level design document.
* `input/` – Example datasets and test cases from the WInte.r framework for development and testing:
  * `entitymatching/` – Movie data from academy awards and actors datasets with training/test splits for entity matching tasks.
  * `fusion/` – Movie datasets with pre-computed correspondences and gold standard for data fusion evaluation.
  * `schemamatching/` – Source datasets and predefined target schemas for schema matching experiments.
* `output/` – Example outputs and expected results from the WInte.r framework for validation:
  * `entitymatching/` – Correspondence files, debug logs, and evaluation results from entity matching runs.
  * `fusion/` – Fused datasets, consistency reports, and fusion evaluation artifacts.
* `winter/` – A copy of the WInte.r framework's public documentation (README and LICENSE) **plus** the original source code. It is not used directly by PyDI but is provided for reference; you can use it to explore the full WInte.r codebase.
* `docs/high_level_design.md` – A copy of the high‑level design document used to design this project. Consult this document for design details.
* `pyproject.toml` – Python packaging metadata and dependencies.
* `.claude/settings.json` – Optional Claude Code settings for this project.

## Design guidelines

When extending this project, please adhere to the following high‑level design principles (see `docs/high_level_design.md` for details):

1. **Simplicity over inheritance.** Prefer simple functions and small classes over deep class hierarchies. Abstract base classes should be thin and intuitive.
2. **DataFrames first.** All core functions and methods should accept and return `pandas.DataFrame` objects. Avoid custom container types unless necessary.
3. **Native metadata.** Use `DataFrame.attrs` and `Series.attrs` to store dataset‑ and column‑level metadata, including provenance.
4. **Deterministic APIs.** Keep method signatures stable and fully type‑hinted. Write rich docstrings using the NumPy style so that LLMs and future developers can understand them.
5. **Observability by default.** Every step should log its activity to the console and write artifacts (CSV/JSON/HTML/XML) to disk. Return file paths for downstream consumers.
6. **Configurable and reproducible.** Provide configuration via JSON/YAML manifests where appropriate. Always allow runs to be reproduced using the same inputs and parameters.
7. **Modular adoption.** Each component should be callable directly from a Python script. Do not require an orchestrator for basic use cases.

## Workflow

A typical pipeline looks like this:

```python
import pandas as pd
from PyDI.profiling import DataProfiler
from PyDI.schema import SimpleSchemaMatcher
from PyDI.matching import SortedNeighbourhood, RuleBasedMatcher
from PyDI.fusion import DataFuser, FusionRule

# load data
df_a = pd.read_csv("input/a.csv")
df_a.attrs["dataset_name"] = "a"
df_b = pd.read_csv("input/b.csv")
df_b.attrs["dataset_name"] = "b"

# profile
DataProfiler().profile(df_a, "runs/default/profiling")
DataProfiler().profile(df_b, "runs/default/profiling")

# schema matching
matcher = SimpleSchemaMatcher()
corr = matcher.match([df_a, df_b], method="label", preprocess=str.lower, threshold=0.8)

# align schemas (placeholder)
# ...

# blocking and matching
blocker = SortedNeighbourhood(df_a, df_b, key="title", window=5)
candidates = list(blocker)
matcher = RuleBasedMatcher()
matches = matcher.match(df_a, df_b, candidates, comparators=[], threshold=0.7)

# fusion (placeholder)
# ...
```

This example is intentionally high‑level; see `docs/high_level_design.md` for the full design.

## Code style and tooling

* Use **PEP 8** style; we recommend formatting code with `black`.
* Write **NumPy‑style docstrings** for all public functions and classes.
* Keep functions short and focused.
* All new code should be accompanied by unit tests placed in `tests/`.
* Use `pytest` for testing.

## Development environment

To set up the environment (use a virtualenv called PyDI):

```bash
# install dependencies
python -m pip install -e .[dev]
```

This project uses a `pyproject.toml` file to manage dependencies. The `[project.dependencies]` section lists runtime dependencies; `[project.optional-dependencies.dev]` lists tools for development such as `pytest` and `black`.
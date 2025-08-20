# PyDI – Python Data Integration Framework

PyDI is a lightweight, extensible framework for end‑to‑end data integration in Python. It is inspired by the WInte.r Java framework and is designed with pandas‑first APIs, simple abstractions, and strong support for reproducible data integration workflows.

## Features

* Data profiling via ydata‑profiling and Sweetviz.
* Schema matching algorithms with evaluation tools.
* Blocking and entity matching strategies.
* Data translation, information extraction, normalization and validation components.
* Conflict resolution and data fusion engine.
* Designed for use by students, researchers, and LLM agents.

## Installation

```bash
git clone <this repository>
cd pydi_project
python -m pip install -e .
```

For optional development tools:

```bash
python -m pip install -e .[dev]
```

## Usage

See the high‑level design document in `docs/high_level_design.md` for an overview of the expected API. The modules in `PyDI` contain stub implementations that you can extend. Example notebooks and scripts will be added to the `examples/` directory over time.
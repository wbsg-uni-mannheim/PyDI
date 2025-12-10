# Tests

This directory contains unit and integration tests for PyDI’s core functionality, built around the example “usecases” datasets (movies, music, games, companies) and some smaller synthetic inputs in `testdata`.

## What the tests cover

- `movies_test/`
  - `test_low_level.py`: low‑level matching and fusion building blocks on small movie examples.
  - `test_low_level_matching.py`: low‑level entity matching on small movie examples test sbubsets (comparators, blocking, thresholds, score normalization).
  - `test_low_level_fusion.py`: low‑level data fusion on small movie examples test sbubsets (fusion strategies, conflict resolution, evaluation).
  - `test_blocker.py`: behaviour of different blocking strategies on the movie data.
  - `test_workflow_movies.py`: end‑to‑end workflow on the movies use case (blocking → matching → fusion).
- `games_test/`, `music_test/`, `companies_test/`
  - Similar workflows as `movies_test/`, but for the games, music, and companies use cases (I/O, blocking, matching, fusion, evaluation).
- `normalization_test/`
  - Tests for normalization utilities (types, transforms, validators, etc.).
- `testdata/`
  - Small XML/CSV inputs, correspondences, and test sets used by the fixtures in `conftest.py`.
- `conftest.py`
  - Shared pytest fixtures to load XML/CSV inputs from `usecases/input` and `tests/testdata`, plus helpers for correspondences and fusion gold standards.

## How to run the tests

- **Run all tests:** `pytest`
- **Run only the tests in this directory:** `pytest tests`
- **Run tests for a specific use case:**
  - Movies: `pytest tests/movies_test`
  - Games: `pytest tests/games_test`
  - Music: `pytest tests/music_test`
  - Companies: `pytest tests/companies_test`
- **Run a single test file:**
  - `pytest tests/movies_test/test_low_level_matching.py`
- **Run a single test function within a file:**
  - `pytest tests/movies_test/test_low_level_matching.py::test_matcher_title_only`
  - `pytest tests/normalization_test/test_normalization.py::test_normalize_email`

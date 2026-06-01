"""Shared fixtures for usecases_synthetic tests."""

from __future__ import annotations

# faiss-cpu's libomp collides with torch's libomp on macOS arm64
# (Darwin 25.x): the faiss search loop crashes with
# ``OMP: Error #179: pthread_mutex_init failed`` once any prior import
# has initialised an OpenMP thread pool. Forcing single-threaded
# operation skips the pool init entirely. Set before any
# numpy/pandas/torch/faiss import so the env is in place at runtime.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root (parent of usecases_synthetic/)."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Return a temp directory for provenance / output files."""
    out = tmp_path / "output" / "provenance"
    out.mkdir(parents=True)
    return out


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a deterministic RNG for tests."""
    return np.random.default_rng(12345)


@pytest.fixture
def companies_sources() -> dict[str, pd.DataFrame]:
    """Return 3 small synthetic DataFrames mimicking companies schema (~20 rows)."""
    rng = np.random.default_rng(99)
    n = 20

    names = [f"Company_{i}" for i in range(n)]
    countries = rng.choice(
        ["United States", "Germany", "Japan", "China", "Brazil"], size=n
    ).tolist()
    cities = rng.choice(
        ["New York", "Berlin", "Tokyo", "Beijing", "Sao Paulo"], size=n
    ).tolist()

    dbpedia = pd.DataFrame(
        {
            "id": [f"http://dbpedia.org/resource/{name}" for name in names],
            "name": names,
            "country": countries,
            "city": cities,
            "revenue": (rng.random(n) * 1e10).tolist(),
            "founded": [f"{1950 + i}" for i in range(n)],
        }
    )
    dbpedia.attrs["dataset_name"] = "dbpedia"

    forbes = pd.DataFrame(
        {
            "id": [
                f"http://www.forbes.com/companies/{name.lower().replace('_', '-')}/"
                for name in names
            ],
            "name": [n.replace("_", " ") for n in names],
            "country": countries,
            "assets": (rng.random(n) * 1e11).tolist(),
            "revenue": (rng.random(n) * 1e10).tolist(),
        }
    )
    forbes.attrs["dataset_name"] = "forbes"

    fullcontact = pd.DataFrame(
        {
            "id": [f"fullcontact_{i}" for i in range(n)],
            "name": names[:n],
            "country": countries,
            "city": cities,
        }
    )
    fullcontact.attrs["dataset_name"] = "fullcontact"

    return {"dbpedia": dbpedia, "forbes": forbes, "fullcontact": fullcontact}


@pytest.fixture
def mock_protection_set() -> set[str]:
    """Return a small expanded_positives set with known IDs."""
    return {
        "http://dbpedia.org/resource/Company_0",
        "http://dbpedia.org/resource/Company_1",
        "http://www.forbes.com/companies/company-0/",
        "http://www.forbes.com/companies/company-1/",
        "fullcontact_0",
        "fullcontact_1",
    }

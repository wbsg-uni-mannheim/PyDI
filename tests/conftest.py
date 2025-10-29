# tests/conftest.py
import pytest
from pathlib import Path

from PyDI.io import load_xml

@pytest.fixture
def input_dir():
    return Path(__file__).resolve().parents[1] / "usecases" / "input"

@pytest.fixture
def return_input_data(input_dir):
    def _loader(usecase, dataset: str):
        return load_xml(
            input_dir / usecase / "data" / f"{dataset}.xml",
            name=dataset,
            nested_handling="aggregate",
        )
    return _loader


@pytest.fixture
def games_input_dir():
    return Path(__file__).resolve().parents[1] / "usecases" / "input" / "games" / "data"

@pytest.fixture
def music_input_dir():
    return Path(__file__).resolve().parents[1] / "usecases" / "input" / "music" / "data"

@pytest.fixture
def companies_input_dir():
    return Path(__file__).resolve().parents[1] / "usecases" / "input" / "companies" / "data"



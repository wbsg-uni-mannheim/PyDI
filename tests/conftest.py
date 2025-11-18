# tests/conftest.py
import pytest
from pathlib import Path

from PyDI.io import load_xml, load_csv

@pytest.fixture
def repo_root():
    current = Path.cwd()
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    return Path.cwd()

@pytest.fixture
def input_dir(repo_root):
    return repo_root / "usecases" / "input"

@pytest.fixture
def get_input_data(input_dir):
    def _loader(usecase, dataset: str):
        return load_xml(
            input_dir / usecase / "data" / f"{dataset}.xml",
            name=dataset,
            nested_handling="aggregate",
        )
    return _loader

@pytest.fixture
def get_correspondences(input_dir):
    def _loader(usecase, dataset_1: str, dataset_2: str):
        return load_csv(
            input_dir / usecase / "entitymatching" / f"{dataset_1}_2_{dataset_2}_test.csv",
            name=f"test_set", header=None, names=['id1', 'id2', 'label'], add_index=False
        )
    return _loader

@pytest.fixture
def get_fusion_test_set(input_dir):
    def _loader(usecase):
        return load_xml(
            input_dir / usecase / "fusion" / f"test_set.xml",
            name='fusion_test_set',
            nested_handling="aggregate",
        )
    return _loader

@pytest.fixture
def get_test_input():
    def _loader(usecase, dataset: str):
        return load_xml(
            Path(__file__).resolve().parents[0] / "testdata" / usecase / "input" / f"{dataset}.xml",
            name=dataset,
            nested_handling="aggregate",
        )
    return _loader

@pytest.fixture
def get_test_correspondences():
    def _loader(usecase, dataset_1: str, dataset_2: str):
        return load_csv(
            Path(__file__).resolve().parents[0] / "testdata" / usecase / "correspondences" / f"{dataset_1}_2_{dataset_2}.csv",
            name=f"{dataset_1}_2_{dataset_2}", header=None, names=['id1', 'id2', 'score'], add_index=False
        )
    return _loader

@pytest.fixture
def get_test_fusion_goldstandard():
    def _loader(usecase):
        return load_xml(
            Path(__file__).resolve().parents[0] / "testdata" / usecase / "goldstandard" / f"fused.xml",
            name='fusion_goldstandard',
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

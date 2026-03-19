"""Shared fixtures for backend tests."""

import pytest
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def project_name():
    return "doped-tio2"


@pytest.fixture
def sample_xyz_path():
    return DATA_DIR / "geo" / "pristine-TiO2.xyz"


@pytest.fixture
def sample_h2_xyz_path():
    return DATA_DIR / "geo" / "pristine-TiO2-H2.xyz"

"""Shared fixtures for the Methane Leak Opportunistic Tasking test suite."""

import sys
import os
import pytest
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def small_grid():
    """A small 200m x 200m grid at 10m resolution for fast tests."""
    half = 100.0
    coords = np.arange(-half, half + 10, 10)
    X, Y = np.meshgrid(coords, coords)
    return X, Y


@pytest.fixture
def single_source():
    """A single ground-level leak source at the origin."""
    return {
        "name": "Test Source",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "emission_rate": 0.5,
    }


@pytest.fixture
def default_wind():
    """Default wind parameters: moderate west wind, neutral stability."""
    return {
        "wind_speed": 3.0,
        "wind_direction_deg": 270.0,
        "stability_class": "D",
    }


@pytest.fixture
def simple_path():
    """A simple straight-line baseline path along the x-axis."""
    return np.array([
        [-100.0, 0.0],
        [-50.0, 0.0],
        [0.0, 0.0],
        [50.0, 0.0],
        [100.0, 0.0],
    ])


@pytest.fixture
def mock_sources():
    """The full set of 5 mock leak sources."""
    from data.mock_data import get_leak_sources
    return get_leak_sources()


@pytest.fixture
def mock_baseline_path():
    """The full mock baseline path."""
    from data.mock_data import get_baseline_path
    return get_baseline_path()

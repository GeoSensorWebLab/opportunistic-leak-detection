"""Tests for state serialization / deserialization."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.state_io import serialize_state, deserialize_state
from models.measurement import Measurement


@pytest.fixture
def sample_grid():
    coords = np.arange(-50, 51, 10.0)
    return np.meshgrid(coords, coords)


@pytest.fixture
def sample_belief(sample_grid):
    X, Y = sample_grid
    return np.random.default_rng(42).random(X.shape)


@pytest.fixture
def sample_measurements():
    return [
        Measurement(x=10.0, y=20.0, concentration_ppm=5.0,
                    detected=True, wind_speed=3.0,
                    wind_direction_deg=270.0, stability_class="D"),
        Measurement(x=-30.0, y=0.0, concentration_ppm=0.5,
                    detected=False, wind_speed=3.0,
                    wind_direction_deg=270.0, stability_class="D"),
    ]


class TestRoundtrip:
    def test_basic_roundtrip(self, sample_grid, sample_belief):
        X, Y = sample_grid
        data = serialize_state(sample_belief, X, Y)
        assert isinstance(data, bytes)
        assert len(data) > 0

        result = deserialize_state(data)
        np.testing.assert_array_almost_equal(result["belief"], sample_belief)
        np.testing.assert_array_almost_equal(result["grid_x"], X)
        np.testing.assert_array_almost_equal(result["grid_y"], Y)

    def test_with_entropy_history(self, sample_grid, sample_belief):
        X, Y = sample_grid
        hist = [100.0, 95.0, 88.0, 80.0]
        data = serialize_state(sample_belief, X, Y, entropy_history=hist)
        result = deserialize_state(data)
        assert result["entropy_history"] == pytest.approx(hist)

    def test_with_measurements(self, sample_grid, sample_belief, sample_measurements):
        X, Y = sample_grid
        data = serialize_state(
            sample_belief, X, Y, measurements=sample_measurements,
        )
        result = deserialize_state(data)
        assert len(result["measurements"]) == 2
        m0 = result["measurements"][0]
        assert m0.x == pytest.approx(10.0)
        assert m0.detected is True

    def test_with_metadata(self, sample_grid, sample_belief):
        X, Y = sample_grid
        meta = {"scoring_mode": "EER", "version": "1.0"}
        data = serialize_state(sample_belief, X, Y, metadata=meta)
        result = deserialize_state(data)
        assert result["metadata"]["scoring_mode"] == "EER"

    def test_full_roundtrip(self, sample_grid, sample_belief, sample_measurements):
        X, Y = sample_grid
        hist = [100.0, 90.0]
        meta = {"test": True}
        data = serialize_state(
            sample_belief, X, Y,
            measurements=sample_measurements,
            entropy_history=hist,
            metadata=meta,
        )
        result = deserialize_state(data)

        np.testing.assert_array_almost_equal(result["belief"], sample_belief)
        assert len(result["measurements"]) == 2
        assert result["entropy_history"] == pytest.approx(hist)
        assert result["metadata"]["test"] is True

    def test_empty_optional_fields(self, sample_grid, sample_belief):
        X, Y = sample_grid
        data = serialize_state(sample_belief, X, Y)
        result = deserialize_state(data)
        assert result["measurements"] == []
        assert result["entropy_history"] == []
        assert result["metadata"] == {}

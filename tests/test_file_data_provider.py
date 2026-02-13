"""Tests for the FileDataProvider class."""

import sys
import os
import json
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.interfaces import DataProvider, FileDataProvider


# ---------------------------------------------------------------------------
# Fixtures — paths to the bundled sample data files
# ---------------------------------------------------------------------------

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


@pytest.fixture
def sample_provider():
    """A FileDataProvider loaded from the bundled sample files."""
    return FileDataProvider(
        sources_path=os.path.join(SAMPLES_DIR, "sources.json"),
        path_path=os.path.join(SAMPLES_DIR, "path.csv"),
        wind_scenarios_path=os.path.join(SAMPLES_DIR, "wind_scenarios.json"),
        wind_distribution_path=os.path.join(SAMPLES_DIR, "wind_distribution.json"),
    )


# ---------------------------------------------------------------------------
# ABC conformance
# ---------------------------------------------------------------------------

class TestABCConformance:
    def test_is_data_provider(self, sample_provider):
        assert isinstance(sample_provider, DataProvider)


# ---------------------------------------------------------------------------
# Return types and shapes
# ---------------------------------------------------------------------------

class TestReturnTypes:
    def test_get_leak_sources_returns_list_of_dicts(self, sample_provider):
        sources = sample_provider.get_leak_sources()
        assert isinstance(sources, list)
        assert len(sources) == 5
        for s in sources:
            assert isinstance(s, dict)
            assert "name" in s
            assert "x" in s
            assert "emission_rate" in s

    def test_get_baseline_path_returns_ndarray(self, sample_provider):
        path = sample_provider.get_baseline_path()
        assert isinstance(path, np.ndarray)
        assert path.ndim == 2
        assert path.shape[1] == 2
        assert path.shape[0] >= 2

    def test_get_wind_scenarios_returns_list(self, sample_provider):
        scenarios = sample_provider.get_wind_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 1
        for sc in scenarios:
            assert "name" in sc
            assert "speed" in sc

    def test_get_wind_distribution_returns_list(self, sample_provider):
        dist = sample_provider.get_wind_distribution()
        assert isinstance(dist, list)
        assert len(dist) >= 1
        for d in dist:
            assert "direction" in d
            assert "weight" in d

    def test_wind_distribution_weights_sum_to_one(self, sample_provider):
        dist = sample_provider.get_wind_distribution()
        total = sum(d["weight"] for d in dist)
        assert abs(total - 1.0) < 1e-6

    def test_get_wind_fan_returns_list(self, sample_provider):
        fan = sample_provider.get_wind_fan(center_direction=270.0)
        assert isinstance(fan, list)
        assert len(fan) >= 1


# ---------------------------------------------------------------------------
# Immutability contract — returns copies
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_sources_returns_copy(self, sample_provider):
        s1 = sample_provider.get_leak_sources()
        s1[0]["name"] = "MUTATED"
        s2 = sample_provider.get_leak_sources()
        assert s2[0]["name"] != "MUTATED"

    def test_path_returns_copy(self, sample_provider):
        p1 = sample_provider.get_baseline_path()
        p1[0, 0] = 999999.0
        p2 = sample_provider.get_baseline_path()
        assert p2[0, 0] != 999999.0

    def test_wind_scenarios_returns_copy(self, sample_provider):
        w1 = sample_provider.get_wind_scenarios()
        w1[0]["name"] = "MUTATED"
        w2 = sample_provider.get_wind_scenarios()
        assert w2[0]["name"] != "MUTATED"

    def test_wind_distribution_returns_copy(self, sample_provider):
        d1 = sample_provider.get_wind_distribution()
        d1[0]["weight"] = 999.0
        d2 = sample_provider.get_wind_distribution()
        assert d2[0]["weight"] != 999.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_missing_source_key_raises(self, tmp_path):
        bad_sources = [{"name": "Test", "x": 0.0}]  # missing y, z, emission_rate
        src_file = tmp_path / "bad_sources.json"
        src_file.write_text(json.dumps(bad_sources))
        with pytest.raises(ValueError, match="missing required keys"):
            FileDataProvider(
                sources_path=str(src_file),
                path_path=os.path.join(SAMPLES_DIR, "path.csv"),
                wind_scenarios_path=os.path.join(SAMPLES_DIR, "wind_scenarios.json"),
            )

    def test_empty_path_csv_raises(self, tmp_path):
        csv_file = tmp_path / "empty_path.csv"
        csv_file.write_text("x,y\n1.0,2.0\n")  # only 1 row
        with pytest.raises(ValueError, match="at least 2"):
            FileDataProvider(
                sources_path=os.path.join(SAMPLES_DIR, "sources.json"),
                path_path=str(csv_file),
                wind_scenarios_path=os.path.join(SAMPLES_DIR, "wind_scenarios.json"),
            )

    def test_invalid_weight_sum_raises(self, tmp_path):
        bad_dist = [
            {"direction": 0, "speed": 3.0, "stability_class": "D", "weight": 0.5},
            {"direction": 180, "speed": 3.0, "stability_class": "D", "weight": 0.3},
        ]
        dist_file = tmp_path / "bad_dist.json"
        dist_file.write_text(json.dumps(bad_dist))
        with pytest.raises(ValueError, match="sum to 1.0"):
            FileDataProvider(
                sources_path=os.path.join(SAMPLES_DIR, "sources.json"),
                path_path=os.path.join(SAMPLES_DIR, "path.csv"),
                wind_scenarios_path=os.path.join(SAMPLES_DIR, "wind_scenarios.json"),
                wind_distribution_path=str(dist_file),
            )

    def test_empty_sources_array_raises(self, tmp_path):
        src_file = tmp_path / "empty.json"
        src_file.write_text("[]")
        with pytest.raises(ValueError, match="non-empty"):
            FileDataProvider(
                sources_path=str(src_file),
                path_path=os.path.join(SAMPLES_DIR, "path.csv"),
                wind_scenarios_path=os.path.join(SAMPLES_DIR, "wind_scenarios.json"),
            )

    def test_no_wind_distribution_uses_fallback(self):
        """Without a wind_distribution_path, fallback to mock data 8-direction rose."""
        provider = FileDataProvider(
            sources_path=os.path.join(SAMPLES_DIR, "sources.json"),
            path_path=os.path.join(SAMPLES_DIR, "path.csv"),
            wind_scenarios_path=os.path.join(SAMPLES_DIR, "wind_scenarios.json"),
        )
        dist = provider.get_wind_distribution()
        assert len(dist) == 8
        total = sum(d["weight"] for d in dist)
        assert abs(total - 1.0) < 1e-6

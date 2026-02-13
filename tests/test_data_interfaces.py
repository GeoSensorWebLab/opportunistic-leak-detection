"""Tests for the abstract data layer."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.interfaces import DataProvider, MockDataProvider


class TestMockDataProvider:
    def test_implements_interface(self):
        provider = MockDataProvider()
        assert isinstance(provider, DataProvider)

    def test_get_leak_sources(self):
        provider = MockDataProvider()
        sources = provider.get_leak_sources()
        assert isinstance(sources, list)
        assert len(sources) > 0
        for s in sources:
            assert "name" in s
            assert "x" in s
            assert "y" in s
            assert "z" in s
            assert "emission_rate" in s

    def test_get_baseline_path(self):
        provider = MockDataProvider()
        path = provider.get_baseline_path()
        assert isinstance(path, np.ndarray)
        assert path.ndim == 2
        assert path.shape[1] == 2
        assert len(path) >= 2

    def test_get_wind_scenarios(self):
        provider = MockDataProvider()
        scenarios = provider.get_wind_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        for s in scenarios:
            assert "name" in s
            assert "speed" in s
            assert "direction" in s
            assert "stability_class" in s

    def test_get_wind_distribution(self):
        provider = MockDataProvider()
        dist = provider.get_wind_distribution()
        assert isinstance(dist, list)
        assert len(dist) > 0
        for d in dist:
            assert "direction" in d
            assert "speed" in d
            assert "stability_class" in d
            assert "weight" in d
        total_weight = sum(d["weight"] for d in dist)
        assert abs(total_weight - 1.0) < 1e-6

    def test_get_wind_fan(self):
        provider = MockDataProvider()
        fan = provider.get_wind_fan(center_direction=270.0, spread_deg=30.0, num_scenarios=5)
        assert isinstance(fan, list)
        assert len(fan) == 5
        for d in fan:
            assert "direction" in d
            assert "speed" in d
            assert "stability_class" in d
            assert "weight" in d
        total_weight = sum(d["weight"] for d in fan)
        assert abs(total_weight - 1.0) < 1e-6

    def test_consistency_with_mock_data(self):
        """Ensure MockDataProvider returns the same data as raw mock_data functions."""
        from data.mock_data import (
            get_leak_sources, get_baseline_path, get_wind_scenarios,
            get_wind_distribution, get_wind_fan,
        )

        provider = MockDataProvider()
        assert provider.get_leak_sources() == get_leak_sources()
        np.testing.assert_array_equal(provider.get_baseline_path(), get_baseline_path())
        assert provider.get_wind_scenarios() == get_wind_scenarios()
        assert provider.get_wind_distribution() == get_wind_distribution()
        assert provider.get_wind_fan(270.0) == get_wind_fan(270.0)

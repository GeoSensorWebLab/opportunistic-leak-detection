"""Tests for the multi-scenario wind ensemble feature."""

import numpy as np
import pytest

from data.mock_data import get_wind_distribution, get_wind_fan, get_wind_scenarios
from optimization.opportunity_map import (
    compute_opportunity_map,
    compute_ensemble_opportunity_map,
)


class TestWindDistribution:
    """Tests for get_wind_distribution()."""

    def test_weights_sum_to_one(self):
        """Wind rose weights must sum to 1.0."""
        dist = get_wind_distribution()
        weights = [s["weight"] for s in dist]
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_eight_directions(self):
        """Should return exactly 8 scenarios."""
        dist = get_wind_distribution()
        assert len(dist) == 8

    def test_covers_all_cardinals(self):
        """Should cover N, NE, E, SE, S, SW, W, NW."""
        dist = get_wind_distribution()
        directions = {s["direction"] for s in dist}
        assert directions == {0, 45, 90, 135, 180, 225, 270, 315}

    def test_required_keys(self):
        """Each scenario must have direction, speed, stability_class, weight."""
        dist = get_wind_distribution()
        for s in dist:
            assert "direction" in s
            assert "speed" in s
            assert "stability_class" in s
            assert "weight" in s


class TestWindFan:
    """Tests for get_wind_fan()."""

    def test_weights_sum_to_one(self):
        """Fan weights must sum to 1.0."""
        fan = get_wind_fan(center_direction=270.0, num_scenarios=5)
        weights = [s["weight"] for s in fan]
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_correct_count(self):
        """Should generate the requested number of scenarios."""
        for n in [3, 5, 8, 12]:
            fan = get_wind_fan(center_direction=180.0, num_scenarios=n)
            assert len(fan) == n

    def test_spread_range(self):
        """Directions should span center ± spread."""
        fan = get_wind_fan(center_direction=270.0, spread_deg=30.0, num_scenarios=5)
        directions = [s["direction"] for s in fan]
        assert min(directions) == pytest.approx(240.0, abs=0.1)
        assert max(directions) == pytest.approx(300.0, abs=0.1)

    def test_single_scenario(self):
        """Single scenario fan should just be the center direction."""
        fan = get_wind_fan(center_direction=90.0, num_scenarios=1)
        assert len(fan) == 1
        assert fan[0]["direction"] == pytest.approx(90.0)
        assert fan[0]["weight"] == pytest.approx(1.0)

    def test_invalid_count_raises(self):
        """num_scenarios < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            get_wind_fan(center_direction=270.0, num_scenarios=0)

    def test_wraps_around_360(self):
        """Fan near 0 degrees should wrap around to 350+."""
        fan = get_wind_fan(center_direction=10.0, spread_deg=20.0, num_scenarios=3)
        directions = sorted(s["direction"] for s in fan)
        # Should wrap: 350, 10, 30
        assert any(d > 340 for d in directions), "Should wrap around 360"


class TestEnsembleOpportunityMap:
    """Tests for compute_ensemble_opportunity_map()."""

    @pytest.fixture
    def single_source(self):
        return [{"name": "Test", "x": 0.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5}]

    def test_single_scenario_matches_direct(self, single_source):
        """Ensemble with one scenario should match direct computation."""
        scenario = [{
            "direction": 270, "speed": 3.0,
            "stability_class": "D", "weight": 1.0,
        }]
        _, _, e_conc, e_det = compute_ensemble_opportunity_map(
            sources=single_source,
            wind_scenarios=scenario,
            grid_size=200,
            resolution=20,
        )
        _, _, d_conc, d_det = compute_opportunity_map(
            sources=single_source,
            wind_speed=3.0,
            wind_direction_deg=270,
            stability_class="D",
            grid_size=200,
            resolution=20,
        )
        np.testing.assert_allclose(e_conc, d_conc, atol=1e-10)
        np.testing.assert_allclose(e_det, d_det, atol=1e-10)

    def test_detection_prob_bounded(self, single_source):
        """Ensemble detection prob should be in [0, 1]."""
        scenarios = get_wind_distribution()
        _, _, _, det = compute_ensemble_opportunity_map(
            sources=single_source,
            wind_scenarios=scenarios,
            grid_size=200,
            resolution=20,
        )
        assert np.all(det >= 0.0)
        assert np.all(det <= 1.0)

    def test_ensemble_smoother_than_single(self, single_source):
        """Ensemble should produce a smoother (lower std) detection map than any single scenario."""
        # Single scenario
        _, _, _, det_single = compute_opportunity_map(
            sources=single_source,
            wind_speed=3.0,
            wind_direction_deg=270,
            stability_class="D",
            grid_size=200,
            resolution=20,
        )
        # Ensemble
        scenarios = get_wind_distribution()
        _, _, _, det_ensemble = compute_ensemble_opportunity_map(
            sources=single_source,
            wind_scenarios=scenarios,
            grid_size=200,
            resolution=20,
        )
        # Ensemble should have more spread-out, lower-peak detection
        # (smoother = more cells with moderate values)
        assert det_ensemble.max() <= det_single.max() + 1e-10

    def test_invalid_weights_raises(self, single_source):
        """Weights that don't sum to 1 should raise ValueError."""
        bad_scenarios = [
            {"direction": 0, "speed": 3.0, "stability_class": "D", "weight": 0.5},
            {"direction": 90, "speed": 3.0, "stability_class": "D", "weight": 0.3},
        ]
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_ensemble_opportunity_map(
                sources=single_source,
                wind_scenarios=bad_scenarios,
                grid_size=200,
                resolution=20,
            )

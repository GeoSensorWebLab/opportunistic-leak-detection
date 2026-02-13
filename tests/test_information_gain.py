"""Tests for the information-theoretic scoring (EER) module."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimization.information_gain import (
    compute_cell_entropy,
    compute_total_entropy,
    compute_information_value_grid,
    compute_information_scores,
    compute_ensemble_information_scores,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_grid():
    """A tiny 100m x 100m grid at 10m resolution for fast tests."""
    half = 50.0
    coords = np.arange(-half, half + 10, 10)
    X, Y = np.meshgrid(coords, coords)
    return X, Y


@pytest.fixture
def uniform_belief(tiny_grid):
    """Uniform belief at 0.1 (uncertain but not maximal)."""
    X, _ = tiny_grid
    return np.full_like(X, 0.1)


@pytest.fixture
def mixed_belief(tiny_grid):
    """Belief map with a gradient from 0.05 to 0.5."""
    X, _ = tiny_grid
    rows = X.shape[0]
    gradient = np.linspace(0.05, 0.5, rows)
    return np.tile(gradient[:, None], (1, X.shape[1]))


@pytest.fixture
def default_wind():
    """Default wind parameters: west wind, neutral stability."""
    return {
        "wind_speed": 3.0,
        "wind_direction_deg": 270.0,
        "stability_class": "D",
    }


@pytest.fixture
def simple_deviation(tiny_grid):
    """Deviation grid: distance from center row (y=0)."""
    _, Y = tiny_grid
    return np.abs(Y)


# ---------------------------------------------------------------------------
# Cell Entropy Tests
# ---------------------------------------------------------------------------

class TestCellEntropy:
    """Tests for compute_cell_entropy."""

    def test_maximum_at_half(self):
        """Entropy should be maximal (1.0 bit) at p=0.5."""
        h = compute_cell_entropy(np.array([0.5]))
        assert np.isclose(h[0], 1.0, atol=1e-10)

    def test_near_zero_at_extremes(self):
        """Entropy should be near zero for p close to 0 or 1."""
        h = compute_cell_entropy(np.array([0.001, 0.999]))
        assert h[0] < 0.02
        assert h[1] < 0.02

    def test_symmetric(self):
        """H(p) should equal H(1-p)."""
        p_vals = np.array([0.1, 0.2, 0.3, 0.4])
        h_p = compute_cell_entropy(p_vals)
        h_1_minus_p = compute_cell_entropy(1.0 - p_vals)
        np.testing.assert_allclose(h_p, h_1_minus_p, atol=1e-12)

    def test_non_negative(self):
        """Entropy must be non-negative for all valid probabilities."""
        p_vals = np.linspace(0.001, 0.999, 100)
        h = compute_cell_entropy(p_vals)
        assert np.all(h >= 0)

    def test_output_shape(self):
        """Output shape should match input shape."""
        belief = np.random.uniform(0.01, 0.99, size=(5, 7))
        h = compute_cell_entropy(belief)
        assert h.shape == (5, 7)


# ---------------------------------------------------------------------------
# Total Entropy Tests
# ---------------------------------------------------------------------------

class TestTotalEntropy:
    """Tests for compute_total_entropy."""

    def test_uniform_belief(self, tiny_grid, uniform_belief):
        """Total entropy of uniform belief should equal N * H(p)."""
        h_total = compute_total_entropy(uniform_belief)
        n_cells = uniform_belief.size
        h_single = float(compute_cell_entropy(np.array([0.1]))[0])
        assert np.isclose(h_total, n_cells * h_single, rtol=1e-10)

    def test_certain_belief_zero_entropy(self, tiny_grid):
        """All-zero or all-one belief should have near-zero entropy."""
        X, _ = tiny_grid
        h_zero = compute_total_entropy(np.zeros_like(X))
        h_one = compute_total_entropy(np.ones_like(X))
        # Will be very small but not exactly zero due to clipping
        assert h_zero < 0.01
        assert h_one < 0.01

    def test_positive(self, uniform_belief):
        """Total entropy should be positive for non-trivial belief."""
        assert compute_total_entropy(uniform_belief) > 0


# ---------------------------------------------------------------------------
# Information Value Grid Tests
# ---------------------------------------------------------------------------

class TestInformationValueGrid:
    """Tests for compute_information_value_grid."""

    def test_output_shape(self, tiny_grid, uniform_belief, default_wind):
        """EER grid should match input grid shape."""
        X, Y = tiny_grid
        eer = compute_information_value_grid(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            wind_speed=default_wind["wind_speed"],
            wind_direction_deg=default_wind["wind_direction_deg"],
            stability_class=default_wind["stability_class"],
            avg_emission=0.5,
            subsample=2,
        )
        assert eer.shape == X.shape

    def test_non_negative(self, tiny_grid, uniform_belief, default_wind):
        """EER values should be non-negative."""
        X, Y = tiny_grid
        eer = compute_information_value_grid(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            wind_speed=default_wind["wind_speed"],
            wind_direction_deg=default_wind["wind_direction_deg"],
            stability_class=default_wind["stability_class"],
            avg_emission=0.5,
            subsample=2,
        )
        assert np.all(eer >= -1e-10)

    def test_zero_for_certain_belief(self, tiny_grid, default_wind):
        """EER should be near zero when belief is already certain."""
        X, Y = tiny_grid
        certain = np.full_like(X, 1e-10)  # nearly certain no leak
        eer = compute_information_value_grid(
            grid_x=X, grid_y=Y, belief=certain,
            wind_speed=default_wind["wind_speed"],
            wind_direction_deg=default_wind["wind_direction_deg"],
            stability_class=default_wind["stability_class"],
            avg_emission=0.5,
            subsample=2,
        )
        assert np.max(eer) < 0.01

    def test_masking_with_deviation(
        self, tiny_grid, uniform_belief, default_wind, simple_deviation,
    ):
        """Cells beyond max_deviation should have EER = 0."""
        X, Y = tiny_grid
        max_dev = 25.0
        eer = compute_information_value_grid(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            wind_speed=default_wind["wind_speed"],
            wind_direction_deg=default_wind["wind_direction_deg"],
            stability_class=default_wind["stability_class"],
            avg_emission=0.5,
            deviation=simple_deviation,
            max_deviation=max_dev,
            subsample=1,
        )
        beyond = simple_deviation > max_dev
        assert np.all(eer[beyond] == 0.0)

    def test_subsample_produces_nonzero(
        self, tiny_grid, uniform_belief, default_wind,
    ):
        """Subsampled computation should still produce non-zero EER."""
        X, Y = tiny_grid
        eer = compute_information_value_grid(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            wind_speed=default_wind["wind_speed"],
            wind_direction_deg=default_wind["wind_direction_deg"],
            stability_class=default_wind["stability_class"],
            avg_emission=0.5,
            subsample=3,
        )
        assert np.max(eer) > 0


# ---------------------------------------------------------------------------
# Information Scores Tests
# ---------------------------------------------------------------------------

class TestInformationScores:
    """Tests for compute_information_scores."""

    def test_output_shape(
        self, tiny_grid, uniform_belief, default_wind, simple_deviation,
    ):
        """Score grid should match input shape."""
        X, Y = tiny_grid
        scores = compute_information_scores(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            deviation=simple_deviation, max_deviation=40.0,
            wind_speed=default_wind["wind_speed"],
            wind_direction_deg=default_wind["wind_direction_deg"],
            stability_class=default_wind["stability_class"],
            avg_emission=0.5,
            subsample=2,
        )
        assert scores.shape == X.shape

    def test_non_negative(
        self, tiny_grid, uniform_belief, default_wind, simple_deviation,
    ):
        """Scores should be non-negative."""
        X, Y = tiny_grid
        scores = compute_information_scores(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            deviation=simple_deviation, max_deviation=40.0,
            wind_speed=default_wind["wind_speed"],
            wind_direction_deg=default_wind["wind_direction_deg"],
            stability_class=default_wind["stability_class"],
            avg_emission=0.5,
            subsample=2,
        )
        assert np.all(scores >= -1e-10)

    def test_beyond_max_deviation_zero(
        self, tiny_grid, uniform_belief, default_wind, simple_deviation,
    ):
        """Cells beyond max_deviation must score 0."""
        X, Y = tiny_grid
        max_dev = 20.0
        scores = compute_information_scores(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            deviation=simple_deviation, max_deviation=max_dev,
            wind_speed=default_wind["wind_speed"],
            wind_direction_deg=default_wind["wind_direction_deg"],
            stability_class=default_wind["stability_class"],
            avg_emission=0.5,
            subsample=1,
        )
        beyond = simple_deviation > max_dev
        assert np.all(scores[beyond] == 0.0)

    def test_closer_to_path_scores_higher(
        self, tiny_grid, uniform_belief, default_wind, simple_deviation,
    ):
        """Given uniform belief, cells closer to path should score higher
        (same EER numerator, lower deviation denominator)."""
        X, Y = tiny_grid
        scores = compute_information_scores(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            deviation=simple_deviation, max_deviation=40.0,
            wind_speed=default_wind["wind_speed"],
            wind_direction_deg=default_wind["wind_direction_deg"],
            stability_class=default_wind["stability_class"],
            avg_emission=0.5,
            subsample=1,
        )
        # Middle row (y=0, deviation=0) should have higher scores than edge
        mid_row = X.shape[0] // 2
        edge_row = 0
        mid_max = np.max(scores[mid_row, :])
        edge_max = np.max(scores[edge_row, :])
        if mid_max > 0 and edge_max > 0:
            assert mid_max >= edge_max


# ---------------------------------------------------------------------------
# Ensemble Information Scores Tests
# ---------------------------------------------------------------------------

class TestEnsembleInformationScores:
    """Tests for compute_ensemble_information_scores."""

    def test_output_shape(
        self, tiny_grid, uniform_belief, simple_deviation,
    ):
        """Ensemble scores should match input shape."""
        X, Y = tiny_grid
        scenarios = [
            {"direction": 270.0, "speed": 3.0, "stability_class": "D", "weight": 0.5},
            {"direction": 90.0, "speed": 3.0, "stability_class": "D", "weight": 0.5},
        ]
        scores = compute_ensemble_information_scores(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            deviation=simple_deviation, max_deviation=40.0,
            wind_scenarios=scenarios, avg_emission=0.5,
            subsample=2,
        )
        assert scores.shape == X.shape

    def test_non_negative(
        self, tiny_grid, uniform_belief, simple_deviation,
    ):
        """Ensemble scores should be non-negative."""
        X, Y = tiny_grid
        scenarios = [
            {"direction": 270.0, "speed": 3.0, "stability_class": "D", "weight": 0.5},
            {"direction": 90.0, "speed": 3.0, "stability_class": "D", "weight": 0.5},
        ]
        scores = compute_ensemble_information_scores(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            deviation=simple_deviation, max_deviation=40.0,
            wind_scenarios=scenarios, avg_emission=0.5,
            subsample=2,
        )
        assert np.all(scores >= -1e-10)

    def test_single_scenario_matches_non_ensemble(
        self, tiny_grid, uniform_belief, simple_deviation, default_wind,
    ):
        """Single-scenario ensemble should match non-ensemble result."""
        X, Y = tiny_grid
        scenarios = [{
            "direction": default_wind["wind_direction_deg"],
            "speed": default_wind["wind_speed"],
            "stability_class": default_wind["stability_class"],
            "weight": 1.0,
        }]

        scores_ens = compute_ensemble_information_scores(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            deviation=simple_deviation, max_deviation=40.0,
            wind_scenarios=scenarios, avg_emission=0.5,
            subsample=1,
        )
        scores_single = compute_information_scores(
            grid_x=X, grid_y=Y, belief=uniform_belief,
            deviation=simple_deviation, max_deviation=40.0,
            wind_speed=default_wind["wind_speed"],
            wind_direction_deg=default_wind["wind_direction_deg"],
            stability_class=default_wind["stability_class"],
            avg_emission=0.5,
            subsample=1,
        )
        np.testing.assert_allclose(scores_ens, scores_single, atol=1e-10)

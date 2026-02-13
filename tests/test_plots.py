"""Smoke tests for visualization plot functions.

Each test verifies that the function returns a valid Plotly Figure
without raising exceptions. These are not pixel-perfect tests —
they just confirm the functions work end-to-end with representative
inputs.
"""

import sys
import os
import pytest
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualization.plots import (
    create_single_map_figure,
    create_concentration_figure,
    create_site_figure,
    create_score_profile,
    create_entropy_figure,
    create_prior_posterior_figure,
    create_convergence_figure,
)


@pytest.fixture
def plot_grid():
    """Small 100m grid at 20m resolution for fast plot tests."""
    coords = np.arange(-50, 51, 20.0)
    X, Y = np.meshgrid(coords, coords)
    return X, Y


@pytest.fixture
def plot_sources():
    return [
        {"name": "Source A", "x": -20.0, "y": 10.0, "z": 0.0, "emission_rate": 0.5},
        {"name": "Source B", "x": 30.0, "y": -15.0, "z": 0.0, "emission_rate": 0.3},
    ]


@pytest.fixture
def plot_paths():
    baseline = np.array([[-50, 0], [0, 0], [50, 0]], dtype=float)
    optimized = np.array([[-50, 0], [-20, 10], [0, 0], [30, -15], [50, 0]], dtype=float)
    return baseline, optimized


@pytest.fixture
def plot_recommendations():
    return [
        {"x": -20.0, "y": 10.0, "score": 0.8, "detection_prob": 0.7, "concentration_ppm": 12.0},
        {"x": 30.0, "y": -15.0, "score": 0.5, "detection_prob": 0.4, "concentration_ppm": 5.0},
    ]


@pytest.fixture
def detection_prob(plot_grid):
    X, _ = plot_grid
    return np.random.default_rng(42).random(X.shape)


@pytest.fixture
def concentration_ppm(plot_grid):
    X, _ = plot_grid
    return np.random.default_rng(42).random(X.shape) * 20.0


@pytest.fixture
def belief(plot_grid):
    X, _ = plot_grid
    return np.random.default_rng(42).random(X.shape) * 0.5


class TestCreateSingleMapFigure:
    def test_returns_figure(self, plot_grid, detection_prob, plot_sources,
                            plot_paths, plot_recommendations):
        X, Y = plot_grid
        baseline, optimized = plot_paths
        fig = create_single_map_figure(
            grid_x=X, grid_y=Y, detection_prob=detection_prob,
            sources=plot_sources, baseline_path=baseline,
            optimized_path=optimized, recommendations=plot_recommendations,
            wind_speed=3.0, wind_direction_deg=270.0,
        )
        assert isinstance(fig, go.Figure)

    def test_no_recommendations(self, plot_grid, detection_prob, plot_sources, plot_paths):
        X, Y = plot_grid
        baseline, optimized = plot_paths
        fig = create_single_map_figure(
            grid_x=X, grid_y=Y, detection_prob=detection_prob,
            sources=plot_sources, baseline_path=baseline,
            optimized_path=optimized, recommendations=[],
            wind_speed=3.0, wind_direction_deg=270.0,
        )
        assert isinstance(fig, go.Figure)


class TestCreateConcentrationFigure:
    def test_returns_figure(self, plot_grid, concentration_ppm, plot_sources):
        X, Y = plot_grid
        fig = create_concentration_figure(
            grid_x=X, grid_y=Y, concentration_ppm=concentration_ppm,
            sources=plot_sources, wind_speed=3.0, wind_direction_deg=270.0,
        )
        assert isinstance(fig, go.Figure)


class TestCreateSiteFigure:
    def test_returns_figure(self, plot_grid, concentration_ppm, detection_prob,
                            plot_sources, plot_paths, plot_recommendations):
        X, Y = plot_grid
        baseline, optimized = plot_paths
        fig = create_site_figure(
            grid_x=X, grid_y=Y, concentration_ppm=concentration_ppm,
            detection_prob=detection_prob, sources=plot_sources,
            baseline_path=baseline, optimized_path=optimized,
            recommendations=plot_recommendations,
            wind_speed=3.0, wind_direction_deg=270.0,
        )
        assert isinstance(fig, go.Figure)


class TestCreateEntropyFigure:
    def test_returns_figure(self, plot_grid, belief, plot_sources):
        X, Y = plot_grid
        fig = create_entropy_figure(
            grid_x=X, grid_y=Y, belief=belief,
            sources=plot_sources, wind_speed=3.0, wind_direction_deg=270.0,
        )
        assert isinstance(fig, go.Figure)

    def test_uniform_belief(self, plot_grid, plot_sources):
        X, Y = plot_grid
        uniform = np.full(X.shape, 0.5)
        fig = create_entropy_figure(
            grid_x=X, grid_y=Y, belief=uniform,
            sources=plot_sources, wind_speed=3.0, wind_direction_deg=270.0,
        )
        assert isinstance(fig, go.Figure)


class TestCreatePriorPosteriorFigure:
    def test_returns_figure(self, plot_grid, belief, plot_sources):
        X, Y = plot_grid
        prior = np.full(X.shape, 0.1)
        fig = create_prior_posterior_figure(
            grid_x=X, grid_y=Y, prior=prior, posterior=belief,
            sources=plot_sources, wind_speed=3.0, wind_direction_deg=270.0,
        )
        assert isinstance(fig, go.Figure)

    def test_identical_prior_posterior(self, plot_grid, belief, plot_sources):
        X, Y = plot_grid
        fig = create_prior_posterior_figure(
            grid_x=X, grid_y=Y, prior=belief, posterior=belief,
            sources=plot_sources, wind_speed=3.0, wind_direction_deg=270.0,
        )
        assert isinstance(fig, go.Figure)


class TestCreateConvergenceFigure:
    def test_returns_figure(self):
        history = [100.0, 90.0, 75.0, 60.0, 50.0]
        fig = create_convergence_figure(history)
        assert isinstance(fig, go.Figure)

    def test_two_points(self):
        fig = create_convergence_figure([100.0, 80.0])
        assert isinstance(fig, go.Figure)

    def test_single_point(self):
        fig = create_convergence_figure([100.0])
        assert isinstance(fig, go.Figure)


class TestCreateScoreProfile:
    def test_returns_figure(self, plot_recommendations):
        fig = create_score_profile(plot_recommendations)
        assert isinstance(fig, go.Figure)

    def test_empty_recommendations(self):
        fig = create_score_profile([])
        assert isinstance(fig, go.Figure)

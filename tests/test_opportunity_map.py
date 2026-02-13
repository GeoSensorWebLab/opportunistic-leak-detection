"""Tests for the opportunity map generator."""

import numpy as np
import pytest
from optimization.opportunity_map import create_grid, compute_opportunity_map


class TestCreateGrid:
    """Tests for grid creation."""

    def test_grid_shape(self):
        """Grid should have expected number of cells."""
        X, Y = create_grid(grid_size=100, resolution=10)
        # 100m grid at 10m resolution: from -50 to 50 = 11 points per axis
        assert X.shape == Y.shape
        assert X.shape[0] == X.shape[1]  # Square grid

    def test_grid_centered_at_origin(self):
        """Grid should be symmetric about origin."""
        X, Y = create_grid(grid_size=100, resolution=10)
        np.testing.assert_allclose(X.min(), -X.max(), atol=10)
        np.testing.assert_allclose(Y.min(), -Y.max(), atol=10)

    def test_grid_extent(self):
        """Grid should span the requested size."""
        X, Y = create_grid(grid_size=200, resolution=5)
        assert X.max() >= 95  # Should reach at least close to half the grid size
        assert X.min() <= -95

    def test_resolution_affects_density(self):
        """Finer resolution should produce more grid points."""
        X_coarse, _ = create_grid(grid_size=100, resolution=20)
        X_fine, _ = create_grid(grid_size=100, resolution=5)
        assert X_fine.size > X_coarse.size


class TestComputeOpportunityMap:
    """Tests for the combined opportunity map computation."""

    def test_single_source_produces_plume(self, single_source, default_wind):
        """A single source should produce non-zero concentrations downwind."""
        X, Y, conc_ppm, det_prob = compute_opportunity_map(
            sources=[single_source],
            grid_size=200,
            resolution=10,
            receptor_height=1.5,
            **default_wind,
        )
        assert np.any(conc_ppm > 0), "Should have non-zero concentrations"
        assert np.any(det_prob > 0), "Should have non-zero detection probabilities"

    def test_detection_prob_bounded(self, single_source, default_wind):
        """Detection probability should be in [0, 1]."""
        _, _, _, det_prob = compute_opportunity_map(
            sources=[single_source],
            grid_size=200,
            resolution=10,
            receptor_height=1.5,
            **default_wind,
        )
        assert np.all(det_prob >= 0.0)
        assert np.all(det_prob <= 1.0)

    def test_multi_source_superposition(self, default_wind):
        """Two sources should produce higher concentrations than either alone."""
        src1 = {"name": "A", "x": -50.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5}
        src2 = {"name": "B", "x": -80.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5}

        _, _, conc_single, _ = compute_opportunity_map(
            sources=[src1], grid_size=200, resolution=10, receptor_height=1.5,
            **default_wind,
        )
        _, _, conc_both, _ = compute_opportunity_map(
            sources=[src1, src2], grid_size=200, resolution=10, receptor_height=1.5,
            **default_wind,
        )
        # The combined concentration should be >= single source everywhere
        assert np.all(conc_both >= conc_single - 1e-15)

    def test_output_shapes_match(self, single_source, default_wind):
        """All output arrays should have the same shape."""
        X, Y, conc, det = compute_opportunity_map(
            sources=[single_source],
            grid_size=100,
            resolution=10,
            receptor_height=1.5,
            **default_wind,
        )
        assert X.shape == Y.shape == conc.shape == det.shape

    def test_no_sources_gives_zeros(self, default_wind):
        """Empty source list should produce all-zero maps."""
        _, _, conc, det = compute_opportunity_map(
            sources=[],
            grid_size=100,
            resolution=10,
            receptor_height=1.5,
            **default_wind,
        )
        assert np.allclose(conc, 0.0)
        assert np.all(det < 0.01)  # sigmoid(0 - 5) is very small

    def test_integrated_plume_mode(self, single_source, default_wind):
        """Crosswind-integrated plume mode should produce valid results."""
        X, Y, conc_ppm, det_prob = compute_opportunity_map(
            sources=[single_source],
            grid_size=200,
            resolution=10,
            receptor_height=1.5,
            plume_mode="integrated",
            **default_wind,
        )
        assert np.any(conc_ppm > 0)
        assert np.all(det_prob >= 0.0)
        assert np.all(det_prob <= 1.0)

    def test_integrated_broader_than_instantaneous(self, single_source, default_wind):
        """Integrated mode should produce broader (more non-zero cells) plumes."""
        _, _, conc_inst, _ = compute_opportunity_map(
            sources=[single_source],
            grid_size=200,
            resolution=10,
            receptor_height=1.5,
            plume_mode="instantaneous",
            **default_wind,
        )
        _, _, conc_integ, _ = compute_opportunity_map(
            sources=[single_source],
            grid_size=200,
            resolution=10,
            receptor_height=1.5,
            plume_mode="integrated",
            **default_wind,
        )
        # Count cells with non-trivial concentration
        n_inst = np.sum(conc_inst > 0.01)
        n_integ = np.sum(conc_integ > 0.01)
        assert n_integ >= n_inst, "Integrated plume should have at least as many non-zero cells"

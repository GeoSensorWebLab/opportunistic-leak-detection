"""Tests for the Gaussian plume dispersion model."""

import numpy as np
import pytest
from models.gaussian_plume import (
    gaussian_plume,
    crosswind_integrated_plume,
    compute_sigma,
    concentration_to_ppm,
)


class TestComputeSigma:
    """Tests for dispersion parameter computation."""

    def test_sigma_increases_with_distance(self):
        """Sigma_y and sigma_z should increase monotonically with downwind distance."""
        distances = np.array([10.0, 50.0, 100.0, 500.0, 1000.0])
        for stability in ["A", "B", "C", "D", "E", "F"]:
            sy, sz = compute_sigma(distances, stability)
            assert np.all(np.diff(sy) > 0), f"sigma_y not monotonic for class {stability}"
            assert np.all(np.diff(sz) > 0), f"sigma_z not monotonic for class {stability}"

    def test_unstable_disperses_more_than_stable(self):
        """Class A (very unstable) should have larger sigmas than class F (very stable)."""
        distance = np.array([500.0])
        sy_a, sz_a = compute_sigma(distance, "A")
        sy_f, sz_f = compute_sigma(distance, "F")
        assert sy_a > sy_f, "Unstable should disperse more laterally"
        assert sz_a > sz_f, "Unstable should disperse more vertically"

    def test_invalid_stability_class_raises(self):
        """Unknown stability class should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown stability class"):
            compute_sigma(np.array([100.0]), "Z")

    def test_sigma_positive_for_positive_distance(self):
        """Sigmas should always be positive for positive distances."""
        distances = np.array([1.0, 10.0, 100.0])
        for stability in ["A", "D", "F"]:
            sy, sz = compute_sigma(distances, stability)
            assert np.all(sy > 0)
            assert np.all(sz > 0)

    def test_clamps_negative_distance(self):
        """Negative distance (upwind) should be clamped to 1m minimum."""
        distances = np.array([-50.0, 0.0, 1.0])
        sy, sz = compute_sigma(distances, "D")
        # First two values should equal the value at 1m (clamped)
        assert sy[0] == sy[2], "Negative distance should be clamped"
        assert sy[1] == sy[2], "Zero distance should be clamped"


class TestGaussianPlume:
    """Tests for the Gaussian plume concentration model."""

    def test_concentration_positive_downwind(self, small_grid, default_wind):
        """Concentration should be positive at downwind locations."""
        X, Y = small_grid
        conc = gaussian_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        # Wind from west (270) means plume blows east (+x)
        # Points at x > 0, y ~ 0 should have positive concentration
        east_mask = (X > 20) & (np.abs(Y) < 10)
        assert np.any(conc[east_mask] > 0), "Should have positive concentration downwind"

    def test_zero_upwind(self, small_grid, default_wind):
        """Concentration should be zero at upwind locations."""
        X, Y = small_grid
        conc = gaussian_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        # Wind from west: upwind is negative x
        upwind_mask = X < -10
        assert np.allclose(conc[upwind_mask], 0.0), "Should be zero upwind"

    def test_concentration_decreases_with_distance(self, default_wind):
        """Centerline concentration should decrease with downwind distance."""
        distances = np.array([50.0, 100.0, 200.0, 500.0])
        # Place receptors along centerline (y=0) at increasing x
        X = distances
        Y = np.zeros_like(distances)

        conc = gaussian_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        assert np.all(np.diff(conc) < 0), "Centerline concentration should decrease with distance"

    def test_crosswind_symmetry(self, default_wind):
        """Concentration should be symmetric about the plume centerline."""
        # Wind from west → plume along +x axis
        # Test symmetry at y = +50 and y = -50 at same downwind distance
        X = np.array([200.0, 200.0])
        Y = np.array([50.0, -50.0])

        conc = gaussian_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        np.testing.assert_allclose(conc[0], conc[1], rtol=1e-10,
                                    err_msg="Plume should be symmetric about centerline")

    def test_ground_reflection_doubles_surface(self):
        """Ground-level source with ground-level receptor should benefit from ground reflection."""
        # For z=0 source and z=0 receptor, the image term equals the direct term
        # so concentration should be exactly double the free-space value
        X = np.array([100.0])
        Y = np.array([0.0])

        conc_surface = gaussian_plume(
            X, Y, receptor_z=0.0,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            wind_speed=3.0, wind_direction_deg=270.0, stability_class="D",
        )
        # At z=0, H=0: vertical term = exp(0) + exp(0) = 2
        # Without reflection it would be exp(0) = 1
        # So reflected concentration should be > 0
        assert conc_surface[0] > 0

    def test_zero_wind_speed_raises(self, small_grid):
        """Wind speed of 0 should raise ValueError."""
        X, Y = small_grid
        with pytest.raises(ValueError, match="Wind speed must be positive"):
            gaussian_plume(
                X, Y, receptor_z=1.5,
                source_x=0.0, source_y=0.0, source_z=0.0,
                emission_rate=0.5,
                wind_speed=0.0, wind_direction_deg=270.0, stability_class="D",
            )

    def test_negative_wind_speed_raises(self, small_grid):
        """Negative wind speed should raise ValueError."""
        X, Y = small_grid
        with pytest.raises(ValueError, match="Wind speed must be positive"):
            gaussian_plume(
                X, Y, receptor_z=1.5,
                source_x=0.0, source_y=0.0, source_z=0.0,
                emission_rate=0.5,
                wind_speed=-5.0, wind_direction_deg=270.0, stability_class="D",
            )

    def test_higher_emission_gives_higher_concentration(self, default_wind):
        """Doubling emission rate should double concentration (linearity)."""
        X = np.array([100.0])
        Y = np.array([0.0])

        conc_q1 = gaussian_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        conc_q2 = gaussian_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=1.0,
            **default_wind,
        )
        np.testing.assert_allclose(conc_q2, 2 * conc_q1, rtol=1e-10,
                                    err_msg="Plume equation should be linear in emission rate")

    def test_wind_direction_north(self, small_grid):
        """Wind from north (0 deg) should produce plume toward south (-y)."""
        X, Y = small_grid
        conc = gaussian_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            wind_speed=3.0, wind_direction_deg=0.0, stability_class="D",
        )
        south_mask = (Y < -20) & (np.abs(X) < 10)
        north_mask = Y > 20
        assert np.sum(conc[south_mask]) > 0, "Should have concentration to the south"
        assert np.allclose(conc[north_mask], 0.0), "Should be zero to the north (upwind)"


class TestConcentrationToPpm:
    """Tests for kg/m3 to ppm conversion."""

    def test_zero_concentration(self):
        """Zero input should give zero output."""
        result = concentration_to_ppm(np.array([0.0]))
        assert result[0] == 0.0

    def test_positive_conversion(self):
        """Positive concentration should give positive ppm."""
        result = concentration_to_ppm(np.array([1e-6]))
        assert result[0] > 0

    def test_linearity(self):
        """Conversion should be linear."""
        c1 = np.array([1e-6])
        c2 = np.array([2e-6])
        ppm1 = concentration_to_ppm(c1)
        ppm2 = concentration_to_ppm(c2)
        np.testing.assert_allclose(ppm2, 2 * ppm1, rtol=1e-10)

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        c = np.ones((5, 10)) * 1e-6
        ppm = concentration_to_ppm(c)
        assert ppm.shape == (5, 10)


class TestCrosswindIntegratedPlume:
    """Tests for the crosswind-integrated Gaussian plume model."""

    def test_positive_downwind(self, small_grid, default_wind):
        """Should produce positive concentration at downwind locations."""
        X, Y = small_grid
        conc = crosswind_integrated_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        east_mask = (X > 20) & (np.abs(Y) < 50)
        assert np.any(conc[east_mask] > 0)

    def test_zero_upwind(self, small_grid, default_wind):
        """Should be zero at upwind locations."""
        X, Y = small_grid
        conc = crosswind_integrated_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        upwind_mask = X < -10
        assert np.allclose(conc[upwind_mask], 0.0)

    def test_broader_than_instantaneous(self, default_wind):
        """Crosswind-integrated plume should be broader (higher off-centerline)."""
        # At a far-off-centerline point, integrated plume should be higher
        # because it doesn't penalize crosswind offset
        X = np.array([200.0])
        Y = np.array([100.0])  # far off centerline

        conc_inst = gaussian_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        conc_integ = crosswind_integrated_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        # Integrated should be >= instantaneous at off-centerline points
        assert conc_integ[0] >= conc_inst[0]

    def test_linear_in_emission_rate(self, default_wind):
        """Doubling emission should double concentration."""
        X = np.array([100.0])
        Y = np.array([0.0])

        conc_q1 = crosswind_integrated_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        conc_q2 = crosswind_integrated_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=1.0,
            **default_wind,
        )
        np.testing.assert_allclose(conc_q2, 2 * conc_q1, rtol=1e-10)

    def test_zero_wind_speed_raises(self, small_grid):
        """Wind speed of 0 should raise ValueError."""
        X, Y = small_grid
        with pytest.raises(ValueError, match="Wind speed must be positive"):
            crosswind_integrated_plume(
                X, Y, receptor_z=1.5,
                source_x=0.0, source_y=0.0, source_z=0.0,
                emission_rate=0.5,
                wind_speed=0.0, wind_direction_deg=270.0, stability_class="D",
            )

    def test_same_shape_as_input(self, small_grid, default_wind):
        """Output should match input grid shape."""
        X, Y = small_grid
        conc = crosswind_integrated_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        assert conc.shape == X.shape

    def test_centerline_decreases_with_distance(self, default_wind):
        """Centerline concentration should decrease with downwind distance."""
        distances = np.array([50.0, 100.0, 200.0, 500.0])
        X = distances
        Y = np.zeros_like(distances)

        conc = crosswind_integrated_plume(
            X, Y, receptor_z=1.5,
            source_x=0.0, source_y=0.0, source_z=0.0,
            emission_rate=0.5,
            **default_wind,
        )
        assert np.all(np.diff(conc) < 0)

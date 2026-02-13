"""Tests for temporal / intermittent leak features.

Covers:
  - Gaussian puff dispersion model
  - Duty cycle in opportunity map
  - Puff mode in opportunity map
  - Duty cycle in synthetic twin (time-averaged and time-resolved)
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.gaussian_plume import gaussian_puff, concentration_to_ppm
from optimization.opportunity_map import compute_opportunity_map
from config import DEFAULT_PUFF_MASS_KG, DEFAULT_PUFF_TIME_S


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_source(**overrides):
    """Create a single test source with optional overrides."""
    base = {
        "name": "Test",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "emission_rate": 0.5,
        "duty_cycle": 1.0,
    }
    base.update(overrides)
    return base


# ===========================================================================
# TestGaussianPuff
# ===========================================================================

class TestGaussianPuff:
    """Tests for the gaussian_puff() function."""

    def test_positive_at_puff_center(self):
        """Concentration should be positive near the puff center."""
        # Puff center is at downwind distance = u * t
        u, t = 3.0, 60.0
        travel = u * t  # 180 m downwind
        # Wind from west (270) -> blowing east (+x)
        conc = gaussian_puff(
            receptor_x=np.array([travel]),
            receptor_y=np.array([0.0]),
            receptor_z=0.0,
            source_x=0.0,
            source_y=0.0,
            source_z=0.0,
            total_mass=5.0,
            wind_speed=u,
            wind_direction_deg=270.0,
            stability_class="D",
            time_since_release=t,
        )
        assert conc[0] > 0.0

    def test_zero_far_away(self):
        """Concentration should be negligible far from the puff."""
        conc = gaussian_puff(
            receptor_x=np.array([5000.0]),
            receptor_y=np.array([5000.0]),
            receptor_z=0.0,
            source_x=0.0,
            source_y=0.0,
            source_z=0.0,
            total_mass=5.0,
            wind_speed=3.0,
            wind_direction_deg=270.0,
            stability_class="D",
            time_since_release=60.0,
        )
        assert conc[0] < 1e-20

    def test_linear_in_mass(self):
        """Doubling mass should double concentration everywhere."""
        kwargs = dict(
            receptor_x=np.array([180.0]),
            receptor_y=np.array([0.0]),
            receptor_z=1.5,
            source_x=0.0,
            source_y=0.0,
            source_z=0.0,
            wind_speed=3.0,
            wind_direction_deg=270.0,
            stability_class="D",
            time_since_release=60.0,
        )
        c1 = gaussian_puff(total_mass=5.0, **kwargs)
        c2 = gaussian_puff(total_mass=10.0, **kwargs)
        np.testing.assert_allclose(c2, 2.0 * c1, rtol=1e-10)

    def test_disperses_over_time(self):
        """Peak concentration should decrease as the puff disperses."""
        kwargs = dict(
            receptor_x=np.array([0.0]),
            receptor_y=np.array([0.0]),
            receptor_z=0.0,
            source_x=0.0,
            source_y=0.0,
            source_z=0.0,
            total_mass=5.0,
            wind_speed=3.0,
            wind_direction_deg=270.0,
            stability_class="D",
        )
        # At the puff center (which moves downwind), measure at successive times
        # by placing receptor at the moving center
        c_early = gaussian_puff(
            **{**kwargs, "receptor_x": np.array([3.0 * 10.0])},
            time_since_release=10.0,
        )
        c_late = gaussian_puff(
            **{**kwargs, "receptor_x": np.array([3.0 * 120.0])},
            time_since_release=120.0,
        )
        assert c_early[0] > c_late[0]

    def test_center_moves_downwind(self):
        """Puff center should be at u*t downwind of source."""
        u, t = 3.0, 100.0
        travel = u * t  # 300 m
        # Sample along downwind axis
        xs = np.linspace(100, 500, 50)
        conc = gaussian_puff(
            receptor_x=xs,
            receptor_y=np.zeros_like(xs),
            receptor_z=0.0,
            source_x=0.0,
            source_y=0.0,
            source_z=0.0,
            total_mass=5.0,
            wind_speed=u,
            wind_direction_deg=270.0,
            stability_class="D",
            time_since_release=t,
        )
        peak_x = xs[np.argmax(conc)]
        assert abs(peak_x - travel) < 20.0  # within 20m of expected

    def test_crosswind_symmetry(self):
        """Concentration should be symmetric about the wind axis."""
        travel = 3.0 * 60.0
        conc_pos = gaussian_puff(
            receptor_x=np.array([travel]),
            receptor_y=np.array([50.0]),
            receptor_z=0.0,
            source_x=0.0,
            source_y=0.0,
            source_z=0.0,
            total_mass=5.0,
            wind_speed=3.0,
            wind_direction_deg=270.0,
            stability_class="D",
            time_since_release=60.0,
        )
        conc_neg = gaussian_puff(
            receptor_x=np.array([travel]),
            receptor_y=np.array([-50.0]),
            receptor_z=0.0,
            source_x=0.0,
            source_y=0.0,
            source_z=0.0,
            total_mass=5.0,
            wind_speed=3.0,
            wind_direction_deg=270.0,
            stability_class="D",
            time_since_release=60.0,
        )
        np.testing.assert_allclose(conc_pos, conc_neg, rtol=1e-10)

    def test_ground_reflection(self):
        """Elevated source should produce higher ground conc than without reflection."""
        # With ground reflection, concentration at z=0 from elevated source
        # should be higher than the single-term exponential (the reflection adds)
        travel = 3.0 * 60.0
        conc_ground = gaussian_puff(
            receptor_x=np.array([travel]),
            receptor_y=np.array([0.0]),
            receptor_z=0.0,
            source_x=0.0,
            source_y=0.0,
            source_z=5.0,  # elevated source
            total_mass=5.0,
            wind_speed=3.0,
            wind_direction_deg=270.0,
            stability_class="D",
            time_since_release=60.0,
        )
        # Ground-level source should give higher concentration at z=0
        conc_surface = gaussian_puff(
            receptor_x=np.array([travel]),
            receptor_y=np.array([0.0]),
            receptor_z=0.0,
            source_x=0.0,
            source_y=0.0,
            source_z=0.0,
            total_mass=5.0,
            wind_speed=3.0,
            wind_direction_deg=270.0,
            stability_class="D",
            time_since_release=60.0,
        )
        # Both should be positive and surface source gives higher conc at z=0
        assert conc_ground[0] > 0.0
        assert conc_surface[0] >= conc_ground[0]

    def test_raises_on_negative_wind_speed(self):
        """Should raise ValueError for non-positive wind speed."""
        with pytest.raises(ValueError, match="Wind speed"):
            gaussian_puff(
                receptor_x=np.array([100.0]),
                receptor_y=np.array([0.0]),
                receptor_z=0.0,
                source_x=0.0,
                source_y=0.0,
                source_z=0.0,
                total_mass=5.0,
                wind_speed=-1.0,
                wind_direction_deg=270.0,
                stability_class="D",
                time_since_release=60.0,
            )

    def test_raises_on_negative_time(self):
        """Should raise ValueError for non-positive time."""
        with pytest.raises(ValueError, match="Time since release"):
            gaussian_puff(
                receptor_x=np.array([100.0]),
                receptor_y=np.array([0.0]),
                receptor_z=0.0,
                source_x=0.0,
                source_y=0.0,
                source_z=0.0,
                total_mass=5.0,
                wind_speed=3.0,
                wind_direction_deg=270.0,
                stability_class="D",
                time_since_release=-10.0,
            )

    def test_raises_on_negative_mass(self):
        """Should raise ValueError for negative total mass."""
        with pytest.raises(ValueError, match="Total mass"):
            gaussian_puff(
                receptor_x=np.array([100.0]),
                receptor_y=np.array([0.0]),
                receptor_z=0.0,
                source_x=0.0,
                source_y=0.0,
                source_z=0.0,
                total_mass=-1.0,
                wind_speed=3.0,
                wind_direction_deg=270.0,
                stability_class="D",
                time_since_release=60.0,
            )

    def test_output_shape_matches_input(self):
        """Output shape should match receptor grid shape."""
        X, Y = np.meshgrid(np.linspace(-50, 50, 20), np.linspace(-50, 50, 20))
        conc = gaussian_puff(
            receptor_x=X,
            receptor_y=Y,
            receptor_z=1.5,
            source_x=0.0,
            source_y=0.0,
            source_z=0.0,
            total_mass=5.0,
            wind_speed=3.0,
            wind_direction_deg=270.0,
            stability_class="D",
            time_since_release=60.0,
        )
        assert conc.shape == X.shape


# ===========================================================================
# TestDutyCycleInOpportunityMap
# ===========================================================================

class TestDutyCycleInOpportunityMap:
    """Tests for duty_cycle integration in compute_opportunity_map."""

    def test_dc1_matches_original(self):
        """duty_cycle=1.0 should produce same result as no duty_cycle key."""
        src_with = [_single_source(duty_cycle=1.0)]
        src_without = [{"name": "Test", "x": 0.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5}]

        _, _, conc_with, _ = compute_opportunity_map(
            sources=src_with, wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=200, resolution=10,
        )
        _, _, conc_without, _ = compute_opportunity_map(
            sources=src_without, wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=200, resolution=10,
        )
        np.testing.assert_allclose(conc_with, conc_without, rtol=1e-10)

    def test_dc_half_halves_concentration(self):
        """duty_cycle=0.5 should produce half the concentration."""
        src_full = [_single_source(duty_cycle=1.0)]
        src_half = [_single_source(duty_cycle=0.5)]

        _, _, conc_full, _ = compute_opportunity_map(
            sources=src_full, wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=200, resolution=10,
        )
        _, _, conc_half, _ = compute_opportunity_map(
            sources=src_half, wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=200, resolution=10,
        )
        # Where concentration is nonzero, the ratio should be ~0.5
        mask = conc_full > 1e-15
        if np.any(mask):
            np.testing.assert_allclose(
                conc_half[mask] / conc_full[mask], 0.5, rtol=1e-6,
            )

    def test_dc_zero_gives_zero(self):
        """duty_cycle=0.0 should produce zero concentration everywhere."""
        src = [_single_source(duty_cycle=0.0)]
        _, _, conc, det = compute_opportunity_map(
            sources=src, wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=200, resolution=10,
        )
        assert np.all(conc == 0.0)
        assert np.all(det == 0.0)

    def test_default_dc_is_one(self):
        """Source without duty_cycle key should default to 1.0."""
        src_no_key = [{"name": "T", "x": 0.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5}]
        src_explicit = [_single_source(duty_cycle=1.0)]

        _, _, c1, _ = compute_opportunity_map(
            sources=src_no_key, wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=200, resolution=10,
        )
        _, _, c2, _ = compute_opportunity_map(
            sources=src_explicit, wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=200, resolution=10,
        )
        np.testing.assert_allclose(c1, c2, rtol=1e-10)

    def test_mixed_duty_cycles(self):
        """Two sources with different duty cycles should produce intermediate result."""
        src_a = _single_source(name="A", x=-100.0, duty_cycle=1.0)
        src_b = _single_source(name="B", x=100.0, duty_cycle=0.5)

        _, _, conc, det = compute_opportunity_map(
            sources=[src_a, src_b], wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=400, resolution=10,
        )
        # Should have nonzero detections (source A is always on)
        assert np.max(det) > 0.0


# ===========================================================================
# TestPuffModeInOpportunityMap
# ===========================================================================

class TestPuffModeInOpportunityMap:
    """Tests for plume_mode='puff' in compute_opportunity_map."""

    def test_puff_mode_produces_valid_output(self):
        """Puff mode should produce finite, non-negative concentrations."""
        src = [_single_source()]
        X, Y, conc, det = compute_opportunity_map(
            sources=src, wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=400, resolution=10,
            plume_mode="puff",
        )
        assert np.all(np.isfinite(conc))
        assert np.all(conc >= 0.0)
        assert np.max(conc) > 0.0

    def test_puff_detection_bounded(self):
        """Detection probabilities should be in [0, 1]."""
        src = [_single_source()]
        _, _, _, det = compute_opportunity_map(
            sources=src, wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=400, resolution=10,
            plume_mode="puff",
        )
        assert np.all(det >= 0.0)
        assert np.all(det <= 1.0)

    def test_puff_mode_different_from_instantaneous(self):
        """Puff mode should give different results than standard plume."""
        src = [_single_source()]
        kwargs = dict(
            sources=src, wind_speed=3.0, wind_direction_deg=270.0,
            stability_class="D", grid_size=200, resolution=10,
        )
        _, _, conc_inst, _ = compute_opportunity_map(plume_mode="instantaneous", **kwargs)
        _, _, conc_puff, _ = compute_opportunity_map(plume_mode="puff", **kwargs)
        # They should not be identical
        assert not np.allclose(conc_inst, conc_puff)


# ===========================================================================
# TestDutyCycleInSyntheticTwin
# ===========================================================================

class TestDutyCycleInSyntheticTwin:
    """Tests for duty cycle behavior in SyntheticExperiment."""

    def _make_experiment(self, duty_cycle=1.0, time_resolved=False):
        """Create a minimal synthetic experiment."""
        from validation.synthetic_twin import SyntheticExperiment

        src = {
            "name": "Test",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "emission_rate": 0.5,
            "equipment_type": "wellhead",
            "age_years": 10,
            "production_rate_mcfd": 1000.0,
            "last_inspection_days": 30,
            "duty_cycle": duty_cycle,
        }
        wind = {"wind_speed": 3.0, "wind_direction_deg": 270.0, "stability_class": "D"}
        return SyntheticExperiment(
            ground_truth=[src],
            all_equipment=[src],
            wind_params=wind,
            grid_size=200,
            resolution=20,
            time_resolved=time_resolved,
        )

    def test_time_averaged_reduces_signal(self):
        """duty_cycle < 1 in time-averaged mode should reduce measured ppm."""
        exp_full = self._make_experiment(duty_cycle=1.0)
        exp_half = self._make_experiment(duty_cycle=0.5)
        rng = np.random.default_rng(42)
        wind = {"wind_speed": 3.0, "wind_direction_deg": 270.0, "stability_class": "D"}

        # Measure 100m downwind (east of source with west wind)
        _, _, ppm_full = exp_full.simulate_measurement(100.0, 0.0, wind, rng)
        rng2 = np.random.default_rng(42)
        _, _, ppm_half = exp_half.simulate_measurement(100.0, 0.0, wind, rng2)

        assert ppm_full > 0.0
        np.testing.assert_allclose(ppm_half, ppm_full * 0.5, rtol=1e-6)

    def test_time_resolved_binary(self):
        """In time-resolved mode, emission should be full-rate or zero."""
        exp = self._make_experiment(duty_cycle=0.5, time_resolved=True)
        wind = {"wind_speed": 3.0, "wind_direction_deg": 270.0, "stability_class": "D"}

        ppms = []
        for seed in range(100):
            rng = np.random.default_rng(seed)
            _, _, ppm = exp.simulate_measurement(100.0, 0.0, wind, rng)
            ppms.append(ppm)

        ppms = np.array(ppms)
        # Should have a mix of zero and nonzero values
        assert np.any(ppms == 0.0), "Expected some zero readings"
        assert np.any(ppms > 0.0), "Expected some nonzero readings"
        # All nonzero values should be the same (full emission rate)
        nonzero = ppms[ppms > 0.0]
        np.testing.assert_allclose(nonzero, nonzero[0], rtol=1e-10)

    def test_dc_zero_never_detected(self):
        """duty_cycle=0.0 in time-averaged mode should give zero concentration."""
        exp = self._make_experiment(duty_cycle=0.0)
        wind = {"wind_speed": 3.0, "wind_direction_deg": 270.0, "stability_class": "D"}
        rng = np.random.default_rng(42)
        _, detected, ppm = exp.simulate_measurement(100.0, 0.0, wind, rng)
        assert ppm == 0.0
        assert not detected

    def test_backward_compatible(self):
        """Experiment without duty_cycle key should behave as dc=1.0."""
        from validation.synthetic_twin import SyntheticExperiment

        src_no_dc = {
            "name": "Test",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "emission_rate": 0.5,
            "equipment_type": "wellhead",
            "age_years": 10,
            "production_rate_mcfd": 1000.0,
            "last_inspection_days": 30,
        }
        src_dc1 = dict(src_no_dc, duty_cycle=1.0)
        wind = {"wind_speed": 3.0, "wind_direction_deg": 270.0, "stability_class": "D"}

        exp_no_dc = SyntheticExperiment(
            ground_truth=[src_no_dc], all_equipment=[src_no_dc],
            wind_params=wind, grid_size=200, resolution=20,
        )
        exp_dc1 = SyntheticExperiment(
            ground_truth=[src_dc1], all_equipment=[src_dc1],
            wind_params=wind, grid_size=200, resolution=20,
        )

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        _, _, ppm1 = exp_no_dc.simulate_measurement(100.0, 0.0, wind, rng1)
        _, _, ppm2 = exp_dc1.simulate_measurement(100.0, 0.0, wind, rng2)
        np.testing.assert_allclose(ppm1, ppm2, rtol=1e-10)


# ===========================================================================
# TestDutyCycleInBayesian
# ===========================================================================

class TestDutyCycleInBayesian:
    """Tests for duty cycle in the Bayesian belief map."""

    def test_avg_emission_weighted_by_dc(self):
        """BayesianBeliefMap._avg_emission should weight by duty_cycle."""
        from models.bayesian import BayesianBeliefMap
        from optimization.opportunity_map import create_grid

        X, Y = create_grid(100, 20)
        prior = np.full_like(X, 0.1)

        sources = [
            {"name": "A", "emission_rate": 1.0, "duty_cycle": 0.5},
            {"name": "B", "emission_rate": 1.0, "duty_cycle": 1.0},
        ]
        bbm = BayesianBeliefMap(X, Y, prior, sources)
        # Expected: mean([1.0*0.5, 1.0*1.0]) = 0.75
        np.testing.assert_allclose(bbm._avg_emission, 0.75, rtol=1e-10)

    def test_avg_emission_defaults_without_dc(self):
        """Without duty_cycle key, avg should use emission_rate * 1.0."""
        from models.bayesian import BayesianBeliefMap
        from optimization.opportunity_map import create_grid

        X, Y = create_grid(100, 20)
        prior = np.full_like(X, 0.1)

        sources = [
            {"name": "A", "emission_rate": 0.8},
            {"name": "B", "emission_rate": 0.4},
        ]
        bbm = BayesianBeliefMap(X, Y, prior, sources)
        np.testing.assert_allclose(bbm._avg_emission, 0.6, rtol=1e-10)

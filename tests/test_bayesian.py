"""Tests for the Bayesian belief map and measurement model."""

import numpy as np
import pytest

from models.measurement import Measurement
from models.bayesian import BayesianBeliefMap
from models.prior import create_spatial_prior


class TestMeasurement:
    """Tests for the Measurement dataclass."""

    def test_valid_measurement(self):
        """Valid measurement should be created without error."""
        m = Measurement(
            x=100.0, y=50.0, concentration_ppm=10.0,
            detected=True, wind_speed=3.0,
            wind_direction_deg=270.0, stability_class="D",
        )
        assert m.x == 100.0
        assert m.detected is True

    def test_negative_concentration_raises(self):
        """Negative concentration should raise ValueError."""
        with pytest.raises(ValueError, match="concentration_ppm"):
            Measurement(
                x=0.0, y=0.0, concentration_ppm=-1.0,
                detected=False, wind_speed=3.0,
                wind_direction_deg=270.0, stability_class="D",
            )

    def test_zero_wind_speed_raises(self):
        """Zero wind speed should raise ValueError."""
        with pytest.raises(ValueError, match="wind_speed"):
            Measurement(
                x=0.0, y=0.0, concentration_ppm=5.0,
                detected=True, wind_speed=0.0,
                wind_direction_deg=270.0, stability_class="D",
            )

    def test_negative_wind_speed_raises(self):
        """Negative wind speed should raise ValueError."""
        with pytest.raises(ValueError, match="wind_speed"):
            Measurement(
                x=0.0, y=0.0, concentration_ppm=5.0,
                detected=True, wind_speed=-2.0,
                wind_direction_deg=270.0, stability_class="D",
            )

    def test_invalid_stability_class_raises(self):
        """Invalid stability class should raise ValueError."""
        with pytest.raises(ValueError, match="stability class"):
            Measurement(
                x=0.0, y=0.0, concentration_ppm=5.0,
                detected=True, wind_speed=3.0,
                wind_direction_deg=270.0, stability_class="Z",
            )

    def test_stability_class_uppercased(self):
        """Lowercase stability class should be uppercased."""
        m = Measurement(
            x=0.0, y=0.0, concentration_ppm=5.0,
            detected=True, wind_speed=3.0,
            wind_direction_deg=270.0, stability_class="d",
        )
        assert m.stability_class == "D"

    def test_zero_concentration_valid(self):
        """Zero concentration should be accepted."""
        m = Measurement(
            x=0.0, y=0.0, concentration_ppm=0.0,
            detected=False, wind_speed=1.0,
            wind_direction_deg=0.0, stability_class="A",
        )
        assert m.concentration_ppm == 0.0


@pytest.fixture
def belief_setup():
    """Create a small grid, prior, and sources for Bayesian testing."""
    half = 100.0
    coords = np.arange(-half, half + 10, 10)
    X, Y = np.meshgrid(coords, coords)

    sources = [
        {"name": "Src1", "x": -50.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5},
        {"name": "Src2", "x": 50.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5},
    ]

    prior_probs = [0.3, 0.3]
    prior = create_spatial_prior(X, Y, sources, prior_probs, kernel_radius=50.0)

    return X, Y, prior, sources


class TestBayesianBeliefMap:
    """Tests for the BayesianBeliefMap class."""

    def test_initial_belief_equals_prior(self, belief_setup):
        """Before any updates, belief should equal the prior."""
        X, Y, prior, sources = belief_setup
        bbm = BayesianBeliefMap(X, Y, prior, sources)
        np.testing.assert_array_equal(bbm.get_belief_map(), prior)

    def test_belief_bounded_after_positive(self, belief_setup):
        """Belief should remain in [0, 1] after a positive detection."""
        X, Y, prior, sources = belief_setup
        bbm = BayesianBeliefMap(X, Y, prior, sources)

        m = Measurement(
            x=50.0, y=20.0, concentration_ppm=15.0,
            detected=True, wind_speed=3.0,
            wind_direction_deg=270.0, stability_class="D",
        )
        bbm.update(m)
        belief = bbm.get_belief_map()
        assert np.all(belief >= 0.0)
        assert np.all(belief <= 1.0)

    def test_belief_bounded_after_negative(self, belief_setup):
        """Belief should remain in [0, 1] after a non-detection."""
        X, Y, prior, sources = belief_setup
        bbm = BayesianBeliefMap(X, Y, prior, sources)

        m = Measurement(
            x=50.0, y=20.0, concentration_ppm=0.5,
            detected=False, wind_speed=3.0,
            wind_direction_deg=270.0, stability_class="D",
        )
        bbm.update(m)
        belief = bbm.get_belief_map()
        assert np.all(belief >= 0.0)
        assert np.all(belief <= 1.0)

    def test_positive_detection_increases_upwind_belief(self, belief_setup):
        """Detection downwind of a source should increase belief near that source."""
        X, Y, prior, sources = belief_setup
        bbm = BayesianBeliefMap(X, Y, prior, sources)

        # Source at (-50, 0), wind from west (270) = blowing east
        # Detection at (0, 0) is downwind of source at (-50, 0)
        m = Measurement(
            x=0.0, y=0.0, concentration_ppm=20.0,
            detected=True, wind_speed=3.0,
            wind_direction_deg=270.0, stability_class="D",
        )
        bbm.update(m)
        belief = bbm.get_belief_map()

        # Belief near the upwind source (-50, 0) should increase
        # Find grid cell closest to (-50, 0)
        center_row = X.shape[0] // 2
        src_col = np.argmin(np.abs(X[0, :] - (-50.0)))
        assert belief[center_row, src_col] > prior[center_row, src_col]

    def test_negative_decreases_belief_at_would_be_detected(self, belief_setup):
        """Non-detection should decrease belief at cells that would have caused detection."""
        X, Y, prior, sources = belief_setup
        bbm = BayesianBeliefMap(X, Y, prior, sources)

        # Non-detection very close to source at (-50, 0)
        # Wind from west: if there were a leak at (-50, 0), sensor at (-40, 0)
        # would be downwind and should detect
        m = Measurement(
            x=-40.0, y=0.0, concentration_ppm=0.0,
            detected=False, wind_speed=3.0,
            wind_direction_deg=270.0, stability_class="D",
        )
        bbm.update(m)
        belief = bbm.get_belief_map()

        # Belief at cells upwind of measurement that would have caused detection
        # should decrease. The source at (-50, 0) is upwind of (-40, 0).
        center_row = X.shape[0] // 2
        src_col = np.argmin(np.abs(X[0, :] - (-50.0)))
        assert belief[center_row, src_col] <= prior[center_row, src_col]

    def test_reset_restores_prior(self, belief_setup):
        """Reset should restore belief to the original prior."""
        X, Y, prior, sources = belief_setup
        bbm = BayesianBeliefMap(X, Y, prior, sources)

        m = Measurement(
            x=0.0, y=0.0, concentration_ppm=10.0,
            detected=True, wind_speed=3.0,
            wind_direction_deg=270.0, stability_class="D",
        )
        bbm.update(m)

        # Belief should have changed
        assert not np.allclose(bbm.get_belief_map(), prior)

        bbm.reset()
        np.testing.assert_array_equal(bbm.get_belief_map(), prior)

    def test_reset_clears_measurements(self, belief_setup):
        """Reset should clear the measurement history."""
        X, Y, prior, sources = belief_setup
        bbm = BayesianBeliefMap(X, Y, prior, sources)

        bbm.update(Measurement(
            x=0.0, y=0.0, concentration_ppm=10.0,
            detected=True, wind_speed=3.0,
            wind_direction_deg=270.0, stability_class="D",
        ))
        assert len(bbm.measurements) == 1

        bbm.reset()
        assert len(bbm.measurements) == 0

    def test_measurement_history_tracking(self, belief_setup):
        """Each update should append to measurement history."""
        X, Y, prior, sources = belief_setup
        bbm = BayesianBeliefMap(X, Y, prior, sources)

        for i in range(3):
            bbm.update(Measurement(
                x=float(i * 10), y=0.0, concentration_ppm=5.0,
                detected=(i % 2 == 0), wind_speed=3.0,
                wind_direction_deg=270.0, stability_class="D",
            ))

        assert len(bbm.measurements) == 3

    def test_multiple_updates_stay_bounded(self, belief_setup):
        """Belief should stay in [0, 1] after many sequential updates."""
        X, Y, prior, sources = belief_setup
        bbm = BayesianBeliefMap(X, Y, prior, sources)

        # Alternate positive and negative detections
        for i in range(10):
            bbm.update(Measurement(
                x=float(i * 5 - 25), y=float(i * 3 - 15),
                concentration_ppm=float(i * 2),
                detected=(i % 2 == 0), wind_speed=3.0,
                wind_direction_deg=270.0, stability_class="D",
            ))

        belief = bbm.get_belief_map()
        assert np.all(belief >= 0.0)
        assert np.all(belief <= 1.0)

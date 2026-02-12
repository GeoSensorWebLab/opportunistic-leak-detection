"""Tests for the detection probability model."""

import numpy as np
import pytest
from models.detection import detection_probability


class TestDetectionProbability:
    """Tests for sigmoid detection probability."""

    def test_at_threshold_gives_half(self):
        """Probability should be exactly 0.5 at the detection threshold."""
        result = detection_probability(np.array([5.0]), threshold_ppm=5.0)
        np.testing.assert_allclose(result[0], 0.5, rtol=1e-10)

    def test_far_below_threshold_near_zero(self):
        """Concentration far below threshold should give P near 0."""
        result = detection_probability(np.array([0.0]), threshold_ppm=5.0)
        assert result[0] < 0.01

    def test_far_above_threshold_near_one(self):
        """Concentration far above threshold should give P near 1."""
        result = detection_probability(np.array([50.0]), threshold_ppm=5.0)
        assert result[0] > 0.99

    def test_monotonic_increase(self):
        """Probability should increase monotonically with concentration."""
        concentrations = np.array([0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0])
        probs = detection_probability(concentrations, threshold_ppm=5.0)
        assert np.all(np.diff(probs) > 0), "Probability should be monotonically increasing"

    def test_output_bounded_zero_one(self):
        """All probabilities should be in [0, 1]."""
        concentrations = np.linspace(-10, 100, 500)
        probs = detection_probability(concentrations, threshold_ppm=5.0)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_steepness_effect(self):
        """Higher steepness should produce sharper transition."""
        c = np.array([4.0])  # Just below threshold
        prob_gentle = detection_probability(c, threshold_ppm=5.0, steepness=0.5)
        prob_steep = detection_probability(c, threshold_ppm=5.0, steepness=5.0)
        # Both should be < 0.5 (below threshold), but gentle should be closer to 0.5
        assert prob_gentle > prob_steep, "Gentler slope should give higher P below threshold"

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        c = np.ones((3, 4, 5)) * 10.0
        probs = detection_probability(c)
        assert probs.shape == (3, 4, 5)

    def test_handles_extreme_values(self):
        """Should not overflow/NaN for very large or very negative concentrations."""
        c = np.array([-1000.0, 0.0, 1000.0])
        probs = detection_probability(c, threshold_ppm=5.0)
        assert not np.any(np.isnan(probs)), "Should not produce NaN"
        assert not np.any(np.isinf(probs)), "Should not produce Inf"
        assert probs[0] < 1e-10  # Very low for far below threshold
        assert probs[2] > 1 - 1e-10  # Very high for far above

    def test_different_threshold(self):
        """Changing threshold should shift the sigmoid."""
        c = np.array([10.0])
        prob_low = detection_probability(c, threshold_ppm=5.0)
        prob_high = detection_probability(c, threshold_ppm=15.0)
        assert prob_low > prob_high, "Higher threshold should give lower P at same concentration"

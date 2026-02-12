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


class TestMinimumDetectionLimit:
    """Tests for the sensor MDL hard cutoff."""

    def test_below_mdl_is_zero(self):
        """Concentrations below MDL should give exactly P = 0."""
        c = np.array([0.0, 0.5, 0.99])
        probs = detection_probability(c, threshold_ppm=5.0, mdl_ppm=1.0)
        np.testing.assert_array_equal(probs, 0.0)

    def test_at_mdl_is_nonzero(self):
        """Concentration exactly at MDL should give P > 0 (sigmoid value)."""
        c = np.array([1.0])
        prob = detection_probability(c, threshold_ppm=5.0, mdl_ppm=1.0)
        assert prob[0] > 0.0, "At MDL, sigmoid should give non-zero probability"

    def test_above_mdl_follows_sigmoid(self):
        """Concentrations above MDL should follow the normal sigmoid curve."""
        c = np.array([5.0])  # At threshold
        prob_with_mdl = detection_probability(c, threshold_ppm=5.0, mdl_ppm=1.0)
        prob_no_mdl = detection_probability(c, threshold_ppm=5.0, mdl_ppm=0.0)
        np.testing.assert_allclose(prob_with_mdl, prob_no_mdl, rtol=1e-10,
                                    err_msg="Above MDL, behavior should match sigmoid")

    def test_mdl_zero_disables_cutoff(self):
        """MDL = 0 should produce same results as original sigmoid."""
        c = np.array([0.1, 0.5, 1.0, 3.0, 5.0, 10.0])
        prob_mdl_zero = detection_probability(c, threshold_ppm=5.0, mdl_ppm=0.0)
        # With mdl=0, no cutoff applied — all values follow sigmoid
        assert np.all(prob_mdl_zero > 0), "MDL=0 should not zero out anything"

    def test_mdl_creates_dead_zone(self):
        """MDL should create a clear dead zone in the detection map."""
        c = np.linspace(0, 10, 100)
        probs = detection_probability(c, threshold_ppm=5.0, mdl_ppm=2.0)
        # All values below 2 ppm should be exactly zero
        below_mdl = c < 2.0
        above_mdl = c >= 2.0
        assert np.all(probs[below_mdl] == 0.0), "Below MDL should be exactly zero"
        assert np.all(probs[above_mdl] > 0.0), "Above MDL should be positive"

    def test_higher_mdl_reduces_detectable_area(self):
        """Higher MDL should result in fewer detectable cells."""
        c = np.linspace(0, 20, 200)
        probs_low_mdl = detection_probability(c, threshold_ppm=5.0, mdl_ppm=0.5)
        probs_high_mdl = detection_probability(c, threshold_ppm=5.0, mdl_ppm=3.0)
        detectable_low = np.sum(probs_low_mdl > 0)
        detectable_high = np.sum(probs_high_mdl > 0)
        assert detectable_low > detectable_high, "Higher MDL should shrink detectable area"

    def test_mdl_preserves_shape(self):
        """Output shape should match input regardless of MDL."""
        c = np.ones((3, 4)) * 2.0
        probs = detection_probability(c, mdl_ppm=1.0)
        assert probs.shape == (3, 4)

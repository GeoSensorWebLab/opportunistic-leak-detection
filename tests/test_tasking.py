"""Tests for the tasking optimizer."""

import numpy as np
import pytest
from optimization.tasking import (
    compute_path_deviation,
    compute_tasking_scores,
    recommend_waypoints,
    build_optimized_path,
    _project_onto_path,
)


class TestComputePathDeviation:
    """Tests for path deviation computation."""

    def test_on_path_deviation_near_zero(self, small_grid, simple_path):
        """Points on the path should have near-zero deviation."""
        X, Y = small_grid
        deviation = compute_path_deviation(X, Y, simple_path)
        # Check cell closest to path (y=0 axis)
        center_row = deviation.shape[0] // 2
        # The center row corresponds to y=0, which is the path
        assert deviation[center_row, :].min() < 10.0  # Within one cell of path

    def test_deviation_increases_away_from_path(self, small_grid, simple_path):
        """Deviation should increase as we move away from the path."""
        X, Y = small_grid
        deviation = compute_path_deviation(X, Y, simple_path)
        center_row = deviation.shape[0] // 2
        # Moving up (increasing row index = positive y) from the path
        if center_row + 3 < deviation.shape[0]:
            assert deviation[center_row + 3, 10] > deviation[center_row + 1, 10]

    def test_deviation_non_negative(self, small_grid, simple_path):
        """All deviations should be non-negative."""
        X, Y = small_grid
        deviation = compute_path_deviation(X, Y, simple_path)
        assert np.all(deviation >= 0)

    def test_deviation_shape_matches_grid(self, small_grid, simple_path):
        """Output shape should match grid shape."""
        X, Y = small_grid
        deviation = compute_path_deviation(X, Y, simple_path)
        assert deviation.shape == X.shape


class TestComputeTaskingScores:
    """Tests for tasking score computation."""

    def test_scores_non_negative(self, small_grid, simple_path):
        """All scores should be non-negative."""
        X, Y = small_grid
        det_prob = np.random.rand(*X.shape) * 0.5
        scores = compute_tasking_scores(X, Y, det_prob, simple_path)
        assert np.all(scores >= 0)

    def test_zero_detection_gives_zero_score(self, small_grid, simple_path):
        """Zero detection probability should give zero score."""
        X, Y = small_grid
        det_prob = np.zeros_like(X)
        scores = compute_tasking_scores(X, Y, det_prob, simple_path)
        assert np.allclose(scores, 0.0)

    def test_beyond_max_deviation_is_zero(self, small_grid, simple_path):
        """Cells beyond max_deviation should have zero score."""
        X, Y = small_grid
        det_prob = np.ones_like(X)  # Perfect detection everywhere
        scores = compute_tasking_scores(
            X, Y, det_prob, simple_path, max_deviation=30.0,
        )
        deviation = compute_path_deviation(X, Y, simple_path)
        far_mask = deviation > 30.0
        assert np.allclose(scores[far_mask], 0.0)

    def test_on_path_scores_higher(self, small_grid, simple_path):
        """On-path cells should score higher than off-path cells at same detection prob."""
        X, Y = small_grid
        det_prob = np.ones_like(X) * 0.5  # Uniform detection
        scores = compute_tasking_scores(X, Y, det_prob, simple_path, max_deviation=200.0)
        deviation = compute_path_deviation(X, Y, simple_path)
        near_mask = deviation < 15
        far_mask = (deviation > 50) & (deviation < 200)
        if np.any(near_mask) and np.any(far_mask):
            assert scores[near_mask].mean() > scores[far_mask].mean()

    def test_precomputed_deviation(self, small_grid, simple_path):
        """Precomputed deviation should give same results as computed."""
        X, Y = small_grid
        det_prob = np.random.rand(*X.shape)
        scores_auto = compute_tasking_scores(X, Y, det_prob, simple_path)
        deviation = compute_path_deviation(X, Y, simple_path)
        scores_pre = compute_tasking_scores(
            X, Y, det_prob, simple_path, precomputed_deviation=deviation,
        )
        np.testing.assert_array_equal(scores_auto, scores_pre)


class TestRecommendWaypoints:
    """Tests for waypoint recommendation with non-maximum suppression."""

    def _make_scored_grid(self):
        """Create a small grid with known score peaks."""
        coords = np.arange(-100, 110, 10)
        X, Y = np.meshgrid(coords, coords)
        scores = np.zeros_like(X, dtype=float)
        det_prob = np.zeros_like(X, dtype=float)
        conc = np.zeros_like(X, dtype=float)

        # Place two peaks far apart
        scores[5, 5] = 1.0   # at (-50, -50)
        scores[15, 15] = 0.8  # at (50, 50)
        det_prob[5, 5] = 0.9
        det_prob[15, 15] = 0.7
        conc[5, 5] = 20.0
        conc[15, 15] = 15.0

        return X, Y, scores, det_prob, conc

    def test_returns_top_k(self):
        """Should return at most top_k waypoints."""
        X, Y, scores, det_prob, conc = self._make_scored_grid()
        recs = recommend_waypoints(X, Y, scores, det_prob, conc, top_k=2)
        assert len(recs) <= 2

    def test_returns_sorted_by_score(self):
        """Recommendations should be in descending score order."""
        X, Y, scores, det_prob, conc = self._make_scored_grid()
        recs = recommend_waypoints(X, Y, scores, det_prob, conc, top_k=5)
        if len(recs) >= 2:
            assert recs[0]["score"] >= recs[1]["score"]

    def test_min_separation_enforced(self):
        """Waypoints closer than min_separation should be suppressed."""
        coords = np.arange(-100, 110, 10)
        X, Y = np.meshgrid(coords, coords)
        scores = np.zeros_like(X, dtype=float)
        det_prob = np.ones_like(X, dtype=float)
        conc = np.ones_like(X, dtype=float) * 10.0

        # Place two high-score peaks only 10m apart
        scores[10, 10] = 1.0
        scores[10, 11] = 0.9  # Only 10m away
        recs = recommend_waypoints(X, Y, scores, det_prob, conc, top_k=5, min_separation=50.0)
        assert len(recs) == 1, "Close peaks should be suppressed"

    def test_no_scores_returns_empty(self):
        """All-zero scores should return empty list."""
        coords = np.arange(-50, 60, 10)
        X, Y = np.meshgrid(coords, coords)
        scores = np.zeros_like(X)
        det_prob = np.zeros_like(X)
        conc = np.zeros_like(X)
        recs = recommend_waypoints(X, Y, scores, det_prob, conc, top_k=5)
        assert len(recs) == 0

    def test_recommendation_dict_keys(self):
        """Each recommendation should have expected keys."""
        X, Y, scores, det_prob, conc = self._make_scored_grid()
        recs = recommend_waypoints(X, Y, scores, det_prob, conc, top_k=1)
        assert len(recs) == 1
        rec = recs[0]
        assert "x" in rec
        assert "y" in rec
        assert "score" in rec
        assert "detection_prob" in rec
        assert "concentration_ppm" in rec


class TestProjectOntoPath:
    """Tests for point-to-path projection."""

    def test_project_point_on_segment(self):
        """A point directly on a segment should project to itself."""
        path = np.array([[0, 0], [100, 0]])
        point = np.array([50, 0])
        seg_idx, t, closest, cum_dist = _project_onto_path(point, path)
        np.testing.assert_allclose(closest, [50, 0], atol=1e-10)
        assert abs(t - 0.5) < 1e-10

    def test_project_perpendicular_point(self):
        """A point perpendicular to a segment should project to the foot."""
        path = np.array([[0, 0], [100, 0]])
        point = np.array([50, 30])
        seg_idx, t, closest, cum_dist = _project_onto_path(point, path)
        np.testing.assert_allclose(closest, [50, 0], atol=1e-10)

    def test_project_beyond_segment_end(self):
        """A point past the endpoint should clamp to the endpoint."""
        path = np.array([[0, 0], [100, 0]])
        point = np.array([150, 0])
        seg_idx, t, closest, cum_dist = _project_onto_path(point, path)
        np.testing.assert_allclose(closest, [100, 0], atol=1e-10)
        assert abs(t - 1.0) < 1e-10


class TestBuildOptimizedPath:
    """Tests for path optimization."""

    def test_no_waypoints_returns_baseline(self, simple_path):
        """No waypoints should return the baseline unchanged."""
        result = build_optimized_path(simple_path, [])
        np.testing.assert_array_equal(result, simple_path)

    def test_includes_waypoints(self, simple_path):
        """Optimized path should include all waypoint coordinates."""
        waypoints = [{"x": 25.0, "y": 30.0, "score": 1.0}]
        result = build_optimized_path(simple_path, waypoints)
        # The waypoint should appear somewhere in the result
        found = any(
            np.allclose(result[i], [25.0, 30.0], atol=1e-10)
            for i in range(len(result))
        )
        assert found, "Waypoint should be in the optimized path"

    def test_optimized_longer_than_baseline(self, simple_path):
        """Adding a detour waypoint should make the path longer."""
        waypoints = [{"x": 0.0, "y": 50.0, "score": 1.0}]
        result = build_optimized_path(simple_path, waypoints)
        assert len(result) > len(simple_path)

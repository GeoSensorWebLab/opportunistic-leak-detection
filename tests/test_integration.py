"""End-to-end integration tests for the full pipeline."""

import numpy as np
import pytest
from data.mock_data import (
    get_leak_sources,
    get_baseline_path,
    get_road_following_path,
    get_inspection_targets,
)
from optimization.opportunity_map import compute_opportunity_map
from optimization.tasking import (
    compute_path_deviation,
    compute_tasking_scores,
    recommend_waypoints,
    build_optimized_path,
)
from optimization.metrics import compute_route_metrics, find_nearest_source


class TestFullPipeline:
    """Test the complete pipeline: sources -> opportunity map -> scoring -> path."""

    def test_end_to_end_produces_recommendations(self):
        """Full pipeline should produce at least one recommendation."""
        sources = get_leak_sources()
        baseline_path = get_baseline_path()

        X, Y, conc_ppm, det_prob = compute_opportunity_map(
            sources=sources,
            wind_speed=3.0,
            wind_direction_deg=270.0,
            stability_class="D",
            grid_size=1000,
            resolution=10,
        )

        scores = compute_tasking_scores(X, Y, det_prob, baseline_path)
        recs = recommend_waypoints(X, Y, scores, det_prob, conc_ppm, top_k=5)

        assert len(recs) > 0, "Should produce at least one recommendation"
        assert all(r["score"] > 0 for r in recs)
        assert all(0 <= r["detection_prob"] <= 1 for r in recs)

    def test_optimized_path_is_valid(self):
        """Optimized path should be a valid numpy array with at least as many points."""
        sources = get_leak_sources()
        baseline_path = get_baseline_path()

        X, Y, conc_ppm, det_prob = compute_opportunity_map(
            sources=sources,
            wind_speed=3.0,
            wind_direction_deg=270.0,
            stability_class="D",
            grid_size=1000,
            resolution=10,
        )

        scores = compute_tasking_scores(X, Y, det_prob, baseline_path)
        recs = recommend_waypoints(X, Y, scores, det_prob, conc_ppm, top_k=5)
        optimized = build_optimized_path(baseline_path, recs)

        assert isinstance(optimized, np.ndarray)
        assert optimized.shape[1] == 2
        assert len(optimized) >= len(baseline_path)

    def test_different_wind_gives_different_results(self):
        """Different wind directions should produce different recommendations."""
        sources = get_leak_sources()
        baseline_path = get_baseline_path()

        def run_pipeline(wind_dir):
            X, Y, conc_ppm, det_prob = compute_opportunity_map(
                sources=sources,
                wind_speed=3.0,
                wind_direction_deg=wind_dir,
                stability_class="D",
                grid_size=1000,
                resolution=10,
            )
            scores = compute_tasking_scores(X, Y, det_prob, baseline_path)
            return recommend_waypoints(X, Y, scores, det_prob, conc_ppm, top_k=3)

        recs_west = run_pipeline(270)  # Wind from west
        recs_north = run_pipeline(0)   # Wind from north

        if recs_west and recs_north:
            # At least the top recommendation should differ
            loc_west = (recs_west[0]["x"], recs_west[0]["y"])
            loc_north = (recs_north[0]["x"], recs_north[0]["y"])
            dist = np.hypot(loc_west[0] - loc_north[0], loc_west[1] - loc_north[1])
            assert dist > 10, "Different wind should produce different recommendations"

    def test_pipeline_with_all_stability_classes(self):
        """Pipeline should work for all stability classes without error."""
        sources = get_leak_sources()
        baseline_path = get_baseline_path()

        for stability in ["A", "B", "C", "D", "E", "F"]:
            X, Y, conc_ppm, det_prob = compute_opportunity_map(
                sources=sources,
                wind_speed=3.0,
                wind_direction_deg=270.0,
                stability_class=stability,
                grid_size=200,
                resolution=20,
            )
            scores = compute_tasking_scores(X, Y, det_prob, baseline_path)
            recs = recommend_waypoints(X, Y, scores, det_prob, conc_ppm, top_k=3)
            # Just verify no errors — some classes may produce 0 recommendations
            assert isinstance(recs, list)


class TestPaths:
    """Test path data functions."""

    def test_inspection_targets_has_five_stops(self):
        """Inspection targets should return exactly 5 mandatory stops."""
        targets = get_inspection_targets()
        assert len(targets) == 5
        assert all("name" in t and "x" in t and "y" in t for t in targets)

    def test_baseline_path_visits_all_targets(self):
        """Free-walking baseline path should pass near all inspection targets."""
        path = get_baseline_path()
        targets = get_inspection_targets()

        for target in targets:
            dists = np.sqrt(
                (path[:, 0] - target["x"]) ** 2
                + (path[:, 1] - target["y"]) ** 2
            )
            assert dists.min() < 1.0, (
                f"Baseline path should pass within 1m of {target['name']}, "
                f"but closest is {dists.min():.1f}m"
            )

    def test_road_following_path_is_longer(self):
        """Road-following path should be longer than the free-walking path."""
        free = get_baseline_path()
        road = get_road_following_path()

        def path_len(p):
            return float(np.sum(np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))))

        assert path_len(road) > path_len(free), (
            "Road-following path should be longer than free-walking path"
        )

    def test_baseline_path_shape(self):
        """Baseline path should be a 2D array with (N, 2) shape."""
        path = get_baseline_path()
        assert isinstance(path, np.ndarray)
        assert path.ndim == 2
        assert path.shape[1] == 2
        assert len(path) >= 8  # at least start, end, 5 targets + midpoints

    def test_road_following_path_shape(self):
        """Road-following path should be a 2D array."""
        path = get_road_following_path()
        assert isinstance(path, np.ndarray)
        assert path.ndim == 2
        assert path.shape[1] == 2


class TestRouteMetrics:
    """Test route metrics computation."""

    def test_metrics_keys(self):
        """compute_route_metrics should return all expected keys."""
        baseline = np.array([[0, 0], [100, 0], [200, 0]], dtype=float)
        optimized = np.array([[0, 0], [50, 50], [100, 0], [200, 0]], dtype=float)
        recs = [{"x": 50, "y": 50, "detection_prob": 0.8, "score": 0.5, "concentration_ppm": 10}]

        m = compute_route_metrics(baseline, optimized, recs)

        expected_keys = {
            "baseline_distance_m",
            "optimized_distance_m",
            "added_detour_m",
            "added_detour_pct",
            "time_impact_min",
            "num_detour_points",
            "avg_detection_prob",
        }
        assert set(m.keys()) == expected_keys

    def test_baseline_distance_correct(self):
        """Baseline distance should be computed correctly."""
        baseline = np.array([[0, 0], [100, 0]], dtype=float)
        m = compute_route_metrics(baseline, baseline, [])
        assert abs(m["baseline_distance_m"] - 100.0) < 0.01

    def test_detour_adds_distance(self):
        """Optimized path with detour should be longer than baseline."""
        baseline = np.array([[0, 0], [100, 0]], dtype=float)
        optimized = np.array([[0, 0], [50, 50], [100, 0]], dtype=float)
        recs = [{"x": 50, "y": 50, "detection_prob": 0.9, "score": 1.0, "concentration_ppm": 20}]

        m = compute_route_metrics(baseline, optimized, recs)

        assert m["optimized_distance_m"] > m["baseline_distance_m"]
        assert m["added_detour_m"] > 0
        assert m["added_detour_pct"] > 0
        assert m["time_impact_min"] > 0

    def test_no_recommendations(self):
        """With no recommendations, metrics should show zero detour."""
        path = np.array([[0, 0], [100, 0]], dtype=float)
        m = compute_route_metrics(path, path, [])

        assert m["added_detour_m"] == 0.0
        assert m["num_detour_points"] == 0
        assert m["avg_detection_prob"] == 0.0

    def test_find_nearest_source(self):
        """Should return the name of the closest source."""
        sources = [
            {"name": "A", "x": 0, "y": 0},
            {"name": "B", "x": 100, "y": 0},
        ]
        rec = {"x": 90, "y": 0}
        assert find_nearest_source(rec, sources) == "B"

    def test_find_nearest_source_empty(self):
        """Should return None for empty sources list."""
        assert find_nearest_source({"x": 0, "y": 0}, []) is None

"""End-to-end integration tests for the full pipeline."""

import numpy as np
import pytest
from data.mock_data import get_leak_sources, get_baseline_path
from optimization.opportunity_map import compute_opportunity_map
from optimization.tasking import (
    compute_path_deviation,
    compute_tasking_scores,
    recommend_waypoints,
    build_optimized_path,
)


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

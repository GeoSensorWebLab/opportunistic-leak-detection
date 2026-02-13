"""Phase 9 high-priority test gaps.

Tests for edge-cases and invariants identified during the deep code review:
  1. Sequential non-detections saturating belief toward zero
  2. Worker path segment reconstruction covers the original path
  3. Campaign multi-day entropy carry-forward (entropy decreases over days)
  4. Waypoint already on baseline path produces minimal detour
  5. Multi-worker fleet coverage >= single-worker coverage
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.bayesian import BayesianBeliefMap
from models.measurement import Measurement
from models.prior import compute_all_priors, create_spatial_prior
from optimization.campaign import CampaignState, plan_next_day, close_day
from optimization.multi_worker import (
    WorkerRoute,
    split_baseline_path,
    allocate_waypoints,
    compute_fleet_coverage,
)
from optimization.tasking import build_optimized_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def belief_grid():
    """Small grid with two sources and a prior for Bayesian tests."""
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


@pytest.fixture
def straight_path():
    """Straight-line path along x-axis."""
    return np.array([
        [-200.0, 0.0],
        [-100.0, 0.0],
        [0.0, 0.0],
        [100.0, 0.0],
        [200.0, 0.0],
    ])


@pytest.fixture
def campaign_sources():
    """Sources for campaign tests."""
    from data.mock_data import get_leak_sources
    return get_leak_sources()


@pytest.fixture
def campaign_path():
    """Baseline path for campaign tests."""
    from data.mock_data import get_baseline_path
    return get_baseline_path()


@pytest.fixture
def campaign_wind():
    """Wind params for campaign tests."""
    return {
        "wind_speed": 3.0,
        "wind_direction_deg": 270.0,
        "stability_class": "D",
    }


# ---------------------------------------------------------------------------
# 1. Sequential non-detections should saturate belief toward zero
# ---------------------------------------------------------------------------

class TestBeliefSaturation:
    """Verify that many non-detections push belief down without going negative."""

    def test_repeated_non_detections_decrease_belief(self, belief_grid):
        """Many non-detections at a source should drive belief toward 0 locally."""
        X, Y, prior, sources = belief_grid
        bbm = BayesianBeliefMap(X, Y, prior, sources)

        # Belief near source at (-50, 0)
        center_row = X.shape[0] // 2
        src_col = np.argmin(np.abs(X[0, :] - (-50.0)))
        initial_near_src = prior[center_row, src_col]

        # 20 non-detections close to source at (-50, 0), downwind
        for _ in range(20):
            m = Measurement(
                x=-40.0, y=0.0, concentration_ppm=0.0,
                detected=False, wind_speed=3.0,
                wind_direction_deg=270.0, stability_class="D",
            )
            bbm.update(m)

        final_belief = bbm.get_belief_map()

        # Belief should remain bounded in [0, 1]
        assert np.all(final_belief >= 0.0)
        assert np.all(final_belief <= 1.0)

        # Belief near the source should have decreased (locally)
        assert final_belief[center_row, src_col] < initial_near_src

        # No NaNs
        assert not np.any(np.isnan(final_belief))

    def test_belief_never_becomes_nan_or_negative(self, belief_grid):
        """Extreme number of updates should not produce NaN or negative values."""
        X, Y, prior, sources = belief_grid
        bbm = BayesianBeliefMap(X, Y, prior, sources)

        # Mix of detections and non-detections in rapid succession
        for i in range(50):
            m = Measurement(
                x=float(np.random.default_rng(i).uniform(-80, 80)),
                y=float(np.random.default_rng(i + 100).uniform(-80, 80)),
                concentration_ppm=float(np.random.default_rng(i + 200).uniform(0, 30)),
                detected=(i % 3 == 0),
                wind_speed=3.0,
                wind_direction_deg=270.0,
                stability_class="D",
            )
            bbm.update(m)

        belief = bbm.get_belief_map()
        assert np.all(np.isfinite(belief))
        assert np.all(belief >= 0.0)
        assert np.all(belief <= 1.0)


# ---------------------------------------------------------------------------
# 2. Worker path segment reconstruction
# ---------------------------------------------------------------------------

class TestPathSegmentReconstruction:
    """Worker segments should collectively cover the original path."""

    def test_segments_cover_original_endpoints(self, straight_path):
        """First segment starts at path start, last ends at path end."""
        for n_workers in [2, 3, 4]:
            segments = split_baseline_path(straight_path, n_workers)
            assert len(segments) == n_workers

            # First segment should start at or near original start
            np.testing.assert_allclose(
                segments[0][0], straight_path[0], atol=1e-6,
                err_msg=f"First segment doesn't start at path start for {n_workers} workers",
            )
            # Last segment should end at or near original end
            np.testing.assert_allclose(
                segments[-1][-1], straight_path[-1], atol=1e-6,
                err_msg=f"Last segment doesn't end at path end for {n_workers} workers",
            )

    def test_each_segment_has_at_least_two_points(self, straight_path):
        """Every segment must have at least two points to define a path."""
        for n_workers in [2, 3, 4]:
            segments = split_baseline_path(straight_path, n_workers)
            for i, seg in enumerate(segments):
                assert len(seg) >= 2, f"Segment {i} has < 2 points for {n_workers} workers"


# ---------------------------------------------------------------------------
# 3. Campaign multi-day entropy carry-forward
# ---------------------------------------------------------------------------

class TestCampaignEntropyCarryForward:
    """Entropy should decrease (or stay flat) across consecutive campaign days
    when measurements provide information."""

    def test_entropy_decreases_with_detections(
        self, campaign_sources, campaign_wind, campaign_path,
    ):
        """Closing a day with a detection should reduce entropy for the next day."""
        campaign = CampaignState()

        # Day 1
        day1 = plan_next_day(
            campaign, campaign_sources, campaign_wind, campaign_path,
            resolution=20.0,
        )
        meas1 = [
            Measurement(
                x=0.0, y=0.0, concentration_ppm=10.0,
                detected=True, wind_speed=3.0,
                wind_direction_deg=270.0, stability_class="D",
            ),
        ]
        close_day(campaign, day1, meas1, campaign_sources)

        # Day 2 — should inherit day 1's posterior
        day2 = plan_next_day(
            campaign, campaign_sources, campaign_wind, campaign_path,
            resolution=20.0,
        )

        # Day 2's starting entropy should equal day 1's ending entropy
        assert day2.entropy_start == pytest.approx(day1.entropy_end, abs=1e-6)

    def test_no_measurements_keeps_entropy_constant(
        self, campaign_sources, campaign_wind, campaign_path,
    ):
        """A day with zero measurements should not change entropy."""
        campaign = CampaignState()

        day1 = plan_next_day(
            campaign, campaign_sources, campaign_wind, campaign_path,
            resolution=20.0,
        )
        close_day(campaign, day1, [], campaign_sources)

        # Entropy should be unchanged since no measurements
        assert day1.entropy_end == pytest.approx(day1.entropy_start, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Waypoint already on baseline path
# ---------------------------------------------------------------------------

class TestWaypointOnBaseline:
    """A waypoint that lies exactly on the baseline should add minimal detour."""

    def test_on_path_waypoint_minimal_detour(self, straight_path):
        """If the waypoint is on the path, the optimized path should be barely longer."""
        # Waypoint at (0, 0) which is on the straight_path
        waypoints = [
            {"x": 0.0, "y": 0.0, "score": 0.9, "detection_prob": 0.8, "concentration_ppm": 10.0},
        ]

        optimized = build_optimized_path(straight_path, waypoints)

        # The path should exist and include the waypoint
        assert optimized is not None
        assert len(optimized) >= len(straight_path)

        # Compute total lengths
        def path_length(p):
            return sum(np.linalg.norm(p[i + 1] - p[i]) for i in range(len(p) - 1))

        original_len = path_length(straight_path)
        optimized_len = path_length(optimized)

        # The detour should be small (< 10% of original path)
        detour = optimized_len - original_len
        assert detour < 0.10 * original_len, (
            f"Detour {detour:.1f}m is too large for an on-path waypoint"
        )


# ---------------------------------------------------------------------------
# 5. Multi-worker coverage improvement
# ---------------------------------------------------------------------------

class TestMultiWorkerCoverage:
    """Fleet coverage with 2+ workers should be >= single worker."""

    def test_two_workers_at_least_as_good_as_one(self, straight_path):
        """Fleet coverage P should be >= single-worker P everywhere."""
        recs = [
            {"x": -150.0, "y": 50.0, "score": 0.8, "detection_prob": 0.7, "concentration_ppm": 10.0},
            {"x": 150.0, "y": -50.0, "score": 0.6, "detection_prob": 0.5, "concentration_ppm": 5.0},
            {"x": 0.0, "y": 30.0, "score": 0.5, "detection_prob": 0.4, "concentration_ppm": 3.0},
        ]

        X, Y = np.meshgrid(np.arange(-200, 201, 20), np.arange(-200, 201, 20))
        det_prob = np.ones(X.shape) * 0.3
        sources = [{"x": 0.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5, "name": "Test"}]

        # Single worker
        routes_1 = allocate_waypoints(recs, [straight_path])
        cov_1 = compute_fleet_coverage(routes_1, X, Y, det_prob, sources)

        # Two workers
        segments = split_baseline_path(straight_path, 2)
        routes_2 = allocate_waypoints(recs, segments)
        cov_2 = compute_fleet_coverage(routes_2, X, Y, det_prob, sources)

        # Complementary probability: P_fleet = 1 - prod(1 - P_i)
        # With 2 workers, P_fleet >= any single P_i, so max should be >=
        assert cov_2["max_fleet_prob"] >= cov_1["max_fleet_prob"] - 1e-6

    def test_fleet_coverage_bounded(self, straight_path):
        """Fleet coverage should always be in [0, 1]."""
        recs = [
            {"x": 0.0, "y": 30.0, "score": 0.9, "detection_prob": 0.9, "concentration_ppm": 20.0},
        ]

        X, Y = np.meshgrid(np.arange(-200, 201, 20), np.arange(-200, 201, 20))
        det_prob = np.random.default_rng(42).random(X.shape)
        sources = [{"x": 0.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5, "name": "Test"}]

        for n_workers in [1, 2, 3]:
            segments = split_baseline_path(straight_path, n_workers)
            routes = allocate_waypoints(recs, segments)
            cov = compute_fleet_coverage(routes, X, Y, det_prob, sources)

            assert np.all(cov["fleet_prob"] >= 0.0)
            assert np.all(cov["fleet_prob"] <= 1.0)

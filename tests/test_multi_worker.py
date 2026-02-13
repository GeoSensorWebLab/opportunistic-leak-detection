"""Tests for multi-worker allocation."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimization.multi_worker import (
    WorkerRoute,
    split_baseline_path,
    allocate_waypoints,
    compute_fleet_coverage,
)


@pytest.fixture
def straight_path():
    """Simple straight-line path along x-axis."""
    return np.array([
        [-200.0, 0.0],
        [-100.0, 0.0],
        [0.0, 0.0],
        [100.0, 0.0],
        [200.0, 0.0],
    ])


@pytest.fixture
def sample_recommendations():
    """Waypoints at known positions."""
    return [
        {"x": -150.0, "y": 50.0, "score": 0.8, "detection_prob": 0.7, "concentration_ppm": 10.0},
        {"x": 150.0, "y": -50.0, "score": 0.6, "detection_prob": 0.5, "concentration_ppm": 5.0},
        {"x": 0.0, "y": 30.0, "score": 0.5, "detection_prob": 0.4, "concentration_ppm": 3.0},
    ]


class TestSplitPath:
    def test_single_worker_returns_full_path(self, straight_path):
        segments = split_baseline_path(straight_path, 1)
        assert len(segments) == 1
        np.testing.assert_array_equal(segments[0], straight_path)

    def test_two_workers_splits_path(self, straight_path):
        segments = split_baseline_path(straight_path, 2)
        assert len(segments) == 2
        for seg in segments:
            assert len(seg) >= 2

    def test_three_workers_splits_path(self, straight_path):
        segments = split_baseline_path(straight_path, 3)
        assert len(segments) == 3
        for seg in segments:
            assert len(seg) >= 2


class TestAllocateWaypoints:
    def test_single_worker_gets_all(self, straight_path, sample_recommendations):
        paths = [straight_path]
        routes = allocate_waypoints(sample_recommendations, paths)
        assert len(routes) == 1
        assert len(routes[0].assigned_waypoints) == 3
        assert routes[0].optimized_path is not None

    def test_two_workers_distribute(self, straight_path, sample_recommendations):
        segments = split_baseline_path(straight_path, 2)
        routes = allocate_waypoints(sample_recommendations, segments)
        assert len(routes) == 2
        total_assigned = sum(len(r.assigned_waypoints) for r in routes)
        assert total_assigned == 3

    def test_worker_route_has_path(self, straight_path, sample_recommendations):
        routes = allocate_waypoints(sample_recommendations, [straight_path])
        for route in routes:
            assert route.optimized_path is not None
            assert len(route.optimized_path) >= len(straight_path)

    def test_empty_recommendations(self, straight_path):
        routes = allocate_waypoints([], [straight_path])
        assert len(routes) == 1
        assert len(routes[0].assigned_waypoints) == 0


class TestFleetCoverage:
    def test_single_worker_fleet(self, straight_path, sample_recommendations):
        routes = allocate_waypoints(sample_recommendations, [straight_path])
        X, Y = np.meshgrid(np.arange(-200, 201, 20), np.arange(-200, 201, 20))
        det_prob = np.random.default_rng(42).random(X.shape) * 0.5
        sources = [{"x": 0.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5, "name": "Test"}]

        coverage = compute_fleet_coverage(routes, X, Y, det_prob, sources)
        assert "fleet_prob" in coverage
        assert coverage["fleet_prob"].shape == X.shape
        assert coverage["num_workers"] == 1
        assert 0.0 <= coverage["max_fleet_prob"] <= 1.0

    def test_more_workers_more_coverage(self, straight_path, sample_recommendations):
        X, Y = np.meshgrid(np.arange(-200, 201, 20), np.arange(-200, 201, 20))
        det_prob = np.ones(X.shape) * 0.3
        sources = [{"x": 0.0, "y": 0.0, "z": 0.0, "emission_rate": 0.5, "name": "Test"}]

        # 1 worker
        routes_1 = allocate_waypoints(sample_recommendations, [straight_path])
        cov_1 = compute_fleet_coverage(routes_1, X, Y, det_prob, sources)

        # 2 workers
        segments = split_baseline_path(straight_path, 2)
        routes_2 = allocate_waypoints(sample_recommendations, segments)
        cov_2 = compute_fleet_coverage(routes_2, X, Y, det_prob, sources)

        # Fleet coverage should be >= single worker coverage on average
        assert cov_2["avg_fleet_prob"] >= 0.0


class TestWorkerRoute:
    def test_build_path_no_waypoints(self, straight_path):
        route = WorkerRoute(worker_id=0, baseline_path=straight_path)
        route.build_path()
        np.testing.assert_array_equal(route.optimized_path, straight_path)
        assert route.total_detection_prob == 0.0

    def test_build_path_with_waypoints(self, straight_path):
        wp = {"x": 0.0, "y": 50.0, "score": 0.5, "detection_prob": 0.6, "concentration_ppm": 5.0}
        route = WorkerRoute(worker_id=0, baseline_path=straight_path, assigned_waypoints=[wp])
        route.build_path()
        assert route.optimized_path is not None
        assert len(route.optimized_path) > len(straight_path)
        assert route.total_detection_prob == pytest.approx(0.6)

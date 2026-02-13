"""
Multi-Worker Allocation for fleet-based opportunistic leak detection.

Assigns recommended waypoints to multiple workers based on path deviation,
then builds an optimized path for each worker. Also computes fleet-level
complementary detection probability.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.spatial.distance import cdist

from optimization.tasking import build_optimized_path, compute_path_deviation
from config import MAX_DEVIATION_M


@dataclass
class WorkerRoute:
    """Route assignment for a single field worker."""

    worker_id: int
    baseline_path: np.ndarray
    assigned_waypoints: List[dict] = field(default_factory=list)
    optimized_path: Optional[np.ndarray] = None
    total_detection_prob: float = 0.0

    def build_path(self) -> None:
        """Build the optimized path from baseline + assigned waypoints."""
        self.optimized_path = build_optimized_path(
            self.baseline_path, self.assigned_waypoints,
        )
        if self.assigned_waypoints:
            self.total_detection_prob = float(np.mean(
                [w["detection_prob"] for w in self.assigned_waypoints]
            ))


def split_baseline_path(
    baseline_path: np.ndarray,
    num_workers: int,
) -> List[np.ndarray]:
    """Split a baseline path into roughly equal segments for each worker.

    Each segment shares overlap at the split point so workers have
    continuous paths.

    Args:
        baseline_path: (N, 2) array of waypoints.
        num_workers: Number of workers to split across.

    Returns:
        List of (M_i, 2) arrays, one per worker.
    """
    n = len(baseline_path)
    if num_workers <= 1 or n < 2:
        return [baseline_path.copy()]

    # Compute cumulative distance along path
    diffs = np.diff(baseline_path, axis=0)
    seg_lens = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_dist = cum_dist[-1]

    segments = []
    for w in range(num_workers):
        start_dist = w * total_dist / num_workers
        end_dist = (w + 1) * total_dist / num_workers

        # Find indices bounding this segment (use consistent side="right" for both)
        start_idx = max(0, int(np.searchsorted(cum_dist, start_dist, side="right")) - 1)
        end_idx = min(n - 1, int(np.searchsorted(cum_dist, end_dist, side="right")))

        segment = baseline_path[start_idx:end_idx + 1].copy()
        if len(segment) < 2:
            segment = baseline_path[max(0, start_idx - 1):end_idx + 1].copy()
        segments.append(segment)

    return segments


def allocate_waypoints(
    recommendations: List[dict],
    worker_paths: List[np.ndarray],
    max_deviation: float = MAX_DEVIATION_M,
) -> List[WorkerRoute]:
    """Assign waypoints to workers by greedy minimum-deviation.

    For each waypoint, assigns it to the worker whose baseline path
    has the lowest deviation to that waypoint.

    Args:
        recommendations: List of recommendation dicts with 'x', 'y' keys.
        worker_paths: List of baseline paths, one per worker.
        max_deviation: Max allowable deviation (waypoints beyond this
                       are assigned to the nearest worker anyway).

    Returns:
        List of WorkerRoute objects with paths built.
    """
    num_workers = len(worker_paths)
    routes = [
        WorkerRoute(worker_id=i, baseline_path=path)
        for i, path in enumerate(worker_paths)
    ]

    for rec in recommendations:
        pt = np.array([[rec["x"], rec["y"]]])
        best_worker = -1
        best_dist = float("inf")

        for i, path in enumerate(worker_paths):
            dists = cdist(pt, path, metric="euclidean")
            min_d = float(dists.min())
            if min_d < best_dist:
                best_dist = min_d
                best_worker = i

        # Only assign if the waypoint is within max_deviation of some worker
        if best_worker >= 0 and best_dist <= max_deviation:
            routes[best_worker].assigned_waypoints.append(rec)

    for route in routes:
        route.build_path()

    return routes


def compute_fleet_coverage(
    worker_routes: List[WorkerRoute],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    detection_prob: np.ndarray,
    sources: List[dict],
) -> dict:
    """Compute fleet-level detection using complementary probability.

    P_fleet(x,y) = 1 - prod_i(1 - P_worker_i(x,y))

    For each worker, P_worker is detection_prob weighted by proximity
    to their optimized path.

    Args:
        worker_routes: List of WorkerRoute objects with optimized paths.
        grid_x, grid_y: 2D meshgrid arrays.
        detection_prob: 2D detection probability from opportunity map.
        sources: List of source dicts.

    Returns:
        Dict with 'fleet_prob' (2D array), 'max_fleet_prob' (float),
        'avg_fleet_prob' (float), 'num_workers' (int).
    """
    p_miss = np.ones_like(detection_prob)

    for route in worker_routes:
        if route.optimized_path is None or len(route.optimized_path) < 2:
            continue
        deviation = compute_path_deviation(grid_x, grid_y, route.optimized_path)
        # Worker only contributes detection within max_deviation of their path
        worker_reach = deviation <= MAX_DEVIATION_M
        worker_prob = np.where(worker_reach, detection_prob, 0.0)
        p_miss *= (1.0 - worker_prob)

    fleet_prob = 1.0 - p_miss

    return {
        "fleet_prob": fleet_prob,
        "max_fleet_prob": float(np.max(fleet_prob)),
        "avg_fleet_prob": float(np.mean(fleet_prob[fleet_prob > 0])) if np.any(fleet_prob > 0) else 0.0,
        "num_workers": len(worker_routes),
    }

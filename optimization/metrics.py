"""
Route Metrics for the Methane Leak Opportunistic Tasking System.

Computes distance, time, and detour statistics for baseline vs optimized paths.
"""

import numpy as np
from typing import Dict, List, Optional

from config import WALKING_SPEED_MPS


def _path_length(path: np.ndarray) -> float:
    """Total Euclidean length of a polyline path in meters."""
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


def compute_route_metrics(
    baseline: np.ndarray,
    optimized: np.ndarray,
    recommendations: List[dict],
    walking_speed: float = WALKING_SPEED_MPS,
) -> Dict[str, float]:
    """
    Compute comparison metrics between the baseline and optimized routes.

    Args:
        baseline: (N, 2) array of baseline waypoints.
        optimized: (M, 2) array of optimized waypoints (includes detours).
        recommendations: List of recommendation dicts from the tasking optimizer.
        walking_speed: Walking speed in m/s.

    Returns:
        Dict with keys:
            baseline_distance_m: Total baseline path length (m).
            optimized_distance_m: Total optimized path length (m).
            added_detour_m: Extra distance from detours (m).
            added_detour_pct: Detour as percentage of baseline distance.
            time_impact_min: Additional walking time from detours (min).
            num_detour_points: Number of recommended waypoints inserted.
            avg_detection_prob: Mean detection probability across recommendations.
    """
    baseline_dist = _path_length(baseline)
    optimized_dist = _path_length(optimized)
    added = optimized_dist - baseline_dist

    avg_prob = 0.0
    if recommendations:
        avg_prob = float(np.mean([r["detection_prob"] for r in recommendations]))

    return {
        "baseline_distance_m": baseline_dist,
        "optimized_distance_m": optimized_dist,
        "added_detour_m": max(added, 0.0),
        "added_detour_pct": (added / baseline_dist * 100.0) if baseline_dist > 0 else 0.0,
        "time_impact_min": (added / walking_speed / 60.0) if walking_speed > 0 else 0.0,
        "num_detour_points": len(recommendations),
        "avg_detection_prob": avg_prob,
    }


def find_nearest_source(
    recommendation: dict,
    sources: List[dict],
) -> Optional[str]:
    """
    Return the name of the closest leak source to a recommendation point.

    Args:
        recommendation: Dict with 'x', 'y' keys.
        sources: List of source dicts with 'name', 'x', 'y' keys.

    Returns:
        Name of the nearest source, or None if sources is empty.
    """
    if not sources:
        return None

    rx, ry = recommendation["x"], recommendation["y"]
    best_name = None
    best_dist = float("inf")

    for src in sources:
        d = np.hypot(rx - src["x"], ry - src["y"])
        if d < best_dist:
            best_dist = d
            best_name = src["name"]

    return best_name

"""
Validation metrics for synthetic twin experiments.

Provides detection, efficiency, and learning metrics to quantify how
well a routing strategy localises leak sources.
"""

import numpy as np
from typing import List, Optional

from optimization.information_gain import compute_total_entropy


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def source_detection_rate(
    belief: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    true_sources: List[dict],
    belief_threshold: float = 0.3,
    radius_m: float = 50.0,
) -> float:
    """Fraction of ground-truth sources with high posterior belief nearby.

    A source is considered "detected" if any cell within *radius_m* has
    belief >= *belief_threshold*.

    Args:
        belief: 2-D posterior belief map.
        grid_x, grid_y: Meshgrid arrays.
        true_sources: List of source dicts with 'x', 'y' keys.
        belief_threshold: Minimum belief to count as detected.
        radius_m: Search radius around each source (meters).

    Returns:
        Detection rate in [0, 1].
    """
    if not true_sources:
        return 1.0

    detected = 0
    for src in true_sources:
        dist = np.hypot(grid_x - src["x"], grid_y - src["y"])
        nearby = dist <= radius_m
        if np.any(belief[nearby] >= belief_threshold):
            detected += 1

    return detected / len(true_sources)


def localization_rmse(
    belief: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    true_sources: List[dict],
    top_k: Optional[int] = None,
) -> float:
    """RMSE between belief-map peaks and true source locations.

    Extracts the *top_k* (default = number of true sources) highest-belief
    cells and matches each to the nearest ground-truth source.  Returns
    the root-mean-square distance of these matches.

    Args:
        belief: 2-D posterior belief map.
        grid_x, grid_y: Meshgrid arrays.
        true_sources: List of source dicts with 'x', 'y' keys.
        top_k: Number of peaks to extract (default: len(true_sources)).

    Returns:
        RMSE in meters.  Returns 0.0 if no true sources.
    """
    if not true_sources:
        return 0.0

    k = top_k or len(true_sources)
    flat = belief.ravel()
    peak_idx = np.argsort(flat)[::-1][:k]
    peak_x = grid_x.ravel()[peak_idx]
    peak_y = grid_y.ravel()[peak_idx]

    true_xy = np.array([[s["x"], s["y"]] for s in true_sources])

    # Greedy nearest-source matching
    used = set()
    sq_errors = []
    for sx, sy in true_xy:
        best_dist = float("inf")
        for i in range(len(peak_x)):
            if i in used:
                continue
            d = np.hypot(peak_x[i] - sx, peak_y[i] - sy)
            if d < best_dist:
                best_dist = d
                best_i = i
        used.add(best_i)
        sq_errors.append(best_dist ** 2)

    return float(np.sqrt(np.mean(sq_errors)))


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def information_efficiency(
    entropy_history: List[float],
    cumulative_distance: List[float],
) -> float:
    """Total entropy reduction per unit distance traveled.

    Args:
        entropy_history: Entropy at each step (length = num_steps + 1).
        cumulative_distance: Cumulative distance at each step.

    Returns:
        Bits of entropy reduced per meter.  Returns 0.0 if no distance.
    """
    total_reduction = entropy_history[0] - entropy_history[-1]
    total_dist = cumulative_distance[-1]
    if total_dist < 1e-6:
        return 0.0
    return total_reduction / total_dist


def operational_overhead(
    cumulative_distance: List[float],
    baseline_distance: float,
) -> float:
    """Ratio of total distance traveled to a baseline (e.g., shortest path).

    Returns:
        Overhead ratio (1.0 = same as baseline, 2.0 = twice as far).
    """
    if baseline_distance < 1e-6:
        return 0.0
    return cumulative_distance[-1] / baseline_distance


# ---------------------------------------------------------------------------
# Learning metrics
# ---------------------------------------------------------------------------

def entropy_reduction_fraction(entropy_history: List[float]) -> float:
    """Fraction of initial entropy removed by the end of the experiment.

    Returns:
        Value in [0, 1].  1.0 means all uncertainty eliminated.
    """
    h0 = entropy_history[0]
    if h0 < 1e-15:
        return 1.0
    return (h0 - entropy_history[-1]) / h0


def convergence_step(
    entropy_history: List[float],
    threshold_fraction: float = 0.5,
) -> Optional[int]:
    """Step at which entropy drops below a fraction of the initial value.

    Args:
        entropy_history: Entropy at each step.
        threshold_fraction: Target fraction (e.g., 0.5 = 50% reduction).

    Returns:
        Step index, or None if the threshold was never reached.
    """
    h0 = entropy_history[0]
    target = h0 * (1.0 - threshold_fraction)
    for i, h in enumerate(entropy_history):
        if h <= target:
            return i
    return None


def first_detection_step(detection_events: List[bool]) -> Optional[int]:
    """Step index of the first positive detection (0-indexed).

    Returns:
        Step index, or None if no detection occurred.
    """
    for i, det in enumerate(detection_events):
        if det:
            return i
    return None

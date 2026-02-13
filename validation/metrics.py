"""
Validation metrics for synthetic twin experiments.

Provides detection, efficiency, and learning metrics to quantify how
well a routing strategy localises leak sources.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_rel

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

    # Optimal matching via the Hungarian algorithm (minimises total distance)
    peak_xy = np.column_stack([peak_x, peak_y])
    cost = np.zeros((len(true_xy), len(peak_xy)))
    for i, (sx, sy) in enumerate(true_xy):
        cost[i] = np.hypot(peak_xy[:, 0] - sx, peak_xy[:, 1] - sy)

    row_ind, col_ind = linear_sum_assignment(cost)
    sq_errors = cost[row_ind, col_ind] ** 2

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


# ---------------------------------------------------------------------------
# Statistical significance helpers
# ---------------------------------------------------------------------------

def paired_significance_test(
    metric_a: List[float],
    metric_b: List[float],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired t-test for two matched metric vectors.

    Args:
        metric_a: Metric values for strategy/condition A.
        metric_b: Metric values for strategy/condition B (same length).
        alpha: Significance level.

    Returns:
        Dict with keys: mean_diff, p_value, significant, ci_lower, ci_upper.

    Raises:
        ValueError: If inputs have different lengths or fewer than 2 elements.
    """
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"Input lengths must match: got {len(a)} and {len(b)}"
        )
    if len(a) < 2:
        raise ValueError("Need at least 2 paired observations")

    diff = a - b
    mean_diff = float(np.mean(diff))
    t_stat, p_value = ttest_rel(a, b)
    p_value = float(p_value)

    # 95% confidence interval for the mean difference
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    from scipy.stats import t as t_dist
    t_crit = float(t_dist.ppf(1.0 - alpha / 2.0, df=len(diff) - 1))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "mean_diff": mean_diff,
        "p_value": p_value,
        "significant": p_value < alpha,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def bootstrap_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for the mean.

    Args:
        values: Sample values.
        confidence: Confidence level (e.g. 0.95 for 95%).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: mean, ci_lower, ci_upper, std.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        raise ValueError("Cannot bootstrap from empty array")

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean": float(np.mean(arr)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }

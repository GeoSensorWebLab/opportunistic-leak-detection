"""
Prior Belief Model for Leak Probability.

Computes a prior probability of leak at each source based on equipment
attributes (type, age, production rate, inspection recency). Projects
source-level priors onto a 2D spatial grid using Gaussian kernels.

This implements Stage 1 of the Hierarchical Bayesian Architecture:
    Historical well data -> GEV-inspired prior -> Geospatial probability field.

References:
    - EPA GHGRP emission factors for equipment types
    - John et al. (2016) GEV distribution for emission priors
"""

import numpy as np
from typing import List, Tuple

from config import (
    EQUIPMENT_LEAK_RATES,
    AGE_REFERENCE_YEARS,
    AGE_FACTOR_SCALE,
    AGE_FACTOR_EXPONENT,
    PRODUCTION_RATE_REFERENCE_MCFD,
    PRODUCTION_RATE_SCALE,
    INSPECTION_DECAY_DAYS,
    PRIOR_KERNEL_RADIUS_M,
)


def compute_source_prior(source: dict) -> float:
    """
    Compute the prior leak probability for a single source.

    Combines four risk factors multiplicatively:
        P_prior = P_base * F_age * F_production * F_inspection

    Each factor >= 1.0, so the prior is always >= the base rate.
    Result is clipped to [0, 1].

    Args:
        source: Source dict with keys: equipment_type, age_years,
                production_rate_mcfd, last_inspection_days.
                Missing keys use safe defaults.

    Returns:
        Prior leak probability in [0, 1].
    """
    # Base rate from equipment type
    eq_type = source.get("equipment_type", "default")
    p_base = EQUIPMENT_LEAK_RATES.get(eq_type, EQUIPMENT_LEAK_RATES["default"])

    # Age factor: older equipment has higher leak probability
    age = source.get("age_years", 0)
    f_age = 1.0 + AGE_FACTOR_SCALE * (age / AGE_REFERENCE_YEARS) ** AGE_FACTOR_EXPONENT

    # Production rate factor: higher throughput increases mechanical stress
    prod_rate = source.get("production_rate_mcfd", 0.0)
    f_production = 1.0 + PRODUCTION_RATE_SCALE * (prod_rate / PRODUCTION_RATE_REFERENCE_MCFD)

    # Inspection recency factor: exponential decay of inspection benefit
    days_since = source.get("last_inspection_days", 0)
    f_inspection = 1.0 + (1.0 - np.exp(-days_since / INSPECTION_DECAY_DAYS))

    prior = p_base * f_age * f_production * f_inspection
    return float(np.clip(prior, 0.0, 1.0))


def compute_all_priors(sources: List[dict]) -> List[float]:
    """
    Compute prior leak probabilities for all sources.

    Args:
        sources: List of source dicts.

    Returns:
        List of prior probabilities, one per source.
    """
    return [compute_source_prior(src) for src in sources]


def create_spatial_prior(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sources: List[dict],
    prior_probs: List[float],
    kernel_radius: float = PRIOR_KERNEL_RADIUS_M,
) -> np.ndarray:
    """
    Project source-level priors onto a 2D spatial grid.

    Each source contributes a Gaussian kernel centered at its location,
    weighted by its prior probability. The result is the combined spatial
    prior using complementary probability (independent events):

        P_spatial(x,y) = 1 - prod_i(1 - P_i * K_i(x,y))

    where K_i is a normalized Gaussian kernel for source i.

    Args:
        grid_x, grid_y: 2D meshgrid arrays (meters).
        sources: List of source dicts with 'x', 'y' keys.
        prior_probs: Prior probability for each source.
        kernel_radius: Gaussian kernel standard deviation (meters).

    Returns:
        2D array of spatial prior probabilities, same shape as grid_x.
    """
    # Start with probability of no leak at any cell
    p_no_leak = np.ones_like(grid_x, dtype=float)

    for src, p_prior in zip(sources, prior_probs):
        sx, sy = src["x"], src["y"]

        # Distance from this source to each grid cell
        dist_sq = (grid_x - sx) ** 2 + (grid_y - sy) ** 2

        # Gaussian kernel (normalized to peak=1 at source location)
        kernel = np.exp(-0.5 * dist_sq / (kernel_radius ** 2))

        # Contribution: prior probability weighted by spatial proximity
        p_leak_here = p_prior * kernel

        # Complementary probability for independent sources
        p_no_leak *= (1.0 - p_leak_here)

    spatial_prior = 1.0 - p_no_leak
    return np.clip(spatial_prior, 0.0, 1.0)

"""
Information-Theoretic Scoring using Expected Entropy Reduction (EER).

For each candidate measurement location, computes the expected reduction
in total belief-map entropy, considering both detection and non-detection
outcomes weighted by their probability.

    EER(m) = H(current) - [ P(detect) * H(posterior|detect)
                          + P(no_detect) * H(posterior|no_detect) ]

Each grid cell is treated as an independent binary hypothesis (leak / no
leak), so the total EER is the *sum* of per-cell expected entropy
reductions.  This factorisation keeps the computation vectorised and
avoids intractable joint-state enumeration.

Subsampling + bilinear interpolation makes the grid-wide computation
practical at interactive speeds (~1-3 s for a 1 km site at 5 m base
resolution with subsample=4).

This implements Stage 3 of the Hierarchical Bayesian Architecture.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import List, Optional, Tuple

from models.detection import detection_probability
from config import (
    DISPERSION_COEFFICIENTS,
    AIR_MOLAR_MASS,
    METHANE_MOLAR_MASS,
    AIR_DENSITY,
    RECEPTOR_HEIGHT_M,
    FALSE_ALARM_RATE,
    DETECTION_THRESHOLD_PPM,
    SENSOR_MDL_PPM,
    DEVIATION_EPSILON,
    EER_SUBSAMPLE,
)


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def compute_cell_entropy(belief: np.ndarray) -> np.ndarray:
    """Per-cell binary Shannon entropy.

    H_binary(p) = -(p * log2(p) + (1-p) * log2(1-p))

    Args:
        belief: Array of probabilities in [0, 1] (any shape).

    Returns:
        Array of entropy values in bits, same shape as input.
        Ranges from 0 (certain) to 1 (maximum uncertainty at p=0.5).
    """
    p = np.clip(belief, 1e-15, 1.0 - 1e-15)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def compute_total_entropy(belief: np.ndarray) -> float:
    """Total Shannon entropy of a belief map (sum of cell entropies).

    Args:
        belief: 2D array of P(leak) at each cell.

    Returns:
        Total entropy in bits.
    """
    return float(np.sum(compute_cell_entropy(belief)))


# ---------------------------------------------------------------------------
# Core EER computation (inner loop, vectorised per candidate)
# ---------------------------------------------------------------------------

def _compute_eer_at_candidates(
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    grid_x_flat: np.ndarray,
    grid_y_flat: np.ndarray,
    belief_flat: np.ndarray,
    h_current_flat: np.ndarray,
    wind_ux: float,
    wind_uy: float,
    a_y: float,
    b_y: float,
    a_z: float,
    b_z: float,
    Q: float,
    u: float,
    z: float,
) -> np.ndarray:
    """Compute EER for an array of candidate measurement locations.

    For each candidate *m*, the algorithm:
      1. Computes the reverse-plume concentration at *m* from every grid
         cell treated as a hypothetical source.
      2. Converts to P(detect | leak_i) via the sigmoid sensor model.
      3. Derives per-cell posteriors for both the detection and
         non-detection outcomes.
      4. Sums the per-cell expected entropy reductions.

    All inner operations are fully vectorised over the grid (N cells).

    Args:
        candidate_x, candidate_y: 1-D arrays of candidate coords (M,).
        grid_x_flat, grid_y_flat: Flattened grid coordinates (N,).
        belief_flat: Flattened current belief P(leak_i) (N,).
        h_current_flat: Pre-computed current cell entropy (N,).
        wind_ux, wind_uy: Wind unit-vector components (toward).
        a_y, b_y, a_z, b_z: Dispersion power-law coefficients.
        Q: Representative emission rate for reverse plume (kg/s).
        u: Wind speed (m/s).
        z: Receptor height (m).

    Returns:
        1-D array of EER values (M,), in bits.
    """
    M = len(candidate_x)
    eer_values = np.empty(M)

    p_leak = belief_flat
    p_no_leak = 1.0 - p_leak

    for j in range(M):
        # --- Reverse plume: concentration at candidate from each cell ---
        dx = candidate_x[j] - grid_x_flat
        dy = candidate_y[j] - grid_y_flat

        downwind = dx * wind_ux + dy * wind_uy
        crosswind = -dx * wind_uy + dy * wind_ux

        x_safe = np.maximum(downwind, 1.0)
        sigma_y = a_y * np.power(x_safe, b_y)
        sigma_z = a_z * np.power(x_safe, b_z)

        norm = Q / (2.0 * np.pi * u * sigma_y * sigma_z)
        lateral = np.exp(-0.5 * (crosswind / sigma_y) ** 2)
        # Ground-level source (H=0): vertical = 2*exp(-0.5*(z/σz)^2)
        vertical = 2.0 * np.exp(-0.5 * (z / sigma_z) ** 2)

        conc_kg_m3 = np.where(downwind > 1.0, norm * lateral * vertical, 0.0)
        conc_ppm = (
            (conc_kg_m3 / AIR_DENSITY)
            * (AIR_MOLAR_MASS / METHANE_MOLAR_MASS)
            * 1e6
        )

        # --- P(detect | leak at cell i) ---
        p_det_given_leak = detection_probability(
            conc_ppm,
            threshold_ppm=DETECTION_THRESHOLD_PPM,
            mdl_ppm=SENSOR_MDL_PPM,
        )

        # --- Per-cell observation probabilities ---
        p_obs_detect = (
            p_det_given_leak * p_leak + FALSE_ALARM_RATE * p_no_leak
        )
        p_no_det_given_leak = 1.0 - p_det_given_leak
        p_obs_no_detect = (
            p_no_det_given_leak * p_leak
            + (1.0 - FALSE_ALARM_RATE) * p_no_leak
        )

        # --- Posteriors ---
        safe_det = np.maximum(p_obs_detect, 1e-15)
        posterior_detect = np.clip(
            p_det_given_leak * p_leak / safe_det, 0.0, 1.0
        )

        safe_no_det = np.maximum(p_obs_no_detect, 1e-15)
        posterior_no_detect = np.clip(
            p_no_det_given_leak * p_leak / safe_no_det, 0.0, 1.0
        )

        # --- Per-cell expected posterior entropy ---
        h_detect = compute_cell_entropy(posterior_detect)
        h_no_detect = compute_cell_entropy(posterior_no_detect)

        eer_per_cell = h_current_flat - (
            p_obs_detect * h_detect + p_obs_no_detect * h_no_detect
        )

        # Sum non-negative contributions (numerical noise can make tiny negatives)
        eer_values[j] = np.sum(np.maximum(eer_per_cell, 0.0))

    return eer_values


# ---------------------------------------------------------------------------
# Grid-wide EER with subsampling + interpolation
# ---------------------------------------------------------------------------

def compute_information_value_grid(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    belief: np.ndarray,
    wind_speed: float,
    wind_direction_deg: float,
    stability_class: str,
    avg_emission: float,
    deviation: Optional[np.ndarray] = None,
    max_deviation: Optional[float] = None,
    subsample: int = EER_SUBSAMPLE,
) -> np.ndarray:
    """Compute Expected Entropy Reduction at each grid cell.

    Uses subsampling of the candidate grid followed by bilinear
    interpolation back to the full resolution.  When *deviation* and
    *max_deviation* are supplied only reachable cells are evaluated.

    Args:
        grid_x, grid_y: 2-D meshgrid arrays (full resolution).
        belief: 2-D belief map P(leak) at each cell, same shape.
        wind_speed: Wind speed (m/s).
        wind_direction_deg: Meteorological wind direction (degrees).
        stability_class: Pasquill-Gifford class A-F.
        avg_emission: Representative emission rate for reverse plume (kg/s).
        deviation: Optional 2-D path-deviation grid (meters).
        max_deviation: Optional max-deviation cutoff (meters).
        subsample: Candidate grid subsample factor (1 = full resolution).

    Returns:
        2-D array of EER values (bits), same shape as grid_x.
    """
    full_shape = grid_x.shape

    # --- Subsample candidate grid ----------------------------------------
    sub_rows = np.arange(0, full_shape[0], max(subsample, 1))
    sub_cols = np.arange(0, full_shape[1], max(subsample, 1))
    sub_ix = np.ix_(sub_rows, sub_cols)

    sub_x = grid_x[sub_ix]
    sub_y = grid_y[sub_ix]

    # Build candidate mask
    if deviation is not None and max_deviation is not None:
        sub_dev = deviation[sub_ix]
        mask = sub_dev <= max_deviation
    else:
        mask = np.ones(sub_x.shape, dtype=bool)

    candidate_indices = np.argwhere(mask)

    if len(candidate_indices) == 0:
        return np.zeros(full_shape)

    # --- Pre-compute shared quantities -----------------------------------
    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()
    belief_flat = belief.ravel()
    h_current_flat = compute_cell_entropy(belief).ravel()

    wind_toward_deg = (wind_direction_deg + 180.0) % 360.0
    wind_toward_rad = np.radians(wind_toward_deg)
    wind_ux = np.sin(wind_toward_rad)
    wind_uy = np.cos(wind_toward_rad)

    sc = stability_class.upper()
    coeffs = DISPERSION_COEFFICIENTS[sc]
    a_y, b_y = coeffs["sigma_y"]
    a_z, b_z = coeffs["sigma_z"]

    # Candidate coordinates
    cx = sub_x[candidate_indices[:, 0], candidate_indices[:, 1]]
    cy = sub_y[candidate_indices[:, 0], candidate_indices[:, 1]]

    # --- Compute EER at all candidates -----------------------------------
    eer_values = _compute_eer_at_candidates(
        cx, cy,
        grid_x_flat, grid_y_flat, belief_flat, h_current_flat,
        wind_ux, wind_uy, a_y, b_y, a_z, b_z,
        avg_emission, wind_speed, RECEPTOR_HEIGHT_M,
    )

    # --- Place into sub-grid and interpolate to full resolution ----------
    eer_sub = np.zeros(sub_x.shape)
    for k, (r, c) in enumerate(candidate_indices):
        eer_sub[r, c] = eer_values[k]

    if subsample > 1 and eer_sub.shape != full_shape:
        # Bilinear interpolation via RegularGridInterpolator
        row_coords = grid_y[sub_rows, 0]   # Y varies along rows
        col_coords = grid_x[0, sub_cols]   # X varies along columns

        interp = RegularGridInterpolator(
            (row_coords, col_coords),
            eer_sub,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        full_row_coords = grid_y[:, 0]
        full_col_coords = grid_x[0, :]
        rr, cc = np.meshgrid(full_row_coords, full_col_coords, indexing="ij")
        pts = np.column_stack([rr.ravel(), cc.ravel()])
        eer_full = interp(pts).reshape(full_shape)
    else:
        eer_full = eer_sub

    # Zero out unreachable cells
    if deviation is not None and max_deviation is not None:
        eer_full[deviation > max_deviation] = 0.0

    return eer_full


# ---------------------------------------------------------------------------
# Scoring wrappers (drop-in alternative to compute_tasking_scores)
# ---------------------------------------------------------------------------

def compute_information_scores(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    belief: np.ndarray,
    deviation: np.ndarray,
    max_deviation: float,
    wind_speed: float,
    wind_direction_deg: float,
    stability_class: str,
    avg_emission: float,
    epsilon: float = DEVIATION_EPSILON,
    subsample: int = EER_SUBSAMPLE,
) -> np.ndarray:
    """Compute information-theoretic tasking scores.

        Score(x,y) = EER(x,y) / (PathDeviation(x,y) + epsilon)

    Cells beyond *max_deviation* are scored 0.  This is the EER-based
    alternative to the heuristic ``compute_tasking_scores()``.

    Args:
        grid_x, grid_y: 2-D meshgrid arrays.
        belief: 2-D belief map P(leak) at each cell.
        deviation: 2-D path-deviation grid (meters).
        max_deviation: Max allowable deviation (meters).
        wind_speed, wind_direction_deg, stability_class: Wind conditions.
        avg_emission: Representative emission rate (kg/s).
        epsilon: Division-by-zero guard (meters).
        subsample: EER grid subsample factor.

    Returns:
        2-D score array, same shape as grid_x.
    """
    eer = compute_information_value_grid(
        grid_x=grid_x,
        grid_y=grid_y,
        belief=belief,
        wind_speed=wind_speed,
        wind_direction_deg=wind_direction_deg,
        stability_class=stability_class,
        avg_emission=avg_emission,
        deviation=deviation,
        max_deviation=max_deviation,
        subsample=subsample,
    )

    scores = eer / (deviation + epsilon)
    scores[deviation > max_deviation] = 0.0
    return scores


def compute_ensemble_information_scores(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    belief: np.ndarray,
    deviation: np.ndarray,
    max_deviation: float,
    wind_scenarios: List[dict],
    avg_emission: float,
    epsilon: float = DEVIATION_EPSILON,
    subsample: int = EER_SUBSAMPLE,
) -> np.ndarray:
    """Weighted-average information scores across wind scenarios.

    Each scenario dict must have keys: 'direction', 'speed',
    'stability_class', 'weight'.  Weights must sum to 1.0.

    Returns:
        2-D score array, same shape as grid_x.
    """
    total_eer = np.zeros_like(grid_x)

    for scenario in wind_scenarios:
        eer = compute_information_value_grid(
            grid_x=grid_x,
            grid_y=grid_y,
            belief=belief,
            wind_speed=scenario["speed"],
            wind_direction_deg=scenario["direction"],
            stability_class=scenario["stability_class"],
            avg_emission=avg_emission,
            deviation=deviation,
            max_deviation=max_deviation,
            subsample=subsample,
        )
        total_eer += scenario["weight"] * eer

    scores = total_eer / (deviation + epsilon)
    scores[deviation > max_deviation] = 0.0
    return scores

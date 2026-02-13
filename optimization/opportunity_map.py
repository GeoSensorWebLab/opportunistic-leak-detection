"""
Opportunity Map Generator.

Creates a 2D grid over the site and computes aggregate detection probability
from all potential leak sources under current wind conditions.
"""

import numpy as np
import streamlit as st
from typing import List, Tuple

from models.gaussian_plume import (
    gaussian_plume,
    crosswind_integrated_plume,
    gaussian_puff,
    concentration_to_ppm,
)
from models.detection import detection_probability
from config import (
    GRID_SIZE_M,
    GRID_RESOLUTION_M,
    RECEPTOR_HEIGHT_M,
    CACHE_MAX_ENTRIES,
    SENSOR_MDL_PPM,
    DETECTION_THRESHOLD_PPM,
    DEFAULT_PUFF_MASS_KG,
    DEFAULT_PUFF_TIME_S,
)


def create_grid(
    grid_size: float = GRID_SIZE_M,
    resolution: float = GRID_RESOLUTION_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D meshgrid for the site.

    Args:
        grid_size: Total extent in meters (square grid centered at origin).
        resolution: Grid cell size in meters.

    Returns:
        (X, Y) meshgrid arrays in meters.
    """
    half = grid_size / 2.0
    coords = np.arange(-half, half + resolution, resolution)
    X, Y = np.meshgrid(coords, coords)
    return X, Y


def compute_opportunity_map(
    sources: List[dict],
    wind_speed: float,
    wind_direction_deg: float,
    stability_class: str,
    grid_size: float = GRID_SIZE_M,
    resolution: float = GRID_RESOLUTION_M,
    receptor_height: float = RECEPTOR_HEIGHT_M,
    mdl_ppm: float = SENSOR_MDL_PPM,
    threshold_ppm: float = DETECTION_THRESHOLD_PPM,
    plume_mode: str = "instantaneous",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a combined detection-probability heat map from all leak sources.

    Concentrations from individual plumes are **summed** (the Gaussian plume
    equation is linear, so superposition applies).  Detection probability is
    then computed once on the total concentration field, which correctly
    models a sensor responding to the aggregate methane present at each point.

    Args:
        sources: List of source dicts, each with keys:
                 'x', 'y', 'z' (meters), 'emission_rate' (kg/s), 'name' (str).
        wind_speed: Wind speed in m/s.
        wind_direction_deg: Meteorological wind direction (degrees).
        stability_class: Pasquill-Gifford class A-F.
        grid_size: Site extent in meters.
        resolution: Grid cell size in meters.
        receptor_height: Sensor height above ground (meters).
        mdl_ppm: Sensor Minimum Detection Limit in ppm.
        threshold_ppm: Sigmoid midpoint for detection probability.
        plume_mode: ``"instantaneous"`` for the standard Gaussian plume or
                    ``"integrated"`` for the crosswind-integrated formulation
                    (broader, lower-peak plumes matching time-averaged data).

    Returns:
        (X, Y, concentration_ppm, detection_prob) — all 2D arrays.
        concentration_ppm is the summed concentration from all sources at each cell.
        detection_prob is computed on the total concentration field.
    """
    X, Y = create_grid(grid_size, resolution)

    # Accumulate total concentration via superposition (kg/m^3)
    total_conc = np.zeros_like(X, dtype=float)

    for src in sources:
        duty_cycle = src.get("duty_cycle", 1.0)
        effective_rate = src["emission_rate"] * duty_cycle

        if plume_mode == "puff":
            total_conc += gaussian_puff(
                receptor_x=X,
                receptor_y=Y,
                receptor_z=receptor_height,
                source_x=src["x"],
                source_y=src["y"],
                source_z=src.get("z", 0.0),
                total_mass=src.get("puff_mass", DEFAULT_PUFF_MASS_KG) * duty_cycle,
                wind_speed=wind_speed,
                wind_direction_deg=wind_direction_deg,
                stability_class=stability_class,
                time_since_release=src.get("puff_time", DEFAULT_PUFF_TIME_S),
            )
        else:
            plume_fn = (
                crosswind_integrated_plume if plume_mode == "integrated"
                else gaussian_plume
            )
            total_conc += plume_fn(
                receptor_x=X,
                receptor_y=Y,
                receptor_z=receptor_height,
                source_x=src["x"],
                source_y=src["y"],
                source_z=src.get("z", 0.0),
                emission_rate=effective_rate,
                wind_speed=wind_speed,
                wind_direction_deg=wind_direction_deg,
                stability_class=stability_class,
            )

    # Convert summed concentration to ppm, then derive detection probability
    concentration_ppm = concentration_to_ppm(total_conc)
    combined_detection_prob = detection_probability(
        concentration_ppm, threshold_ppm=threshold_ppm, mdl_ppm=mdl_ppm,
    )

    return X, Y, concentration_ppm, combined_detection_prob


def compute_ensemble_opportunity_map(
    sources: List[dict],
    wind_scenarios: List[dict],
    grid_size: float = GRID_SIZE_M,
    resolution: float = GRID_RESOLUTION_M,
    receptor_height: float = RECEPTOR_HEIGHT_M,
    mdl_ppm: float = SENSOR_MDL_PPM,
    threshold_ppm: float = DETECTION_THRESHOLD_PPM,
    plume_mode: str = "instantaneous",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a weighted-average opportunity map across multiple wind scenarios.

    Each scenario is a dict with keys: 'direction', 'speed', 'stability_class', 'weight'.
    The weights must sum to 1.0 (within tolerance).

    Returns:
        (X, Y, avg_concentration_ppm, avg_detection_prob) — all 2D arrays.
    """
    weights = [s["weight"] for s in wind_scenarios]
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(
            f"Wind scenario weights must sum to 1.0, got {weight_sum:.4f}"
        )

    X, Y = create_grid(grid_size, resolution)
    avg_conc = np.zeros_like(X, dtype=float)
    avg_det = np.zeros_like(X, dtype=float)

    for scenario in wind_scenarios:
        _, _, conc_ppm, det_prob = compute_opportunity_map(
            sources=sources,
            wind_speed=scenario["speed"],
            wind_direction_deg=scenario["direction"],
            stability_class=scenario["stability_class"],
            grid_size=grid_size,
            resolution=resolution,
            receptor_height=receptor_height,
            mdl_ppm=mdl_ppm,
            threshold_ppm=threshold_ppm,
            plume_mode=plume_mode,
        )
        w = scenario["weight"]
        avg_conc += w * conc_ppm
        avg_det += w * det_prob

    return X, Y, avg_conc, avg_det


@st.cache_data(max_entries=CACHE_MAX_ENTRIES)
def cached_ensemble_opportunity_map(
    sources_key: Tuple[Tuple, ...],
    scenarios_key: Tuple[Tuple, ...],
    grid_size: float = GRID_SIZE_M,
    resolution: float = GRID_RESOLUTION_M,
    receptor_height: float = RECEPTOR_HEIGHT_M,
    mdl_ppm: float = SENSOR_MDL_PPM,
    threshold_ppm: float = DETECTION_THRESHOLD_PPM,
    plume_mode: str = "instantaneous",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cached wrapper around compute_ensemble_opportunity_map.

    Accepts hashable keys for Streamlit caching.
    """
    sources = [
        {
            "name": s[0],
            "x": s[1],
            "y": s[2],
            "z": s[3],
            "emission_rate": s[4],
            "duty_cycle": s[5] if len(s) > 5 else 1.0,
        }
        for s in sources_key
    ]
    wind_scenarios = [
        {
            "direction": sc[0],
            "speed": sc[1],
            "stability_class": sc[2],
            "weight": sc[3],
        }
        for sc in scenarios_key
    ]
    return compute_ensemble_opportunity_map(
        sources=sources,
        wind_scenarios=wind_scenarios,
        grid_size=grid_size,
        resolution=resolution,
        receptor_height=receptor_height,
        mdl_ppm=mdl_ppm,
        threshold_ppm=threshold_ppm,
        plume_mode=plume_mode,
    )


@st.cache_data(max_entries=CACHE_MAX_ENTRIES)
def cached_opportunity_map(
    sources_key: Tuple[Tuple, ...],
    wind_speed: float,
    wind_direction_deg: float,
    stability_class: str,
    grid_size: float = GRID_SIZE_M,
    resolution: float = GRID_RESOLUTION_M,
    receptor_height: float = RECEPTOR_HEIGHT_M,
    mdl_ppm: float = SENSOR_MDL_PPM,
    threshold_ppm: float = DETECTION_THRESHOLD_PPM,
    plume_mode: str = "instantaneous",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cached wrapper around compute_opportunity_map.

    Accepts a hashable *sources_key* (tuple of tuples) instead of a list
    of dicts so Streamlit can hash the arguments for its cache.
    """
    sources = [
        {
            "name": s[0],
            "x": s[1],
            "y": s[2],
            "z": s[3],
            "emission_rate": s[4],
            "duty_cycle": s[5] if len(s) > 5 else 1.0,
        }
        for s in sources_key
    ]
    return compute_opportunity_map(
        sources=sources,
        wind_speed=wind_speed,
        wind_direction_deg=wind_direction_deg,
        stability_class=stability_class,
        grid_size=grid_size,
        resolution=resolution,
        receptor_height=receptor_height,
        mdl_ppm=mdl_ppm,
        threshold_ppm=threshold_ppm,
        plume_mode=plume_mode,
    )

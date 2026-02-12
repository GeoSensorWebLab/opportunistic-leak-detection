"""
Gaussian Plume Dispersion Model.

Implements the standard Gaussian plume equation for a continuous point source
at ground level, with Pasquill-Gifford dispersion coefficients.

Convention:
  - Wind direction uses METEOROLOGICAL convention (direction wind comes FROM).
  - Wind blows in the opposite direction (FROM 270 = blowing East = +x direction).
  - All coordinates in meters, concentrations in kg/m^3.
"""

import numpy as np
from config import DISPERSION_COEFFICIENTS, AIR_MOLAR_MASS, METHANE_MOLAR_MASS, AIR_DENSITY


def _get_dispersion_coeffs(stability_class: str) -> dict:
    """Return (a, b) coefficients for sigma_y and sigma_z."""
    sc = stability_class.upper()
    if sc not in DISPERSION_COEFFICIENTS:
        raise ValueError(f"Unknown stability class '{sc}'. Use A-F.")
    return DISPERSION_COEFFICIENTS[sc]


def compute_sigma(distance_downwind: np.ndarray, stability_class: str):
    """
    Compute lateral (sigma_y) and vertical (sigma_z) dispersion parameters.

    Args:
        distance_downwind: Downwind distances in meters (must be > 0 for valid results).
        stability_class: Pasquill-Gifford class A-F.

    Returns:
        (sigma_y, sigma_z) arrays in meters.
    """
    coeffs = _get_dispersion_coeffs(stability_class)
    a_y, b_y = coeffs["sigma_y"]
    a_z, b_z = coeffs["sigma_z"]

    # Clamp distance to avoid log(0); negative distances mean upwind (no plume)
    x_safe = np.maximum(distance_downwind, 1.0)

    sigma_y = a_y * np.power(x_safe, b_y)
    sigma_z = a_z * np.power(x_safe, b_z)

    return sigma_y, sigma_z


def gaussian_plume(
    receptor_x: np.ndarray,
    receptor_y: np.ndarray,
    receptor_z: float,
    source_x: float,
    source_y: float,
    source_z: float,
    emission_rate: float,
    wind_speed: float,
    wind_direction_deg: float,
    stability_class: str,
) -> np.ndarray:
    """
    Calculate concentration at receptor points from a single point source.

    Uses the standard Gaussian plume equation with ground reflection:
        C = (Q / (2*pi*u*sigma_y*sigma_z)) *
            exp(-0.5*(crosswind/sigma_y)^2) *
            [exp(-0.5*((z-H)/sigma_z)^2) + exp(-0.5*((z+H)/sigma_z)^2)]

    Args:
        receptor_x, receptor_y: Receptor coordinates (meters), can be 2D grids.
        receptor_z: Receptor height above ground (meters), scalar.
        source_x, source_y: Source location (meters).
        source_z: Source release height (meters), typically 0 for ground-level leaks.
        emission_rate: Emission rate Q in kg/s.
        wind_speed: Wind speed in m/s (must be > 0).
        wind_direction_deg: Meteorological wind direction (degrees, 0=N, 90=E, etc.).
        stability_class: Pasquill-Gifford stability class (A-F).

    Returns:
        concentration: Array of concentrations in kg/m^3, same shape as receptor_x.
    """
    if wind_speed <= 0:
        raise ValueError("Wind speed must be positive.")

    # Convert meteorological direction to the direction wind is blowing TOWARD
    # Met convention: 270 means wind FROM the west, blowing toward east (+x)
    wind_toward_deg = (wind_direction_deg + 180.0) % 360.0
    wind_toward_rad = np.radians(wind_toward_deg)

    # Unit vector of wind direction (toward)
    # In our coordinate system: x=East, y=North
    wind_ux = np.sin(wind_toward_rad)
    wind_uy = np.cos(wind_toward_rad)

    # Vector from source to each receptor
    dx = receptor_x - source_x
    dy = receptor_y - source_y

    # Project onto downwind (along wind) and crosswind (perpendicular) axes
    downwind = dx * wind_ux + dy * wind_uy
    crosswind = -dx * wind_uy + dy * wind_ux

    # Dispersion parameters (only valid for downwind > 0)
    sigma_y, sigma_z = compute_sigma(downwind, stability_class)

    # Gaussian plume equation with ground reflection
    concentration = np.zeros_like(receptor_x, dtype=float)

    # Only compute where receptor is downwind of source
    mask = downwind > 1.0  # At least 1m downwind

    if np.any(mask):
        sy = sigma_y[mask]
        sz = sigma_z[mask]
        cw = crosswind[mask]
        u = wind_speed

        # Normalization
        norm = emission_rate / (2.0 * np.pi * u * sy * sz)

        # Lateral Gaussian
        lateral = np.exp(-0.5 * (cw / sy) ** 2)

        # Vertical Gaussian with ground reflection (image source method)
        z = receptor_z
        H = source_z
        vertical = np.exp(-0.5 * ((z - H) / sz) ** 2) + np.exp(
            -0.5 * ((z + H) / sz) ** 2
        )

        concentration[mask] = norm * lateral * vertical

    return concentration


def concentration_to_ppm(concentration_kg_m3: np.ndarray) -> np.ndarray:
    """
    Convert methane concentration from kg/m^3 to parts per million (ppm) by volume.

    Uses ideal gas approximation at standard conditions:
        ppm = (C / rho_air) * (M_air / M_CH4) * 1e6

    Args:
        concentration_kg_m3: Concentration in kg/m^3.

    Returns:
        Concentration in ppm (volume).
    """
    ppm = (concentration_kg_m3 / AIR_DENSITY) * (AIR_MOLAR_MASS / METHANE_MOLAR_MASS) * 1e6
    return ppm

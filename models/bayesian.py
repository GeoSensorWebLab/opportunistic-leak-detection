"""
Bayesian Belief Map for iterative leak localization.

Implements cell-wise Bayes' theorem to update a spatial belief map
based on field observations (detections and non-detections). Uses a
vectorized reverse-plume computation — for each grid cell as a
hypothetical source, compute the expected concentration at the
measurement location.

This is Stage 2 of the Hierarchical Bayesian Architecture:
    Prior (Stage 1) -> Field observations -> Posterior belief map.
"""

import numpy as np
from typing import List

from models.measurement import Measurement
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
)


class BayesianBeliefMap:
    """Maintains and updates a spatial belief map via Bayesian inference.

    Each grid cell holds P(leak at cell i), updated as measurements
    arrive using cell-wise Bayes' theorem.

    Args:
        grid_x: 2D meshgrid of x-coordinates (meters).
        grid_y: 2D meshgrid of y-coordinates (meters).
        prior: 2D array of prior leak probabilities, same shape as grid.
        sources: List of source dicts (used for default emission rates).
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        prior: np.ndarray,
        sources: List[dict],
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.prior = prior.copy()
        self.belief = prior.copy()
        self.sources = sources
        self.measurements: List[Measurement] = []

        # Average effective emission rate across known sources for reverse plume
        if sources:
            self._avg_emission = np.mean(
                [s["emission_rate"] * s.get("duty_cycle", 1.0) for s in sources]
            )
        else:
            self._avg_emission = 0.5

    def update(self, measurement: Measurement) -> None:
        """Update the belief map with a new measurement.

        Args:
            measurement: A Measurement instance (detected or not).
        """
        self.measurements.append(measurement)
        if measurement.detected:
            self._update_positive(measurement)
        else:
            self._update_negative(measurement)

    def _update_positive(self, measurement: Measurement) -> None:
        """Bayesian update for a positive detection.

        P(leak_i | detect) = P(detect | leak_i) * P(leak_i) / P(detect)
        where:
            P(detect | leak_i) = detection_prob from reverse plume
            P(detect | no_leak_i) = FALSE_ALARM_RATE
        """
        reverse_conc = self._compute_reverse_plume(measurement)
        p_detect_given_leak = detection_probability(
            reverse_conc,
            threshold_ppm=DETECTION_THRESHOLD_PPM,
            mdl_ppm=SENSOR_MDL_PPM,
        )

        p_leak = self.belief
        p_no_leak = 1.0 - p_leak

        numerator = p_detect_given_leak * p_leak
        denominator = numerator + FALSE_ALARM_RATE * p_no_leak

        # Avoid division by zero
        safe_denom = np.maximum(denominator, 1e-15)
        self.belief = np.clip(numerator / safe_denom, 0.0, 1.0)

    def _update_negative(self, measurement: Measurement) -> None:
        """Bayesian update for a non-detection.

        P(leak_i | no_detect) = P(no_detect | leak_i) * P(leak_i) / P(no_detect)
        where:
            P(no_detect | leak_i) = 1 - detection_prob from reverse plume
            P(no_detect | no_leak_i) ≈ 1.0
        """
        reverse_conc = self._compute_reverse_plume(measurement)
        p_detect_given_leak = detection_probability(
            reverse_conc,
            threshold_ppm=DETECTION_THRESHOLD_PPM,
            mdl_ppm=SENSOR_MDL_PPM,
        )
        p_no_detect_given_leak = 1.0 - p_detect_given_leak

        p_leak = self.belief
        p_no_leak = 1.0 - p_leak
        p_no_detect_given_no_leak = 1.0 - FALSE_ALARM_RATE

        numerator = p_no_detect_given_leak * p_leak
        denominator = numerator + p_no_detect_given_no_leak * p_no_leak

        safe_denom = np.maximum(denominator, 1e-15)
        self.belief = np.clip(numerator / safe_denom, 0.0, 1.0)

    def _compute_reverse_plume(self, measurement: Measurement) -> np.ndarray:
        """Vectorized reverse-plume: treat each grid cell as a hypothetical source.

        For each cell (i,j), compute the concentration that a source at (i,j)
        would produce at the measurement location, using the same Gaussian
        plume physics. Returns concentration in ppm.
        """
        # Wind direction: meteorological -> blowing toward
        wind_toward_deg = (measurement.wind_direction_deg + 180.0) % 360.0
        wind_toward_rad = np.radians(wind_toward_deg)

        wind_ux = np.sin(wind_toward_rad)
        wind_uy = np.cos(wind_toward_rad)

        # Vector from each hypothetical source (grid cell) to measurement point
        dx = measurement.x - self.grid_x
        dy = measurement.y - self.grid_y

        # Downwind and crosswind distances
        downwind = dx * wind_ux + dy * wind_uy
        crosswind = -dx * wind_uy + dy * wind_ux

        # Get dispersion coefficients
        sc = measurement.stability_class.upper()
        coeffs = DISPERSION_COEFFICIENTS[sc]
        a_y, b_y = coeffs["sigma_y"]
        a_z, b_z = coeffs["sigma_z"]

        # Only valid where measurement is downwind of hypothetical source
        x_safe = np.maximum(downwind, 1.0)
        sigma_y = a_y * np.power(x_safe, b_y)
        sigma_z = a_z * np.power(x_safe, b_z)

        # Gaussian plume (ground-level source, receptor at measurement height)
        Q = self._avg_emission
        u = measurement.wind_speed
        z = RECEPTOR_HEIGHT_M
        H = 0.0  # assume ground-level hypothetical sources

        norm = Q / (2.0 * np.pi * u * sigma_y * sigma_z)
        lateral = np.exp(-0.5 * (crosswind / sigma_y) ** 2)
        vertical = np.exp(-0.5 * ((z - H) / sigma_z) ** 2) + np.exp(
            -0.5 * ((z + H) / sigma_z) ** 2
        )

        concentration_kg_m3 = np.where(
            downwind > 1.0,
            norm * lateral * vertical,
            0.0,
        )

        # Convert to ppm
        ppm = (concentration_kg_m3 / AIR_DENSITY) * (AIR_MOLAR_MASS / METHANE_MOLAR_MASS) * 1e6
        return ppm

    def get_belief_map(self) -> np.ndarray:
        """Return the current posterior belief map."""
        return self.belief.copy()

    def set_belief(self, belief: np.ndarray) -> None:
        """Restore a saved posterior as the current belief.

        Args:
            belief: 2D array of P(leak), same shape as grid.
        """
        self.belief = np.clip(belief.copy(), 0.0, 1.0)

    def reset(self) -> None:
        """Reset belief to the original prior and clear measurement history."""
        self.belief = self.prior.copy()
        self.measurements.clear()

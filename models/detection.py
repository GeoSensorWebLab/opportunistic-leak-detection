"""
Detection Probability Model.

Converts methane concentration fields into probability-of-detection maps
based on sensor characteristics.

Uses a sigmoid-style detection curve: probability transitions from 0 to 1
around the sensor's detection threshold, modeling real sensor noise.
"""

import numpy as np
from config import DETECTION_THRESHOLD_PPM, DETECTION_STEEPNESS


def detection_probability(
    concentration_ppm: np.ndarray,
    threshold_ppm: float = DETECTION_THRESHOLD_PPM,
    steepness: float = DETECTION_STEEPNESS,
) -> np.ndarray:
    """
    Compute probability of detection at each grid point.

    Uses a logistic (sigmoid) function centered at the detection threshold:
        P(detect) = 1 / (1 + exp(-steepness * (C - threshold)))

    This models the gradual transition from undetectable to detectable
    concentrations, accounting for sensor noise and variability.

    Args:
        concentration_ppm: Methane concentration in ppm (any shape).
        threshold_ppm: Sensor detection threshold in ppm.
        steepness: Controls how sharp the transition is.
                   Higher = more binary (detect/not-detect).
                   Lower = more gradual.

    Returns:
        Probability of detection, same shape as input, values in [0, 1].
    """
    exponent = -steepness * (concentration_ppm - threshold_ppm)
    # Clip to avoid overflow in exp
    exponent = np.clip(exponent, -50, 50)
    prob = 1.0 / (1.0 + np.exp(exponent))
    return prob

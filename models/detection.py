"""
Detection Probability Model.

Converts methane concentration fields into probability-of-detection maps
based on sensor characteristics.

Uses a sigmoid-style detection curve with a hard Minimum Detection Limit
(MDL) cutoff.  Below the MDL the sensor physically cannot respond, so
P(detect) = 0 regardless of noise.  Above the MDL, a logistic sigmoid
models the gradual transition from marginal to certain detection.

Typical handheld methane detectors:
    MDL       ~1 ppm   (instrument noise floor)
    Threshold ~5 ppm   (reliable detection, P = 50%)
"""

import numpy as np
from config import DETECTION_THRESHOLD_PPM, DETECTION_STEEPNESS, SENSOR_MDL_PPM


def detection_probability(
    concentration_ppm: np.ndarray,
    threshold_ppm: float = DETECTION_THRESHOLD_PPM,
    steepness: float = DETECTION_STEEPNESS,
    mdl_ppm: float = SENSOR_MDL_PPM,
) -> np.ndarray:
    """
    Compute probability of detection at each grid point.

    Model:
        if C < MDL:   P = 0   (below instrument noise floor)
        if C >= MDL:  P = sigmoid(C)  (logistic transition)

    The sigmoid is: P = 1 / (1 + exp(-steepness * (C - threshold)))

    Args:
        concentration_ppm: Methane concentration in ppm (any shape).
        threshold_ppm: Sigmoid midpoint — concentration at which P = 0.5.
        steepness: Controls how sharp the sigmoid transition is.
                   Higher = more binary (detect/not-detect).
                   Lower = more gradual.
        mdl_ppm: Minimum Detection Limit in ppm.  Concentrations below
                 this value yield P = 0.  Set to 0 to disable.

    Returns:
        Probability of detection, same shape as input, values in [0, 1].
    """
    exponent = -steepness * (concentration_ppm - threshold_ppm)
    # Clip to avoid overflow in exp
    exponent = np.clip(exponent, -50, 50)
    prob = 1.0 / (1.0 + np.exp(exponent))

    # Hard cutoff: zero probability below the minimum detection limit
    if mdl_ppm > 0:
        prob = np.where(concentration_ppm < mdl_ppm, 0.0, prob)

    return prob

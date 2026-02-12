"""
Measurement data model for Bayesian belief updates.

Represents a single field observation (detection or non-detection) at
a specific location under known wind conditions.
"""

from dataclasses import dataclass


@dataclass
class Measurement:
    """A single field measurement of methane concentration.

    Args:
        x: Measurement location, East (meters).
        y: Measurement location, North (meters).
        concentration_ppm: Observed methane concentration in ppm.
        detected: Whether the sensor triggered a detection.
        wind_speed: Wind speed at time of measurement (m/s).
        wind_direction_deg: Meteorological wind direction (degrees).
        stability_class: Pasquill-Gifford stability class (A-F).
    """

    x: float
    y: float
    concentration_ppm: float
    detected: bool
    wind_speed: float
    wind_direction_deg: float
    stability_class: str

    def __post_init__(self):
        if self.concentration_ppm < 0:
            raise ValueError("concentration_ppm must be >= 0")
        if self.wind_speed <= 0:
            raise ValueError("wind_speed must be > 0")
        if self.stability_class.upper() not in "ABCDEF":
            raise ValueError(f"Invalid stability class: {self.stability_class}")
        self.stability_class = self.stability_class.upper()

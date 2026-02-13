"""
Weather API abstraction for wind observation and forecast data.

Provides a pluggable interface for live weather integration.
The StubWeatherProvider returns configurable hardcoded values
for development and testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class WindObservation:
    """A single wind observation at a point in time."""

    speed: float                    # m/s
    direction: float                # Meteorological degrees (0-360)
    stability_class: str            # Pasquill-Gifford class A-F
    timestamp: Optional[datetime] = None
    station_id: Optional[str] = None

    def __post_init__(self):
        if self.speed < 0:
            raise ValueError("Wind speed must be >= 0")
        self.stability_class = self.stability_class.upper()
        if self.stability_class not in "ABCDEF":
            raise ValueError(f"Invalid stability class: {self.stability_class}")


class WeatherProvider(ABC):
    """Abstract base class for weather data sources."""

    @abstractmethod
    def get_current_wind(self) -> WindObservation:
        """Return the most recent wind observation."""
        ...

    @abstractmethod
    def get_forecast(self, hours_ahead: int = 6) -> List[WindObservation]:
        """Return wind forecast for the next N hours.

        Args:
            hours_ahead: Number of hours to forecast.

        Returns:
            List of WindObservation, one per hour.
        """
        ...


class StubWeatherProvider(WeatherProvider):
    """Configurable stub that returns hardcoded wind data.

    Args:
        speed: Default wind speed (m/s).
        direction: Default wind direction (meteorological degrees).
        stability_class: Default Pasquill-Gifford class.
        station_id: Optional station identifier.
    """

    def __init__(
        self,
        speed: float = 3.0,
        direction: float = 270.0,
        stability_class: str = "D",
        station_id: str = "STUB-001",
    ):
        self.speed = speed
        self.direction = direction
        self.stability_class = stability_class
        self.station_id = station_id

    def get_current_wind(self) -> WindObservation:
        return WindObservation(
            speed=self.speed,
            direction=self.direction,
            stability_class=self.stability_class,
            timestamp=datetime.now(),
            station_id=self.station_id,
        )

    def get_forecast(self, hours_ahead: int = 6) -> List[WindObservation]:
        base_time = datetime.now()
        observations = []
        for h in range(hours_ahead):
            obs = WindObservation(
                speed=self.speed,
                direction=self.direction,
                stability_class=self.stability_class,
                timestamp=base_time.replace(hour=(base_time.hour + h) % 24),
                station_id=self.station_id,
            )
            observations.append(obs)
        return observations

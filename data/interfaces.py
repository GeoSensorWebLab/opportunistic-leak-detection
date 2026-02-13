"""
Abstract Data Provider interface for pluggable data sources.

Allows swapping mock data for real SCADA/SensorUp feeds without
changing downstream code.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List


class DataProvider(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def get_leak_sources(self) -> List[dict]:
        """Return list of potential leak source dicts.

        Each dict must have keys: 'name', 'x', 'y', 'z', 'emission_rate'.
        Optional keys: 'equipment_type', 'age_years', 'production_rate_mcfd',
        'last_inspection_days', 'duty_cycle'.
        """
        ...

    @abstractmethod
    def get_baseline_path(self) -> np.ndarray:
        """Return (N, 2) array of [x, y] waypoints for the worker path."""
        ...

    @abstractmethod
    def get_wind_scenarios(self) -> List[dict]:
        """Return list of wind scenario dicts.

        Each dict must have keys: 'name', 'speed', 'direction', 'stability_class'.
        """
        ...

    @abstractmethod
    def get_wind_distribution(self) -> List[dict]:
        """Return an 8-direction wind rose with equal weights.

        Each dict must have keys: 'direction', 'speed', 'stability_class', 'weight'.
        Weights must sum to 1.0.
        """
        ...

    @abstractmethod
    def get_wind_fan(
        self,
        center_direction: float,
        spread_deg: float = 30.0,
        num_scenarios: int = 8,
        speed: float = 3.0,
        stability_class: str = "D",
    ) -> List[dict]:
        """Generate a fan of wind directions around a center direction.

        Args:
            center_direction: Center direction in meteorological degrees.
            spread_deg: Total angular spread of the fan.
            num_scenarios: Number of scenarios in the fan.
            speed: Wind speed for all scenarios.
            stability_class: Stability class for all scenarios.

        Returns:
            List of dicts with keys: 'direction', 'speed', 'stability_class', 'weight'.
        """
        ...


class MockDataProvider(DataProvider):
    """Wraps existing mock_data.py functions."""

    def get_leak_sources(self) -> List[dict]:
        from data.mock_data import get_leak_sources
        return get_leak_sources()

    def get_baseline_path(self) -> np.ndarray:
        from data.mock_data import get_baseline_path
        return get_baseline_path()

    def get_wind_scenarios(self) -> List[dict]:
        from data.mock_data import get_wind_scenarios
        return get_wind_scenarios()

    def get_wind_distribution(self) -> List[dict]:
        from data.mock_data import get_wind_distribution
        return get_wind_distribution()

    def get_wind_fan(
        self,
        center_direction: float,
        spread_deg: float = 30.0,
        num_scenarios: int = 8,
        speed: float = 3.0,
        stability_class: str = "D",
    ) -> List[dict]:
        from data.mock_data import get_wind_fan
        return get_wind_fan(
            center_direction=center_direction,
            spread_deg=spread_deg,
            num_scenarios=num_scenarios,
            speed=speed,
            stability_class=stability_class,
        )

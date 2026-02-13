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

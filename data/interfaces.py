"""
Abstract Data Provider interface for pluggable data sources.

Allows swapping mock data for real SCADA/SensorUp feeds without
changing downstream code.
"""

import csv
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional


class DataProvider(ABC):
    """Abstract base class for data sources.

    **Immutability contract:** All methods return *fresh* data.  Callers
    must NOT mutate the returned lists or dicts in-place because the
    provider may cache them internally.  If mutation is needed (e.g. UI
    sliders overriding duty_cycle), copy first::

        sources = [s.copy() for s in provider.get_leak_sources()]
    """

    @abstractmethod
    def get_leak_sources(self) -> List[dict]:
        """Return list of potential leak source dicts.

        Each dict must have keys: 'name', 'x', 'y', 'z', 'emission_rate'.
        Optional keys: 'equipment_type', 'age_years', 'production_rate_mcfd',
        'last_inspection_days', 'duty_cycle'.

        Note: Callers must copy before mutating (see class docstring).
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
            spread_deg: Half-spread in degrees (fan spans center +/- spread_deg).
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


class FileDataProvider(DataProvider):
    """Load site data from JSON/CSV files on disk.

    Args:
        sources_path: Path to a JSON file with an array of source dicts.
        path_path: Path to a CSV file with ``x,y`` columns for the baseline path.
        wind_scenarios_path: Path to a JSON file with an array of wind scenario dicts.
        wind_distribution_path: Optional path to a JSON file with an 8-direction
            wind rose (weights must sum to 1.0).  If not provided,
            ``get_wind_distribution()`` returns a uniform 8-direction rose.

    Raises:
        ValueError: If required keys are missing or data is invalid.
        FileNotFoundError: If any file does not exist.
    """

    _REQUIRED_SOURCE_KEYS = {"name", "x", "y", "z", "emission_rate"}
    _REQUIRED_WIND_SCENARIO_KEYS = {"name", "speed", "direction", "stability_class"}
    _REQUIRED_WIND_DIST_KEYS = {"direction", "speed", "stability_class", "weight"}

    def __init__(
        self,
        sources_path: str,
        path_path: str,
        wind_scenarios_path: str,
        wind_distribution_path: Optional[str] = None,
    ):
        self._sources = self._load_sources(sources_path)
        self._path = self._load_path(path_path)
        self._wind_scenarios = self._load_wind_scenarios(wind_scenarios_path)
        if wind_distribution_path:
            self._wind_distribution = self._load_wind_distribution(
                wind_distribution_path
            )
        else:
            self._wind_distribution = None

    # -- loaders with validation ------------------------------------------

    @classmethod
    def _load_sources(cls, path: str) -> List[dict]:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"Sources file must contain a non-empty JSON array: {path}")
        for i, src in enumerate(data):
            missing = cls._REQUIRED_SOURCE_KEYS - set(src.keys())
            if missing:
                raise ValueError(
                    f"Source #{i} missing required keys {missing} in {path}"
                )
        return data

    @classmethod
    def _load_path(cls, path: str) -> np.ndarray:
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append([float(row["x"]), float(row["y"])])
        if len(rows) < 2:
            raise ValueError(
                f"Path CSV must have at least 2 waypoints, got {len(rows)}: {path}"
            )
        return np.array(rows)

    @classmethod
    def _load_wind_scenarios(cls, path: str) -> List[dict]:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(
                f"Wind scenarios file must contain a non-empty JSON array: {path}"
            )
        for i, sc in enumerate(data):
            missing = cls._REQUIRED_WIND_SCENARIO_KEYS - set(sc.keys())
            if missing:
                raise ValueError(
                    f"Wind scenario #{i} missing required keys {missing} in {path}"
                )
        return data

    @classmethod
    def _load_wind_distribution(cls, path: str) -> List[dict]:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(
                f"Wind distribution file must contain a non-empty JSON array: {path}"
            )
        for i, entry in enumerate(data):
            missing = cls._REQUIRED_WIND_DIST_KEYS - set(entry.keys())
            if missing:
                raise ValueError(
                    f"Wind distribution entry #{i} missing required keys {missing} in {path}"
                )
        total_weight = sum(e["weight"] for e in data)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Wind distribution weights must sum to 1.0, got {total_weight:.6f}"
            )
        return data

    # -- DataProvider interface -------------------------------------------

    def get_leak_sources(self) -> List[dict]:
        return [s.copy() for s in self._sources]

    def get_baseline_path(self) -> np.ndarray:
        return self._path.copy()

    def get_wind_scenarios(self) -> List[dict]:
        return [s.copy() for s in self._wind_scenarios]

    def get_wind_distribution(self) -> List[dict]:
        if self._wind_distribution is not None:
            return [d.copy() for d in self._wind_distribution]
        # Fallback: uniform 8-direction rose
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

"""
Synthetic Twin Experiment Engine.

Runs controlled experiments with known ground-truth leaks to validate
detection strategies.  The engine:

  1. Accepts ground-truth sources (the actual leaks) and all known
     equipment (used for prior computation and opportunity map).
  2. Simulates sensor readings at each measurement location using
     the forward Gaussian plume model plus sensor noise.
  3. Updates a Bayesian belief map after each measurement.
  4. Records entropy, detections, and distance for post-hoc analysis.

Strategies are pluggable — each implements ``next_location(state)``
and returns (x, y) for the next measurement.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from models.gaussian_plume import gaussian_plume, concentration_to_ppm
from models.detection import detection_probability
from models.prior import compute_all_priors, create_spatial_prior
from models.measurement import Measurement
from models.bayesian import BayesianBeliefMap
from optimization.opportunity_map import compute_opportunity_map, create_grid
from optimization.information_gain import (
    compute_total_entropy,
    compute_information_value_grid,
)
from config import (
    GRID_SIZE_M,
    GRID_RESOLUTION_M,
    RECEPTOR_HEIGHT_M,
    DETECTION_THRESHOLD_PPM,
    SENSOR_MDL_PPM,
    DEVIATION_EPSILON,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Holds the output of a single experiment run."""

    strategy_name: str
    num_steps: int
    entropy_history: List[float]
    detection_events: List[bool]
    cumulative_distance: List[float]
    measurements: List[Measurement]
    final_belief: np.ndarray

    @property
    def total_distance(self) -> float:
        return self.cumulative_distance[-1]

    @property
    def total_detections(self) -> int:
        return sum(self.detection_events)

    @property
    def entropy_reduction(self) -> float:
        return self.entropy_history[0] - self.entropy_history[-1]


# ---------------------------------------------------------------------------
# Strategy protocol & implementations
# ---------------------------------------------------------------------------

class Strategy(Protocol):
    """Protocol for measurement strategies."""

    name: str

    def next_location(self, state: dict) -> Tuple[float, float]: ...


class RandomStrategy:
    """Pick a uniformly random location within the grid."""

    name = "Random"

    def next_location(self, state: dict) -> Tuple[float, float]:
        rng = state["rng"]
        lo, hi = state["grid_bounds"]
        x = rng.uniform(lo, hi)
        y = rng.uniform(lo, hi)
        return float(x), float(y)


class GridSearchStrategy:
    """Visit grid cells in raster order (left-to-right, bottom-to-top).

    Uses a coarse step size to avoid redundant closely-spaced visits.
    """

    name = "Grid Search"

    def __init__(self, step_m: float = 50.0):
        self.step_m = step_m
        self._cells: Optional[List[Tuple[float, float]]] = None
        self._index = 0

    def _build_cells(self, lo: float, hi: float) -> None:
        xs = np.arange(lo, hi + self.step_m, self.step_m)
        ys = np.arange(lo, hi + self.step_m, self.step_m)
        self._cells = [(float(x), float(y)) for y in ys for x in xs]
        self._index = 0

    def next_location(self, state: dict) -> Tuple[float, float]:
        lo, hi = state["grid_bounds"]
        if self._cells is None:
            self._build_cells(lo, hi)
        if self._index >= len(self._cells):
            self._index = 0  # wrap around
        loc = self._cells[self._index]
        self._index += 1
        return loc


class MaxDetectionStrategy:
    """Go to the cell with highest detection probability.

    Ignores path deviation — pure "go where signal is strongest".
    A minimum travel distance prevents the strategy from staying put.
    """

    name = "Max Detection"

    def __init__(self, min_travel_m: float = 30.0):
        self.min_travel_m = min_travel_m

    def next_location(self, state: dict) -> Tuple[float, float]:
        det = state["detection_prob"]
        gx = state["grid_x"]
        gy = state["grid_y"]
        cx, cy = state["current_x"], state["current_y"]

        dist = np.hypot(gx - cx, gy - cy)
        score = det / (dist + DEVIATION_EPSILON)
        score[dist < self.min_travel_m] = 0.0

        idx = np.unravel_index(np.argmax(score), score.shape)
        return float(gx[idx]), float(gy[idx])


class OpportunisticStrategy:
    """Heuristic scoring: belief * detection_prob / distance.

    This mirrors the production heuristic but uses distance-from-current
    instead of path deviation. A minimum travel distance prevents the
    strategy from staying at the same location.
    """

    name = "Opportunistic"

    def __init__(self, min_travel_m: float = 30.0):
        self.min_travel_m = min_travel_m

    def next_location(self, state: dict) -> Tuple[float, float]:
        belief = state["belief"]
        det = state["detection_prob"]
        gx = state["grid_x"]
        gy = state["grid_y"]
        cx, cy = state["current_x"], state["current_y"]

        dist = np.hypot(gx - cx, gy - cy)
        score = belief * det / (dist + DEVIATION_EPSILON)
        score[dist < self.min_travel_m] = 0.0

        idx = np.unravel_index(np.argmax(score), score.shape)
        return float(gx[idx]), float(gy[idx])


class InformationTheoreticStrategy:
    """EER scoring: Expected Entropy Reduction / distance.

    Picks the location that maximises expected information gain per
    unit travel distance. A minimum travel distance prevents the
    strategy from re-measuring at the same location.
    """

    name = "EER"

    def __init__(self, subsample: int = 4, min_travel_m: float = 30.0):
        self.subsample = subsample
        self.min_travel_m = min_travel_m

    def next_location(self, state: dict) -> Tuple[float, float]:
        gx = state["grid_x"]
        gy = state["grid_y"]
        belief = state["belief"]
        wp = state["wind_params"]
        equip = state["all_equipment"]
        cx, cy = state["current_x"], state["current_y"]

        avg_emission = float(np.mean(
            [s["emission_rate"] * s.get("duty_cycle", 1.0) for s in equip]
        ))
        dist = np.hypot(gx - cx, gy - cy)

        eer = compute_information_value_grid(
            grid_x=gx,
            grid_y=gy,
            belief=belief,
            wind_speed=wp["wind_speed"],
            wind_direction_deg=wp["wind_direction_deg"],
            stability_class=wp["stability_class"],
            avg_emission=avg_emission,
            subsample=self.subsample,
        )

        score = eer / (dist + DEVIATION_EPSILON)
        score[dist < self.min_travel_m] = 0.0

        idx = np.unravel_index(np.argmax(score), score.shape)
        return float(gx[idx]), float(gy[idx])


# ---------------------------------------------------------------------------
# Experiment engine
# ---------------------------------------------------------------------------

class SyntheticExperiment:
    """Runs a single synthetic twin experiment.

    Args:
        ground_truth: Sources that are *actually* leaking.
        all_equipment: All known equipment (for prior & opportunity map).
        wind_params: Dict with wind_speed, wind_direction_deg, stability_class.
        wind_sequence: Optional list of wind_params dicts (cycled per step).
        grid_size: Site extent in meters.
        resolution: Grid cell size in meters (coarser = faster).
        sensor_noise_ppm: Std-dev of Gaussian sensor noise (ppm).
    """

    def __init__(
        self,
        ground_truth: List[dict],
        all_equipment: List[dict],
        wind_params: dict,
        wind_sequence: Optional[List[dict]] = None,
        grid_size: float = GRID_SIZE_M,
        resolution: float = 10.0,
        sensor_noise_ppm: float = 0.5,
        time_resolved: bool = False,
    ):
        self.ground_truth = ground_truth
        self.all_equipment = all_equipment
        self.wind_params = wind_params
        self.wind_sequence = wind_sequence
        self.grid_size = grid_size
        self.resolution = resolution
        self.sensor_noise_ppm = sensor_noise_ppm
        self.time_resolved = time_resolved

        # Build grid
        self.grid_x, self.grid_y = create_grid(grid_size, resolution)

        # Compute opportunity map from all known equipment
        _, _, self.concentration_ppm, self.detection_prob = compute_opportunity_map(
            sources=all_equipment,
            wind_speed=wind_params["wind_speed"],
            wind_direction_deg=wind_params["wind_direction_deg"],
            stability_class=wind_params["stability_class"],
            grid_size=grid_size,
            resolution=resolution,
        )

        # Compute spatial prior
        prior_probs = compute_all_priors(all_equipment)
        self.spatial_prior = create_spatial_prior(
            self.grid_x, self.grid_y, all_equipment, prior_probs,
        )

    def _get_wind(self, step: int) -> dict:
        """Return wind params for a given step (cycles through sequence)."""
        if self.wind_sequence:
            return self.wind_sequence[step % len(self.wind_sequence)]
        return self.wind_params

    def simulate_measurement(
        self,
        x: float,
        y: float,
        wind: dict,
        rng: np.random.Generator,
    ) -> Tuple[float, bool, float]:
        """Simulate a sensor reading at (x, y) using ground-truth sources.

        If ``time_resolved`` is True, each source's emission is gated by a
        Bernoulli draw based on its duty cycle (on/off per measurement).
        Otherwise, the time-averaged effective rate (emission_rate * duty_cycle)
        is used deterministically.

        Returns:
            (measured_ppm, detected, true_ppm)
        """
        total_conc = np.zeros(1)
        for src in self.ground_truth:
            duty_cycle = src.get("duty_cycle", 1.0)

            if self.time_resolved:
                # Bernoulli draw: source is on or off this instant
                if rng.random() >= duty_cycle:
                    continue  # source is off this measurement
                effective_rate = src["emission_rate"]
            else:
                # Time-averaged: scale emission by duty cycle
                effective_rate = src["emission_rate"] * duty_cycle

            total_conc += gaussian_plume(
                receptor_x=np.array([x]),
                receptor_y=np.array([y]),
                receptor_z=RECEPTOR_HEIGHT_M,
                source_x=src["x"],
                source_y=src["y"],
                source_z=src.get("z", 0.0),
                emission_rate=effective_rate,
                wind_speed=wind["wind_speed"],
                wind_direction_deg=wind["wind_direction_deg"],
                stability_class=wind["stability_class"],
            )

        true_ppm = float(concentration_to_ppm(total_conc)[0])

        # Probabilistic detection via sigmoid + MDL
        p_det = float(detection_probability(
            np.array([true_ppm]),
            threshold_ppm=DETECTION_THRESHOLD_PPM,
            mdl_ppm=SENSOR_MDL_PPM,
        )[0])
        detected = bool(rng.random() < p_det)

        # Noisy concentration (only meaningful when detected)
        if detected:
            measured_ppm = max(0.0, true_ppm + rng.normal(0, self.sensor_noise_ppm))
        else:
            measured_ppm = 0.0

        return measured_ppm, detected, true_ppm

    def run(
        self,
        strategy: Strategy,
        num_steps: int = 20,
        seed: int = 42,
        start_x: float = 0.0,
        start_y: float = 0.0,
    ) -> ExperimentResult:
        """Run the experiment with a given strategy.

        Args:
            strategy: Object with ``next_location(state)`` method.
            num_steps: Number of measurement steps.
            seed: Random seed for reproducibility.
            start_x, start_y: Worker starting position.

        Returns:
            ExperimentResult with full history.
        """
        rng = np.random.default_rng(seed)

        # Initialise belief map
        belief_obj = BayesianBeliefMap(
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            prior=self.spatial_prior,
            sources=self.all_equipment,
        )

        half = self.grid_size / 2.0
        current_x, current_y = start_x, start_y

        # History tracking
        entropy_history = [compute_total_entropy(belief_obj.get_belief_map())]
        detection_events: List[bool] = []
        cumulative_distance = [0.0]
        measurements: List[Measurement] = []

        for step in range(num_steps):
            wind = self._get_wind(step)

            state = {
                "grid_x": self.grid_x,
                "grid_y": self.grid_y,
                "belief": belief_obj.get_belief_map(),
                "detection_prob": self.detection_prob,
                "wind_params": wind,
                "current_x": current_x,
                "current_y": current_y,
                "all_equipment": self.all_equipment,
                "step": step,
                "grid_bounds": (-half, half),
                "rng": rng,
            }

            # Strategy picks next location
            next_x, next_y = strategy.next_location(state)

            # Clip to grid bounds
            next_x = float(np.clip(next_x, -half, half))
            next_y = float(np.clip(next_y, -half, half))

            # Travel
            dist = float(np.hypot(next_x - current_x, next_y - current_y))
            current_x, current_y = next_x, next_y

            # Simulate measurement
            measured_ppm, detected, true_ppm = self.simulate_measurement(
                current_x, current_y, wind, rng,
            )

            meas = Measurement(
                x=current_x,
                y=current_y,
                concentration_ppm=measured_ppm,
                detected=detected,
                wind_speed=wind["wind_speed"],
                wind_direction_deg=wind["wind_direction_deg"],
                stability_class=wind["stability_class"],
            )

            belief_obj.update(meas)

            # Record
            measurements.append(meas)
            detection_events.append(detected)
            cumulative_distance.append(cumulative_distance[-1] + dist)
            entropy_history.append(
                compute_total_entropy(belief_obj.get_belief_map())
            )

        return ExperimentResult(
            strategy_name=strategy.name,
            num_steps=num_steps,
            entropy_history=entropy_history,
            detection_events=detection_events,
            cumulative_distance=cumulative_distance,
            measurements=measurements,
            final_belief=belief_obj.get_belief_map(),
        )


# ---------------------------------------------------------------------------
# Strategy comparator
# ---------------------------------------------------------------------------

class StrategyComparator:
    """Run multiple strategies on the same experiment and compare results.

    Args:
        experiment: A SyntheticExperiment instance.
        strategies: List of Strategy objects to compare.
        num_steps: Number of measurement steps per run.
    """

    def __init__(
        self,
        experiment: SyntheticExperiment,
        strategies: List[Strategy],
        num_steps: int = 20,
    ):
        self.experiment = experiment
        self.strategies = strategies
        self.num_steps = num_steps

    def run(self, seed: int = 42) -> Dict[str, ExperimentResult]:
        """Run all strategies and return results keyed by strategy name.

        Each strategy uses the same seed for fair comparison (the random
        number generator is re-seeded per run so measurement noise is
        reproducible but strategy choices differ).
        """
        results = {}
        for strategy in self.strategies:
            result = self.experiment.run(
                strategy=strategy,
                num_steps=self.num_steps,
                seed=seed,
            )
            results[strategy.name] = result
        return results

    def summary_table(
        self,
        results: Dict[str, ExperimentResult],
    ) -> List[dict]:
        """Generate a summary comparison table.

        Returns:
            List of dicts, one per strategy, with key metrics.
        """
        rows = []
        for name, res in results.items():
            rows.append({
                "strategy": name,
                "total_distance_m": res.total_distance,
                "total_detections": res.total_detections,
                "entropy_reduction": res.entropy_reduction,
                "final_entropy": res.entropy_history[-1],
                "entropy_reduction_pct": (
                    100.0 * res.entropy_reduction / max(res.entropy_history[0], 1e-15)
                ),
            })
        return rows

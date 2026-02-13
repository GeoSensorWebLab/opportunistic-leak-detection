"""
Multi-Day Campaign Planning for iterative leak detection.

Maintains state across multiple inspection days, using the previous
day's posterior as the next day's prior.  Supports serialization for
persistence across sessions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.measurement import Measurement
from models.bayesian import BayesianBeliefMap
from models.prior import compute_all_priors, create_spatial_prior
from optimization.opportunity_map import create_grid, compute_opportunity_map
from optimization.tasking import (
    compute_tasking_scores,
    cached_path_deviation,
    recommend_waypoints,
)
from optimization.information_gain import compute_total_entropy, compute_information_scores
from config import (
    GRID_SIZE_M,
    GRID_RESOLUTION_M,
    DEVIATION_EPSILON,
    MAX_DEVIATION_M,
    TOP_K_RECOMMENDATIONS,
)


@dataclass
class DayPlan:
    """Record of a single inspection day."""

    day_index: int
    starting_belief: np.ndarray
    measurements: List[Measurement] = field(default_factory=list)
    recommendations: List[dict] = field(default_factory=list)
    ending_belief: Optional[np.ndarray] = None
    entropy_start: float = 0.0
    entropy_end: float = 0.0


@dataclass
class CampaignState:
    """Persistent state for a multi-day campaign."""

    days: List[DayPlan] = field(default_factory=list)
    current_belief: Optional[np.ndarray] = None
    grid_x: Optional[np.ndarray] = None
    grid_y: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Serialize to a dict suitable for NPZ storage.

        Grid arrays and belief are stored as numpy arrays.
        Day-level metadata is stored as lists of dicts.
        """
        days_data = []
        for d in self.days:
            day_entry = {
                "day_index": d.day_index,
                "entropy_start": d.entropy_start,
                "entropy_end": d.entropy_end,
                "num_measurements": len(d.measurements),
                "num_recommendations": len(d.recommendations),
            }
            if d.starting_belief is not None:
                day_entry["starting_belief"] = d.starting_belief.tolist()
            if d.ending_belief is not None:
                day_entry["ending_belief"] = d.ending_belief.tolist()
            days_data.append(day_entry)

        result = {
            "num_days": len(self.days),
            "days": days_data,
            "has_belief": self.current_belief is not None,
        }
        if self.current_belief is not None:
            result["current_belief"] = self.current_belief.tolist()
        if self.grid_x is not None:
            result["grid_x"] = self.grid_x.tolist()
        if self.grid_y is not None:
            result["grid_y"] = self.grid_y.tolist()
        return result

    @staticmethod
    def from_dict(data: dict, grid_x: Optional[np.ndarray] = None, grid_y: Optional[np.ndarray] = None) -> "CampaignState":
        """Reconstruct a CampaignState from serialized data.

        Args:
            data: Dict produced by to_dict().
            grid_x: Optional override grid (uses stored grid if None).
            grid_y: Optional override grid (uses stored grid if None).

        Returns:
            Restored CampaignState with belief and day history.
        """
        # Restore grids
        gx = grid_x
        gy = grid_y
        if gx is None and "grid_x" in data:
            gx = np.array(data["grid_x"])
        if gy is None and "grid_y" in data:
            gy = np.array(data["grid_y"])

        # Restore current belief
        belief = None
        if "current_belief" in data:
            belief = np.array(data["current_belief"])

        # Restore days (lightweight — no measurements/recommendations)
        days = []
        for dd in data.get("days", []):
            starting = None
            ending = None
            if "starting_belief" in dd:
                starting = np.array(dd["starting_belief"])
            if "ending_belief" in dd:
                ending = np.array(dd["ending_belief"])
            day = DayPlan(
                day_index=dd["day_index"],
                starting_belief=starting if starting is not None else np.array([]),
                ending_belief=ending,
                entropy_start=dd.get("entropy_start", 0.0),
                entropy_end=dd.get("entropy_end", 0.0),
            )
            days.append(day)

        state = CampaignState(
            days=days,
            current_belief=belief,
            grid_x=gx,
            grid_y=gy,
        )
        return state


def plan_next_day(
    campaign: CampaignState,
    sources: List[dict],
    wind_params: dict,
    baseline_path: np.ndarray,
    max_deviation: float = MAX_DEVIATION_M,
    scoring_mode: str = "heuristic",
    top_k: int = TOP_K_RECOMMENDATIONS,
    grid_size: float = GRID_SIZE_M,
    resolution: float = GRID_RESOLUTION_M,
) -> DayPlan:
    """Generate recommendations for the next inspection day.

    Uses the campaign's accumulated posterior as the prior for scoring.

    Args:
        campaign: Current campaign state.
        sources: List of source dicts.
        wind_params: Dict with wind_speed, wind_direction_deg, stability_class.
        baseline_path: Worker's baseline inspection path.
        max_deviation: Max allowable deviation from path.
        scoring_mode: "heuristic" or "eer".
        top_k: Number of recommendations.
        grid_size: Site extent in meters.
        resolution: Grid resolution in meters.

    Returns:
        DayPlan with starting_belief and recommendations.
    """
    # Ensure grid is set
    if campaign.grid_x is None or campaign.grid_y is None:
        campaign.grid_x, campaign.grid_y = create_grid(grid_size, resolution)

    # Use accumulated posterior as starting belief, or compute fresh prior
    if campaign.current_belief is not None:
        starting_belief = campaign.current_belief.copy()
    else:
        prior_probs = compute_all_priors(sources)
        starting_belief = create_spatial_prior(
            campaign.grid_x, campaign.grid_y, sources, prior_probs,
        )
        campaign.current_belief = starting_belief.copy()

    # Compute opportunity map
    _, _, concentration_ppm, detection_prob = compute_opportunity_map(
        sources=sources,
        wind_speed=wind_params["wind_speed"],
        wind_direction_deg=wind_params["wind_direction_deg"],
        stability_class=wind_params["stability_class"],
        grid_size=grid_size,
        resolution=resolution,
    )

    # Compute path deviation
    from optimization.tasking import compute_path_deviation
    deviation = compute_path_deviation(campaign.grid_x, campaign.grid_y, baseline_path)

    # Compute scores using accumulated posterior
    if scoring_mode == "eer":
        avg_emission = float(np.mean(
            [s["emission_rate"] * s.get("duty_cycle", 1.0) for s in sources]
        )) if sources else 0.5
        scores = compute_information_scores(
            grid_x=campaign.grid_x,
            grid_y=campaign.grid_y,
            belief=starting_belief,
            deviation=deviation,
            max_deviation=max_deviation,
            wind_speed=wind_params["wind_speed"],
            wind_direction_deg=wind_params["wind_direction_deg"],
            stability_class=wind_params["stability_class"],
            avg_emission=avg_emission,
            epsilon=DEVIATION_EPSILON,
        )
    else:
        scores = compute_tasking_scores(
            grid_x=campaign.grid_x,
            grid_y=campaign.grid_y,
            detection_prob=detection_prob,
            baseline_path=baseline_path,
            epsilon=DEVIATION_EPSILON,
            max_deviation=max_deviation,
            precomputed_deviation=deviation,
            prior_weight=starting_belief,
        )

    recommendations = recommend_waypoints(
        grid_x=campaign.grid_x,
        grid_y=campaign.grid_y,
        scores=scores,
        detection_prob=detection_prob,
        concentration_ppm=concentration_ppm,
        top_k=top_k,
    )

    day = DayPlan(
        day_index=len(campaign.days),
        starting_belief=starting_belief.copy(),
        recommendations=recommendations,
        entropy_start=compute_total_entropy(starting_belief),
    )

    return day


def close_day(
    campaign: CampaignState,
    day_plan: DayPlan,
    measurements: List[Measurement],
    sources: List[dict],
) -> None:
    """Finalize a day by running Bayesian updates and advancing the campaign.

    Args:
        campaign: Campaign state to update in place.
        day_plan: The day plan being closed.
        measurements: Measurements collected during the day.
        sources: List of source dicts.
    """
    # Guard against calling close_day on an already-closed plan
    if day_plan.ending_belief is not None:
        raise RuntimeError(
            f"Day {day_plan.day_index} has already been closed."
        )

    day_plan.measurements = measurements

    # Run Bayesian updates from the day's starting belief
    bayesian_obj = BayesianBeliefMap(
        grid_x=campaign.grid_x,
        grid_y=campaign.grid_y,
        prior=day_plan.starting_belief,
        sources=sources,
    )

    for m in measurements:
        bayesian_obj.update(m)

    ending_belief = bayesian_obj.get_belief_map()
    day_plan.ending_belief = ending_belief
    day_plan.entropy_end = compute_total_entropy(ending_belief)

    # Advance campaign: next day starts from this posterior
    campaign.current_belief = ending_belief.copy()
    campaign.days.append(day_plan)


def campaign_summary(campaign: CampaignState) -> dict:
    """Aggregate metrics across all days in a campaign.

    Returns:
        Dict with total_days, total_measurements, entropy_per_day,
        total_entropy_reduction.
    """
    total_measurements = sum(len(d.measurements) for d in campaign.days)
    entropy_per_day = [
        {"day": d.day_index, "start": d.entropy_start, "end": d.entropy_end}
        for d in campaign.days
    ]

    total_reduction = 0.0
    if campaign.days:
        total_reduction = campaign.days[0].entropy_start - campaign.days[-1].entropy_end

    return {
        "total_days": len(campaign.days),
        "total_measurements": total_measurements,
        "entropy_per_day": entropy_per_day,
        "total_entropy_reduction": total_reduction,
    }

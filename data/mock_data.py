"""
Mock Data for the Methane Leak Opportunistic Tasking System.

Provides synthetic leak sources and a worker baseline path.
Designed to be swapped out for real SCADA/SensorUp data later.
"""

import numpy as np
from typing import List, Tuple


def get_leak_sources() -> List[dict]:
    """
    Return a list of potential methane leak sources (static assets).

    Each source represents infrastructure like wellheads, compressor stations,
    or pipeline junctions that could develop leaks.

    Returns:
        List of dicts with keys: 'name', 'x', 'y', 'z', 'emission_rate'.
        Coordinates in meters, emission_rate in kg/s.
    """
    return [
        {
            "name": "Wellhead A",
            "x": -260.0,
            "y": 130.0,
            "z": 0.0,
            "emission_rate": 0.3,
            "equipment_type": "wellhead",
            "age_years": 22,
            "production_rate_mcfd": 850.0,
            "last_inspection_days": 120,
            "duty_cycle": 0.7,
        },
        {
            "name": "Compressor Station B",
            "x": 75.0,
            "y": -140.0,
            "z": 2.0,
            "emission_rate": 0.8,
            "equipment_type": "compressor",
            "age_years": 8,
            "production_rate_mcfd": 5200.0,
            "last_inspection_days": 45,
            "duty_cycle": 0.95,
        },
        {
            "name": "Pipeline Junction C",
            "x": 280.0,
            "y": -30.0,
            "z": 0.0,
            "emission_rate": 0.15,
            "equipment_type": "pipeline_junction",
            "age_years": 15,
            "production_rate_mcfd": 3100.0,
            "last_inspection_days": 200,
            "duty_cycle": 0.3,
        },
        {
            "name": "Storage Tank D",
            "x": -110.0,
            "y": -270.0,
            "z": 3.0,
            "emission_rate": 1.2,
            "equipment_type": "storage_tank",
            "age_years": 30,
            "production_rate_mcfd": 0.0,
            "last_inspection_days": 90,
            "duty_cycle": 1.0,
        },
        {
            "name": "Valve Cluster E",
            "x": 270.0,
            "y": -60.0,
            "z": 0.5,
            "emission_rate": 0.4,
            "equipment_type": "valve",
            "age_years": 12,
            "production_rate_mcfd": 2400.0,
            "last_inspection_days": 60,
            "duty_cycle": 0.6,
        },
    ]


def get_inspection_targets() -> List[dict]:
    """
    Return the 5 mandatory equipment inspection stops.

    These are the locations the field worker must visit regardless of
    opportunistic methane detection. Order reflects a logical south-to-north
    walk across the facility.

    Returns:
        List of dicts with keys: 'name', 'x', 'y'.
    """
    return [
        {"name": "Storage Tank D",       "x": -110.0, "y": -270.0},
        {"name": "Compressor Station B", "x":   75.0, "y": -140.0},
        {"name": "Valve Cluster E",      "x":  270.0, "y":  -60.0},
        {"name": "Pipeline Junction C",  "x":  280.0, "y":  -30.0},
        {"name": "Wellhead A",           "x": -260.0, "y":  130.0},
    ]


def get_baseline_path() -> np.ndarray:
    """
    Return a free-walking baseline path between inspection targets.

    Workers carry handheld methane detectors and walk freely across
    facility grounds — they are not constrained to roads. This path
    represents direct point-to-point walking between mandatory
    equipment inspection stops with midpoints on long segments
    for deviation accuracy.

    Route: South Gate → Tank D → Compressor B → Valve E →
           Pipeline C → Wellhead A → North Gate

    Returns:
        (N, 2) array of [x, y] coordinates in meters.
        Total distance ~1,600 m.
    """
    return np.array(
        [
            # ── South gate entry ──
            [0.0, -380.0],          # South gate
            # ── Walk to Storage Tank D ──
            [-55.0, -325.0],        # Midpoint: gate → tank
            [-110.0, -270.0],       # Storage Tank D (inspection stop)
            # ── Walk to Compressor Station B ──
            [-18.0, -205.0],        # Midpoint: tank → compressor
            [75.0, -140.0],         # Compressor Station B (inspection stop)
            # ── Walk to Valve Cluster E ──
            [172.0, -100.0],        # Midpoint: compressor → valve
            [270.0, -60.0],         # Valve Cluster E (inspection stop)
            # ── Walk to Pipeline Junction C ──
            [280.0, -30.0],         # Pipeline Junction C (inspection stop, nearby)
            # ── Walk to Wellhead A ──
            [10.0, 50.0],           # Midpoint: pipeline → wellhead
            [-260.0, 130.0],        # Wellhead A (inspection stop)
            # ── Walk to North gate ──
            [-130.0, 190.0],        # Midpoint: wellhead → north gate
            [0.0, 250.0],           # North gate — route complete
        ]
    )


def get_road_following_path() -> np.ndarray:
    """
    Return the legacy road-following baseline path (27 waypoints).

    This path rigidly follows facility roads with out-and-back detours.
    Preserved for comparison with the free-walking path.
    Total distance ~2,100 m.

    Returns:
        (N, 2) array of [x, y] coordinates in meters.
    """
    return np.array(
        [
            # ── South gate entry ──
            [0.0, -380.0],       # Gate entrance
            [0.0, -340.0],       # Main Access Road heading north
            # ── Tank Farm detour (west along Tank Farm Road) ──
            [0.0, -270.0],       # Junction: Main Access Rd / Tank Farm Rd
            [-60.0, -270.0],     # Tank Farm Rd — heading west
            [-110.0, -270.0],    # Near Storage Tank D
            [-180.0, -270.0],    # Far end of tank containment berm
            [-110.0, -270.0],    # Return east past tank
            [0.0, -270.0],       # Back on Main Access Rd
            # ── Compressor area ──
            [0.0, -160.0],       # Main Access Rd — passing compressor zone
            [75.0, -140.0],      # Compressor Pad (near Compressor Station B)
            [0.0, -120.0],       # Back to Main Access Rd
            # ── Pipe rack spine (east-west) ──
            [0.0, -30.0],        # Main Pipe Rack junction
            [150.0, -30.0],      # Pipe rack heading east
            [280.0, -30.0],      # Valve / Metering Station (Pipeline Junction C)
            [270.0, -60.0],      # Valve Cluster E
            [280.0, -30.0],      # Back to pipe rack
            [150.0, -30.0],      # Return west along pipe rack
            [0.0, -30.0],        # Center of pipe rack
            # ── North toward Well Pad ──
            [0.0, 0.0],          # Site center
            [0.0, 120.0],        # Junction: Main Access Rd / Well Pad Access
            [-100.0, 120.0],     # Well Pad Access heading west
            [-200.0, 130.0],     # Approaching Well Pad
            [-260.0, 130.0],     # Wellhead A
            [-200.0, 130.0],     # Return east
            [0.0, 120.0],        # Back on Main Access Rd
            # ── Control Room & north gate exit ──
            [0.0, 50.0],         # Near Control Room
            [0.0, 120.0],        # Heading to north exit
            [0.0, 250.0],        # North gate — route complete
        ]
    )


def get_wind_distribution() -> List[dict]:
    """
    Return an 8-direction wind rose with equal weights.

    Each scenario has a cardinal/intercardinal direction, moderate speed,
    and neutral stability. Weights sum to 1.0.

    Returns:
        List of dicts with keys: 'direction', 'speed', 'stability_class', 'weight'.
    """
    directions = [0, 45, 90, 135, 180, 225, 270, 315]  # N, NE, E, SE, S, SW, W, NW
    return [
        {
            "direction": d,
            "speed": 3.0,
            "stability_class": "D",
            "weight": 1.0 / len(directions),
        }
        for d in directions
    ]


def get_wind_fan(
    center_direction: float,
    spread_deg: float = 30.0,
    num_scenarios: int = 8,
    speed: float = 3.0,
    stability_class: str = "D",
) -> List[dict]:
    """
    Generate a fan of wind directions around a center direction.

    Creates equally spaced scenarios spanning center ± spread_deg,
    with equal weights that sum to 1.0.

    Args:
        center_direction: Center wind direction in degrees (meteorological).
        spread_deg: Half-spread in degrees (fan covers center ± spread_deg).
        num_scenarios: Number of wind scenarios to generate.
        speed: Wind speed for all scenarios (m/s).
        stability_class: Pasquill-Gifford class for all scenarios.

    Returns:
        List of dicts with keys: 'direction', 'speed', 'stability_class', 'weight'.
    """
    if num_scenarios < 1:
        raise ValueError("num_scenarios must be >= 1")

    if num_scenarios == 1:
        directions = [center_direction]
    else:
        directions = np.linspace(
            center_direction - spread_deg,
            center_direction + spread_deg,
            num_scenarios,
        )

    return [
        {
            "direction": float(d % 360),
            "speed": speed,
            "stability_class": stability_class,
            "weight": 1.0 / num_scenarios,
        }
        for d in directions
    ]


def get_wind_scenarios() -> List[dict]:
    """
    Return preset wind scenarios for quick testing.

    Returns:
        List of dicts with keys: 'name', 'speed', 'direction', 'stability_class'.
    """
    return [
        {
            "name": "Light SW Breeze (Unstable)",
            "speed": 2.0,
            "direction": 225,
            "stability_class": "A",
        },
        {
            "name": "Moderate West Wind (Neutral)",
            "speed": 4.0,
            "direction": 270,
            "stability_class": "D",
        },
        {
            "name": "Strong North Wind (Stable)",
            "speed": 8.0,
            "direction": 0,
            "stability_class": "F",
        },
        {
            "name": "East Wind (Slightly Unstable)",
            "speed": 3.0,
            "direction": 90,
            "stability_class": "B",
        },
    ]

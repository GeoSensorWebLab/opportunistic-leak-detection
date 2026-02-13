"""
Pre-defined synthetic experiment scenarios for validation.

Each scenario function returns a dict with:
    - ground_truth: list of source dicts that are ACTUALLY leaking
    - all_equipment: list of all known equipment (for prior model)
    - wind_params: dict with wind_speed, wind_direction_deg, stability_class
    - description: human-readable summary
"""

from typing import List
from data.mock_data import get_leak_sources


def _all_equipment() -> List[dict]:
    """Return the full set of known equipment locations."""
    return get_leak_sources()


def scenario_a_simple() -> dict:
    """Scenario A: Single source, steady west wind.

    The simplest case — one wellhead leaking, moderate wind, neutral
    stability.  Tests basic detection and localisation.
    """
    equipment = _all_equipment()
    return {
        "ground_truth": [equipment[0]],  # Wellhead A
        "all_equipment": equipment,
        "wind_params": {
            "wind_speed": 3.0,
            "wind_direction_deg": 270.0,
            "stability_class": "D",
        },
        "description": "Single source (Wellhead A), steady west wind, neutral stability",
    }


def scenario_b_multi_source() -> dict:
    """Scenario B: Three sources, steady west wind.

    Tests multi-source localisation — the belief map must identify
    separate peaks for each leaking source.
    """
    equipment = _all_equipment()
    return {
        "ground_truth": [
            equipment[0],  # Wellhead A (0.3 kg/s)
            equipment[2],  # Pipeline Junction C (0.15 kg/s)
            equipment[3],  # Storage Tank D (1.2 kg/s)
        ],
        "all_equipment": equipment,
        "wind_params": {
            "wind_speed": 3.0,
            "wind_direction_deg": 270.0,
            "stability_class": "D",
        },
        "description": "Three sources (A, C, D), steady west wind, neutral stability",
    }


def scenario_c_variable_wind() -> dict:
    """Scenario C: Single source, variable wind.

    Wind changes direction between steps.  Returns a list of wind
    parameter dicts in the 'wind_sequence' key.  The experiment runner
    should cycle through them.
    """
    equipment = _all_equipment()
    wind_sequence = [
        {"wind_speed": 3.0, "wind_direction_deg": 270.0, "stability_class": "D"},
        {"wind_speed": 2.5, "wind_direction_deg": 250.0, "stability_class": "C"},
        {"wind_speed": 4.0, "wind_direction_deg": 290.0, "stability_class": "D"},
        {"wind_speed": 2.0, "wind_direction_deg": 225.0, "stability_class": "B"},
        {"wind_speed": 3.5, "wind_direction_deg": 300.0, "stability_class": "D"},
    ]
    return {
        "ground_truth": [equipment[1]],  # Compressor Station B
        "all_equipment": equipment,
        "wind_params": wind_sequence[0],  # default / initial
        "wind_sequence": wind_sequence,
        "description": "Single source (Compressor B), rotating wind, mixed stability",
    }


def scenario_d_complex() -> dict:
    """Scenario D: Three sources, variable wind.

    The most challenging case — multiple leaks with varying wind
    conditions.  Tests the full Bayesian inference pipeline.
    """
    equipment = _all_equipment()
    wind_sequence = [
        {"wind_speed": 3.0, "wind_direction_deg": 270.0, "stability_class": "D"},
        {"wind_speed": 2.5, "wind_direction_deg": 250.0, "stability_class": "C"},
        {"wind_speed": 4.0, "wind_direction_deg": 290.0, "stability_class": "D"},
        {"wind_speed": 2.0, "wind_direction_deg": 180.0, "stability_class": "B"},
        {"wind_speed": 3.5, "wind_direction_deg": 315.0, "stability_class": "D"},
    ]
    return {
        "ground_truth": [
            equipment[0],  # Wellhead A
            equipment[1],  # Compressor Station B
            equipment[4],  # Valve Cluster E
        ],
        "all_equipment": equipment,
        "wind_params": wind_sequence[0],
        "wind_sequence": wind_sequence,
        "description": "Three sources (A, B, E), rotating wind, mixed stability",
    }


def scenario_e_intermittent() -> dict:
    """Scenario E: Two intermittent sources, steady wind.

    Tests duty-cycle / intermittent leak behavior.  Pipeline Junction C
    has a low duty cycle (0.3 — intermittent pressure cycling), while
    Compressor Station B is nearly continuous (0.9).  Steady west wind
    keeps plume physics simple so the test isolates temporal effects.
    """
    equipment = _all_equipment()
    # Override duty cycles for ground truth sources
    gt_pipeline = dict(equipment[2], duty_cycle=0.3)   # Pipeline Junction C
    gt_compressor = dict(equipment[1], duty_cycle=0.9)  # Compressor Station B
    return {
        "ground_truth": [gt_pipeline, gt_compressor],
        "all_equipment": equipment,
        "wind_params": {
            "wind_speed": 3.0,
            "wind_direction_deg": 270.0,
            "stability_class": "D",
        },
        "description": "Two intermittent sources (C dc=0.3, B dc=0.9), steady west wind",
    }

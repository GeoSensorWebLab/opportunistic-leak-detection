#!/usr/bin/env python3
"""
One-at-a-Time Sensitivity Analysis.

Sweeps individual parameters while holding others at default values.
For each parameter value, runs ``scenario_a_simple`` with the
``InformationTheoreticStrategy`` and reports key performance metrics.

Usage:
    uv run python experiments/run_sensitivity_analysis.py
    uv run python experiments/run_sensitivity_analysis.py --steps 10 --seed 42
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import config

from validation.synthetic_twin import (
    SyntheticExperiment,
    InformationTheoreticStrategy,
)
from validation.scenarios import scenario_a_simple
from validation.metrics import (
    source_detection_rate,
    localization_rmse,
    information_efficiency,
    entropy_reduction_fraction,
)


# ---------------------------------------------------------------------------
# Parameter sweep definitions
# ---------------------------------------------------------------------------

PARAM_SWEEPS = {
    "DETECTION_THRESHOLD_PPM": [3.0, 5.0, 7.0],
    "SENSOR_MDL_PPM": [0.5, 1.0, 2.0],
    "FALSE_ALARM_RATE": [0.001, 0.01, 0.1],
    "DEVIATION_SCALE_M": [25.0, 50.0, 100.0],
    "EER_SUBSAMPLE": [1, 2, 4, 8],
    "PRIOR_KERNEL_RADIUS_M": [50.0, 100.0, 200.0],
}

# Modules that import these config values by value need patching too.
# Maps config attribute name -> list of (module_path, attr_name) to patch.
_VALUE_IMPORT_PATCHES = {
    "DETECTION_THRESHOLD_PPM": [
        ("validation.synthetic_twin", "DETECTION_THRESHOLD_PPM"),
    ],
    "SENSOR_MDL_PPM": [
        ("validation.synthetic_twin", "SENSOR_MDL_PPM"),
    ],
}


def _patch_config(attr: str, value):
    """Set a config attribute and any by-value copies in consuming modules."""
    setattr(config, attr, value)
    for mod_path, mod_attr in _VALUE_IMPORT_PATCHES.get(attr, []):
        mod = sys.modules.get(mod_path)
        if mod is not None:
            setattr(mod, mod_attr, value)


def _get_originals() -> dict:
    """Snapshot all swept attributes from config."""
    return {attr: getattr(config, attr) for attr in PARAM_SWEEPS}


def _restore_originals(originals: dict):
    """Restore all attributes to their original values."""
    for attr, val in originals.items():
        _patch_config(attr, val)


# ---------------------------------------------------------------------------
# Single run helper
# ---------------------------------------------------------------------------

def run_single(
    num_steps: int,
    resolution: float,
    seed: int,
) -> dict:
    """Run scenario_a_simple with EER strategy and return metrics."""
    sc = scenario_a_simple()
    exp = SyntheticExperiment(
        ground_truth=sc["ground_truth"],
        all_equipment=sc["all_equipment"],
        wind_params=sc["wind_params"],
        resolution=resolution,
    )
    strategy = InformationTheoreticStrategy(subsample=getattr(config, "EER_SUBSAMPLE", 4))
    result = exp.run(strategy, num_steps=num_steps, seed=seed)

    det_rate = source_detection_rate(
        result.final_belief,
        exp.grid_x,
        exp.grid_y,
        sc["ground_truth"],
    )
    rmse = localization_rmse(
        result.final_belief,
        exp.grid_x,
        exp.grid_y,
        sc["ground_truth"],
    )
    eff = information_efficiency(
        result.entropy_history,
        result.cumulative_distance,
    )
    ent_frac = entropy_reduction_fraction(result.entropy_history)

    return {
        "total_detections": result.total_detections,
        "total_distance_m": result.total_distance,
        "entropy_reduction_pct": 100.0 * ent_frac,
        "localization_rmse_m": rmse,
        "source_detection_rate": det_rate,
        "info_efficiency_bits_per_m": eff,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sensitivity(
    num_steps: int = 20,
    resolution: float = 20.0,
    seed: int = 42,
    verbose: bool = True,
):
    """Run one-at-a-time sensitivity analysis.

    Returns:
        List of dicts, one per (parameter, value) pair.
    """
    originals = _get_originals()
    all_rows = []

    for param_name, values in PARAM_SWEEPS.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"Sweeping: {param_name}")
            print(f"  Default: {originals[param_name]}")
            print(f"  Values:  {values}")
            print(f"{'='*70}")
            print(f"  {'Value':>10}  {'Detections':>10}  {'Distance':>10}  "
                  f"{'Ent %':>8}  {'RMSE':>8}  {'DetRate':>8}  {'Eff':>10}")
            print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

        for val in values:
            # Patch this parameter only; all others stay at default
            _restore_originals(originals)
            _patch_config(param_name, val)

            try:
                t0 = time.time()
                metrics = run_single(num_steps, resolution, seed)
                elapsed = time.time() - t0
            finally:
                _restore_originals(originals)

            row = {"parameter": param_name, "value": val, **metrics}
            all_rows.append(row)

            if verbose:
                print(
                    f"  {val:>10}  "
                    f"{metrics['total_detections']:>10}  "
                    f"{metrics['total_distance_m']:>10.0f}  "
                    f"{metrics['entropy_reduction_pct']:>7.1f}%  "
                    f"{metrics['localization_rmse_m']:>8.1f}  "
                    f"{metrics['source_detection_rate']:>7.0%}  "
                    f"{metrics['info_efficiency_bits_per_m']:>10.4f}"
                    f"  ({elapsed:.1f}s)"
                )

    return all_rows


def print_summary(rows: list):
    """Print a summary of the sensitivity analysis."""
    print(f"\n\n{'='*70}")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*70}")

    # Group by parameter
    params = {}
    for row in rows:
        p = row["parameter"]
        if p not in params:
            params[p] = []
        params[p].append(row)

    for param_name, param_rows in params.items():
        ent_vals = [r["entropy_reduction_pct"] for r in param_rows]
        val_labels = [r["value"] for r in param_rows]
        best_idx = int(np.argmax(ent_vals))
        print(f"\n  {param_name}:")
        print(f"    Best value: {val_labels[best_idx]} "
              f"(entropy reduction: {ent_vals[best_idx]:.1f}%)")
        print(f"    Range: {min(ent_vals):.1f}% — {max(ent_vals):.1f}%")


def main():
    parser = argparse.ArgumentParser(description="One-at-a-Time Sensitivity Analysis")
    parser.add_argument("--steps", type=int, default=20, help="Steps per experiment")
    parser.add_argument("--resolution", type=float, default=20.0, help="Grid resolution (m)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-value output")
    args = parser.parse_args()

    print("Sensitivity Analysis")
    print(f"Steps: {args.steps}, Resolution: {args.resolution}m, Seed: {args.seed}")
    print(f"Parameters: {list(PARAM_SWEEPS.keys())}")

    rows = run_sensitivity(
        num_steps=args.steps,
        resolution=args.resolution,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print_summary(rows)

    total_runs = sum(len(v) for v in PARAM_SWEEPS.values())
    print(f"\nTotal: {total_runs} experiment runs completed.")


if __name__ == "__main__":
    main()

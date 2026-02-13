#!/usr/bin/env python3
"""
Strategy Comparison Benchmark.

Runs all routing strategies on all synthetic scenarios and produces
a comparison table with key performance metrics.  Supports multiple
seeds for variance reporting (mean +/- std).

Usage:
    uv run python experiments/run_strategy_comparison.py
    uv run python experiments/run_strategy_comparison.py --steps 30 --seed 123
    uv run python experiments/run_strategy_comparison.py --seeds 42 123 456 --steps 20
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from validation.synthetic_twin import (
    SyntheticExperiment,
    StrategyComparator,
    RandomStrategy,
    GridSearchStrategy,
    MaxDetectionStrategy,
    OpportunisticStrategy,
    InformationTheoreticStrategy,
)
from validation.scenarios import (
    scenario_a_simple,
    scenario_b_multi_source,
    scenario_c_variable_wind,
    scenario_d_complex,
    scenario_e_intermittent,
    scenario_f_no_leaks,
    scenario_g_extreme,
)
from validation.metrics import (
    source_detection_rate,
    localization_rmse,
    information_efficiency,
    entropy_reduction_fraction,
    convergence_step,
    first_detection_step,
)


ALL_SCENARIOS = {
    "A (simple)": scenario_a_simple,
    "B (multi-source)": scenario_b_multi_source,
    "C (variable wind)": scenario_c_variable_wind,
    "D (complex)": scenario_d_complex,
    "E (intermittent)": scenario_e_intermittent,
    "F (no leaks)": scenario_f_no_leaks,
    "G (extreme)": scenario_g_extreme,
}


def build_strategies():
    """Return all strategies to compare."""
    return [
        RandomStrategy(),
        GridSearchStrategy(step_m=80.0),
        MaxDetectionStrategy(),
        OpportunisticStrategy(),
        InformationTheoreticStrategy(subsample=4),
    ]


def run_comparison(
    num_steps: int = 20,
    resolution: float = 20.0,
    seed: int = 42,
    verbose: bool = True,
):
    """Run all strategies on all scenarios and return detailed results.

    Returns:
        List of dicts, one per (scenario, strategy) pair.
    """
    strategies = build_strategies()
    all_rows = []

    for sc_name, sc_fn in ALL_SCENARIOS.items():
        sc = sc_fn()

        if verbose:
            print(f"\n{'='*70}")
            print(f"Scenario {sc_name}: {sc['description']}")
            print(f"{'='*70}")

        t0 = time.time()
        experiment = SyntheticExperiment(
            ground_truth=sc["ground_truth"],
            all_equipment=sc["all_equipment"],
            wind_params=sc["wind_params"],
            wind_sequence=sc.get("wind_sequence"),
            resolution=resolution,
        )

        comparator = StrategyComparator(
            experiment, strategies, num_steps=num_steps,
        )
        results = comparator.run(seed=seed)
        elapsed = time.time() - t0

        if verbose:
            print(f"  Completed in {elapsed:.1f}s\n")
            print(f"  {'Strategy':<20} {'Detections':>10} {'Distance':>10} "
                  f"{'Entropy %':>10} {'RMSE':>10} {'Det Rate':>10} "
                  f"{'Eff (b/m)':>10} {'1st Det':>8}")
            print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} "
                  f"{'-'*10} {'-'*10} {'-'*8}")

        for strat_name, result in results.items():
            det_rate = source_detection_rate(
                result.final_belief,
                experiment.grid_x,
                experiment.grid_y,
                sc["ground_truth"],
            )
            rmse = localization_rmse(
                result.final_belief,
                experiment.grid_x,
                experiment.grid_y,
                sc["ground_truth"],
            )
            eff = information_efficiency(
                result.entropy_history,
                result.cumulative_distance,
            )
            ent_frac = entropy_reduction_fraction(result.entropy_history)
            first_det = first_detection_step(result.detection_events)
            conv = convergence_step(result.entropy_history, 0.5)

            row = {
                "scenario": sc_name,
                "strategy": strat_name,
                "total_detections": result.total_detections,
                "total_distance_m": result.total_distance,
                "entropy_reduction_pct": 100.0 * ent_frac,
                "localization_rmse_m": rmse,
                "source_detection_rate": det_rate,
                "info_efficiency_bits_per_m": eff,
                "first_detection_step": first_det,
                "convergence_step_50pct": conv,
            }
            all_rows.append(row)

            if verbose:
                first_str = str(first_det) if first_det is not None else "—"
                print(
                    f"  {strat_name:<20} {result.total_detections:>10} "
                    f"{result.total_distance:>10.0f} "
                    f"{100*ent_frac:>9.1f}% "
                    f"{rmse:>10.1f} "
                    f"{det_rate:>9.0%}  "
                    f"{eff:>10.4f} "
                    f"{first_str:>8}"
                )

    return all_rows


def run_multi_seed(
    seeds: list,
    num_steps: int = 20,
    resolution: float = 20.0,
    verbose: bool = True,
):
    """Run the benchmark across multiple seeds and aggregate statistics.

    Args:
        seeds: List of random seeds.
        num_steps: Steps per experiment.
        resolution: Grid resolution in meters.
        verbose: Print per-seed output.

    Returns:
        Dict mapping (scenario, strategy) -> dict of metric lists.
    """
    aggregated = {}  # (scenario, strategy) -> {metric_name: [values]}

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'#'*70}")
            print(f"  SEED {seed} ({i+1}/{len(seeds)})")
            print(f"{'#'*70}")

        rows = run_comparison(
            num_steps=num_steps,
            resolution=resolution,
            seed=seed,
            verbose=verbose,
        )

        for row in rows:
            key = (row["scenario"], row["strategy"])
            if key not in aggregated:
                aggregated[key] = {
                    "entropy_reduction_pct": [],
                    "localization_rmse_m": [],
                    "source_detection_rate": [],
                    "info_efficiency_bits_per_m": [],
                    "total_detections": [],
                    "total_distance_m": [],
                }
            for metric in aggregated[key]:
                aggregated[key][metric].append(row[metric])

    return aggregated


def print_summary(rows):
    """Print a cross-scenario summary ranking strategies."""
    strategy_names = sorted(set(r["strategy"] for r in rows))

    print(f"\n\n{'='*70}")
    print("SUMMARY: Average across all scenarios")
    print(f"{'='*70}")
    print(f"  {'Strategy':<20} {'Avg Ent %':>10} {'Avg RMSE':>10} "
          f"{'Avg DetRate':>12} {'Avg Eff':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    summaries = []
    for name in strategy_names:
        strat_rows = [r for r in rows if r["strategy"] == name]
        avg_ent = np.mean([r["entropy_reduction_pct"] for r in strat_rows])
        avg_rmse = np.mean([r["localization_rmse_m"] for r in strat_rows])
        avg_det = np.mean([r["source_detection_rate"] for r in strat_rows])
        avg_eff = np.mean([r["info_efficiency_bits_per_m"] for r in strat_rows])
        summaries.append((name, avg_ent, avg_rmse, avg_det, avg_eff))

    # Sort by entropy reduction (best first)
    summaries.sort(key=lambda x: x[1], reverse=True)

    for name, avg_ent, avg_rmse, avg_det, avg_eff in summaries:
        print(
            f"  {name:<20} {avg_ent:>9.1f}% {avg_rmse:>10.1f} "
            f"{avg_det:>11.0%}  {avg_eff:>10.4f}"
        )


def print_multi_seed_summary(aggregated: dict):
    """Print mean +/- std across seeds for each (scenario, strategy)."""
    print(f"\n\n{'='*80}")
    print("MULTI-SEED SUMMARY: mean +/- std across seeds")
    print(f"{'='*80}")

    # Group by strategy across all scenarios
    strategy_agg = {}
    for (sc_name, strat_name), metrics in aggregated.items():
        if strat_name not in strategy_agg:
            strategy_agg[strat_name] = {
                "entropy_reduction_pct": [],
                "localization_rmse_m": [],
                "source_detection_rate": [],
                "info_efficiency_bits_per_m": [],
            }
        for metric in strategy_agg[strat_name]:
            strategy_agg[strat_name][metric].extend(metrics[metric])

    print(f"\n  {'Strategy':<20} {'Ent % (mean±std)':>20} {'RMSE (mean±std)':>20} "
          f"{'DetRate (mean±std)':>22} {'Eff (mean±std)':>20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*22} {'-'*20}")

    summaries = []
    for name, metrics in strategy_agg.items():
        ent_vals = metrics["entropy_reduction_pct"]
        rmse_vals = metrics["localization_rmse_m"]
        det_vals = metrics["source_detection_rate"]
        eff_vals = metrics["info_efficiency_bits_per_m"]
        summaries.append((
            name,
            np.mean(ent_vals), np.std(ent_vals),
            np.mean(rmse_vals), np.std(rmse_vals),
            np.mean(det_vals), np.std(det_vals),
            np.mean(eff_vals), np.std(eff_vals),
        ))

    summaries.sort(key=lambda x: x[1], reverse=True)

    for (name, ent_m, ent_s, rmse_m, rmse_s, det_m, det_s, eff_m, eff_s) in summaries:
        print(
            f"  {name:<20} "
            f"{ent_m:>8.1f}% ± {ent_s:>5.1f}  "
            f"{rmse_m:>8.1f} ± {rmse_s:>5.1f}  "
            f"{100*det_m:>8.0f}% ± {100*det_s:>4.0f}%  "
            f"{eff_m:>8.4f} ± {eff_s:>.4f}"
        )

    # Per-scenario breakdown
    scenarios = sorted(set(sc for sc, _ in aggregated.keys()))
    for sc_name in scenarios:
        print(f"\n  --- {sc_name} ---")
        print(f"  {'Strategy':<20} {'Ent %':>16} {'RMSE':>16} {'DetRate':>16}")

        for strat_name in sorted(set(s for _, s in aggregated.keys())):
            key = (sc_name, strat_name)
            if key not in aggregated:
                continue
            m = aggregated[key]
            ent = m["entropy_reduction_pct"]
            rmse = m["localization_rmse_m"]
            det = m["source_detection_rate"]
            print(
                f"  {strat_name:<20} "
                f"{np.mean(ent):>6.1f} ± {np.std(ent):>4.1f}  "
                f"{np.mean(rmse):>6.1f} ± {np.std(rmse):>4.1f}  "
                f"{100*np.mean(det):>5.0f}% ± {100*np.std(det):>3.0f}%"
            )


def main():
    parser = argparse.ArgumentParser(description="Strategy Comparison Benchmark")
    parser.add_argument("--steps", type=int, default=20, help="Steps per experiment")
    parser.add_argument("--resolution", type=float, default=20.0, help="Grid resolution (m)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (single-seed mode)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for variance reporting (e.g., --seeds 42 123 456)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-scenario output")
    args = parser.parse_args()

    if args.seeds and len(args.seeds) > 1:
        # Multi-seed mode
        print("Strategy Comparison Benchmark (Multi-Seed)")
        print(f"Steps: {args.steps}, Resolution: {args.resolution}m, "
              f"Seeds: {args.seeds}")

        aggregated = run_multi_seed(
            seeds=args.seeds,
            num_steps=args.steps,
            resolution=args.resolution,
            verbose=not args.quiet,
        )
        print_multi_seed_summary(aggregated)

        total_runs = len(aggregated) * len(args.seeds)
        print(f"\nTotal: {total_runs} experiment runs across {len(args.seeds)} seeds.")
    else:
        # Single-seed mode (original behavior)
        seed = args.seeds[0] if args.seeds else args.seed
        print("Strategy Comparison Benchmark")
        print(f"Steps: {args.steps}, Resolution: {args.resolution}m, Seed: {seed}")

        rows = run_comparison(
            num_steps=args.steps,
            resolution=args.resolution,
            seed=seed,
            verbose=not args.quiet,
        )
        print_summary(rows)

        print(f"\nTotal: {len(rows)} experiment runs completed.")


if __name__ == "__main__":
    main()

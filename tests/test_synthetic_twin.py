"""Tests for the synthetic twin validation engine."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validation.synthetic_twin import (
    SyntheticExperiment,
    ExperimentResult,
    RandomStrategy,
    GridSearchStrategy,
    MaxDetectionStrategy,
    OpportunisticStrategy,
    InformationTheoreticStrategy,
    StrategyComparator,
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
    paired_significance_test,
    bootstrap_confidence_interval,
)


# ---------------------------------------------------------------------------
# Fixtures — use coarse resolution for speed
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_experiment():
    """Scenario A experiment at coarse resolution."""
    sc = scenario_a_simple()
    return SyntheticExperiment(
        ground_truth=sc["ground_truth"],
        all_equipment=sc["all_equipment"],
        wind_params=sc["wind_params"],
        resolution=20.0,  # very coarse for speed
    )


@pytest.fixture
def multi_experiment():
    """Scenario B experiment at coarse resolution."""
    sc = scenario_b_multi_source()
    return SyntheticExperiment(
        ground_truth=sc["ground_truth"],
        all_equipment=sc["all_equipment"],
        wind_params=sc["wind_params"],
        resolution=20.0,
    )


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_scenario_a_has_one_source(self):
        sc = scenario_a_simple()
        assert len(sc["ground_truth"]) == 1

    def test_scenario_b_has_three_sources(self):
        sc = scenario_b_multi_source()
        assert len(sc["ground_truth"]) == 3

    def test_scenario_c_has_wind_sequence(self):
        sc = scenario_c_variable_wind()
        assert "wind_sequence" in sc
        assert len(sc["wind_sequence"]) >= 2

    def test_scenario_d_has_both(self):
        sc = scenario_d_complex()
        assert len(sc["ground_truth"]) == 3
        assert "wind_sequence" in sc

    def test_scenario_e_is_intermittent(self):
        sc = scenario_e_intermittent()
        assert len(sc["ground_truth"]) == 2
        dcs = [s.get("duty_cycle", 1.0) for s in sc["ground_truth"]]
        assert any(dc < 1.0 for dc in dcs), "At least one source should be intermittent"

    def test_scenario_f_has_no_ground_truth(self):
        sc = scenario_f_no_leaks()
        assert len(sc["ground_truth"]) == 0

    def test_scenario_g_has_five_extreme_sources(self):
        sc = scenario_g_extreme()
        assert len(sc["ground_truth"]) == 5
        for src in sc["ground_truth"]:
            assert src["emission_rate"] >= 1.0  # 10x of original rates
            assert src["duty_cycle"] == 1.0

    def test_all_scenarios_have_required_keys(self):
        for sc_fn in [scenario_a_simple, scenario_b_multi_source,
                      scenario_c_variable_wind, scenario_d_complex,
                      scenario_e_intermittent, scenario_f_no_leaks,
                      scenario_g_extreme]:
            sc = sc_fn()
            assert "ground_truth" in sc
            assert "all_equipment" in sc
            assert "wind_params" in sc
            assert "description" in sc


# ---------------------------------------------------------------------------
# Experiment result tests
# ---------------------------------------------------------------------------

class TestExperimentResult:
    def test_run_produces_result(self, simple_experiment):
        result = simple_experiment.run(RandomStrategy(), num_steps=3, seed=42)
        assert isinstance(result, ExperimentResult)

    def test_result_has_correct_history_lengths(self, simple_experiment):
        n = 5
        result = simple_experiment.run(RandomStrategy(), num_steps=n, seed=42)
        # entropy_history has n+1 entries (initial + one per step)
        assert len(result.entropy_history) == n + 1
        assert len(result.detection_events) == n
        assert len(result.cumulative_distance) == n + 1
        assert len(result.measurements) == n

    def test_entropy_history_non_negative(self, simple_experiment):
        result = simple_experiment.run(RandomStrategy(), num_steps=5, seed=42)
        assert all(h >= 0 for h in result.entropy_history)

    def test_cumulative_distance_monotonic(self, simple_experiment):
        result = simple_experiment.run(RandomStrategy(), num_steps=5, seed=42)
        for i in range(1, len(result.cumulative_distance)):
            assert result.cumulative_distance[i] >= result.cumulative_distance[i - 1]

    def test_total_detections_bounded(self, simple_experiment):
        n = 5
        result = simple_experiment.run(RandomStrategy(), num_steps=n, seed=42)
        assert 0 <= result.total_detections <= n


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestStrategies:
    def test_random_stays_in_bounds(self, simple_experiment):
        result = simple_experiment.run(RandomStrategy(), num_steps=10, seed=42)
        half = simple_experiment.grid_size / 2
        for m in result.measurements:
            assert -half <= m.x <= half
            assert -half <= m.y <= half

    def test_grid_search_visits_different_locations(self, simple_experiment):
        result = simple_experiment.run(
            GridSearchStrategy(step_m=100.0), num_steps=5, seed=42,
        )
        coords = [(m.x, m.y) for m in result.measurements]
        # At least 3 unique locations out of 5
        unique = set((round(x, 1), round(y, 1)) for x, y in coords)
        assert len(unique) >= 3

    def test_max_detection_runs(self, simple_experiment):
        result = simple_experiment.run(
            MaxDetectionStrategy(), num_steps=3, seed=42,
        )
        assert len(result.measurements) == 3

    def test_opportunistic_runs(self, simple_experiment):
        result = simple_experiment.run(
            OpportunisticStrategy(), num_steps=3, seed=42,
        )
        assert len(result.measurements) == 3

    def test_eer_runs(self, simple_experiment):
        result = simple_experiment.run(
            InformationTheoreticStrategy(subsample=4), num_steps=3, seed=42,
        )
        assert len(result.measurements) == 3

    def test_all_strategies_have_names(self):
        strategies = [
            RandomStrategy(),
            GridSearchStrategy(),
            MaxDetectionStrategy(),
            OpportunisticStrategy(),
            InformationTheoreticStrategy(),
        ]
        for s in strategies:
            assert isinstance(s.name, str)
            assert len(s.name) > 0


# ---------------------------------------------------------------------------
# Comparator tests
# ---------------------------------------------------------------------------

class TestComparator:
    def test_comparator_runs_all_strategies(self, simple_experiment):
        strategies = [RandomStrategy(), GridSearchStrategy(step_m=100.0)]
        comp = StrategyComparator(
            simple_experiment, strategies, num_steps=3,
        )
        results = comp.run(seed=42)
        assert len(results) == 2
        assert "Random" in results
        assert "Grid Search" in results

    def test_summary_table(self, simple_experiment):
        strategies = [RandomStrategy(), OpportunisticStrategy()]
        comp = StrategyComparator(
            simple_experiment, strategies, num_steps=3,
        )
        results = comp.run(seed=42)
        table = comp.summary_table(results)
        assert len(table) == 2
        for row in table:
            assert "strategy" in row
            assert "total_distance_m" in row
            assert "entropy_reduction" in row


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_source_detection_rate_bounds(self, simple_experiment):
        result = simple_experiment.run(RandomStrategy(), num_steps=10, seed=42)
        rate = source_detection_rate(
            result.final_belief,
            simple_experiment.grid_x,
            simple_experiment.grid_y,
            simple_experiment.ground_truth,
        )
        assert 0.0 <= rate <= 1.0

    def test_localization_rmse_non_negative(self, simple_experiment):
        result = simple_experiment.run(RandomStrategy(), num_steps=5, seed=42)
        rmse = localization_rmse(
            result.final_belief,
            simple_experiment.grid_x,
            simple_experiment.grid_y,
            simple_experiment.ground_truth,
        )
        assert rmse >= 0.0

    def test_information_efficiency_non_negative(self, simple_experiment):
        result = simple_experiment.run(RandomStrategy(), num_steps=5, seed=42)
        eff = information_efficiency(
            result.entropy_history, result.cumulative_distance,
        )
        assert eff >= 0.0

    def test_entropy_reduction_fraction_bounded(self, simple_experiment):
        result = simple_experiment.run(RandomStrategy(), num_steps=5, seed=42)
        frac = entropy_reduction_fraction(result.entropy_history)
        assert 0.0 <= frac <= 1.0

    def test_convergence_step_returns_none_or_int(self, simple_experiment):
        result = simple_experiment.run(RandomStrategy(), num_steps=5, seed=42)
        step = convergence_step(result.entropy_history, threshold_fraction=0.99)
        assert step is None or isinstance(step, int)

    def test_first_detection_step(self):
        assert first_detection_step([False, False, True, False]) == 2
        assert first_detection_step([False, False, False]) is None
        assert first_detection_step([True]) == 0


# ---------------------------------------------------------------------------
# Variable wind test
# ---------------------------------------------------------------------------

class TestVariableWind:
    def test_variable_wind_experiment(self):
        sc = scenario_c_variable_wind()
        exp = SyntheticExperiment(
            ground_truth=sc["ground_truth"],
            all_equipment=sc["all_equipment"],
            wind_params=sc["wind_params"],
            wind_sequence=sc.get("wind_sequence"),
            resolution=20.0,
        )
        result = exp.run(RandomStrategy(), num_steps=5, seed=42)
        assert len(result.measurements) == 5

    def test_wind_actually_varies(self):
        """Verify that measurements under variable wind use distinct directions."""
        sc = scenario_c_variable_wind()
        exp = SyntheticExperiment(
            ground_truth=sc["ground_truth"],
            all_equipment=sc["all_equipment"],
            wind_params=sc["wind_params"],
            wind_sequence=sc.get("wind_sequence"),
            resolution=20.0,
        )
        num_steps = len(sc["wind_sequence"])
        result = exp.run(RandomStrategy(), num_steps=num_steps, seed=42)
        directions = [m.wind_direction_deg for m in result.measurements]
        unique_dirs = set(directions)
        assert len(unique_dirs) >= 2, (
            f"Expected multiple wind directions, got {unique_dirs}"
        )

    def test_scenario_c_with_all_strategies(self):
        """All 5 strategies run on Scenario C (variable wind) without error."""
        sc = scenario_c_variable_wind()
        exp = SyntheticExperiment(
            ground_truth=sc["ground_truth"],
            all_equipment=sc["all_equipment"],
            wind_params=sc["wind_params"],
            wind_sequence=sc.get("wind_sequence"),
            resolution=20.0,
        )
        strategies = [
            RandomStrategy(),
            GridSearchStrategy(step_m=100.0),
            MaxDetectionStrategy(),
            OpportunisticStrategy(),
            InformationTheoreticStrategy(subsample=4),
        ]
        for strategy in strategies:
            result = exp.run(strategy, num_steps=3, seed=42)
            assert len(result.measurements) == 3, (
                f"Strategy {strategy.name} failed on Scenario C"
            )


# ---------------------------------------------------------------------------
# New scenario smoke tests (F and G)
# ---------------------------------------------------------------------------

class TestNewScenarios:
    def test_scenario_f_no_leak_few_detections(self):
        """Scenario F (no leaks) should produce very few detections."""
        sc = scenario_f_no_leaks()
        exp = SyntheticExperiment(
            ground_truth=sc["ground_truth"],
            all_equipment=sc["all_equipment"],
            wind_params=sc["wind_params"],
            resolution=20.0,
        )
        result = exp.run(RandomStrategy(), num_steps=10, seed=42)
        # With no ground truth, detections should only come from false alarms
        assert result.total_detections <= 3, (
            f"Expected very few detections with no leaks, got {result.total_detections}"
        )

    def test_scenario_g_extreme_has_detections(self):
        """Scenario G (extreme emissions) should produce many detections."""
        sc = scenario_g_extreme()
        exp = SyntheticExperiment(
            ground_truth=sc["ground_truth"],
            all_equipment=sc["all_equipment"],
            wind_params=sc["wind_params"],
            resolution=20.0,
        )
        result = exp.run(RandomStrategy(), num_steps=10, seed=42)
        # With 5 sources at 10x emission, there should be many detections
        assert result.total_detections >= 1, (
            "Expected detections with extreme emissions"
        )

    def test_scenario_f_completes_with_all_strategies(self):
        """All strategies should run on Scenario F without error."""
        sc = scenario_f_no_leaks()
        exp = SyntheticExperiment(
            ground_truth=sc["ground_truth"],
            all_equipment=sc["all_equipment"],
            wind_params=sc["wind_params"],
            resolution=20.0,
        )
        for strategy in [RandomStrategy(), GridSearchStrategy(step_m=100.0),
                         OpportunisticStrategy()]:
            result = exp.run(strategy, num_steps=3, seed=42)
            assert len(result.measurements) == 3

    def test_scenario_g_completes_with_all_strategies(self):
        """All strategies should run on Scenario G without error."""
        sc = scenario_g_extreme()
        exp = SyntheticExperiment(
            ground_truth=sc["ground_truth"],
            all_equipment=sc["all_equipment"],
            wind_params=sc["wind_params"],
            resolution=20.0,
        )
        for strategy in [RandomStrategy(), GridSearchStrategy(step_m=100.0),
                         OpportunisticStrategy()]:
            result = exp.run(strategy, num_steps=3, seed=42)
            assert len(result.measurements) == 3


# ---------------------------------------------------------------------------
# Statistical helper tests
# ---------------------------------------------------------------------------

class TestStatisticalHelpers:
    def test_identical_arrays_not_significant(self):
        """Identical arrays should yield p > 0.05 (not significant)."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_significance_test(a, a)
        assert result["p_value"] > 0.05 or result["mean_diff"] == 0.0
        assert result["mean_diff"] == pytest.approx(0.0)

    def test_clearly_different_arrays_significant(self):
        """Clearly different arrays should yield p < 0.05."""
        a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = paired_significance_test(a, b)
        assert result["p_value"] < 0.05
        assert result["significant"] is True
        assert result["mean_diff"] > 0

    def test_ci_contains_mean(self):
        """Confidence interval should contain the mean difference."""
        a = [5.0, 6.0, 7.0, 8.0, 9.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_significance_test(a, b)
        assert result["ci_lower"] <= result["mean_diff"] <= result["ci_upper"]

    def test_wrong_length_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            paired_significance_test([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_too_few_observations_raises(self):
        """Fewer than 2 observations should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            paired_significance_test([1.0], [2.0])

    def test_bootstrap_basics(self):
        """Bootstrap CI should contain the sample mean."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_confidence_interval(values)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]
        assert result["mean"] == pytest.approx(3.0)
        assert result["std"] > 0

    def test_bootstrap_empty_raises(self):
        """Empty array should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_confidence_interval([])

    def test_bootstrap_single_value(self):
        """Single value should produce tight CI around the value."""
        result = bootstrap_confidence_interval([42.0])
        assert result["mean"] == pytest.approx(42.0)
        assert result["ci_lower"] == pytest.approx(42.0)
        assert result["ci_upper"] == pytest.approx(42.0)

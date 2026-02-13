# Improvement Plan: Methane Leak Opportunistic Tasking System

## Current State Assessment

The existing codebase is a **well-structured research prototype** (~2,000 LOC across 14 Python files) with:
- Correct Gaussian plume dispersion physics (Pasquill-Gifford, ground reflection)
- Sigmoid detection probability model
- Basic scoring optimizer (detection_prob / deviation)
- Path optimization (nearest-neighbor + 2-opt)
- Interactive Streamlit UI with Plotly dual-panel visualization
- Mock data layer (5 sources, 1 path, 4 wind presets)

**Gap analysis against the research plan reveals 6 major capability gaps and several incremental improvements needed.**

---

## Phase 1: Foundation Hardening (Weeks 1-2)

### 1.1 Test Suite

**Why:** Zero tests exist. Every subsequent improvement risks breaking existing functionality.

**Tasks:**
- Create `tests/` directory with `conftest.py` (shared fixtures: grid, sources, wind params)
- `tests/test_gaussian_plume.py` — Validate plume physics:
  - Concentration decreases with distance (monotonic decay downwind)
  - Ground reflection doubles ground-level concentration vs. infinite domain
  - Upwind receptors return zero
  - Symmetry about plume centerline
  - Known analytical solutions for specific stability classes
- `tests/test_detection.py` — Validate sigmoid behavior:
  - P = 0.5 at threshold
  - P → 0 far below threshold
  - P → 1 far above threshold
  - Monotonic increase
- `tests/test_opportunity_map.py` — Validate aggregation:
  - Single source matches direct plume computation
  - Multi-source superposition correctness
  - Grid dimensions match config
- `tests/test_tasking.py` — Validate optimization:
  - On-path cells score higher than off-path cells at same detection probability
  - Cells beyond max_deviation score zero
  - Non-maximum suppression enforces min separation
  - Waypoint count <= top_k
  - Optimized path visits all recommended waypoints
- `tests/test_integration.py` — End-to-end pipeline:
  - Full run from sources → opportunity map → scoring → recommendations → path
  - Regression tests with frozen mock data outputs

**Files created:**
```
tests/
├── conftest.py
├── test_gaussian_plume.py
├── test_detection.py
├── test_opportunity_map.py
├── test_tasking.py
└── test_integration.py
```

**Dependencies:** `pytest`, `pytest-cov` added to `pyproject.toml` dev dependencies.

---

### 1.2 Error Handling & Input Validation

**Why:** No validation exists. Invalid slider values or edge-case inputs could produce nonsensical results silently.

**Tasks:**
- Add input validation in `models/gaussian_plume.py`:
  - `wind_speed > 0` (already partially handled, formalize with ValueError)
  - `stability_class in {'A','B','C','D','E','F'}`
  - `emission_rate >= 0`
- Add validation in `optimization/tasking.py`:
  - `epsilon > 0`
  - `max_deviation > 0`
  - `top_k >= 1`
  - `baseline_path` has at least 2 points
- Add validation in `main.py` sidebar:
  - Wind speed minimum enforced at 0.5 m/s (already done via slider)
  - Display warnings for extreme parameter combinations (e.g., very fine grid + large site = slow)
- Add `try/except` blocks around computation in `main.py` with `st.error()` display

**Files modified:** `models/gaussian_plume.py`, `optimization/tasking.py`, `main.py`

---

### 1.3 Fix Path Deviation Approximation

**Why:** Current implementation uses point-to-nearest-waypoint distance instead of true point-to-line-segment distance. This causes systematic underestimation of deviation for grid cells near long path segments.

**Tasks:**
- Replace `cdist` approach in `compute_path_deviation()` with proper point-to-segment projection
- Reuse the existing `_project_onto_path()` logic from `tasking.py` (vectorized for the full grid)
- Validate with test case: point equidistant from two waypoints but close to the connecting segment

**Files modified:** `optimization/tasking.py`

---

### 1.4 Extract Hardcoded Constants to Config

**Why:** Several magic numbers are scattered across modules, contradicting the project convention.

**Tasks:**
- Move `M_air = 28.97` from `gaussian_plume.py` to `config.py`
- Move clustering threshold `0.08` from `build_optimized_path()` to `config.py` as `CLUSTER_FRACTION`
- Move compass position `(-400, 400)` from `plots.py` to `config.py`
- Move min separation `50.0` default from `recommend_waypoints()` to `config.py` as `MIN_WAYPOINT_SEPARATION_M`
- Move detection steepness `1.0` default to `config.py` as `DETECTION_STEEPNESS`

**Files modified:** `config.py`, `models/gaussian_plume.py`, `optimization/tasking.py`, `visualization/plots.py`

---

## Phase 2: Bayesian Inference Engine (Weeks 3-4)

### 2.1 Prior Belief Model (Stage 1 of Research Architecture)

**Why:** The research plan calls for a Generalized Extreme Value (GEV) distribution as the prior for leak probability, informed by historical well data (age, production rates, equipment type). Currently, all sources are treated equally with no prior probability weighting.

**Tasks:**
- Create `models/prior.py`:
  - `compute_prior_probability(sources)` — Assign prior leak probability to each source based on attributes:
    - Well age factor (older = higher probability, exponential growth)
    - Equipment type risk factor (compressor > valve > wellhead > tank)
    - Production rate factor (higher throughput = higher stress)
    - Maintenance history factor (time since last inspection)
  - `gev_emission_prior(sources)` — GEV distribution for emission rate estimation per source type
  - `create_spatial_prior(grid_x, grid_y, sources, prior_probs)` — Project source-level priors onto a 2D grid using distance-weighted kernel (each source contributes a radial probability field)
- Extend source dict schema in `data/mock_data.py`:
  - Add fields: `age_years`, `equipment_type`, `production_rate_mcfd`, `last_inspection_days`
  - Provide realistic mock values for the 5 existing sources
- Add prior probability as a weighting factor in `compute_tasking_scores()`:
  - New formula: `Score = Prior(x,y) * P_detection(x,y) / (Deviation(x,y) + ε)`

**Files created:** `models/prior.py`
**Files modified:** `data/mock_data.py`, `optimization/tasking.py`, `config.py`

---

### 2.2 Bayesian Update Engine (Stage 4 of Research Architecture)

**Why:** The core novel contribution is recursive Bayesian updating — using field measurements (including non-detections) to refine beliefs about leak locations over time. Nothing like this exists currently.

**Tasks:**
- Create `models/bayesian.py`:
  - `BayesianBeliefMap` class:
    - `__init__(grid_x, grid_y, prior)` — Initialize belief grid from prior
    - `update(measurement_location, concentration_ppm, wind_params)` — Bayes update:
      - Compute likelihood: P(measurement | source_at_each_cell) using plume model
      - Posterior ∝ Prior × Likelihood
      - Normalize
    - `update_negative(measurement_location, non_detection_threshold, wind_params)` — Negative information update:
      - If no detection at location, reduce probability for cells that *would* have produced detectable concentration there
      - Key innovation: absence of evidence *is* evidence of absence (bounded by sensor sensitivity)
    - `get_belief_map()` — Return current posterior grid
    - `get_entropy_map()` — Return Shannon entropy at each cell (uncertainty measure)
    - `reset()` — Reset to prior
  - `compute_expected_entropy_reduction(grid_x, grid_y, belief_map, candidate_location, wind_params)`:
    - For candidate measurement location, compute expected reduction in total entropy
    - Considers both detection and non-detection outcomes weighted by their probability
    - This is the Expected Entropy Reduction (EER) from the research plan
- Create `models/measurement.py`:
  - `Measurement` dataclass: `location (x,y)`, `concentration_ppm`, `timestamp`, `detected (bool)`
  - `MeasurementHistory` class: stores ordered list of measurements, provides serialization

**Files created:** `models/bayesian.py`, `models/measurement.py`
**Files modified:** `config.py` (Bayesian update hyperparameters)

---

### 2.3 Information-Theoretic Scoring (Stage 3 of Research Architecture)

**Why:** The current scoring function `P_detect / (deviation + ε)` is a heuristic. The research plan specifies an information-theoretic cost function using Expected Entropy Reduction (EER).

**Tasks:**
- Create `optimization/information_gain.py`:
  - `compute_information_value(grid_x, grid_y, belief_map, wind_params)` — Compute EER at each grid cell
  - `compute_cost_function(information_value, deviation, params)` — Implement research cost function:
    ```
    J = E_c / (1 + ζ * exp(d - λ * d_min))
    ```
    Where E_c = expected information gain, d = path distance, d_min = minimum distance, ζ and λ are tunable parameters
  - `compute_kl_divergence(prior, posterior)` — KL divergence between prior and posterior (alternative metric)
- Integrate as alternative scoring mode in `optimization/tasking.py`:
  - Add `scoring_mode` parameter: `"heuristic"` (current) or `"information_theoretic"` (new)
  - UI toggle in sidebar

**Files created:** `optimization/information_gain.py`
**Files modified:** `optimization/tasking.py`, `main.py`, `config.py`

---

## Phase 3: Enhanced Physics & Detection (Weeks 4-5)

### 3.1 Cross-Plume Integrated Concentration Model

**Why:** The research plan specifies a cross-plume integrated Gaussian formulation for robustness to turbulent fluctuations. Current model is the standard instantaneous Gaussian which overpredicts peak concentrations and underpredicts detection probability at off-axis locations.

**Tasks:**
- Add to `models/gaussian_plume.py`:
  - `crosswind_integrated_plume()` — Integrates the Gaussian in the crosswind direction, giving a line-averaged concentration:
    ```
    C_integrated = Q / (√(2π) * u * σz) * exp(-½((z-H)/σz)²)
    ```
    (no σy dependence — integrated out)
  - This produces broader, lower-peak plumes that better match time-averaged field measurements
- Add `plume_mode` parameter to `compute_opportunity_map()`: `"instantaneous"` (current) or `"integrated"` (new)
- UI toggle in sidebar under "Advanced Settings" expander

**Files modified:** `models/gaussian_plume.py`, `optimization/opportunity_map.py`, `main.py`

---

### 3.2 Time-Varying Source Model

**Why:** Real leaks are often intermittent (on/off cycling). The research plan identifies temporal dynamics as important. Currently all sources are assumed persistent.

**Tasks:**
- Extend source schema with temporal parameters:
  - `duty_cycle` (0-1): fraction of time leak is active
  - `onset_time_hr`: when leak started (relative to current time)
  - `duration_hr`: how long leak has been active
- Add to `models/gaussian_plume.py`:
  - `time_averaged_concentration()` — Multiply instantaneous concentration by duty_cycle
  - `puff_model()` — For short-duration releases, use Gaussian puff (spherical expansion) instead of plume (continuous)
- Update `data/mock_data.py` with temporal attributes for existing sources
- Update opportunity map computation to account for duty cycles

**Files modified:** `models/gaussian_plume.py`, `data/mock_data.py`, `optimization/opportunity_map.py`, `config.py`

---

### 3.3 Sensor Model Enhancement

**Why:** Current sigmoid model is static. Real sensors have response time, drift, and environmental sensitivity. The research plan calls for sensor-specific calibration.

**Tasks:**
- Extend `models/detection.py`:
  - `SensorModel` class with configurable parameters:
    - `threshold_ppm`: detection threshold
    - `steepness`: sigmoid steepness
    - `response_time_s`: time to reach 90% of final reading
    - `false_positive_rate`: background noise false alarm probability
    - `temperature_factor(temp_c)`: performance degradation at extreme temperatures
  - `detection_probability_with_exposure(concentration_ppm, exposure_time_s, sensor)`:
    - Accounts for dwell time at measurement location
    - Longer exposure → higher effective sensitivity
- Add `sensor_type` dropdown to sidebar: "Handheld (default)", "Fixed point", "OGI Camera", "Custom"
- Each type has preset parameters

**Files modified:** `models/detection.py`, `main.py`, `config.py`

---

## Phase 4: Multi-Worker & Campaign Planning (Weeks 5-6)

### 4.1 Multi-Worker Route Optimization

**Why:** Research Scenario 2 (emergency response) and Scenario 4 (multi-day campaign) require coordinating multiple workers. Current system handles only a single path.

**Tasks:**
- Create `optimization/multi_worker.py`:
  - `allocate_workers(workers, recommendations, constraints)`:
    - Input: List of worker start locations, list of recommended waypoints, worker schedules
    - Output: Assignment of waypoints to workers minimizing total travel while maximizing coverage
    - Algorithm: Modified k-medoids clustering of waypoints by worker proximity, then per-worker 2-opt
  - `compute_complementary_coverage(worker_paths, detection_prob)`:
    - Compute combined detection probability across all workers
    - Use complementary probability: P_combined = 1 - Π(1 - P_i)
    - Identify coverage gaps between workers
  - `deconflict_routes(worker_paths, min_separation_m)`:
    - Ensure workers don't redundantly cover the same area
    - Spatially diverse assignments
- Extend `data/mock_data.py`:
  - `get_worker_fleet()` — Returns list of 2-4 workers with start locations, schedules, capabilities
- Add multi-worker toggle to UI with worker count selector

**Files created:** `optimization/multi_worker.py`
**Files modified:** `data/mock_data.py`, `main.py`

---

### 4.2 Campaign Planning Mode

**Why:** Research Scenario 4 describes multi-day inspection campaigns with weather forecast integration and adaptive replanning. Currently the system operates in single-snapshot mode.

**Tasks:**
- Create `optimization/campaign.py`:
  - `CampaignPlanner` class:
    - `__init__(sources, facility, num_days, workers)` — Initialize multi-day plan
    - `plan_day(day_index, wind_forecast, previous_measurements)`:
      - Uses Bayesian belief map (updated from all previous days)
      - Incorporates wind forecast for that day
      - Allocates workers to maximize cumulative coverage
      - Returns per-worker daily routes
    - `get_coverage_report()` — Summary statistics:
      - Sources inspected / total
      - Cumulative information gain
      - Remaining uncertainty (entropy)
      - Worker utilization
  - `WindForecast` dataclass: `day`, `wind_speed`, `wind_direction`, `stability_class`, `confidence`
- Create multi-day planning UI page (Streamlit multi-page app):
  - Wind forecast input for 5-7 days
  - Day-by-day route visualization
  - Cumulative coverage heatmap
  - Campaign metrics dashboard

**Files created:** `optimization/campaign.py`, `pages/campaign_planner.py`
**Files modified:** `main.py` (convert to multi-page app structure)

---

## Phase 5: Validation Framework (Weeks 6-7)

### 5.1 Synthetic Twin Experiment Engine

**Why:** The research plan requires rigorous validation through synthetic twin experiments. No validation framework exists.

**Tasks:**
- Create `validation/` directory
- Create `validation/synthetic_twin.py`:
  - `SyntheticExperiment` class:
    - `__init__(ground_truth_sources, wind_sequence, worker_model)` — Set up experiment
    - `run_experiment(strategy, num_steps)`:
      - At each step: generate wind, compute optimal route, simulate measurement, update beliefs
      - Returns: detection timeline, belief convergence, path efficiency metrics
    - `simulate_measurement(location, true_sources, wind_params)`:
      - Compute true concentration at location from all active sources
      - Add Gaussian noise (sensor accuracy)
      - Return simulated reading
  - `StrategyComparator` class:
    - Compare multiple routing strategies on same ground truth:
      - `"random"` — Random waypoint selection
      - `"shortest_path"` — Minimum distance routing (no detection optimization)
      - `"grid_search"` — Systematic coverage
      - `"opportunistic"` — Our optimized approach
      - `"information_theoretic"` — EER-based approach
    - Returns comparative metrics table and convergence plots

**Files created:**
```
validation/
├── __init__.py
├── synthetic_twin.py
├── scenarios.py          # Pre-defined test scenarios (A-D from research plan)
└── metrics.py            # Detection metrics, efficiency metrics, learning metrics
```

---

### 5.2 Performance Metrics Module

**Why:** Research plan defines specific KPIs that need systematic tracking.

**Tasks:**
- Create `validation/metrics.py`:
  - **Detection metrics:**
    - `true_positive_rate(detected, ground_truth)` — Leak detection probability
    - `false_alarm_rate(false_positives, total_inspections)` — Spatial false positive rate
    - `time_to_detection(detection_timestamps, leak_start_times)` — Time from leak onset to detection
    - `localization_accuracy(estimated_locations, true_locations)` — RMSE from true source
  - **Efficiency metrics:**
    - `information_per_distance(entropy_reduction, distance_traveled)` — Gain per unit distance
    - `operational_cost_ratio(optimized_distance, baseline_distance)` — Cost overhead
    - `schedule_compliance(actual_time, planned_time)` — Timeliness
  - **Learning metrics:**
    - `posterior_uncertainty(belief_map)` — Total remaining entropy
    - `convergence_rate(entropy_history)` — Rate of entropy reduction over iterations
    - `prediction_accuracy(predicted_leak, actual_leak)` — Classification accuracy
- Integrate metrics display into validation UI

**Files created:** `validation/metrics.py`, `validation/scenarios.py`

---

## Phase 6: Visualization & UX Enhancements (Weeks 7-8)

### 6.1 Belief Map Visualization

**Why:** Bayesian inference produces evolving belief maps that are central to the research contribution. Need visualization for both the probability field and the uncertainty field.

**Tasks:**
- Add to `visualization/plots.py`:
  - `create_belief_figure(grid_x, grid_y, belief_map, entropy_map, measurements)` — Three-panel figure:
    - Left: Prior probability field
    - Center: Current posterior probability field
    - Right: Entropy (uncertainty) field
    - Overlay: measurement locations with detection/non-detection markers
  - `create_convergence_plot(entropy_history)` — Line chart showing entropy reduction over time
  - `create_comparison_plot(strategy_results)` — Side-by-side strategy comparison bar chart

**Files modified:** `visualization/plots.py`

---

### 6.2 Worker Guidance Interface

**Why:** Research Scenarios 1, 3, and 5 describe non-expert-friendly interfaces with simple visual indicators, route guidance, and acceptance/rejection options. Current UI is analyst-focused, not worker-focused.

**Tasks:**
- Create `pages/worker_guidance.py` — Simplified Streamlit page for field workers:
  - Large, simple map with color-coded priority zones (red/yellow/green)
  - Turn-by-turn route directions (text-based)
  - "Accept Route" / "Keep Original" buttons
  - Estimated time impact display ("Your route: 47 min → Recommended: 51 min")
  - Star-based information value indicator
  - Minimal controls (no sliders, no physics parameters)
- This page consumes the same backend computation but presents it differently

**Files created:** `pages/worker_guidance.py`

---

### 6.3 Dashboard & Reporting

**Why:** Research Scenario 4 requires campaign summary reporting. Scenario 6 requires historical analysis and continuous improvement tracking.

**Tasks:**
- Create `pages/dashboard.py`:
  - Campaign progress summary (sources inspected / total)
  - Detection history timeline
  - Worker utilization chart
  - Uncertainty reduction over time
  - Data export to CSV/JSON
- Add `st.download_button()` for results export in main page

**Files created:** `pages/dashboard.py`
**Files modified:** `main.py`

---

## Phase 7: Data Integration Preparation (Weeks 8-9)

### 7.1 Abstract Data Layer

**Why:** The research plan identifies SCADA integration, weather APIs, and GPS tracking as future integration points. Current mock data is hardcoded. An abstract interface makes swapping seamless.

**Tasks:**
- Create `data/interfaces.py`:
  - `DataProvider` abstract base class:
    - `get_sources() -> List[SourceDict]`
    - `get_worker_paths() -> List[np.ndarray]`
    - `get_wind_conditions() -> WindDict`
    - `get_facility_layout() -> LayoutDict`
  - `MockDataProvider(DataProvider)` — Wraps existing mock_data.py functions
  - `FileDataProvider(DataProvider)` — Loads from CSV/JSON files
- Update `main.py` to use `DataProvider` interface instead of direct function calls
- Add data source selector in sidebar: "Mock Data", "Load from File"

**Files created:** `data/interfaces.py`
**Files modified:** `main.py`

---

### 7.2 Weather API Stub

**Why:** Live weather data is a key integration point. Having the interface ready means only the API key/connection is needed later.

**Tasks:**
- Create `data/weather.py`:
  - `WeatherProvider` abstract class:
    - `get_current_wind() -> WindDict`
    - `get_forecast(hours_ahead) -> List[WindDict]`
  - `MockWeatherProvider` — Returns presets from mock_data.py
  - `OpenWeatherProvider` — Stub for OpenWeatherMap API (reads API key from env var, returns mock data if key missing)
- Wire into campaign planner

**Files created:** `data/weather.py`
**Files modified:** `config.py` (API endpoint configuration)

---

## Phase 8: Documentation & Reproducibility (Weeks 9-10)

### 8.1 Algorithm Documentation

**Tasks:**
- Create `docs/algorithms.md`:
  - Mathematical formulation of each algorithm with equation numbers
  - Corresponds to equations in the research paper
  - Cross-references to code implementations
- Create `docs/validation_results.md`:
  - Synthetic twin experiment results
  - Strategy comparison tables
  - Convergence plots

### 8.2 Reproducibility Package

**Tasks:**
- Create `experiments/` directory with runnable scripts:
  - `experiments/run_synthetic_twin.py` — Run all 4 scenarios (A-D)
  - `experiments/run_strategy_comparison.py` — Compare all routing strategies
  - `experiments/run_sensitivity_analysis.py` — Parameter sensitivity sweeps
- Each script saves results to `results/` directory
- Add `Makefile` or script for one-command experiment reproduction

---

## Priority Matrix

| Phase | Effort | Impact | Dependencies | Priority |
|-------|--------|--------|--------------|----------|
| 1. Foundation Hardening | Low | High | None | **P0 — Do First** |
| 2. Bayesian Engine | High | Critical | Phase 1 | **P0 — Core Research** |
| 3. Enhanced Physics | Medium | High | Phase 1 | **P1** |
| 5. Validation Framework | Medium | Critical | Phase 2 | **P1 — For Paper** |
| 6. Visualization & UX | Medium | Medium | Phase 2 | **P2** |
| 4. Multi-Worker | Medium | Medium | Phase 2 | **P2** |
| 7. Data Integration | Low | Medium | Phase 1 | **P2** |
| 8. Documentation | Low | High | Phase 5 | **P3** |

---

## Implementation Order (Recommended)

```
Week 1-2:  Phase 1 (Tests, validation, hardening)
              └─► Ensures all subsequent work doesn't break existing functionality

Week 3-4:  Phase 2 (Bayesian engine — core research contribution)
              └─► Prior model → Bayesian update → Information-theoretic scoring

Week 4-5:  Phase 3 (Enhanced physics)
              └─► Cross-plume integration, time-varying sources, sensor models

Week 5-6:  Phase 4 (Multi-worker) — can be deferred if not needed for paper
              └─► Multi-worker allocation, campaign planning

Week 6-7:  Phase 5 (Validation — required for paper)
              └─► Synthetic twin experiments, strategy comparison, metrics

Week 7-8:  Phase 6 (Visualization for Bayesian results)
              └─► Belief maps, convergence plots, worker guidance page

Week 8-9:  Phase 7 (Data integration preparation)
              └─► Abstract interfaces, weather stubs

Week 9-10: Phase 8 (Documentation, reproducibility)
              └─► Algorithm docs, experiment scripts, results
```

---

## New File Summary

```
New files to create:
├── tests/
│   ├── conftest.py
│   ├── test_gaussian_plume.py
│   ├── test_detection.py
│   ├── test_opportunity_map.py
│   ├── test_tasking.py
│   └── test_integration.py
├── models/
│   ├── prior.py                  # GEV prior probability model
│   ├── bayesian.py               # Bayesian belief map & update engine
│   └── measurement.py            # Measurement data structures
├── optimization/
│   ├── information_gain.py       # EER, KL divergence, cost function
│   ├── multi_worker.py           # Multi-worker route allocation
│   └── campaign.py               # Multi-day campaign planner
├── validation/
│   ├── __init__.py
│   ├── synthetic_twin.py         # Synthetic experiment engine
│   ├── scenarios.py              # Pre-defined test scenarios
│   └── metrics.py                # Detection, efficiency, learning metrics
├── data/
│   ├── interfaces.py             # Abstract data provider
│   └── weather.py                # Weather API integration
├── pages/
│   ├── worker_guidance.py        # Simplified field worker UI
│   ├── campaign_planner.py       # Multi-day planning page
│   └── dashboard.py              # Reporting & analytics
├── experiments/
│   ├── run_synthetic_twin.py
│   ├── run_strategy_comparison.py
│   └── run_sensitivity_analysis.py
└── docs/
    ├── algorithms.md
    └── validation_results.md
```

**Existing files modified:** `config.py`, `models/gaussian_plume.py`, `models/detection.py`, `optimization/opportunity_map.py`, `optimization/tasking.py`, `visualization/plots.py`, `data/mock_data.py`, `main.py`, `pyproject.toml`

---

## Success Criteria (Aligned with Research Plan)

| Criterion | Metric | Target |
|-----------|--------|--------|
| Physics validated | Unit tests pass, plume matches published data | 100% test pass |
| Bayesian inference works | Posterior converges to ground truth in synthetic experiments | Entropy reduction > 50% in 10 updates |
| Information gain scoring | EER-based routing outperforms heuristic scoring | > 15% improvement in detection rate |
| Routing efficiency | Optimized routing vs. random routing | > 30% improvement |
| Negative information | Non-detections reduce uncertainty measurably | Posterior probability decreases for ruled-out cells |
| Multi-worker coordination | Combined coverage exceeds single-worker | > 50% more coverage with 2 workers |
| Non-expert usability | Worker guidance page requires no training | Simple accept/reject interface |
| Reproducibility | All experiments runnable from scripts | Single command reproduction |

---

## Phase 9: Deep Code Review — Bug Fixes & Hardening (2026-02-13)

This section documents findings from a comprehensive code review of the entire codebase. Issues are categorized by severity and module.

---

### 9.1 CRITICAL / HIGH SEVERITY BUGS

#### Bug 1: Bayesian update breaks with tiny detection probabilities
- **File:** `models/bayesian.py:96-101`
- **Problem:** When `FALSE_ALARM_RATE=0` and detection probability is near-zero, the `safe_denom = max(denom, 1e-15)` guard distorts the posterior. A cell with `p_leak=1.0` and `p_detect=1e-20` computes `1e-20 / 1e-15 = 1e-5` instead of ~1.0. Incorrect Bayesian update.
- **Fix:** Floor detection probability at `1e-10` before update; avoid updating cells where signal is below noise floor.

#### Bug 2: `scoring_mode` parameter silently ignored in campaign planning
- **File:** `optimization/campaign.py:146-207`
- **Problem:** `plan_next_day()` accepts `scoring_mode="eer"` but always uses heuristic scoring. Users think they're getting EER-based campaigns but aren't.
- **Fix:** Implement the EER scoring branch using `compute_information_scores()`.

#### Bug 3: `max_deviation` parameter ignored in multi-worker allocation
- **File:** `optimization/multi_worker.py:111-121`
- **Problem:** `allocate_waypoints()` docstring says it respects `max_deviation`, but the code always assigns to the nearest worker regardless of distance. Waypoints 500m from any worker path get assigned anyway.
- **Fix:** Skip waypoints beyond `max_deviation` from all workers, or mark them as unassigned.

#### Bug 4: Streamlit cache crash with unhashable WorkerRoute
- **File:** `visualization/plots.py:269,282`
- **Problem:** `create_single_map_figure()` is `@st.cache_data` but receives `WorkerRoute` objects containing numpy arrays, causing `TypeError: unhashable type`.
- **Fix:** Remove `@st.cache_data` from functions that accept complex objects with numpy arrays.

#### Bug 5: State serialization JSON corruption
- **File:** `data/state_io.py:104,111`
- **Problem:** `str(npz["measurements_json"])` on a numpy char array can produce extra quotes in some numpy versions, causing `json.loads()` to fail or corrupt data.
- **Fix:** Use `.item()` to extract the scalar string: `npz["measurements_json"].item()`.

#### Bug 6: Division by small sigma produces infinite concentrations
- **File:** `models/gaussian_plume.py:126-138`
- **Problem:** At close range with stable conditions, `sigma_y` and `sigma_z` can become <0.01m, making `emission_rate / (2π * u * σy * σz)` approach infinity.
- **Fix:** Add `sigma = np.maximum(sigma, 0.01)` floor after computing sigmas in all plume functions.

#### Bug 7: IndexError on empty entropy history
- **File:** `visualization/plots.py:993`
- **Problem:** `create_convergence_figure()` accesses `steps[0]` and `steps[-1]` without checking for empty input.
- **Fix:** Add early return with empty figure if `len(entropy_history) < 2`.

---

### 9.2 MEDIUM SEVERITY — Logic & Correctness

#### Bug 8: Off-by-one in searchsorted path splitting
- **File:** `optimization/multi_worker.py:71-72`
- **Problem:** `side="right"` vs `side="left"` mismatch in `np.searchsorted` can create gaps/overlaps between worker segments.
- **Fix:** Use consistent `side` parameter and validate segment continuity.

#### Bug 9: Prior factors saturate at 1.0 indistinguishably
- **File:** `models/prior.py:49-66`
- **Problem:** Multiplicative factors for old, high-throughput, uninspected equipment all clip to P=1.0, losing ranking information.
- **Fix:** Cap individual factors before multiplication (e.g., `f_age` max 5.0, `f_production` max 3.0, `f_inspection` max 2.0).

#### Bug 10: Reverse plume hardcodes H=0.0 for all hypothetical sources
- **File:** `models/bayesian.py:166`
- **Problem:** Real sources may be elevated; forward and reverse plume models are asymmetric.
- **Fix:** Use actual source heights or averaged height from source list.

#### Bug 11: `set_belief()` doesn't validate input shape
- **File:** `models/bayesian.py:188-194`
- **Fix:** Add `if belief.shape != self.grid_x.shape: raise ValueError(...)`.

#### Bug 12: No validation that emission_rate >= 0
- **File:** `models/gaussian_plume.py:56-67`
- **Fix:** Add `if emission_rate < 0: raise ValueError(...)` at function entry.

#### Bug 13: Puff model has no guard against zero sigma
- **File:** `models/gaussian_plume.py:282`
- **Fix:** Floor sigma values at `1e-6`; return zeros for non-positive `total_mass`.

#### Bug 14: Ensemble weight validation inconsistent
- **File:** `optimization/opportunity_map.py:162-165` validates weights; `information_gain.py` does not.
- **Fix:** Add same weight validation to `compute_ensemble_information_scores()`.

#### Bug 15: Turn-by-turn waypoint matching uses rounding instead of distance
- **File:** `pages/worker_guidance.py:90,102`
- **Fix:** Use `np.hypot(dx, dy) < tolerance` instead of `round()` comparison.

---

### 9.3 MEDIUM SEVERITY — State Management

#### Bug 16: Sources mutated in-place via UI sliders
- **File:** `main.py:440-449`
- **Problem:** Duty cycle sliders mutate the source dicts returned by `DataProvider`. If the provider caches its list, mutations persist across sessions.
- **Fix:** Deep-copy sources before mutation: `sources = [s.copy() for s in data_provider.get_leak_sources()]`.

#### Bug 17: Measurement serialization drops timestamp/station_id
- **File:** `data/state_io.py:47-61`
- **Problem:** `Measurement` fields `timestamp` and `station_id` are not included in the serialized dict, causing data loss on roundtrip.
- **Fix:** Serialize all `Measurement` dataclass fields.

#### Bug 18: Campaign day close clears measurements but not entropy history
- **File:** `main.py:992-995`
- **Problem:** `st.session_state.bayesian_measurements` is cleared on day close, but entropy history accumulates across days, creating inconsistent state.
- **Fix:** Reset entropy history on day close, or track per-day histories separately.

---

### 9.4 DESIGN IMPROVEMENTS

#### Design 19: Detection scoring penalizes on-path locations
- **File:** `optimization/tasking.py:128` and `information_gain.py:362`
- **Problem:** `score = value / (deviation + epsilon)` with `epsilon=10m` means cells directly on the baseline path get divided by 10, discounting them.
- **Suggestion:** Reformulate as additive cost or use a bonus for low deviation.

#### Design 20: Use `scipy.special.expit()` for numerically stable sigmoid
- **File:** `models/detection.py:48-50`
- **Benefit:** Replaces manual clipping with a numerically stable library function.

#### Design 21: DataProvider contract needs immutability documentation
- **File:** `data/interfaces.py`
- **Action:** Document whether returned lists are safe to mutate; consider returning copies.

#### Design 22: Campaign state needs invariant checks
- **File:** `optimization/campaign.py:259-261`
- **Action:** Guard against calling `close_day()` twice or out of order; add state machine checks.

---

### 9.5 VISUALIZATION ISSUES

#### Viz 23: Start/end markers duplicated in multi-worker mode
- **File:** `visualization/plots.py:348-368`
- **Problem:** `_add_start_end_markers()` called for each worker, creating overlapping markers.
- **Fix:** Color markers by worker ID, or skip in multi-worker mode.

#### Viz 24: Missing aspect ratio control in worker guidance map
- **File:** `pages/worker_guidance.py:221`
- **Fix:** Add `scaleratio=1` to maintain 1:1 aspect ratio.

#### Viz 25: Poor color scaling on uniform/low-value data
- **File:** `visualization/plots.py:918,930`
- **Problem:** `zmax=max(np.max(data), 0.01)` creates extreme color stretching for uniform low-value data.
- **Fix:** Use percentile-based scaling.

#### Viz 26: NaN handling in log-scale concentration plots
- **File:** `visualization/plots.py:472,566`
- **Problem:** NaN in concentration data silently renders as blank cells.
- **Fix:** Pre-filter NaN values with `np.nan_to_num()`.

---

### 9.6 TEST & VALIDATION GAPS

| Gap | Priority |
|-----|----------|
| No test for sequential non-detections saturating belief to zero | High |
| No test that worker path segments reconstruct the original path | High |
| No test for waypoint already on the baseline path | Medium |
| Multi-worker coverage test only asserts `>= 0.0`, not improvement | Medium |
| Campaign multi-day entropy carry-forward not tested | High |
| Experiment runner uses fixed seed=42 with no variance reporting | Medium |
| No statistical significance testing in strategy comparison | Medium |
| No comparison to published Pasquill-Gifford tables (Turner 1970) | Low |
| Only 5 validation scenarios — missing "no leaks", "extreme emissions" | Medium |
| Localization RMSE uses greedy matching instead of Hungarian algorithm | Medium |

---

### 9.7 IMPLEMENTATION STATUS

All Phase 9 items have been implemented and verified (278 tests passing).

```
Step 1: Critical bugs (Bugs 1-7)                          ✅ DONE
   └─► Bayesian update, campaign scoring, multi-worker, cache, serialization, plume, viz

Step 2: Medium-severity logic fixes (Bugs 8-15)           ✅ DONE
   └─► Path splitting, prior saturation, reverse plume, validation, ensemble weights

Step 3: State management fixes (Bugs 16-18)               ✅ DONE
   └─► Source mutation, measurement serialization, campaign state
   └─► Note: Bug 17 was a false positive — all Measurement fields are already serialized

Step 4: Design improvements (Design 19-22)                 ✅ DONE
   └─► Scoring formula (exp decay), expit, DataProvider contract, campaign invariants

Step 5: Visualization fixes (Viz 23-26)                    ✅ DONE
   └─► Worker-colored markers, aspect ratio, percentile color scaling, NaN handling

Step 6: Test gap coverage (Section 9.6)                    ✅ DONE
   └─► tests/test_phase9_gaps.py: 9 new tests covering belief saturation,
       path segments, campaign carry-forward, on-path waypoints, fleet coverage
   └─► validation/metrics.py: Hungarian algorithm (linear_sum_assignment) replaces greedy RMSE
   └─► experiments/run_strategy_comparison.py: --seeds flag for multi-seed variance reporting
```

---

## Phase 7, 8, 9.6 Completion (2026-02-13)

### Phase 9.6: Additional Test Gaps — DONE

```
✅ validation/scenarios.py: Added scenario_f_no_leaks() and scenario_g_extreme()
✅ validation/metrics.py: Added paired_significance_test() and bootstrap_confidence_interval()
✅ experiments/run_strategy_comparison.py: Wired scenarios F and G into ALL_SCENARIOS
✅ tests/test_gaussian_plume.py: Added TestPasquillGiffordValidation (Turner 1970 reference + ordering)
✅ tests/test_synthetic_twin.py: Added TestNewScenarios (F/G smoke tests) + TestStatisticalHelpers (9 tests)
```

### Phase 7: FileDataProvider — DONE

```
✅ data/samples/sources.json, path.csv, wind_scenarios.json, wind_distribution.json
✅ data/interfaces.py: Added FileDataProvider class (eager validation, immutability contract)
✅ tests/test_file_data_provider.py: 16 tests (ABC conformance, types, immutability, validation)
✅ main.py: Added "Data Source" sidebar expander with file upload support
```

### Phase 8: Sensitivity Analysis + Documentation — DONE

```
✅ experiments/run_sensitivity_analysis.py: One-at-a-time parameter sweep (6 parameters)
✅ docs/algorithms.md: Mathematical formulations for all 8 model components
✅ docs/validation_results.md: Template with placeholder tables for results
```

---

## OVERALL COMPLETION STATUS

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation Hardening (tests, validation, config) | ✅ Complete |
| 2 | Bayesian Inference Engine (prior, belief, EER) | ✅ Complete |
| 3.1 | Cross-Plume Integrated Concentration | ✅ Complete |
| 3.2 | Time-Varying Source Model (duty cycle, puff) | ✅ Complete |
| 3.3 | Sensor Model Enhancement (SensorModel class) | ⏭️ Skipped (per user) |
| 4 | Multi-Worker & Campaign Planning | ✅ Complete |
| 5 | Validation Framework (synthetic twin, scenarios) | ✅ Complete |
| 6.1 | Belief Map Visualization | ✅ Complete |
| 6.2 | Worker Guidance Interface | ✅ Complete |
| 6.3 | Dashboard & Reporting Page | ❌ Not implemented |
| 7 | Data Integration (FileDataProvider) | ✅ Complete |
| 8 | Documentation & Reproducibility | ✅ Complete |
| 9 | Deep Code Review — Bug Fixes & Hardening | ✅ Complete |

### Remaining Items

1. **Phase 3.3 — SensorModel class** (skipped per user request): Would add `SensorModel` class with configurable response time, false positive rate, temperature factor, and sensor type presets. Low priority — current sigmoid + MDL model is sufficient.

2. **Phase 6.3 — Dashboard & Reporting Page** (`pages/dashboard.py`): A dedicated Streamlit page for campaign progress summary, detection history timeline, worker utilization chart, and uncertainty reduction over time. Currently these features are partially available in the campaign mode panel of main.py and the export functionality.

### Test Count: 316 tests passing

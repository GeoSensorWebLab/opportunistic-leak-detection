# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

**Methane Leak Opportunistic Tasking System** — a Python Streamlit app that recommends optimal locations for field workers to detect methane leaks from oil & gas infrastructure. It combines Gaussian plume atmospheric dispersion modeling, detection probability, and path optimization.

## Tech Stack

- **Python 3.10+** (managed via `uv`)
- **Streamlit** — web UI
- **NumPy / SciPy** — numerical computation and spatial distance
- **Plotly** — interactive visualization
- **Matplotlib** — fallback plotting

## How to Run

```bash
uv sync                          # install dependencies
uv run streamlit run main.py     # start the app at localhost:8501
```

## Project Structure

```
MethaneSimulator/
├── main.py                       # Streamlit app entry point
├── config.py                     # Global constants & Pasquill-Gifford coefficients
├── pyproject.toml                # Dependencies (uv)
├── models/
│   ├── gaussian_plume.py         # Gaussian plume dispersion (standard + crosswind-integrated)
│   ├── detection.py              # Sigmoid detection probability model
│   ├── prior.py                  # Stage 1 prior: equipment risk → spatial prior
│   ├── measurement.py            # Measurement dataclass for field observations
│   └── bayesian.py               # Stage 2 Bayesian belief map (posterior updates + set_belief)
├── optimization/
│   ├── opportunity_map.py        # 2D grid heatmap aggregation + wind ensemble
│   ├── tasking.py                # Cost function, waypoint ranking, path insertion
│   ├── information_gain.py       # Stage 3: Expected Entropy Reduction (EER) scoring
│   ├── multi_worker.py           # Multi-worker waypoint allocation & fleet coverage
│   └── campaign.py               # Multi-day campaign planning with posterior carry-forward
├── visualization/
│   ├── plots.py                  # Plotly interactive site maps, entropy, convergence charts
│   └── compass_widget.py         # SVG wind compass widget
├── pages/
│   └── worker_guidance.py        # Field worker guidance page (priority zones, turn-by-turn)
├── data/
│   ├── mock_data.py              # Synthetic leak sources, paths, wind presets & distributions
│   ├── interfaces.py             # Abstract DataProvider ABC + MockDataProvider
│   ├── weather.py                # WeatherProvider ABC + StubWeatherProvider
│   ├── state_io.py               # Belief state serialization (NPZ + JSON)
│   └── facility_layout.py        # Facility infrastructure layout definitions
├── validation/
│   ├── synthetic_twin.py         # Synthetic experiment engine & strategy implementations
│   ├── scenarios.py              # Pre-defined test scenarios (A-E)
│   └── metrics.py                # Detection, efficiency, and learning metrics
├── experiments/
│   └── run_strategy_comparison.py # Batch strategy comparison across all scenarios
└── tests/
    ├── conftest.py               # Shared pytest fixtures
    ├── test_gaussian_plume.py    # Plume dispersion tests
    ├── test_detection.py         # Detection probability tests
    ├── test_opportunity_map.py   # Opportunity map tests
    ├── test_tasking.py           # Tasking optimizer tests
    ├── test_prior.py             # Prior model tests
    ├── test_bayesian.py          # Bayesian belief map & measurement tests
    ├── test_ensemble.py          # Wind ensemble tests
    ├── test_information_gain.py  # EER information-theoretic scoring tests
    ├── test_synthetic_twin.py    # Synthetic twin validation + variable wind tests
    ├── test_temporal.py          # Duty cycle, Gaussian puff, intermittent leak tests
    ├── test_integration.py       # End-to-end pipeline tests
    ├── test_plots.py             # Visualization smoke tests (all plot functions)
    ├── test_multi_worker.py      # Multi-worker allocation tests
    ├── test_campaign.py          # Campaign planning & serialization tests
    ├── test_data_interfaces.py   # DataProvider ABC conformance tests
    ├── test_state_io.py          # State serialization roundtrip tests
    └── test_weather.py           # Weather API abstraction tests
```

## Architecture & Data Flow

1. **Data layer** (`data/`) provides leak sources, worker paths, facility layout, and wind distributions via abstract `DataProvider` interface (`data/interfaces.py`); `MockDataProvider` wraps synthetic data for development
2. **Physics engine** (`models/gaussian_plume.py`) computes plume concentrations using Gaussian dispersion; supports both standard (instantaneous) and crosswind-integrated formulations
3. **Detection model** (`models/detection.py`) converts concentrations to detection probabilities via sigmoid
4. **Prior model** (`models/prior.py`) computes per-source leak probability from equipment attributes (Stage 1 Bayesian)
5. **Opportunity map** (`optimization/opportunity_map.py`) aggregates all sources into a site-wide probability grid; supports single-scenario or **wind ensemble** (weighted average across multiple wind conditions)
6. **Bayesian belief map** (`models/bayesian.py`) updates spatial belief via cell-wise Bayes' theorem after field observations (Stage 2 Bayesian); uses vectorized reverse-plume computation; supports `set_belief()` for restoring saved posteriors
7. **Tasking optimizer** (`optimization/tasking.py`) scores grid cells by detection value vs path deviation cost, optionally weighted by prior or posterior belief (heuristic mode)
8. **Information-theoretic scoring** (`optimization/information_gain.py`) computes Expected Entropy Reduction (EER) at each grid cell — answers *"where should I measure to learn the most?"* (Stage 3 Bayesian); uses subsampled grid + bilinear interpolation for efficiency; supports single-wind and ensemble modes
9. **Multi-worker allocation** (`optimization/multi_worker.py`) splits baseline path across workers, assigns waypoints by minimum deviation, and computes fleet coverage via complementary probability `P_fleet = 1 - prod(1 - P_worker_i)`
10. **Campaign planning** (`optimization/campaign.py`) manages multi-day inspection campaigns; carries posterior belief forward as next day's prior; tracks entropy reduction per day; supports full serialization/deserialization for persistence
11. **Visualization** (`visualization/`) renders interactive maps, compass widget, belief map, entropy heatmap, prior-vs-posterior comparison, convergence chart, score profiles, and per-worker colored routes
12. **Worker guidance** (`pages/worker_guidance.py`) provides a field-oriented Streamlit page with priority zones (HIGH/MEDIUM/LOW), turn-by-turn walking directions, simplified map, and measurement input form
13. **Weather abstraction** (`data/weather.py`) defines `WeatherProvider` ABC with `StubWeatherProvider` for development; ready for real API integration
14. **State serialization** (`data/state_io.py`) persists belief arrays, grids, measurements, and entropy history as NPZ archives with JSON-encoded metadata
15. **Streamlit UI** (`main.py`) ties it all together with sidebar controls, scoring mode toggle (Heuristic vs EER), multi-worker mode, campaign mode, time-resolved Bernoulli toggle, results export (CSV/JSON/NPZ), and `st.session_state` for persistence

## Key Conventions

- **Coordinate system**: Local Cartesian, x = East, y = North, origin at site center, all units in meters
- **Wind convention**: Meteorological — direction is where wind comes FROM (270 = from west, blowing east)
- **Configuration**: All tunable parameters live in `config.py` — do not hardcode magic numbers in other modules
- **Caching**: Streamlit `@st.cache_data` is used for expensive computations (opportunity map, path deviation)
- **Multi-source detection**: Uses complementary probability `1 - prod(1 - P_i)` for independent events

## Code Style

- Standard Python conventions (PEP 8)
- Type hints are welcome but not strictly enforced
- Docstrings on public functions (Google style)
- Keep modules focused — physics in `models/`, optimization logic in `optimization/`, rendering in `visualization/`

## Important Physics Notes

- Ground reflection uses image source method (prevents unphysical below-ground plumes)
- Sigma parameterization uses power-law `sigma = a * x^b` (Turner 1970)
- Detection threshold default is 5 ppm (typical handheld methane detector)
- Non-maximum suppression ensures spatially diverse recommendations (min 50m apart)
- **Bayesian reverse plume**: for each grid cell as hypothetical source, compute concentration at measurement point — fully vectorized (no per-cell loop)
- **Bayesian update**: cell-wise `P(leak_i | obs) = P(obs | leak_i) * P(leak_i) / P(obs)`, with `FALSE_ALARM_RATE` (default 0.01) for the no-leak likelihood
- **Wind ensemble**: weighted average `E[P] = Σ w_i * P_i` across scenarios; weights must sum to 1.0
- **Expected Entropy Reduction (EER)**: per-cell binary entropy `H(p) = -p*log2(p) - (1-p)*log2(1-p)`; EER at candidate *m* = `Σ_cells [H_current - P(obs)*H(posterior|obs) - P(¬obs)*H(posterior|¬obs)]`; cells are independent so total EER is sum of per-cell reductions; uses subsampled grid (`EER_SUBSAMPLE=4`) + `RegularGridInterpolator` for performance
- **Crosswind-integrated plume**: `C = Q / (√(2π) * u * σz) * [exp(-½((z-H)/σz)²) + exp(-½((z+H)/σz)²)]` — integrates out σy lateral dispersion, producing broader, lower-peak plumes that better match time-averaged field measurements; selectable via `plume_mode` parameter ("instantaneous", "integrated", or "puff")
- **Gaussian puff model**: `C = Q_total / ((2π)^(3/2) σx σy σz) * exp(-½((x_d - u*t)/σx)²) * exp(-½(y_c/σy)²) * [vertical with ground reflection]` — instantaneous release that drifts downwind as a 3D Gaussian cloud; `σx = σy` (isotropic horizontal dispersion); suitable for intermittent/episodic leaks; selectable via `plume_mode="puff"`
- **Duty cycle**: fraction of time a source actively emits (0–1); `effective_rate = emission_rate * duty_cycle`; all lookups use `.get("duty_cycle", 1.0)` for backward compatibility
- **Time-resolved mode** (synthetic twin): Bernoulli draw per source per measurement (`on` with probability `duty_cycle`, else `off`); contrasts with default time-averaged mode that scales emission by duty cycle deterministically

## Integration Points (for future real data)

- Implement a new `DataProvider` subclass (in `data/interfaces.py`) for real asset data (SCADA) — replaces `MockDataProvider` for `get_leak_sources()`, `get_baseline_path()`, `get_wind_scenarios()`, `get_wind_distribution()`, `get_wind_fan()`
- Implement a new `WeatherProvider` subclass (in `data/weather.py`) for live wind conditions — replaces `StubWeatherProvider` for `get_current_wind()` and `get_forecast()`
- Add `pyproj` for GPS-to-Cartesian coordinate transforms
- Feed real sensor readings into `Measurement` dataclass for live Bayesian updates
- Use `data/state_io.py` to persist and restore campaign state across sessions

## Common Tasks

- **Add a new leak source**: Edit `data/mock_data.py`, add to the list in `get_leak_sources()`
- **Change default parameters**: Edit `config.py` (includes `DEFAULT_NUM_WORKERS`, `MAX_WORKERS`)
- **Add a new visualization**: Add function in `visualization/plots.py`, call from `main.py`
- **Add a new wind preset**: Edit `data/mock_data.py:get_wind_scenarios()`
- **Add a wind ensemble mode**: Create a new function in `data/mock_data.py` returning `List[dict]` with `direction`, `speed`, `stability_class`, `weight` keys (weights must sum to 1.0), then wire it into the ensemble UI in `main.py`
- **Swap data source**: Implement a new `DataProvider` subclass in `data/interfaces.py` and instantiate it in `main.py` instead of `MockDataProvider`
- **Add live weather**: Implement a `WeatherProvider` subclass in `data/weather.py` (e.g., wrapping NOAA ISD or OpenWeatherMap API)
- **Run tests**: `uv run pytest tests/ -v`
- **Run strategy comparison**: `uv run python experiments/run_strategy_comparison.py`

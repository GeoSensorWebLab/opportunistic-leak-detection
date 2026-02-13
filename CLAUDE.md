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
│   └── bayesian.py               # Stage 2 Bayesian belief map (posterior updates)
├── optimization/
│   ├── opportunity_map.py        # 2D grid heatmap aggregation + wind ensemble
│   ├── tasking.py                # Cost function, waypoint ranking, path insertion
│   └── information_gain.py       # Stage 3: Expected Entropy Reduction (EER) scoring
├── visualization/
│   ├── plots.py                  # Plotly interactive site maps & charts
│   └── compass_widget.py         # SVG wind compass widget
├── data/
│   ├── mock_data.py              # Synthetic leak sources, paths, wind presets & distributions
│   └── facility_layout.py        # Facility infrastructure layout definitions
├── validation/
│   ├── synthetic_twin.py         # Synthetic experiment engine & strategy implementations
│   ├── scenarios.py              # Pre-defined test scenarios (A-E)
│   └── metrics.py                # Detection, efficiency, and learning metrics
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
    ├── test_synthetic_twin.py    # Synthetic twin validation tests
    ├── test_temporal.py          # Duty cycle, Gaussian puff, intermittent leak tests
    └── test_integration.py       # End-to-end pipeline tests
```

## Architecture & Data Flow

1. **Data layer** (`data/`) provides leak sources, worker paths, facility layout, and wind distributions
2. **Physics engine** (`models/gaussian_plume.py`) computes plume concentrations using Gaussian dispersion; supports both standard (instantaneous) and crosswind-integrated formulations
3. **Detection model** (`models/detection.py`) converts concentrations to detection probabilities via sigmoid
4. **Prior model** (`models/prior.py`) computes per-source leak probability from equipment attributes (Stage 1 Bayesian)
5. **Opportunity map** (`optimization/opportunity_map.py`) aggregates all sources into a site-wide probability grid; supports single-scenario or **wind ensemble** (weighted average across multiple wind conditions)
6. **Bayesian belief map** (`models/bayesian.py`) updates spatial belief via cell-wise Bayes' theorem after field observations (Stage 2 Bayesian); uses vectorized reverse-plume computation
7. **Tasking optimizer** (`optimization/tasking.py`) scores grid cells by detection value vs path deviation cost, optionally weighted by prior or posterior belief (heuristic mode)
8. **Information-theoretic scoring** (`optimization/information_gain.py`) computes Expected Entropy Reduction (EER) at each grid cell — answers *"where should I measure to learn the most?"* (Stage 3 Bayesian); uses subsampled grid + bilinear interpolation for efficiency; supports single-wind and ensemble modes
9. **Visualization** (`visualization/`) renders interactive maps, compass widget, belief map, and score charts
10. **Streamlit UI** (`main.py`) ties it all together with sidebar controls, scoring mode toggle (Heuristic vs EER), and `st.session_state` for Bayesian measurement persistence

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

- Replace `data/mock_data.py:get_leak_sources()` for real asset data (SCADA)
- Replace `data/mock_data.py:get_baseline_path()` for real work order routes
- Replace `data/mock_data.py:get_wind_distribution()` with real wind rose data (e.g., NOAA ISD)
- Add weather API integration for live wind conditions
- Add `pyproj` for GPS-to-Cartesian coordinate transforms
- Feed real sensor readings into `Measurement` dataclass for live Bayesian updates

## Common Tasks

- **Add a new leak source**: Edit `data/mock_data.py`, add to the list in `get_leak_sources()`
- **Change default parameters**: Edit `config.py`
- **Add a new visualization**: Add function in `visualization/plots.py`, call from `main.py`
- **Add a new wind preset**: Edit `data/mock_data.py:get_wind_scenarios()`
- **Add a wind ensemble mode**: Create a new function in `data/mock_data.py` returning `List[dict]` with `direction`, `speed`, `stability_class`, `weight` keys (weights must sum to 1.0), then wire it into the ensemble UI in `main.py`
- **Run tests**: `uv run pytest tests/ -v`

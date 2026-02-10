# Methane Leak Opportunistic Tasking System

An interactive simulation platform that recommends optimal waypoints for field workers to detect methane leaks from oil & gas infrastructure. The system combines **Gaussian plume atmospheric dispersion modeling**, **probabilistic detection mapping**, and **route optimization** to insert efficient detours into a worker's existing inspection path — maximizing leak detection while minimizing disruption to planned operations.

Built as part of the [GeoSensorWebLab](https://github.com/GeoSensorWebLab) research initiative.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [User Interface](#user-interface)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
  - [Gaussian Plume Model](#1-gaussian-plume-model)
  - [Detection Probability Model](#2-detection-probability-model)
  - [Opportunity Map Generator](#3-opportunity-map-generator)
  - [Tasking Optimizer](#4-tasking-optimizer)
  - [Visualization Engine](#5-visualization-engine)
  - [Facility Layout](#6-facility-layout)
- [Configuration Reference](#configuration-reference)
- [Data Integration Guide](#data-integration-guide)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Motivation

Methane is a potent greenhouse gas — over 80 times more effective at trapping heat than CO2 over a 20-year period. Oil and gas facilities are a major source of fugitive methane emissions from equipment leaks at wellheads, compressor stations, pipelines, storage tanks, and valve assemblies.

Current leak detection and repair (LDAR) programs rely on fixed inspection routes that don't adapt to real-time atmospheric conditions. A leak may produce a detectable plume in one wind condition but be invisible in another. This system addresses that gap by:

1. **Modeling plume dispersion** under current wind and atmospheric stability conditions
2. **Mapping detection probability** across the entire site based on sensor characteristics
3. **Scoring every location** by its detection value relative to the worker's path deviation cost
4. **Recommending optimal waypoints** as efficient detours that maximize the chance of finding leaks

The result is a dynamic, wind-aware decision support tool for field technicians conducting routine inspections.

---

## Key Features

- **Real-time plume simulation** — Gaussian dispersion with Pasquill-Gifford stability classes (A through F), ground reflection via image source method
- **Probabilistic detection** — Sigmoid sensor response model accounting for detection threshold, sensor noise, and atmospheric variability
- **Multi-source aggregation** — Superposition of plumes from multiple simultaneous leak sources with statistically correct combined detection probability
- **Path-aware optimization** — Tasking score balances detection value against walking deviation cost, with configurable maximum deviation constraint
- **Non-maximum suppression** — Ensures spatially diverse recommendations (minimum 50 m separation) rather than clustering in a single hotspot
- **Route optimization** — Nearest-neighbor seeding with 2-opt local search to minimize total detour distance when visiting multiple waypoints
- **Interactive Streamlit UI** — Real-time parameter adjustment with dual-panel Plotly visualizations, SVG wind compass, and ranked recommendation table
- **Facility context** — Renders buildings, roads, pipe racks, equipment pads, and site boundary for operational awareness
- **Preset wind scenarios** — Four built-in atmospheric conditions for rapid what-if analysis
- **Modular architecture** — Clean separation of physics, optimization, visualization, and data layers for easy integration with real sensor feeds

---

## System Architecture

```
                         ┌─────────────────────────┐
                         │     Data Sources         │
                         │  (Mock / SCADA / API)    │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
            ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
            │ Leak Sources │  │ Wind Params  │  │ Baseline     │
            │ (x,y,z,Q)   │  │ (u, θ, class)│  │ Path (N×2)  │
            └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                   │                 │                  │
                   ▼                 ▼                  │
            ┌────────────────────────────────┐         │
            │  Gaussian Plume Model          │         │
            │  C(x,y,z) for each source      │         │
            │  + ground reflection           │         │
            └──────────────┬─────────────────┘         │
                           │                            │
                           ▼                            │
            ┌────────────────────────────────┐         │
            │  Detection Probability Model   │         │
            │  Sigmoid: P = σ(C - threshold) │         │
            └──────────────┬─────────────────┘         │
                           │                            │
                           ▼                            │
            ┌────────────────────────────────┐         │
            │  Opportunity Map               │         │
            │  Superposition + combined P    │         │
            │  P_comb = 1 - ∏(1 - Pᵢ)       │         │
            └──────────────┬─────────────────┘         │
                           │                            │
                           ▼                            ▼
            ┌────────────────────────────────────────────────┐
            │  Tasking Optimizer                              │
            │  Score = P(detect) / (PathDeviation + ε)       │
            │  + Non-maximum suppression (top-K extraction)  │
            │  + Route optimization (NN + 2-opt)             │
            └──────────────┬─────────────────────────────────┘
                           │
                           ▼
            ┌────────────────────────────────┐
            │  Visualization Engine          │
            │  Plotly dual-panel maps        │
            │  + SVG compass + score chart   │
            └──────────────┬─────────────────┘
                           │
                           ▼
            ┌────────────────────────────────┐
            │  Streamlit Web Interface       │
            │  localhost:8501                 │
            └────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- [**uv**](https://docs.astral.sh/uv/) — fast Python package manager

### Installation

```bash
git clone https://github.com/GeoSensorWebLab/MethaneSimulator.git
cd MethaneSimulator
uv sync
```

This creates a `.venv` virtual environment and installs all dependencies from `pyproject.toml`.

### Run the Application

```bash
uv run streamlit run main.py
```

The app opens in your browser at **http://localhost:8501**.

---

## User Interface

### Sidebar Controls

| Control | Range | Default | Description |
|---|---|---|---|
| **Preset Scenario** | Custom + 4 presets | Custom | Quick-select atmospheric conditions |
| **Wind Speed** | 0.5 – 15.0 m/s | 3.0 m/s | Surface wind speed |
| **Wind Direction** | 0 – 359 degrees | 270 (from West) | Meteorological convention (where wind comes FROM) |
| **Atmospheric Stability** | A – F | D (Neutral) | Pasquill-Gifford stability class |
| **Max Path Deviation** | 50 – 500 m | 200 m | Maximum allowable detour distance |
| **Recommendations** | 1 – 10 | 5 | Number of waypoints to suggest |
| **Grid Resolution** | 2, 5, 10, 20 m | 5 m | Simulation grid cell size (lower = finer but slower) |

### Main Display

- **Left Panel** — Log-scale methane concentration heatmap showing plume dispersion from all sources, with leak source markers (green diamonds), facility layout overlay, and wind compass rose
- **Right Panel** — Detection probability heatmap with baseline inspection path (cyan dashed), optimized path with detours (white solid), and recommended waypoints (yellow stars)
- **Score Bar Chart** — Ranked visualization of top waypoints by tasking score with hover details (detection probability, concentration)
- **Recommendations Table** — Metrics for each waypoint: location coordinates, detection probability, concentration (ppm), and tasking score

### Wind Presets

| Preset | Speed | Direction | Stability | Character |
|---|---|---|---|---|
| Light SW Breeze (Unstable) | 2.0 m/s | 225 (SW) | A | Wide, dispersed plumes |
| Moderate West Wind (Neutral) | 4.0 m/s | 270 (W) | D | Standard dispersion |
| Strong North Wind (Stable) | 8.0 m/s | 0 (N) | F | Narrow, concentrated plumes |
| East Wind (Slightly Unstable) | 3.0 m/s | 90 (E) | B | Moderate lateral spread |

---

## Project Structure

```
MethaneSimulator/
├── main.py                          # Streamlit application entry point
├── config.py                        # Global constants & Pasquill-Gifford coefficients
├── pyproject.toml                   # Project metadata & dependencies (uv)
├── CLAUDE.md                        # AI assistant project instructions
│
├── models/                          # Physics engine
│   ├── __init__.py
│   ├── gaussian_plume.py            # Gaussian plume dispersion equation
│   └── detection.py                 # Sigmoid detection probability model
│
├── optimization/                    # Decision engine
│   ├── __init__.py
│   ├── opportunity_map.py           # 2D grid heatmap from all sources
│   └── tasking.py                   # Cost function, recommendations, path insertion
│
├── visualization/                   # Rendering engine
│   ├── __init__.py
│   ├── plots.py                     # Plotly interactive site maps & charts
│   └── compass_widget.py            # SVG wind compass widget
│
└── data/                            # Data layer (swappable)
    ├── __init__.py
    ├── mock_data.py                 # Synthetic leak sources, paths, wind presets
    └── facility_layout.py           # Facility infrastructure definitions
```

---

## Core Components

### 1. Gaussian Plume Model

**Module:** `models/gaussian_plume.py`

Implements the standard Gaussian plume equation for a continuous point source with ground reflection (image source method):

```
C(x,y,z) = Q / (2π · u · σy · σz) × exp(-½(y/σy)²) × [exp(-½((z-H)/σz)²) + exp(-½((z+H)/σz)²)]
```

**Parameters:**
- **Q** — emission rate (kg/s)
- **u** — wind speed (m/s)
- **σy, σz** — lateral and vertical dispersion coefficients
- **H** — source release height (m)

**Dispersion Parameterization:**

Uses power-law form from Turner (1970) rather than lookup tables, enabling smooth interpolation at any downwind distance:

```
σy = ay · x^by
σz = az · x^bz
```

where coefficients (a, b) depend on the Pasquill-Gifford stability class:

| Class | Conditions | σy Spread | σz Spread |
|---|---|---|---|
| A | Very unstable (strong solar, light wind) | Very wide | Very tall |
| B | Moderately unstable | Wide | Tall |
| C | Slightly unstable | Moderate | Moderate |
| D | Neutral (overcast, moderate wind) | Standard | Standard |
| E | Slightly stable | Narrow | Short |
| F | Very stable (clear night, light wind) | Very narrow | Very short |

**Wind Convention:** Meteorological — wind direction indicates where wind comes **FROM** (270 = from the west, plume travels east). Internally converted to blowing direction to prevent the most common dispersion modeling bug.

**Key Functions:**

| Function | Description |
|---|---|
| `gaussian_plume()` | Concentration (kg/m3) at receptor grid from a single source |
| `compute_sigma()` | σy, σz dispersion parameters for given downwind distance |
| `concentration_to_ppm()` | Converts kg/m3 to ppm using ideal gas approximation |

---

### 2. Detection Probability Model

**Module:** `models/detection.py`

Converts concentration fields to probability of detection using a logistic sigmoid centered at the sensor's detection threshold:

```
P(detect) = 1 / (1 + exp(-steepness × (C - threshold)))
```

This models realistic sensor behavior — detection is not binary. Concentrations near the threshold have intermediate probability, accounting for:
- Sensor noise and measurement uncertainty
- Atmospheric turbulence causing concentration fluctuations
- Instrument response time vs. exposure duration

**Default Parameters:**
- Threshold: **5 ppm** (typical handheld methane detector sensitivity)
- Steepness: **1.0** (configurable for different sensor characteristics)

---

### 3. Opportunity Map Generator

**Module:** `optimization/opportunity_map.py`

Creates a 2D grid over the site (default 1 km x 1 km, 5 m resolution) and computes the combined detection probability from all potential leak sources under current atmospheric conditions.

**Multi-Source Aggregation:**

Plume concentrations are summed via superposition (the Gaussian plume equation is linear), then detection probability is computed on the total concentration field:

```
C_total(x,y) = Σ Cᵢ(x,y)     (superposition)
P_detect = sigmoid(C_total)     (detection on total field)
```

This correctly models a sensor responding to the aggregate methane present at each point.

**Performance:** Results are cached using Streamlit's `@st.cache_data` decorator, keyed on wind parameters. Grid recomputation only occurs when atmospheric conditions change.

---

### 4. Tasking Optimizer

**Module:** `optimization/tasking.py`

The core decision engine that scores every grid cell and recommends optimal waypoints.

#### Scoring Function

```
Score(x,y) = P(detection at x,y) / (PathDeviation(x,y) + ε)
```

- **PathDeviation** — minimum Euclidean distance from the grid cell to the nearest point on the baseline inspection path
- **ε** — smoothing constant (default 10 m) preventing division by zero for on-path locations
- Cells beyond **max_deviation** (default 200 m) are scored zero

#### Waypoint Extraction

Uses **non-maximum suppression** (borrowed from computer vision) to ensure spatially diverse recommendations:
1. Sort all cells by score (descending)
2. Select the highest-scoring cell
3. Suppress all cells within 50 m of the selection
4. Repeat until top-K waypoints are found

#### Route Optimization

Recommended waypoints are inserted into the baseline path as efficient detours:

1. **Project** each waypoint onto the nearest baseline path segment
2. **Sort** waypoints by walking order (cumulative distance along baseline)
3. **Cluster** consecutive waypoints within 8% of total path length into single detours
4. **Optimize** visit order within each cluster using nearest-neighbor seeding + 2-opt local search
5. **Splice** optimized detours into the baseline at correct insertion points

**Key Functions:**

| Function | Description |
|---|---|
| `compute_path_deviation()` | Min distance from each grid cell to baseline path |
| `compute_tasking_scores()` | Score = detection / (deviation + ε) |
| `recommend_waypoints()` | Top-K extraction with spatial non-maximum suppression |
| `build_optimized_path()` | Insert waypoints as optimized detours into baseline route |

---

### 5. Visualization Engine

**Module:** `visualization/plots.py`, `visualization/compass_widget.py`

#### Site Map (`plots.py`)

Interactive Plotly-based dual-panel visualization with dark theme:

- **Left Panel:** Log-scale methane concentration heatmap (Hot colorscale, reversed) with:
  - Leak source markers (lime green diamonds)
  - Facility layout overlay (buildings, roads, pipe racks, equipment pads, site fence)
  - Compass rose with wind needle

- **Right Panel:** Detection probability heatmap (YlOrRd colorscale, 0-1 range) with:
  - Baseline path (cyan dashed line with markers)
  - Optimized path (white solid line with markers)
  - Recommended waypoints (yellow stars, numbered)
  - Same facility and compass overlays

- **Score Bar Chart:** Gold bar chart ranking waypoints by tasking score with hover tooltips showing detection probability and concentration

#### Wind Compass (`compass_widget.py`)

SVG-rendered sidebar compass widget displaying:
- Outer ring with cardinal (N/E/S/W) and minor (every 10 degrees) tick marks
- Wind needle pointing in the blowing direction (meteorological direction + 180 degrees)
- Center circle showing current wind speed (m/s)
- "From" direction label below compass

---

### 6. Facility Layout

**Module:** `data/facility_layout.py`

Defines a realistic midstream gas processing / compressor station complex (~700 m x 600 m) providing operational context:

| Layer | Elements | Rendering |
|---|---|---|
| Equipment Pads | Well Pad, Compressor Pad, Tank Containment Berm, Valve/Metering Station | Semi-transparent filled rectangles |
| Roads | Main Access Road (N-S), Tank Farm Road (E-W), Well Pad Access | Gray lines (5-6 px) |
| Pipe Racks | Main E-W Rack, Tank Farm Header (N-S) | Orange dotted lines |
| Buildings | Compressor House, Control Room, MCC/Electrical | Opaque filled rectangles with white borders |
| Site Fence | Rectangular boundary with north gate notch | White dash-dot line |

---

## Configuration Reference

All tunable parameters are centralized in `config.py`:

### Grid / Site

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `GRID_SIZE_M` | 1000 | m | Site grid extent (square, centered at origin) |
| `GRID_RESOLUTION_M` | 5 | m | Grid cell size |
| `RECEPTOR_HEIGHT_M` | 1.5 | m | Sensor height above ground (worker carrying handheld) |

### Plume Defaults

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `DEFAULT_EMISSION_RATE` | 0.5 | kg/s | Default leak rate |
| `DEFAULT_WIND_SPEED` | 3.0 | m/s | Default surface wind speed |
| `DEFAULT_WIND_DIRECTION` | 270 | degrees | Default wind from-direction (meteorological) |
| `DEFAULT_STABILITY_CLASS` | D | — | Default Pasquill-Gifford stability class |

### Detection

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `DETECTION_THRESHOLD_PPM` | 5.0 | ppm | Sensor detection threshold |
| `METHANE_MOLAR_MASS` | 16.04 | g/mol | CH4 molar mass |
| `AIR_DENSITY` | 1.225 | kg/m3 | Standard air density at sea level |

### Optimizer

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `DEVIATION_EPSILON` | 10.0 | m | Cost function smoothing constant |
| `MAX_DEVIATION_M` | 200.0 | m | Maximum allowable path deviation |
| `TOP_K_RECOMMENDATIONS` | 5 | — | Number of waypoints to recommend |

### Cache

| Parameter | Default | Description |
|---|---|---|
| `CACHE_MAX_ENTRIES` | 32 | Max entries for Streamlit opportunity map cache |

---

## Data Integration Guide

The architecture is designed for easy data source replacement. Each data function returns a well-defined format that the physics and optimization layers consume without modification.

### 1. Leak Sources

**Replace:** `data/mock_data.py:get_leak_sources()`

**Expected format:** List of dicts, each with:
```python
{
    "name": str,           # Human-readable identifier
    "x": float,            # East coordinate (meters, local Cartesian)
    "y": float,            # North coordinate (meters, local Cartesian)
    "z": float,            # Release height above ground (meters)
    "emission_rate": float  # Emission rate (kg/s)
}
```

**Integration options:** SCADA asset database, OGI camera alerts, continuous monitoring system API

### 2. Worker Baseline Path

**Replace:** `data/mock_data.py:get_baseline_path()`

**Expected format:** `np.ndarray` of shape `(N, 2)` — ordered `[x, y]` waypoints in meters

**Integration options:** Work order management system, GPS tracking, route planning software

### 3. Wind Conditions

**Current:** Sidebar sliders with manual input

**Integration options:**
- Weather station API (OpenWeather, Environment Canada)
- On-site anemometer feed via SCADA/MQTT
- Numerical weather prediction (NWP) model output

### 4. Coordinate Transform

If using GPS coordinates (lat/lon), add a projection step before passing to the plume model:

```python
# Example using pyproj
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:326XX")  # appropriate UTM zone
x, y = transformer.transform(lat, lon)
```

**No changes to physics, optimization, or visualization layers are required** — only the data layer functions need modification.

---

## Technical Details

### Coordinate System
- **Frame:** Local Cartesian, origin at site center
- **Axes:** x = East, y = North
- **Units:** All distances in meters
- **Receptor height:** 1.5 m (handheld sensor at worker chest level)

### Ground Reflection
The Gaussian plume model uses the **image source method** — a virtual source is placed at -H below ground, and its contribution is added to the real source. This ensures:
- Concentration is always non-negative
- Mass is conserved (no flux through the ground plane)
- Physically correct behavior for ground-level and elevated releases

### Multi-Source Detection Probability
Combined detection uses the **complementary probability** formulation:

```
P_combined = 1 - ∏ᵢ(1 - Pᵢ)
```

This is the statistically correct formulation for independent detection events — it avoids both double-counting (sum) and information loss (max).

### Caching Strategy
- **Opportunity map:** Cached by `(sources, wind_speed, wind_direction, stability, grid_size, resolution)` — recomputed only when atmospheric conditions or sources change
- **Path deviation:** Cached by `(grid_size, resolution, baseline_path)` — independent of wind, computed once per path change
- Both caches use Streamlit's `@st.cache_data` with configurable max entries

### Route Optimization Algorithm

1. **Projection:** Each recommended waypoint is projected onto the nearest baseline path segment using parametric line projection with clamping
2. **Sorting:** Waypoints ordered by cumulative distance along the baseline (walking order)
3. **Clustering:** Consecutive waypoints within 8% of total path length grouped into single detour clusters
4. **Intra-cluster optimization:** Nearest-neighbor heuristic provides initial ordering; 2-opt local search iteratively improves by reversing sub-tours until no improving swap exists
5. **Splicing:** Optimized detour sequences inserted into baseline at correct segment indices (processed in reverse order to maintain index stability)

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| [NumPy](https://numpy.org/) | >= 1.24 | Array computation, vectorized plume equations |
| [SciPy](https://scipy.org/) | >= 1.10 | Spatial distance metrics (`cdist`) for path deviation |
| [Matplotlib](https://matplotlib.org/) | >= 3.7 | Fallback plotting capability |
| [Plotly](https://plotly.com/python/) | >= 5.14 | Interactive visualization (heatmaps, scatter, bar charts) |
| [Streamlit](https://streamlit.io/) | >= 1.28 | Web application framework |

**Package manager:** [uv](https://docs.astral.sh/uv/) (fast Python package manager)

**Python version:** 3.10+ (developed on 3.14)

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Ensure the app runs without errors (`uv run streamlit run main.py`)
5. Commit your changes (`git commit -m "Add your feature"`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

### Code Conventions

- PEP 8 style
- Google-style docstrings on public functions
- All configurable parameters in `config.py` — no hardcoded magic numbers in modules
- Module responsibility: physics in `models/`, optimization in `optimization/`, rendering in `visualization/`, data in `data/`

---

## License

This project is part of the [GeoSensorWebLab](https://github.com/GeoSensorWebLab) research initiative.

---

## References

- Turner, D. B. (1970). *Workbook of Atmospheric Dispersion Estimates*. U.S. Environmental Protection Agency.
- Pasquill, F. (1961). The estimation of the dispersion of windborne material. *The Meteorological Magazine*, 90, 33-49.
- Gifford, F. A. (1961). Use of routine meteorological observations for estimating atmospheric dispersion. *Nuclear Safety*, 2, 47-51.

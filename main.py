"""
Methane Leak Opportunistic Tasking System — Streamlit Interface.

Run with:  streamlit run main.py
"""

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np

from data.mock_data import get_leak_sources, get_baseline_path, get_wind_scenarios
from data.facility_layout import get_facility_layout
from optimization.opportunity_map import cached_opportunity_map
from optimization.tasking import (
    compute_tasking_scores,
    cached_path_deviation,
    recommend_waypoints,
    build_optimized_path,
)
from visualization.plots import create_site_figure, create_score_profile
from visualization.compass_widget import compass_html
from config import (
    DEFAULT_WIND_SPEED,
    DEFAULT_WIND_DIRECTION,
    DEFAULT_STABILITY_CLASS,
    GRID_SIZE_M,
    GRID_RESOLUTION_M,
    DEVIATION_EPSILON,
    MAX_DEVIATION_M,
    TOP_K_RECOMMENDATIONS,
)

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Methane Leak Tasking System",
    page_icon="🔥",
    layout="wide",
)

st.title("Methane Leak Opportunistic Tasking System")
st.markdown(
    "Recommends optimal locations for a field worker to detect methane leaks "
    "based on wind conditions and their existing route."
)

# ── Sidebar Controls ─────────────────────────────────────────────────────────

st.sidebar.header("Wind Conditions")

# Preset scenarios
scenarios = get_wind_scenarios()
scenario_names = ["Custom"] + [s["name"] for s in scenarios]
selected_scenario = st.sidebar.selectbox("Preset Scenario", scenario_names)

if selected_scenario != "Custom":
    scenario = next(s for s in scenarios if s["name"] == selected_scenario)
    default_speed = scenario["speed"]
    default_dir = scenario["direction"]
    default_stab = scenario["stability_class"]
else:
    default_speed = DEFAULT_WIND_SPEED
    default_dir = DEFAULT_WIND_DIRECTION
    default_stab = DEFAULT_STABILITY_CLASS

wind_speed = st.sidebar.slider(
    "Wind Speed (m/s)",
    min_value=0.5,
    max_value=15.0,
    value=default_speed,
    step=0.5,
)

wind_direction = st.sidebar.slider(
    "Wind Direction (degrees, meteorological — direction wind comes FROM)",
    min_value=0,
    max_value=359,
    value=default_dir,
    step=5,
)

stability_class = st.sidebar.select_slider(
    "Atmospheric Stability (A=very unstable, F=very stable)",
    options=["A", "B", "C", "D", "E", "F"],
    value=default_stab,
)

# ── Compass Widget (sidebar) ────────────────────────────────────────────────

st.sidebar.markdown("---")
components.html(compass_html(wind_direction, wind_speed), height=200)

# ── Optimizer Settings ───────────────────────────────────────────────────────

st.sidebar.header("Optimizer Settings")

max_deviation = st.sidebar.slider(
    "Max Path Deviation (m)",
    min_value=50,
    max_value=500,
    value=int(MAX_DEVIATION_M),
    step=25,
)

top_k = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=TOP_K_RECOMMENDATIONS,
)

grid_resolution = st.sidebar.select_slider(
    "Grid Resolution (m) — lower = finer but slower",
    options=[2, 5, 10, 20],
    value=GRID_RESOLUTION_M,
)

# ── Data ─────────────────────────────────────────────────────────────────────

sources = get_leak_sources()
baseline_path = get_baseline_path()
facility_layout = get_facility_layout()

# ── Cached Computation ───────────────────────────────────────────────────────

# Convert sources to a hashable key for the cache
sources_key = tuple(
    (s["name"], s["x"], s["y"], s["z"], s["emission_rate"]) for s in sources
)

# Hashable baseline path key for path-deviation cache
baseline_path_key = tuple(tuple(row) for row in baseline_path)

with st.spinner("Computing plume dispersion and opportunity map..."):
    X, Y, concentration_ppm, detection_prob = cached_opportunity_map(
        sources_key=sources_key,
        wind_speed=wind_speed,
        wind_direction_deg=wind_direction,
        stability_class=stability_class,
        grid_size=GRID_SIZE_M,
        resolution=grid_resolution,
    )

    # Cached path deviation (independent of wind — computed once)
    deviation = cached_path_deviation(
        grid_size=GRID_SIZE_M,
        resolution=grid_resolution,
        baseline_path_key=baseline_path_key,
    )

    scores = compute_tasking_scores(
        grid_x=X,
        grid_y=Y,
        detection_prob=detection_prob,
        baseline_path=baseline_path,
        epsilon=DEVIATION_EPSILON,
        max_deviation=float(max_deviation),
        precomputed_deviation=deviation,
    )

    recommendations = recommend_waypoints(
        grid_x=X,
        grid_y=Y,
        scores=scores,
        detection_prob=detection_prob,
        concentration_ppm=concentration_ppm,
        top_k=top_k,
    )

    optimized_path = build_optimized_path(baseline_path, recommendations)

# ── Visualization ────────────────────────────────────────────────────────────

fig_site = create_site_figure(
    grid_x=X,
    grid_y=Y,
    concentration_ppm=concentration_ppm,
    detection_prob=detection_prob,
    sources=sources,
    baseline_path=baseline_path,
    optimized_path=optimized_path,
    recommendations=recommendations,
    wind_speed=wind_speed,
    wind_direction_deg=wind_direction,
    facility_layout=facility_layout,
)

st.plotly_chart(fig_site, use_container_width=True)

# Score bar chart
fig_scores = create_score_profile(recommendations)
st.plotly_chart(fig_scores, use_container_width=True)

# ── Recommendations Table ────────────────────────────────────────────────────

st.subheader("Recommended Waypoints")

if recommendations:
    for i, rec in enumerate(recommendations):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"#{i+1} Location", f"({rec['x']:.0f}, {rec['y']:.0f}) m")
        col2.metric("Detection Prob.", f"{rec['detection_prob']:.1%}")
        col3.metric("Concentration", f"{rec['concentration_ppm']:.1f} ppm")
        col4.metric("Tasking Score", f"{rec['score']:.4f}")
else:
    st.info(
        "No high-value waypoints found near the baseline path under current conditions. "
        "Try adjusting wind direction or increasing max deviation."
    )

# ── Info Panel ───────────────────────────────────────────────────────────────

with st.expander("About the Model"):
    st.markdown(
        """
        **Gaussian Plume Model** — Standard atmospheric dispersion model for
        continuous point sources. Uses Pasquill-Gifford stability classes (A-F)
        to determine lateral and vertical dispersion coefficients.

        **Detection Probability** — Sigmoid function centered at the sensor's
        detection threshold (5 ppm by default). Accounts for sensor noise.

        **Tasking Score** — `Score = P(detection) / (PathDeviation + epsilon)`
        where `epsilon` prevents division-by-zero for on-path locations.

        **Optimized Path** — Inserts high-score waypoints as detours into the
        worker's baseline inspection route.

        ---
        *Data sources are currently mock. The modular architecture supports
        swapping in real SCADA/SensorUp feeds.*
        """
    )

# ── Source Assets Panel ──────────────────────────────────────────────────────

with st.expander("Leak Source Details"):
    for src in sources:
        st.markdown(
            f"- **{src['name']}**: position ({src['x']}, {src['y']}) m, "
            f"height {src['z']} m, emission rate {src['emission_rate']} kg/s"
        )

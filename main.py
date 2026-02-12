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
from optimization.opportunity_map import cached_opportunity_map, create_grid
from optimization.tasking import (
    compute_tasking_scores,
    cached_path_deviation,
    recommend_waypoints,
    build_optimized_path,
)
from models.prior import compute_all_priors, create_spatial_prior
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
    SENSOR_MDL_PPM,
    DETECTION_THRESHOLD_PPM,
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
    "based on wind conditions, equipment risk profiles, and their existing route."
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

# ── Sensor Settings ─────────────────────────────────────────────────────────

st.sidebar.header("Sensor Characteristics")

sensor_mdl = st.sidebar.slider(
    "Minimum Detection Limit (ppm)",
    min_value=0.0,
    max_value=10.0,
    value=SENSOR_MDL_PPM,
    step=0.5,
    help="Hard noise floor of the sensor. Concentrations below this value "
         "are physically undetectable (P = 0). Typical handheld CH4 detectors: "
         "0.5–2 ppm.",
)

sensor_threshold = st.sidebar.slider(
    "Detection Threshold (ppm)",
    min_value=1.0,
    max_value=50.0,
    value=DETECTION_THRESHOLD_PPM,
    step=0.5,
    help="Concentration at which probability of detection is 50%%. "
         "The sigmoid transition is centered here.",
)

if sensor_mdl >= sensor_threshold:
    st.sidebar.warning(
        "MDL should be below the detection threshold. "
        "Current settings may produce unexpected results."
    )

# ── Prior Model Settings ────────────────────────────────────────────────────

st.sidebar.header("Prior Risk Model")

use_prior = st.sidebar.checkbox(
    "Enable risk-based prior weighting",
    value=True,
    help="When enabled, recommendations are biased toward equipment with "
         "higher leak probability based on age, type, production rate, "
         "and inspection recency.",
)

# ── Data ─────────────────────────────────────────────────────────────────────

sources = get_leak_sources()
baseline_path = get_baseline_path()
facility_layout = get_facility_layout()

# ── Prior Computation ───────────────────────────────────────────────────────

prior_probs = compute_all_priors(sources)
spatial_prior = None

if use_prior:
    X_prior, Y_prior = create_grid(GRID_SIZE_M, grid_resolution)
    spatial_prior = create_spatial_prior(X_prior, Y_prior, sources, prior_probs)

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
        mdl_ppm=sensor_mdl,
        threshold_ppm=sensor_threshold,
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
        prior_weight=spatial_prior,
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

st.plotly_chart(fig_site, width="stretch")

# Score bar chart
fig_scores = create_score_profile(recommendations)
st.plotly_chart(fig_scores, width="stretch")

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

# ── Prior Risk Summary ──────────────────────────────────────────────────────

with st.expander("Equipment Risk Assessment (Prior Model)"):
    st.markdown(
        "Prior leak probabilities are computed from equipment attributes: "
        "**type** (compressor > valve > wellhead), **age** (older = higher risk), "
        "**production rate** (higher throughput = more stress), and "
        "**inspection recency** (longer since inspection = more uncertainty)."
    )
    st.markdown("")

    # Sort sources by prior probability (highest risk first)
    ranked = sorted(
        zip(sources, prior_probs), key=lambda x: x[1], reverse=True
    )

    for src, p in ranked:
        risk_level = "HIGH" if p > 0.3 else "MEDIUM" if p > 0.15 else "LOW"
        risk_color = "red" if p > 0.3 else "orange" if p > 0.15 else "green"
        st.markdown(
            f"- **{src['name']}** — Prior: **{p:.1%}** "
            f":{risk_color}_circle: {risk_level} | "
            f"Type: {src.get('equipment_type', 'unknown')}, "
            f"Age: {src.get('age_years', '?')} yr, "
            f"Production: {src.get('production_rate_mcfd', 0):.0f} mcf/d, "
            f"Last inspected: {src.get('last_inspection_days', '?')} days ago"
        )

# ── Info Panel ───────────────────────────────────────────────────────────────

with st.expander("About the Model"):
    scoring_formula = (
        "`Score = Prior(x,y) * P(detection) / (PathDeviation + epsilon)`"
        if use_prior
        else "`Score = P(detection) / (PathDeviation + epsilon)`"
    )
    st.markdown(
        f"""
        **Gaussian Plume Model** — Standard atmospheric dispersion model for
        continuous point sources. Uses Pasquill-Gifford stability classes (A-F)
        to determine lateral and vertical dispersion coefficients.

        **Detection Probability** — Sigmoid function centered at the sensor's
        detection threshold ({sensor_threshold} ppm), with a hard Minimum
        Detection Limit (MDL = {sensor_mdl} ppm) below which P = 0.
        Accounts for sensor noise floor and response characteristics.

        **Prior Risk Model** — Computes per-source leak probability from
        equipment type, age, production rate, and inspection recency. Projects
        onto a spatial grid using Gaussian kernels (Stage 1 of Bayesian
        architecture).

        **Tasking Score** — {scoring_formula}
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

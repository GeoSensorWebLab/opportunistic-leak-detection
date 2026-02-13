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

from data.mock_data import (
    get_leak_sources,
    get_baseline_path,
    get_wind_scenarios,
    get_wind_distribution,
    get_wind_fan,
)
from data.facility_layout import get_facility_layout
from optimization.opportunity_map import (
    cached_opportunity_map,
    cached_ensemble_opportunity_map,
    create_grid,
)
from optimization.tasking import (
    compute_tasking_scores,
    cached_path_deviation,
    recommend_waypoints,
    build_optimized_path,
)
from optimization.metrics import compute_route_metrics, find_nearest_source
from models.prior import compute_all_priors, create_spatial_prior
from models.measurement import Measurement
from models.bayesian import BayesianBeliefMap
from visualization.plots import (
    create_site_figure,
    create_single_map_figure,
    create_concentration_figure,
    create_score_profile,
)
from optimization.information_gain import (
    compute_information_scores,
    compute_ensemble_information_scores,
    compute_total_entropy,
)
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
    DEFAULT_ENSEMBLE_SCENARIOS,
    DEFAULT_WIND_SPREAD_DEG,
    DEFAULT_DUTY_CYCLE,
    DEFAULT_PUFF_TIME_S,
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

# ── Wind Ensemble ───────────────────────────────────────────────────────────

use_ensemble = st.sidebar.checkbox(
    "Enable wind ensemble",
    value=False,
    help="Average detection probability across multiple wind scenarios "
         "to make recommendations robust to wind variability.",
)

wind_scenarios = None
if use_ensemble:
    ensemble_mode = st.sidebar.radio(
        "Ensemble Mode",
        ["8-Direction Rose", "Directional Fan", "Custom Scenarios"],
    )

    if ensemble_mode == "8-Direction Rose":
        wind_scenarios = get_wind_distribution()
    elif ensemble_mode == "Directional Fan":
        fan_center = st.sidebar.slider(
            "Fan Center Direction",
            min_value=0,
            max_value=359,
            value=wind_direction,
            step=5,
        )
        fan_spread = st.sidebar.slider(
            "Fan Spread (degrees)",
            min_value=5.0,
            max_value=90.0,
            value=DEFAULT_WIND_SPREAD_DEG,
            step=5.0,
        )
        fan_count = st.sidebar.slider(
            "Number of Scenarios",
            min_value=3,
            max_value=16,
            value=DEFAULT_ENSEMBLE_SCENARIOS,
        )
        wind_scenarios = get_wind_fan(
            center_direction=float(fan_center),
            spread_deg=fan_spread,
            num_scenarios=fan_count,
            speed=wind_speed,
            stability_class=stability_class,
        )
    else:  # Custom Scenarios — use existing presets with equal weights
        presets = get_wind_scenarios()
        wind_scenarios = [
            {
                "direction": p["direction"],
                "speed": p["speed"],
                "stability_class": p["stability_class"],
                "weight": 1.0 / len(presets),
            }
            for p in presets
        ]

    st.sidebar.caption(f"Ensemble: {len(wind_scenarios)} scenarios")

# ── Compass Widget (sidebar) ────────────────────────────────────────────────

st.sidebar.markdown("---")
components.html(compass_html(wind_direction, wind_speed), height=200)

# ── Optimizer Settings ───────────────────────────────────────────────────────

st.sidebar.header("Optimizer Settings")

scoring_mode = st.sidebar.radio(
    "Scoring Method",
    ["Heuristic", "Information-Theoretic (EER)"],
    help="**Heuristic**: P(detect) / deviation — fast, uses detection probability. "
         "**EER**: Expected Entropy Reduction — measures where you learn the "
         "most about leak locations. Requires prior risk model.",
)

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

plume_mode = st.sidebar.radio(
    "Plume Model",
    ["Instantaneous (standard)", "Crosswind-Integrated", "Gaussian Puff"],
    help="**Instantaneous**: Standard Gaussian plume — sharp, narrow peaks. "
         "**Crosswind-Integrated**: Integrates out lateral dispersion for "
         "broader, lower-peak plumes that better match time-averaged field "
         "measurements under turbulent conditions. "
         "**Gaussian Puff**: Instantaneous release model — a single mass "
         "puff drifting downwind, suitable for intermittent/episodic leaks.",
)
if "Crosswind" in plume_mode:
    plume_mode_key = "integrated"
elif "Puff" in plume_mode:
    plume_mode_key = "puff"
else:
    plume_mode_key = "instantaneous"

puff_time_s = DEFAULT_PUFF_TIME_S
if plume_mode_key == "puff":
    puff_time_s = st.sidebar.slider(
        "Time Since Puff Release (s)",
        min_value=10.0,
        max_value=600.0,
        value=DEFAULT_PUFF_TIME_S,
        step=10.0,
        help="Time elapsed since the instantaneous puff release. "
             "Larger values mean the puff has traveled farther and dispersed more.",
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

# ── Temporal Behavior ──────────────────────────────────────────────────────

st.sidebar.header("Temporal Behavior")

st.sidebar.caption(
    "Duty cycle: fraction of time each source is actively emitting "
    "(1.0 = continuous, 0.0 = never). Reflects intermittent leaks "
    "from pressure cycling, thermal effects, etc."
)

# ── Bayesian Update Settings ───────────────────────────────────────────────

st.sidebar.header("Bayesian Updates")

use_bayesian = st.sidebar.checkbox(
    "Enable Bayesian belief updating",
    value=False,
    help="Update the belief map with field observations (detections or "
         "non-detections) to refine leak location estimates. Requires "
         "prior risk model to be enabled.",
)

if use_bayesian and not use_prior:
    st.sidebar.warning("Bayesian updates require the prior risk model to be enabled.")
    use_bayesian = False

if use_bayesian:
    st.sidebar.subheader("Add Measurement")
    meas_x = st.sidebar.number_input("Measurement X (m)", value=0.0, step=10.0)
    meas_y = st.sidebar.number_input("Measurement Y (m)", value=0.0, step=10.0)
    meas_conc = st.sidebar.number_input(
        "Concentration (ppm)", value=0.0, min_value=0.0, step=1.0,
    )
    meas_detected = st.sidebar.checkbox("Detection triggered", value=False)

    col_add, col_reset = st.sidebar.columns(2)
    add_measurement = col_add.button("Add Measurement")
    reset_belief = col_reset.button("Reset Belief")

    # Initialize session state for Bayesian belief map
    if "bayesian_measurements" not in st.session_state:
        st.session_state.bayesian_measurements = []

    if reset_belief:
        st.session_state.bayesian_measurements = []
        if "belief_map_obj" in st.session_state:
            del st.session_state["belief_map_obj"]

    if add_measurement:
        new_meas = Measurement(
            x=meas_x,
            y=meas_y,
            concentration_ppm=meas_conc,
            detected=meas_detected,
            wind_speed=wind_speed,
            wind_direction_deg=float(wind_direction),
            stability_class=stability_class,
        )
        st.session_state.bayesian_measurements.append(new_meas)

    n_meas = len(st.session_state.get("bayesian_measurements", []))
    st.sidebar.caption(f"Measurements recorded: {n_meas}")

# ── Data ─────────────────────────────────────────────────────────────────────

sources = get_leak_sources()
baseline_path = get_baseline_path()
facility_layout = get_facility_layout()

# Apply per-source duty cycle sliders (in the Temporal Behavior sidebar section)
for src in sources:
    dc = st.sidebar.slider(
        f"Duty Cycle — {src['name']}",
        min_value=0.0,
        max_value=1.0,
        value=float(src.get("duty_cycle", DEFAULT_DUTY_CYCLE)),
        step=0.05,
        key=f"dc_{src['name']}",
    )
    src["duty_cycle"] = dc

# ── Prior Computation ───────────────────────────────────────────────────────

prior_probs = compute_all_priors(sources)
spatial_prior = None

if use_prior:
    X_prior, Y_prior = create_grid(GRID_SIZE_M, grid_resolution)
    spatial_prior = create_spatial_prior(X_prior, Y_prior, sources, prior_probs)

# ── Cached Computation ───────────────────────────────────────────────────────

# Convert sources to a hashable key for the cache (6-tuple with duty_cycle)
sources_key = tuple(
    (s["name"], s["x"], s["y"], s["z"], s["emission_rate"], s.get("duty_cycle", 1.0))
    for s in sources
)

# Hashable baseline path key for path-deviation cache
baseline_path_key = tuple(tuple(row) for row in baseline_path)

with st.spinner("Computing plume dispersion and opportunity map..."):
    if use_ensemble and wind_scenarios:
        scenarios_key = tuple(
            (s["direction"], s["speed"], s["stability_class"], s["weight"])
            for s in wind_scenarios
        )
        X, Y, concentration_ppm, detection_prob = cached_ensemble_opportunity_map(
            sources_key=sources_key,
            scenarios_key=scenarios_key,
            grid_size=GRID_SIZE_M,
            resolution=grid_resolution,
            mdl_ppm=sensor_mdl,
            threshold_ppm=sensor_threshold,
            plume_mode=plume_mode_key,
        )
    else:
        X, Y, concentration_ppm, detection_prob = cached_opportunity_map(
            sources_key=sources_key,
            wind_speed=wind_speed,
            wind_direction_deg=wind_direction,
            stability_class=stability_class,
            grid_size=GRID_SIZE_M,
            resolution=grid_resolution,
            mdl_ppm=sensor_mdl,
            threshold_ppm=sensor_threshold,
            plume_mode=plume_mode_key,
        )

    # ── Bayesian Belief Update ──────────────────────────────────────────────
    belief_map = None
    scoring_prior = spatial_prior  # default: raw prior (or None)

    if use_bayesian and spatial_prior is not None:
        bayesian_obj = BayesianBeliefMap(
            grid_x=X,
            grid_y=Y,
            prior=spatial_prior,
            sources=sources,
        )
        for m in st.session_state.get("bayesian_measurements", []):
            bayesian_obj.update(m)
        belief_map = bayesian_obj.get_belief_map()
        scoring_prior = belief_map  # posterior replaces raw prior in scoring

    # Cached path deviation (independent of wind — computed once)
    deviation = cached_path_deviation(
        grid_size=GRID_SIZE_M,
        resolution=grid_resolution,
        baseline_path_key=baseline_path_key,
    )

    # ── Scoring ────────────────────────────────────────────────────────────
    use_eer = scoring_mode == "Information-Theoretic (EER)"

    if use_eer and scoring_prior is None:
        st.warning(
            "Information-theoretic scoring requires the prior risk model. "
            "Falling back to heuristic scoring."
        )
        use_eer = False

    if use_eer:
        avg_emission = float(np.mean(
            [s["emission_rate"] * s.get("duty_cycle", 1.0) for s in sources]
        ))
        spinner_msg = "Computing Expected Entropy Reduction (EER)..."
        with st.spinner(spinner_msg):
            if use_ensemble and wind_scenarios:
                scores = compute_ensemble_information_scores(
                    grid_x=X,
                    grid_y=Y,
                    belief=scoring_prior,
                    deviation=deviation,
                    max_deviation=float(max_deviation),
                    wind_scenarios=wind_scenarios,
                    avg_emission=avg_emission,
                    epsilon=DEVIATION_EPSILON,
                )
            else:
                scores = compute_information_scores(
                    grid_x=X,
                    grid_y=Y,
                    belief=scoring_prior,
                    deviation=deviation,
                    max_deviation=float(max_deviation),
                    wind_speed=wind_speed,
                    wind_direction_deg=float(wind_direction),
                    stability_class=stability_class,
                    avg_emission=avg_emission,
                    epsilon=DEVIATION_EPSILON,
                )
    else:
        scores = compute_tasking_scores(
            grid_x=X,
            grid_y=Y,
            detection_prob=detection_prob,
            baseline_path=baseline_path,
            epsilon=DEVIATION_EPSILON,
            max_deviation=float(max_deviation),
            precomputed_deviation=deviation,
            prior_weight=scoring_prior,
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

# ── Route Metrics ────────────────────────────────────────────────────────────

metrics = compute_route_metrics(baseline_path, optimized_path, recommendations)

# ── Summary Metrics Banner ───────────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Baseline Distance", f"{metrics['baseline_distance_m']:.0f} m")
m2.metric("Optimized Distance", f"{metrics['optimized_distance_m']:.0f} m")
m3.metric(
    "Time Impact",
    f"+{metrics['time_impact_min']:.1f} min",
    delta=f"+{metrics['added_detour_m']:.0f} m",
    delta_color="off",
)
m4.metric("Detour Points", f"{metrics['num_detour_points']}")
m5.metric("Avg Detection Prob", f"{metrics['avg_detection_prob']:.1%}")

# ── Visualization ──────────────────────────────────────────────────────────

view_options = ["Detection Map", "Concentration Map", "Side-by-Side"]
if use_bayesian and belief_map is not None:
    view_options.append("Belief Map")

active_view = st.radio(
    "Map View",
    view_options,
    horizontal=True,
    label_visibility="collapsed",
)

if active_view == "Detection Map":
    fig_detect = create_single_map_figure(
        grid_x=X,
        grid_y=Y,
        detection_prob=detection_prob,
        sources=sources,
        baseline_path=baseline_path,
        optimized_path=optimized_path,
        recommendations=recommendations,
        wind_speed=wind_speed,
        wind_direction_deg=wind_direction,
        facility_layout=facility_layout,
    )
    st.plotly_chart(fig_detect, use_container_width=True)

elif active_view == "Concentration Map":
    fig_conc = create_concentration_figure(
        grid_x=X,
        grid_y=Y,
        concentration_ppm=concentration_ppm,
        sources=sources,
        wind_speed=wind_speed,
        wind_direction_deg=wind_direction,
        facility_layout=facility_layout,
    )
    st.plotly_chart(fig_conc, use_container_width=True)

elif active_view == "Side-by-Side":
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

elif active_view == "Belief Map" and belief_map is not None:
    import copy
    fig_belief = copy.deepcopy(create_single_map_figure(
        grid_x=X,
        grid_y=Y,
        detection_prob=belief_map,
        sources=sources,
        baseline_path=baseline_path,
        optimized_path=optimized_path,
        recommendations=recommendations,
        wind_speed=wind_speed,
        wind_direction_deg=wind_direction,
        facility_layout=facility_layout,
        colorbar_title="P(leak)",
    ))
    # Overlay measurement markers
    measurements = st.session_state.get("bayesian_measurements", [])
    if measurements:
        import plotly.graph_objects as go
        det_x = [m.x for m in measurements if m.detected]
        det_y = [m.y for m in measurements if m.detected]
        nodet_x = [m.x for m in measurements if not m.detected]
        nodet_y = [m.y for m in measurements if not m.detected]

        if det_x:
            fig_belief.add_trace(
                go.Scatter(
                    x=det_x,
                    y=det_y,
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="limegreen",
                        symbol="circle",
                        line=dict(width=2, color="white"),
                    ),
                    name="Detection (+)",
                    hovertemplate="Detection<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
                )
            )
        if nodet_x:
            fig_belief.add_trace(
                go.Scatter(
                    x=nodet_x,
                    y=nodet_y,
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="red",
                        symbol="x",
                        line=dict(width=2, color="white"),
                    ),
                    name="Non-detection (-)",
                    hovertemplate="Non-detection<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
                )
            )

    st.plotly_chart(fig_belief, use_container_width=True)

# Score bar chart
fig_scores = create_score_profile(recommendations)
st.plotly_chart(fig_scores, use_container_width=True)

# ── Recommendations Table ────────────────────────────────────────────────────

st.subheader("Recommended Waypoints")

if recommendations:
    for i, rec in enumerate(recommendations):
        nearest = find_nearest_source(rec, sources)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(f"#{i+1} Location", f"({rec['x']:.0f}, {rec['y']:.0f}) m")
        col2.metric("Nearest Equipment", nearest or "—")
        col3.metric("Detection Prob.", f"{rec['detection_prob']:.1%}")
        col4.metric("Concentration", f"{rec['concentration_ppm']:.1f} ppm")
        col5.metric("Tasking Score", f"{rec['score']:.4f}")
        if i < len(recommendations) - 1:
            st.divider()
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
    if use_eer:
        scoring_formula = (
            "`Score = EER(x,y) / (PathDeviation + epsilon)` "
            "where EER = Expected Entropy Reduction (bits of information gained)"
        )
    elif use_prior:
        scoring_formula = (
            "`Score = Prior(x,y) * P(detection) / (PathDeviation + epsilon)`"
        )
    else:
        scoring_formula = (
            "`Score = P(detection) / (PathDeviation + epsilon)`"
        )
    st.markdown(
        f"""
        **Gaussian Plume Model** — Standard atmospheric dispersion model for
        continuous point sources. Uses Pasquill-Gifford stability classes (A-F)
        to determine lateral and vertical dispersion coefficients. The
        *crosswind-integrated* variant integrates out lateral dispersion,
        producing broader plumes that better match time-averaged field data.

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

        **Information-Theoretic Scoring (EER)** — When enabled, replaces
        the heuristic score with Expected Entropy Reduction. For each
        candidate measurement location, computes the expected reduction in
        belief-map uncertainty considering both detection and non-detection
        outcomes. Answers: *"Where should I measure to learn the most?"*
        (Stage 3 of Bayesian architecture).

        **Bayesian Belief Update** — When enabled, field observations
        (detections and non-detections) update the spatial belief map via
        cell-wise Bayes' theorem, refining leak location estimates after
        each measurement (Stage 2 of Bayesian architecture).

        **Temporal Behavior (Duty Cycle)** — Real methane leaks are often
        intermittent due to pressure cycling, thermal effects, and
        operational changes. Each source has a duty cycle (0–1) representing
        the fraction of time it actively emits. Emission rates are scaled
        by duty cycle for time-averaged modeling. The *Gaussian Puff* plume
        model simulates a single instantaneous mass release drifting
        downwind, suitable for episodic/intermittent leak events.

        **Wind Ensemble** — When enabled, averages detection probability
        across multiple wind scenarios (rose, directional fan, or custom)
        to make recommendations robust to wind variability during a
        30-45 min inspection walk.

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
            f"height {src['z']} m, emission rate {src['emission_rate']} kg/s, "
            f"duty cycle {src.get('duty_cycle', 1.0):.0%}"
        )

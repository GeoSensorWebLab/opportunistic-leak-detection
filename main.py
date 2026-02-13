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
import pandas as pd

from data.interfaces import MockDataProvider, FileDataProvider
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
    create_entropy_figure,
    create_prior_posterior_figure,
    create_convergence_figure,
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

# ── Data Provider ────────────────────────────────────────────────────────────
if "data_provider" not in st.session_state:
    st.session_state.data_provider = MockDataProvider()
data_provider = st.session_state.data_provider

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Methane Leak Tasking System",
    page_icon="🔥",
    layout="wide",
)


# ── Custom CSS ──────────────────────────────────────────────────────────────

def _inject_custom_css():
    st.markdown("""
    <style>
    /* Reduce default top padding */
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetric"] label { font-size: 0.78rem; opacity: 0.7; }

    /* Tab accent color */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #00b4d8;
    }
    .stTabs [aria-selected="true"] {
        color: #00b4d8;
    }

    /* Subtler dividers */
    hr { opacity: 0.15; }

    /* Sidebar expander styling */
    .stSidebar [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 6px;
        margin-bottom: 4px;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
    }

    /* Status badge styling */
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 6px;
        background: rgba(0, 180, 216, 0.15);
        color: #00b4d8;
        border: 1px solid rgba(0, 180, 216, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)


_inject_custom_css()

# ── Sidebar Controls ──────────────────────────────────────────────────────────
# (Header rendered after sidebar so we can show active config badges)

with st.sidebar.expander("Data Source", expanded=False):
    data_source_mode = st.radio(
        "Source",
        ["Mock (built-in)", "Load from files"],
        key="data_source_mode",
    )
    if data_source_mode == "Load from files":
        import tempfile
        uploaded_sources = st.file_uploader("Sources JSON", type=["json"], key="up_src")
        uploaded_path = st.file_uploader("Path CSV", type=["csv"], key="up_path")
        uploaded_wind = st.file_uploader("Wind Scenarios JSON", type=["json"], key="up_wind")

        if uploaded_sources and uploaded_path and uploaded_wind:
            try:
                tmp_dir = tempfile.mkdtemp()
                src_path = os.path.join(tmp_dir, "sources.json")
                path_path = os.path.join(tmp_dir, "path.csv")
                wind_path = os.path.join(tmp_dir, "wind_scenarios.json")
                with open(src_path, "wb") as f:
                    f.write(uploaded_sources.getvalue())
                with open(path_path, "wb") as f:
                    f.write(uploaded_path.getvalue())
                with open(wind_path, "wb") as f:
                    f.write(uploaded_wind.getvalue())
                new_provider = FileDataProvider(
                    sources_path=src_path,
                    path_path=path_path,
                    wind_scenarios_path=wind_path,
                )
                st.session_state.data_provider = new_provider
                data_provider = new_provider
                st.success("Loaded custom data files.")
            except Exception as e:
                st.error(f"Failed to load files: {e}")
        else:
            st.caption("Upload all 3 files to switch data source.")
    else:
        if not isinstance(st.session_state.data_provider, MockDataProvider):
            st.session_state.data_provider = MockDataProvider()
            data_provider = st.session_state.data_provider

with st.sidebar.expander("Wind Conditions", expanded=True):
    # Preset scenarios
    scenarios = data_provider.get_wind_scenarios()
    scenario_names = ["Custom"] + [s["name"] for s in scenarios]
    selected_scenario = st.selectbox("Preset Scenario", scenario_names)

    if selected_scenario != "Custom":
        scenario = next(s for s in scenarios if s["name"] == selected_scenario)
        default_speed = scenario["speed"]
        default_dir = scenario["direction"]
        default_stab = scenario["stability_class"]
    else:
        default_speed = DEFAULT_WIND_SPEED
        default_dir = DEFAULT_WIND_DIRECTION
        default_stab = DEFAULT_STABILITY_CLASS

    wind_speed = st.slider(
        "Wind Speed (m/s)",
        min_value=0.5,
        max_value=15.0,
        value=default_speed,
        step=0.5,
    )

    wind_direction = st.slider(
        "Wind Direction (deg, meteorological)",
        min_value=0,
        max_value=359,
        value=default_dir,
        step=5,
    )

    stability_class = st.select_slider(
        "Stability (A=unstable, F=stable)",
        options=["A", "B", "C", "D", "E", "F"],
        value=default_stab,
    )

    use_ensemble = st.checkbox(
        "Enable wind ensemble",
        value=False,
        help="Average detection probability across multiple wind scenarios.",
    )

    wind_scenarios = None
    if use_ensemble:
        ensemble_mode = st.radio(
            "Ensemble Mode",
            ["8-Direction Rose", "Directional Fan", "Custom Scenarios"],
        )

        if ensemble_mode == "8-Direction Rose":
            wind_scenarios = data_provider.get_wind_distribution()
        elif ensemble_mode == "Directional Fan":
            fan_center = st.slider(
                "Fan Center Direction",
                min_value=0,
                max_value=359,
                value=wind_direction,
                step=5,
            )
            fan_spread = st.slider(
                "Fan Spread (degrees)",
                min_value=5.0,
                max_value=90.0,
                value=DEFAULT_WIND_SPREAD_DEG,
                step=5.0,
            )
            fan_count = st.slider(
                "Number of Scenarios",
                min_value=3,
                max_value=16,
                value=DEFAULT_ENSEMBLE_SCENARIOS,
            )
            wind_scenarios = data_provider.get_wind_fan(
                center_direction=float(fan_center),
                spread_deg=fan_spread,
                num_scenarios=fan_count,
                speed=wind_speed,
                stability_class=stability_class,
            )
        else:  # Custom Scenarios — use existing presets with equal weights
            presets = data_provider.get_wind_scenarios()
            wind_scenarios = [
                {
                    "direction": p["direction"],
                    "speed": p["speed"],
                    "stability_class": p["stability_class"],
                    "weight": 1.0 / len(presets),
                }
                for p in presets
            ]

        st.caption(f"Ensemble: {len(wind_scenarios)} scenarios")

    components.html(compass_html(wind_direction, wind_speed), height=190)

# ── Plume & Optimizer ────────────────────────────────────────────────────────

with st.sidebar.expander("Plume & Optimizer", expanded=True):
    scoring_mode = st.radio(
        "Scoring Method",
        ["Heuristic", "Information-Theoretic (EER)"],
        help="**Heuristic**: P(detect) / deviation. "
             "**EER**: Expected Entropy Reduction.",
    )

    plume_mode = st.radio(
        "Plume Model",
        ["Instantaneous (standard)", "Crosswind-Integrated", "Gaussian Puff"],
        help="**Instantaneous**: Sharp Gaussian plume. "
             "**Crosswind-Integrated**: Broader, time-averaged plumes. "
             "**Gaussian Puff**: Episodic/intermittent releases.",
    )
    if "Crosswind" in plume_mode:
        plume_mode_key = "integrated"
    elif "Puff" in plume_mode:
        plume_mode_key = "puff"
    else:
        plume_mode_key = "instantaneous"

    puff_time_s = DEFAULT_PUFF_TIME_S
    if plume_mode_key == "puff":
        puff_time_s = st.slider(
            "Time Since Puff Release (s)",
            min_value=10.0,
            max_value=600.0,
            value=DEFAULT_PUFF_TIME_S,
            step=10.0,
        )

    max_deviation = st.slider(
        "Max Path Deviation (m)",
        min_value=50,
        max_value=500,
        value=int(MAX_DEVIATION_M),
        step=25,
    )

    top_k = st.slider(
        "Recommendations",
        min_value=1,
        max_value=10,
        value=TOP_K_RECOMMENDATIONS,
    )

    grid_resolution = st.select_slider(
        "Grid Resolution (m)",
        options=[2, 5, 10, 20],
        value=GRID_RESOLUTION_M,
    )

    st.divider()
    use_multi_worker = st.checkbox("Multi-Worker Mode", value=False)
    num_workers = 1
    if use_multi_worker:
        from config import DEFAULT_NUM_WORKERS, MAX_WORKERS
        num_workers = st.slider(
            "Number of Workers",
            min_value=2,
            max_value=MAX_WORKERS,
            value=max(2, DEFAULT_NUM_WORKERS),
        )

# ── Sensor Settings ─────────────────────────────────────────────────────────

with st.sidebar.expander("Sensor Settings", expanded=False):
    sensor_mdl = st.slider(
        "Minimum Detection Limit (ppm)",
        min_value=0.0,
        max_value=10.0,
        value=SENSOR_MDL_PPM,
        step=0.5,
        help="Hard noise floor. Concentrations below are undetectable (P=0).",
    )

    sensor_threshold = st.slider(
        "Detection Threshold (ppm)",
        min_value=1.0,
        max_value=50.0,
        value=DETECTION_THRESHOLD_PPM,
        step=0.5,
        help="Concentration at which P(detection) = 50%.",
    )

    if sensor_mdl >= sensor_threshold:
        st.warning(
            "MDL should be below the detection threshold."
        )

# ── Prior & Bayesian ────────────────────────────────────────────────────────

with st.sidebar.expander("Prior & Bayesian", expanded=False):
    use_prior = st.checkbox(
        "Enable risk-based prior",
        value=True,
        help="Bias recommendations toward higher-risk equipment.",
    )

    use_bayesian = st.checkbox(
        "Enable Bayesian belief updating",
        value=False,
        help="Update belief map with field observations.",
    )

    if use_bayesian and not use_prior:
        st.warning("Bayesian updates require the prior risk model.")
        use_bayesian = False

    if use_bayesian:
        st.markdown("**Add Measurement**")
        meas_x = st.number_input("Measurement X (m)", value=0.0, step=10.0)
        meas_y = st.number_input("Measurement Y (m)", value=0.0, step=10.0)
        meas_conc = st.number_input(
            "Concentration (ppm)", value=0.0, min_value=0.0, step=1.0,
        )
        meas_detected = st.checkbox("Detection triggered", value=False)

        col_add, col_reset = st.columns(2)
        add_measurement = col_add.button("Add Measurement")
        reset_belief = col_reset.button("Reset Belief")

        # Initialize session state for Bayesian belief map
        if "bayesian_measurements" not in st.session_state:
            st.session_state.bayesian_measurements = []
        if "entropy_history" not in st.session_state:
            st.session_state.entropy_history = []

        if reset_belief:
            st.session_state.bayesian_measurements = []
            st.session_state.entropy_history = []
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
        st.caption(f"Measurements recorded: {n_meas}")

        st.divider()
        st.markdown("**Restore Saved State**")
        uploaded_state = st.file_uploader(
            "Upload .npz belief state",
            type=["npz"],
            key="state_upload",
        )
        if uploaded_state is not None:
            from data.state_io import deserialize_state
            try:
                state_data = deserialize_state(uploaded_state.read())
                st.session_state.bayesian_measurements = state_data.get("measurements", [])
                st.session_state.entropy_history = state_data.get("entropy_history", [])
                st.session_state["_restored_belief"] = state_data["belief"]
                st.success(f"Restored state with {len(state_data.get('measurements', []))} measurements")
            except Exception as e:
                st.error(f"Failed to load state: {e}")

    st.divider()
    use_campaign = st.checkbox(
        "Campaign Mode (multi-day)",
        value=False,
        help="Carry forward posterior belief across inspection days.",
    )
    if use_campaign:
        if "campaign_state" not in st.session_state:
            from optimization.campaign import CampaignState
            st.session_state.campaign_state = CampaignState()
        campaign_days = len(st.session_state.campaign_state.days)
        st.caption(f"Campaign days completed: {campaign_days}")

        if st.button("Start New Day"):
            st.session_state["_campaign_start_day"] = True
        if st.button("Reset Campaign"):
            from optimization.campaign import CampaignState
            st.session_state.campaign_state = CampaignState()

# ── Data ─────────────────────────────────────────────────────────────────────

sources = [s.copy() for s in data_provider.get_leak_sources()]
baseline_path = data_provider.get_baseline_path()
facility_layout = get_facility_layout()

# ── Temporal Behavior (needs sources loaded) ────────────────────────────────

with st.sidebar.expander("Temporal (Duty Cycles)", expanded=False):
    st.caption(
        "Duty cycle: fraction of time each source actively emits "
        "(1.0 = continuous, 0.0 = never)."
    )
    for src in sources:
        dc = st.slider(
            f"Duty Cycle — {src['name']}",
            min_value=0.0,
            max_value=1.0,
            value=float(src.get("duty_cycle", DEFAULT_DUTY_CYCLE)),
            step=0.05,
            key=f"dc_{src['name']}",
        )
        src["duty_cycle"] = dc

    st.divider()
    time_resolved = st.checkbox(
        "Time-resolved (Bernoulli) mode",
        value=False,
        help="Each source is randomly on/off per its duty cycle instead of time-averaged.",
    )

    if time_resolved:
        if "temporal_seed" not in st.session_state:
            st.session_state.temporal_seed = 42
        if st.button("Resample"):
            st.session_state.temporal_seed += 1
        rng_tr = np.random.default_rng(st.session_state.temporal_seed)
        active_sources = []
        for src in sources:
            dc = src.get("duty_cycle", 1.0)
            if rng_tr.random() < dc:
                active_src = dict(src, duty_cycle=1.0)
                active_sources.append(active_src)
        st.caption(f"Active sources: {len(active_sources)} / {len(sources)}")
    else:
        active_sources = None

# ── Prior Computation ───────────────────────────────────────────────────────

prior_probs = compute_all_priors(sources)
spatial_prior = None

if use_prior:
    X_prior, Y_prior = create_grid(GRID_SIZE_M, grid_resolution)
    spatial_prior = create_spatial_prior(X_prior, Y_prior, sources, prior_probs)

# ── Cached Computation ───────────────────────────────────────────────────────

# Use active_sources in time-resolved mode, otherwise original sources
compute_sources = active_sources if (time_resolved and active_sources is not None) else sources

# Convert sources to a hashable key for the cache (6-tuple with duty_cycle)
sources_key = tuple(
    (s["name"], s["x"], s["y"], s["z"], s["emission_rate"], s.get("duty_cycle", 1.0))
    for s in compute_sources
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
        # Replay measurements and record entropy after each update
        measurements_list = st.session_state.get("bayesian_measurements", [])
        entropy_hist = [compute_total_entropy(bayesian_obj.get_belief_map())]
        for m in measurements_list:
            bayesian_obj.update(m)
            entropy_hist.append(compute_total_entropy(bayesian_obj.get_belief_map()))
        st.session_state.entropy_history = entropy_hist
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

    # Multi-worker allocation
    worker_routes = None
    if use_multi_worker and num_workers > 1:
        from optimization.multi_worker import split_baseline_path, allocate_waypoints
        worker_paths = split_baseline_path(baseline_path, num_workers)
        worker_routes = allocate_waypoints(recommendations, worker_paths, float(max_deviation))

    # Share data to Worker Guidance page via session state
    st.session_state["_wg_grid_x"] = X
    st.session_state["_wg_grid_y"] = Y
    st.session_state["_wg_scores"] = scores
    st.session_state["_wg_optimized_path"] = optimized_path
    st.session_state["_wg_recommendations"] = recommendations
    st.session_state["_wg_baseline_path"] = baseline_path

# ── Route Metrics ────────────────────────────────────────────────────────────

metrics = compute_route_metrics(baseline_path, optimized_path, recommendations)

# ── Header ────────────────────────────────────────────────────────────────────

_hdr_left, _hdr_right = st.columns([3, 2])
with _hdr_left:
    st.title("Methane Leak Opportunistic Tasking")
    st.caption(
        "Optimal locations for field workers to detect methane leaks — "
        "combining atmospheric dispersion, equipment risk, and route planning."
    )
with _hdr_right:
    _wind_label = f"Wind {wind_direction}\u00b0 @ {wind_speed} m/s"
    _scoring_label = "EER" if use_eer else "Heuristic"
    _plume_label = plume_mode_key.capitalize()
    _ensemble_label = "Ensemble" if (use_ensemble and wind_scenarios) else "Single"
    st.markdown(
        f'<div style="text-align:right; padding-top:1.2rem;">'
        f'<span class="status-badge">{_wind_label}</span>'
        f'<span class="status-badge">{_plume_label}</span>'
        f'<span class="status-badge">{_scoring_label}</span>'
        f'<span class="status-badge">{_ensemble_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

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

# ── Visualization (Tabs) ───────────────────────────────────────────────────

_tab_names = ["Detection Map", "Concentration Map", "Side-by-Side"]
if use_bayesian and belief_map is not None:
    _tab_names.append("Belief Map")
    _tab_names.append("Entropy Map")
    _tab_names.append("Prior vs Posterior")
    entropy_history = st.session_state.get("entropy_history", [])
    if len(entropy_history) > 1:
        _tab_names.append("Convergence")

_tabs = st.tabs(_tab_names)

with _tabs[0]:  # Detection Map
    # Convert worker_routes to serializable tuples for caching
    _wr_key = None
    if worker_routes and len(worker_routes) > 1:
        _wr_key = tuple(
            (r.worker_id, tuple(map(tuple, r.optimized_path)) if r.optimized_path is not None else ())
            for r in worker_routes
        )
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
        worker_routes=worker_routes if worker_routes and len(worker_routes) > 1 else None,
    )
    st.plotly_chart(fig_detect, use_container_width=True)

with _tabs[1]:  # Concentration Map
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

with _tabs[2]:  # Side-by-Side
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

if use_bayesian and belief_map is not None:
    with _tabs[3]:  # Belief Map
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

    # Entropy Map tab
    with _tabs[4]:
        fig_entropy = create_entropy_figure(
            grid_x=X,
            grid_y=Y,
            belief=belief_map,
            sources=sources,
            wind_speed=wind_speed,
            wind_direction_deg=wind_direction,
            facility_layout=facility_layout,
        )
        st.plotly_chart(fig_entropy, use_container_width=True)

    # Prior vs Posterior tab
    with _tabs[5]:
        fig_pp = create_prior_posterior_figure(
            grid_x=X,
            grid_y=Y,
            prior=spatial_prior,
            posterior=belief_map,
            sources=sources,
            wind_speed=wind_speed,
            wind_direction_deg=wind_direction,
            facility_layout=facility_layout,
        )
        st.plotly_chart(fig_pp, use_container_width=True)

    # Convergence tab (only when multiple measurements)
    entropy_history = st.session_state.get("entropy_history", [])
    if len(entropy_history) > 1:
        with _tabs[6]:
            fig_conv = create_convergence_figure(entropy_history)
            st.plotly_chart(fig_conv, use_container_width=True)

# Score bar chart
fig_scores = create_score_profile(recommendations)
st.plotly_chart(fig_scores, use_container_width=True)

# ── Recommendations Table ────────────────────────────────────────────────────

st.subheader("Recommended Waypoints")

if recommendations:
    # Build worker assignment lookup if multi-worker
    wp_worker_map = {}
    if worker_routes and len(worker_routes) > 1:
        for route in worker_routes:
            for wp in route.assigned_waypoints:
                key = (round(wp["x"], 1), round(wp["y"], 1))
                wp_worker_map[key] = route.worker_id + 1

    rec_rows = []
    for i, rec in enumerate(recommendations):
        nearest = find_nearest_source(rec, sources)
        row = {
            "Rank": i + 1,
            "Location": f"({rec['x']:.0f}, {rec['y']:.0f})",
            "Nearest Equipment": nearest or "—",
            "P(detect)": rec["detection_prob"],
            "Concentration (ppm)": rec["concentration_ppm"],
            "Score": rec["score"],
        }
        if wp_worker_map:
            key = (round(rec["x"], 1), round(rec["y"], 1))
            row["Worker"] = wp_worker_map.get(key, 1)
        rec_rows.append(row)
    rec_df = pd.DataFrame(rec_rows)
    st.dataframe(
        rec_df,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "P(detect)": st.column_config.ProgressColumn(
                "P(detect)", min_value=0, max_value=1, format="%.2f",
            ),
            "Concentration (ppm)": st.column_config.NumberColumn(
                "Concentration (ppm)", format="%.1f",
            ),
            "Score": st.column_config.NumberColumn("Score", format="%.4f"),
        },
        use_container_width=True,
        hide_index=True,
    )
else:
    rec_df = None
    st.info(
        "No high-value waypoints found near the baseline path under current conditions. "
        "Try adjusting wind direction or increasing max deviation."
    )

# ── Export Results ───────────────────────────────────────────────────────────

with st.expander("Export Results"):
    import json
    import io

    ex_c1, ex_c2, ex_c3 = st.columns(3)

    # CSV export
    with ex_c1:
        if rec_df is not None:
            csv_data = rec_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name="recommendations.csv",
                mime="text/csv",
            )
        else:
            st.caption("No recommendations to export.")

    # JSON export
    with ex_c2:
        report = {
            "scoring_mode": "EER" if use_eer else "Heuristic",
            "wind_speed": wind_speed,
            "wind_direction_deg": wind_direction,
            "stability_class": stability_class,
            "plume_mode": plume_mode_key,
            "ensemble": use_ensemble and wind_scenarios is not None,
            "metrics": metrics,
            "recommendations": recommendations,
        }
        json_data = json.dumps(report, indent=2, default=str)
        st.download_button(
            "Download JSON",
            data=json_data,
            file_name="report.json",
            mime="application/json",
        )

    # NPZ export (belief state when Bayesian active)
    with ex_c3:
        if use_bayesian and belief_map is not None:
            buf = io.BytesIO()
            save_dict = {
                "belief": belief_map,
                "grid_x": X,
                "grid_y": Y,
            }
            ent_hist = st.session_state.get("entropy_history", [])
            if ent_hist:
                save_dict["entropy_history"] = np.array(ent_hist)
            np.savez_compressed(buf, **save_dict)
            buf.seek(0)
            st.download_button(
                "Download NPZ (Belief)",
                data=buf.getvalue(),
                file_name="belief_state.npz",
                mime="application/octet-stream",
            )
        else:
            st.caption("Enable Bayesian mode for belief state export.")

# ── Campaign Mode Panel ─────────────────────────────────────────────────────

if use_campaign:
    from optimization.campaign import plan_next_day, close_day, campaign_summary, CampaignState, DayPlan
    import plotly.graph_objects as go

    with st.expander("Campaign Planning (Multi-Day)", expanded=True):
        camp = st.session_state.campaign_state

        # Day selector
        completed_days = len(camp.days)
        if completed_days > 0:
            day_options = [f"Day {d.day_index + 1}" for d in camp.days]
            selected_day_label = st.selectbox("Review Past Day", day_options)
            selected_day_idx = int(selected_day_label.split()[-1]) - 1
            selected_day = camp.days[selected_day_idx]
            st.caption(
                f"Entropy: {selected_day.entropy_start:.0f} -> {selected_day.entropy_end:.0f} bits  |  "
                f"Measurements: {len(selected_day.measurements)}"
            )

        # Action buttons
        camp_c1, camp_c2 = st.columns(2)
        with camp_c1:
            if st.button("Plan Next Day", key="camp_plan"):
                wind_p = {
                    "wind_speed": wind_speed,
                    "wind_direction_deg": float(wind_direction),
                    "stability_class": stability_class,
                }
                day_plan = plan_next_day(
                    camp, sources, wind_p, baseline_path,
                    max_deviation=float(max_deviation),
                    resolution=float(grid_resolution),
                )
                st.session_state["_active_day_plan"] = day_plan
                st.success(f"Day {day_plan.day_index + 1} planned: {len(day_plan.recommendations)} recommendations")

        with camp_c2:
            if st.button("Close Day", key="camp_close"):
                active_plan = st.session_state.get("_active_day_plan")
                if active_plan is not None:
                    day_measurements = st.session_state.get("bayesian_measurements", [])
                    close_day(camp, active_plan, day_measurements, sources)
                    st.session_state.bayesian_measurements = []
                    st.session_state.entropy_history = []
                    del st.session_state["_active_day_plan"]
                    st.success(f"Day {active_plan.day_index + 1} closed with {len(day_measurements)} measurements")
                else:
                    st.warning("No active day plan. Click 'Plan Next Day' first.")

        # Mini entropy-per-day chart
        summary = campaign_summary(camp)
        if summary["total_days"] > 0:
            epd = summary["entropy_per_day"]
            fig_camp = go.Figure()
            fig_camp.add_trace(go.Bar(
                x=[f"Day {e['day'] + 1}" for e in epd],
                y=[e["start"] for e in epd],
                name="Start Entropy",
                marker_color="rgba(0,180,216,0.5)",
            ))
            fig_camp.add_trace(go.Bar(
                x=[f"Day {e['day'] + 1}" for e in epd],
                y=[e["end"] for e in epd],
                name="End Entropy",
                marker_color="#00b4d8",
            ))
            fig_camp.update_layout(
                barmode="group",
                title="Entropy per Day",
                yaxis_title="Entropy (bits)",
                template="plotly_dark",
                height=250,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_camp, use_container_width=True)

            st.caption(
                f"Total days: {summary['total_days']}  |  "
                f"Total measurements: {summary['total_measurements']}  |  "
                f"Total entropy reduction: {summary['total_entropy_reduction']:.0f} bits"
            )

# ── Prior Risk Summary ──────────────────────────────────────────────────────

with st.expander("Equipment Risk Assessment (Prior Model)"):
    st.caption(
        "Prior leak probabilities from equipment attributes: "
        "type, age, production rate, and inspection recency."
    )

    # Sort sources by prior probability (highest risk first)
    ranked = sorted(
        zip(sources, prior_probs), key=lambda x: x[1], reverse=True
    )

    risk_rows = []
    for src, p in ranked:
        risk_level = "HIGH" if p > 0.3 else "MEDIUM" if p > 0.15 else "LOW"
        risk_rows.append({
            "Equipment": src["name"],
            "Prior P(leak)": p,
            "Risk Level": risk_level,
            "Type": src.get("equipment_type", "unknown"),
            "Age (yr)": src.get("age_years", "?"),
            "Production (mcf/d)": src.get("production_rate_mcfd", 0),
            "Last Inspected (days)": src.get("last_inspection_days", "?"),
        })
    risk_df = pd.DataFrame(risk_rows)
    st.dataframe(
        risk_df,
        column_config={
            "Prior P(leak)": st.column_config.ProgressColumn(
                "Prior P(leak)", min_value=0, max_value=1, format="%.1%%",
            ),
            "Production (mcf/d)": st.column_config.NumberColumn(format="%.0f"),
        },
        use_container_width=True,
        hide_index=True,
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

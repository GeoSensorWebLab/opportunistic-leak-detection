"""
Worker Guidance Page — simplified field-worker interface.

Reads data from st.session_state (set by main.py) and presents:
  - Priority zone map (HIGH / MEDIUM / LOW)
  - Turn-by-turn walking directions
  - Field measurement input form
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Priority zone classification
# ---------------------------------------------------------------------------

def compute_priority_zones(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Classify grid cells into HIGH / MEDIUM / LOW priority zones.

    - HIGH: top 10% of nonzero scores
    - MEDIUM: top 25% (above HIGH)
    - LOW: everything else with nonzero score
    - NONE: zero score

    Returns:
        Integer array: 0=NONE, 1=LOW, 2=MEDIUM, 3=HIGH.
    """
    zones = np.zeros(scores.shape, dtype=int)
    nonzero = scores > 0

    if not np.any(nonzero):
        return zones

    vals = scores[nonzero]
    p90 = np.percentile(vals, 90)
    p75 = np.percentile(vals, 75)

    zones[nonzero] = 1  # LOW
    zones[scores >= p75] = 2  # MEDIUM
    zones[scores >= p90] = 3  # HIGH

    return zones


# ---------------------------------------------------------------------------
# Turn-by-turn directions
# ---------------------------------------------------------------------------

def _bearing_deg(dx: float, dy: float) -> float:
    """Compute bearing in degrees (0=North, 90=East) from dx, dy."""
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360


def _bearing_to_cardinal(bearing: float) -> str:
    """Convert bearing to 8-point cardinal direction."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(bearing / 45) % 8
    return dirs[idx]


def generate_turn_by_turn(
    optimized_path: np.ndarray,
    recommendations: List[dict],
    baseline_path: np.ndarray,
) -> List[dict]:
    """Generate step-by-step walking directions.

    Returns:
        List of dicts with 'step', 'instruction', 'distance_m', 'bearing',
        'cardinal', 'is_waypoint', 'cumulative_m'.
    """
    directions = []
    cum_dist = 0.0

    # Create a set of recommendation coords for matching
    rec_coords = set()
    for r in recommendations:
        rec_coords.add((round(r["x"], 1), round(r["y"], 1)))

    for i in range(len(optimized_path) - 1):
        start = optimized_path[i]
        end = optimized_path[i + 1]

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = float(np.hypot(dx, dy))
        bearing = _bearing_deg(dx, dy)
        cardinal = _bearing_to_cardinal(bearing)

        is_wp = (round(end[0], 1), round(end[1], 1)) in rec_coords

        instruction = f"Walk {cardinal} for {dist:.0f}m"
        if is_wp:
            instruction += " — MEASUREMENT POINT"

        cum_dist += dist
        directions.append({
            "step": i + 1,
            "instruction": instruction,
            "distance_m": dist,
            "bearing": bearing,
            "cardinal": cardinal,
            "is_waypoint": is_wp,
            "cumulative_m": cum_dist,
            "x": float(end[0]),
            "y": float(end[1]),
        })

    return directions


# ---------------------------------------------------------------------------
# Simplified map
# ---------------------------------------------------------------------------

def render_simplified_map(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    zones: np.ndarray,
    optimized_path: np.ndarray,
    recommendations: List[dict],
) -> go.Figure:
    """Render a large, clean priority-zone map for field use.

    Uses discrete colors: green=LOW, yellow=MEDIUM, red=HIGH.
    Large numbered markers for waypoints, no facility clutter.
    """
    # Map zones to colors: 0=transparent, 1=green, 2=yellow, 3=red
    zone_colors = np.where(zones == 3, 0.9, np.where(zones == 2, 0.5, np.where(zones == 1, 0.2, 0.0)))

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=zone_colors,
            colorscale=[
                [0.0, "rgba(0,0,0,0)"],
                [0.2, "rgba(0,180,0,0.3)"],
                [0.5, "rgba(255,200,0,0.4)"],
                [0.9, "rgba(255,50,0,0.5)"],
                [1.0, "rgba(255,50,0,0.5)"],
            ],
            zmin=0, zmax=1.0,
            showscale=False,
            hoverinfo="skip",
        )
    )

    # Path
    fig.add_trace(
        go.Scatter(
            x=optimized_path[:, 0],
            y=optimized_path[:, 1],
            mode="lines",
            line=dict(color="white", width=4),
            name="Walking Route",
        )
    )

    # Waypoint markers (large, numbered)
    if recommendations:
        fig.add_trace(
            go.Scatter(
                x=[r["x"] for r in recommendations],
                y=[r["y"] for r in recommendations],
                mode="markers+text",
                marker=dict(size=24, color="yellow", symbol="circle",
                            line=dict(width=3, color="black")),
                text=[f"{i+1}" for i in range(len(recommendations))],
                textfont=dict(size=14, color="black"),
                textposition="middle center",
                name="Measurement Points",
            )
        )

    # Start / End markers
    fig.add_trace(
        go.Scatter(
            x=[optimized_path[0, 0]], y=[optimized_path[0, 1]],
            mode="markers+text",
            marker=dict(size=18, color="limegreen", symbol="circle",
                        line=dict(width=2, color="white")),
            text=["START"], textposition="top center",
            textfont=dict(size=12, color="limegreen"),
            name="Start", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[optimized_path[-1, 0]], y=[optimized_path[-1, 1]],
            mode="markers+text",
            marker=dict(size=18, color="red", symbol="square",
                        line=dict(width=2, color="white")),
            text=["END"], textposition="top center",
            textfont=dict(size=12, color="red"),
            name="End", showlegend=False,
        )
    )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        font=dict(family="sans-serif", size=14),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=30, b=60),
    )
    fig.update_xaxes(title_text="East (m)", scaleanchor="y")
    fig.update_yaxes(title_text="North (m)")

    return fig


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Worker Guidance", page_icon="🧭", layout="wide")
st.title("Field Worker Guidance")

# Check for required session state data
has_data = all(
    k in st.session_state
    for k in ["_wg_grid_x", "_wg_grid_y", "_wg_scores",
              "_wg_optimized_path", "_wg_recommendations", "_wg_baseline_path"]
)

if not has_data:
    st.info(
        "No guidance data available. Run the main Tasking System first, "
        "then enable **Share to Worker Guidance** in the sidebar."
    )
    st.stop()

grid_x = st.session_state["_wg_grid_x"]
grid_y = st.session_state["_wg_grid_y"]
scores = st.session_state["_wg_scores"]
optimized_path = st.session_state["_wg_optimized_path"]
recommendations = st.session_state["_wg_recommendations"]
baseline_path = st.session_state["_wg_baseline_path"]

# Priority zones
zones = compute_priority_zones(grid_x, grid_y, scores)

tab_map, tab_directions, tab_record = st.tabs(
    ["Priority Map", "Walking Directions", "Record Measurement"]
)

with tab_map:
    fig = render_simplified_map(grid_x, grid_y, zones, optimized_path, recommendations)
    st.plotly_chart(fig, use_container_width=True)

    # Zone legend
    c1, c2, c3 = st.columns(3)
    high_cells = int(np.sum(zones == 3))
    med_cells = int(np.sum(zones == 2))
    low_cells = int(np.sum(zones == 1))
    c1.metric("HIGH Priority Cells", high_cells)
    c2.metric("MEDIUM Priority Cells", med_cells)
    c3.metric("LOW Priority Cells", low_cells)

with tab_directions:
    directions = generate_turn_by_turn(optimized_path, recommendations, baseline_path)
    if directions:
        for d in directions:
            icon = "📍" if d["is_waypoint"] else "🚶"
            st.markdown(
                f"**Step {d['step']}** {icon} {d['instruction']}  \n"
                f"*({d['x']:.0f}, {d['y']:.0f}) — cumulative: {d['cumulative_m']:.0f}m*"
            )
    else:
        st.info("No walking directions available.")

with tab_record:
    st.subheader("Record Field Measurement")
    with st.form("measurement_form"):
        col1, col2 = st.columns(2)
        with col1:
            rec_x = st.number_input("X position (m)", value=0.0, step=10.0)
            rec_y = st.number_input("Y position (m)", value=0.0, step=10.0)
        with col2:
            rec_conc = st.number_input("Concentration (ppm)", value=0.0, min_value=0.0, step=1.0)
            rec_detected = st.checkbox("Detection triggered")

        rec_notes = st.text_area("Notes", placeholder="Equipment condition, weather, etc.")
        submitted = st.form_submit_button("Save Measurement")

        if submitted:
            if "field_measurements" not in st.session_state:
                st.session_state.field_measurements = []
            st.session_state.field_measurements.append({
                "x": rec_x, "y": rec_y,
                "concentration_ppm": rec_conc,
                "detected": rec_detected,
                "notes": rec_notes,
            })
            st.success(f"Measurement saved at ({rec_x:.0f}, {rec_y:.0f})")

    # Show recorded measurements
    if st.session_state.get("field_measurements"):
        st.subheader("Recorded Measurements")
        import pandas as pd
        df = pd.DataFrame(st.session_state.field_measurements)
        st.dataframe(df, use_container_width=True)

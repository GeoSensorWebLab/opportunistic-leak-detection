"""
Visualization module for the Methane Leak Opportunistic Tasking System.

Provides Plotly-based interactive plots for the Streamlit interface.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple

from config import COMPASS_POSITION, PATH_ARROW_INTERVAL_M, DETOUR_TOLERANCE_M


def _rect_coords(element: dict) -> Tuple[list, list]:
    """Return (xs, ys) for a closed rectangle given cx, cy, width, height."""
    cx, cy = element["cx"], element["cy"]
    hw, hh = element["width"] / 2, element["height"] / 2
    xs = [cx - hw, cx + hw, cx + hw, cx - hw, cx - hw]
    ys = [cy - hh, cy - hh, cy + hh, cy + hh, cy - hh]
    return xs, ys


def _add_facility_layout(
    fig: go.Figure,
    layout: Dict,
    row: int,
    col: int,
    show_legend: bool = False,
) -> None:
    """Render facility elements (pads, roads, pipe racks, fence, buildings) onto a subplot."""

    # 1. Equipment pads (lowest layer — drawn first)
    for pad in layout["equipment_pads"]:
        xs, ys = _rect_coords(pad)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="lines",
                fill="toself",
                fillcolor=pad["color"],
                line=dict(color=pad["color"].replace("0.15", "0.5"), width=1),
                name=pad["name"],
                showlegend=False,
                hoverinfo="name",
            ),
            row=row, col=col,
        )

    # 2. Roads
    for road in layout["roads"]:
        pts = road["points"]
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="lines",
                line=dict(color=road["color"], width=road["width"]),
                name=road["name"],
                showlegend=False,
                hoverinfo="name",
            ),
            row=row, col=col,
        )

    # 3. Pipe racks
    for rack in layout["pipe_racks"]:
        pts = rack["points"]
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="lines",
                line=dict(color=rack["color"], width=rack["width"], dash="dot"),
                name=rack["name"],
                showlegend=False,
                hoverinfo="name",
            ),
            row=row, col=col,
        )

    # 4. Fence (site boundary)
    fence = layout["fence"]
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in fence],
            y=[p[1] for p in fence],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.45)", width=2, dash="dashdot"),
            name="Site Boundary",
            showlegend=show_legend,
            hoverinfo="name",
        ),
        row=row, col=col,
    )

    # 5. Buildings (topmost facility layer)
    for bldg in layout["buildings"]:
        xs, ys = _rect_coords(bldg)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="lines",
                fill="toself",
                fillcolor=bldg["color"],
                line=dict(color="white", width=1),
                name=bldg["name"],
                showlegend=False,
                hoverinfo="name",
            ),
            row=row, col=col,
        )


# ── Path enhancement helpers ────────────────────────────────────────────────

def _add_path_arrows(
    fig: go.Figure,
    path: np.ndarray,
    color: str,
    row: int,
    col: int,
    interval_m: float = PATH_ARROW_INTERVAL_M,
) -> None:
    """Add direction arrows along a path at regular distance intervals."""
    xref = f"x{col if col > 1 else ''}"
    yref = f"y{col if col > 1 else ''}"

    cum_dist = 0.0
    next_arrow = interval_m

    for i in range(len(path) - 1):
        seg = path[i + 1] - path[i]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            continue

        while cum_dist + seg_len >= next_arrow:
            frac = (next_arrow - cum_dist) / seg_len
            pt = path[i] + frac * seg
            direction = seg / seg_len
            tip = pt + direction * 12  # small arrow tip offset

            fig.add_annotation(
                x=tip[0], y=tip[1],
                ax=pt[0], ay=pt[1],
                xref=xref, yref=yref,
                axref=xref, ayref=yref,
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=color,
            )
            next_arrow += interval_m

        cum_dist += seg_len


def _add_start_end_markers(
    fig: go.Figure,
    path: np.ndarray,
    row: int,
    col: int,
) -> None:
    """Add green START circle and red END square markers at path endpoints."""
    # START marker
    fig.add_trace(
        go.Scatter(
            x=[path[0, 0]],
            y=[path[0, 1]],
            mode="markers+text",
            marker=dict(size=14, color="limegreen", symbol="circle",
                        line=dict(width=2, color="white")),
            text=["START"],
            textposition="middle right",
            textfont=dict(size=10, color="limegreen"),
            name="Start",
            showlegend=False,
            hovertemplate="START<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ),
        row=row, col=col,
    )
    # END marker
    fig.add_trace(
        go.Scatter(
            x=[path[-1, 0]],
            y=[path[-1, 1]],
            mode="markers+text",
            marker=dict(size=14, color="red", symbol="square",
                        line=dict(width=2, color="white")),
            text=["END"],
            textposition="middle right",
            textfont=dict(size=10, color="red"),
            name="End",
            showlegend=False,
            hovertemplate="END<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ),
        row=row, col=col,
    )


def _identify_detour_segments(
    baseline: np.ndarray,
    optimized: np.ndarray,
    tolerance: float = DETOUR_TOLERANCE_M,
) -> List[Tuple[int, int]]:
    """
    Find contiguous runs of optimized-path points that diverge from baseline.

    Returns list of (start_idx, end_idx) into the optimized path where
    the worker deviates from the baseline by more than tolerance.
    """
    from scipy.spatial.distance import cdist

    dists = cdist(optimized, baseline).min(axis=1)
    is_detour = dists > tolerance

    segments = []
    in_seg = False
    start = 0

    for i, det in enumerate(is_detour):
        if det and not in_seg:
            start = max(0, i - 1)  # include lead-in point
            in_seg = True
        elif not det and in_seg:
            segments.append((start, min(i, len(optimized) - 1)))
            in_seg = False

    if in_seg:
        segments.append((start, len(optimized) - 1))

    return segments


def _add_detour_highlights(
    fig: go.Figure,
    optimized: np.ndarray,
    segments: List[Tuple[int, int]],
    row: int,
    col: int,
) -> None:
    """Render detour segments in gold on the map."""
    for idx, (s, e) in enumerate(segments):
        sub = optimized[s:e + 1]
        fig.add_trace(
            go.Scatter(
                x=sub[:, 0],
                y=sub[:, 1],
                mode="lines",
                line=dict(color="gold", width=5),
                name="Detour" if idx == 0 else None,
                showlegend=(idx == 0),
                hovertemplate="Detour<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
            ),
            row=row, col=col,
        )


# ── Single-panel map (primary view) ────────────────────────────────────────

def create_single_map_figure(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    detection_prob: np.ndarray,
    sources: List[dict],
    baseline_path: np.ndarray,
    optimized_path: np.ndarray,
    recommendations: List[dict],
    wind_speed: float,
    wind_direction_deg: float,
    facility_layout: Optional[Dict] = None,
    colorbar_title: str = "P(detect)",
) -> go.Figure:
    """
    Create a full-width single-panel detection map with both paths,
    direction arrows, start/end markers, and detour highlights.
    """
    fig = make_subplots(rows=1, cols=1)

    # Detection probability heatmap
    fig.add_trace(
        go.Heatmap(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=detection_prob,
            colorscale="YlOrRd",
            zmin=0,
            zmax=1,
            colorbar=dict(title=colorbar_title),
            name="Detection Prob",
            hovertemplate="x: %{x:.0f}m<br>y: %{y:.0f}m<br>P(detect): %{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Facility layout
    if facility_layout:
        _add_facility_layout(fig, facility_layout, row=1, col=1, show_legend=True)

    # Sources
    src_x = [s["x"] for s in sources]
    src_y = [s["y"] for s in sources]
    src_names = [s["name"] for s in sources]
    fig.add_trace(
        go.Scatter(
            x=src_x,
            y=src_y,
            mode="markers+text",
            marker=dict(size=12, color="lime", symbol="diamond",
                        line=dict(width=1, color="black")),
            text=src_names,
            textposition="top center",
            textfont=dict(size=9, color="white"),
            name="Leak Sources",
            hovertemplate="%{text}<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ),
        row=1, col=1,
    )

    # Baseline path
    fig.add_trace(
        go.Scatter(
            x=baseline_path[:, 0],
            y=baseline_path[:, 1],
            mode="lines+markers",
            line=dict(color="cyan", width=3, dash="dash"),
            marker=dict(size=4, color="cyan"),
            name="Baseline Path",
            hovertemplate="Baseline<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ),
        row=1, col=1,
    )

    # Optimized path
    fig.add_trace(
        go.Scatter(
            x=optimized_path[:, 0],
            y=optimized_path[:, 1],
            mode="lines+markers",
            line=dict(color="white", width=3),
            marker=dict(size=4, color="white"),
            name="Optimized Path",
            hovertemplate="Optimized<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ),
        row=1, col=1,
    )

    # Detour highlights
    detour_segs = _identify_detour_segments(baseline_path, optimized_path)
    _add_detour_highlights(fig, optimized_path, detour_segs, row=1, col=1)

    # Direction arrows on both paths
    _add_path_arrows(fig, baseline_path, "cyan", row=1, col=1)
    _add_path_arrows(fig, optimized_path, "white", row=1, col=1)

    # Start / End markers
    _add_start_end_markers(fig, optimized_path, row=1, col=1)

    # Recommended waypoints
    if recommendations:
        rec_x = [r["x"] for r in recommendations]
        rec_y = [r["y"] for r in recommendations]
        rec_text = [
            f"#{i+1}: {r['concentration_ppm']:.1f}ppm<br>P={r['detection_prob']:.2f}"
            for i, r in enumerate(recommendations)
        ]
        fig.add_trace(
            go.Scatter(
                x=rec_x,
                y=rec_y,
                mode="markers+text",
                marker=dict(
                    size=16,
                    color="yellow",
                    symbol="star",
                    line=dict(width=2, color="black"),
                ),
                text=[f"#{i+1}" for i in range(len(recommendations))],
                textposition="bottom center",
                textfont=dict(size=11, color="yellow"),
                name="Recommendations",
                hovertext=rec_text,
                hoverinfo="text",
            ),
            row=1, col=1,
        )

    # Wind compass
    _add_compass_rose(fig, wind_speed, wind_direction_deg, row=1, col=1)

    fig.update_layout(
        height=700,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=60, t=40, b=80),
    )

    fig.update_xaxes(title_text="East (m)")
    fig.update_yaxes(title_text="North (m)")

    return fig


# ── Concentration-only map ──────────────────────────────────────────────────

def create_concentration_figure(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    concentration_ppm: np.ndarray,
    sources: List[dict],
    wind_speed: float,
    wind_direction_deg: float,
    facility_layout: Optional[Dict] = None,
) -> go.Figure:
    """Create a full-width concentration heatmap."""
    fig = make_subplots(rows=1, cols=1)

    conc_display = np.log10(np.maximum(concentration_ppm, 1e-3))

    fig.add_trace(
        go.Heatmap(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=conc_display,
            colorscale="Hot",
            reversescale=True,
            colorbar=dict(title="log10(ppm)"),
            name="Concentration",
            hovertemplate="x: %{x:.0f}m<br>y: %{y:.0f}m<br>log10(ppm): %{z:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    if facility_layout:
        _add_facility_layout(fig, facility_layout, row=1, col=1, show_legend=True)

    src_x = [s["x"] for s in sources]
    src_y = [s["y"] for s in sources]
    src_names = [s["name"] for s in sources]
    fig.add_trace(
        go.Scatter(
            x=src_x,
            y=src_y,
            mode="markers+text",
            marker=dict(size=12, color="lime", symbol="diamond",
                        line=dict(width=1, color="black")),
            text=src_names,
            textposition="top center",
            textfont=dict(size=9, color="white"),
            name="Leak Sources",
            hovertemplate="%{text}<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ),
        row=1, col=1,
    )

    _add_compass_rose(fig, wind_speed, wind_direction_deg, row=1, col=1)

    fig.update_layout(
        height=700,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=60, t=40, b=80),
    )

    fig.update_xaxes(title_text="East (m)")
    fig.update_yaxes(title_text="North (m)")

    return fig


# ── Side-by-side dual panel (preserved) ─────────────────────────────────────

def create_site_figure(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    concentration_ppm: np.ndarray,
    detection_prob: np.ndarray,
    sources: List[dict],
    baseline_path: np.ndarray,
    optimized_path: np.ndarray,
    recommendations: List[dict],
    wind_speed: float,
    wind_direction_deg: float,
    facility_layout: Optional[Dict] = None,
) -> go.Figure:
    """
    Create the dual-panel site visualization with plume heatmap, paths, and assets.

    Returns a Plotly figure with two subplots:
      Left: Concentration heatmap with plumes
      Right: Detection probability heatmap with paths and recommendations
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Methane Concentration (ppm)",
            "Detection Probability & Tasking",
        ),
        horizontal_spacing=0.12,
    )

    # --- Left panel: Concentration heatmap ---
    conc_display = np.log10(np.maximum(concentration_ppm, 1e-3))

    fig.add_trace(
        go.Heatmap(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=conc_display,
            colorscale="Hot",
            reversescale=True,
            colorbar=dict(title="log10(ppm)", x=0.42),
            name="Concentration",
            hovertemplate="x: %{x:.0f}m<br>y: %{y:.0f}m<br>log10(ppm): %{z:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Facility layout on left panel
    if facility_layout:
        _add_facility_layout(fig, facility_layout, row=1, col=1, show_legend=True)

    # Sources on left panel
    src_x = [s["x"] for s in sources]
    src_y = [s["y"] for s in sources]
    src_names = [s["name"] for s in sources]
    fig.add_trace(
        go.Scatter(
            x=src_x,
            y=src_y,
            mode="markers+text",
            marker=dict(size=12, color="lime", symbol="diamond", line=dict(width=1, color="black")),
            text=src_names,
            textposition="top center",
            textfont=dict(size=9, color="white"),
            name="Leak Sources",
            hovertemplate="%{text}<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Wind arrow on left panel
    _add_compass_rose(fig, wind_speed, wind_direction_deg, row=1, col=1)

    # --- Right panel: Detection probability + paths ---
    fig.add_trace(
        go.Heatmap(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=detection_prob,
            colorscale="YlOrRd",
            zmin=0,
            zmax=1,
            colorbar=dict(title="P(detect)", x=1.0),
            name="Detection Prob",
            hovertemplate="x: %{x:.0f}m<br>y: %{y:.0f}m<br>P(detect): %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Facility layout on right panel
    if facility_layout:
        _add_facility_layout(fig, facility_layout, row=1, col=2, show_legend=False)

    # Sources on right panel
    fig.add_trace(
        go.Scatter(
            x=src_x,
            y=src_y,
            mode="markers",
            marker=dict(size=10, color="lime", symbol="diamond", line=dict(width=1, color="black")),
            name="Leak Sources",
            showlegend=False,
            hovertemplate="%{text}<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
            text=src_names,
        ),
        row=1,
        col=2,
    )

    # Baseline path
    fig.add_trace(
        go.Scatter(
            x=baseline_path[:, 0],
            y=baseline_path[:, 1],
            mode="lines+markers",
            line=dict(color="cyan", width=3, dash="dash"),
            marker=dict(size=4, color="cyan"),
            name="Baseline Path",
            hovertemplate="Baseline<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Optimized path
    fig.add_trace(
        go.Scatter(
            x=optimized_path[:, 0],
            y=optimized_path[:, 1],
            mode="lines+markers",
            line=dict(color="white", width=3),
            marker=dict(size=4, color="white"),
            name="Optimized Path",
            hovertemplate="Optimized<br>(%{x:.0f}, %{y:.0f})<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Detour highlights on right panel
    detour_segs = _identify_detour_segments(baseline_path, optimized_path)
    _add_detour_highlights(fig, optimized_path, detour_segs, row=1, col=2)

    # Direction arrows
    _add_path_arrows(fig, baseline_path, "cyan", row=1, col=2)
    _add_path_arrows(fig, optimized_path, "white", row=1, col=2)

    # Start / End markers
    _add_start_end_markers(fig, optimized_path, row=1, col=2)

    # Recommended waypoints
    if recommendations:
        rec_x = [r["x"] for r in recommendations]
        rec_y = [r["y"] for r in recommendations]
        rec_text = [
            f"#{i+1}: {r['concentration_ppm']:.1f}ppm<br>P={r['detection_prob']:.2f}"
            for i, r in enumerate(recommendations)
        ]
        fig.add_trace(
            go.Scatter(
                x=rec_x,
                y=rec_y,
                mode="markers+text",
                marker=dict(
                    size=16,
                    color="yellow",
                    symbol="star",
                    line=dict(width=2, color="black"),
                ),
                text=[f"#{i+1}" for i in range(len(recommendations))],
                textposition="bottom center",
                textfont=dict(size=11, color="yellow"),
                name="Recommendations",
                hovertext=rec_text,
                hoverinfo="text",
            ),
            row=1,
            col=2,
        )

    # Wind arrow on right panel
    _add_compass_rose(fig, wind_speed, wind_direction_deg, row=1, col=2)

    # Layout
    fig.update_layout(
        height=700,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=60, t=60, b=80),
    )

    fig.update_xaxes(title_text="East (m)", row=1, col=1)
    fig.update_yaxes(title_text="North (m)", row=1, col=1)
    fig.update_xaxes(title_text="East (m)", row=1, col=2)
    fig.update_yaxes(title_text="North (m)", row=1, col=2)

    return fig


def _add_compass_rose(
    fig: go.Figure,
    wind_speed: float,
    wind_direction_deg: float,
    row: int,
    col: int,
):
    """Add a compass rose with cardinal labels and a wind-direction needle to a subplot."""
    xref = f"x{col if col > 1 else ''}"
    yref = f"y{col if col > 1 else ''}"

    # Centre of compass rose in data coords (upper-left corner of plot)
    cx, cy = COMPASS_POSITION
    outer_r = 55

    # Outer ring
    fig.add_shape(
        type="circle",
        x0=cx - outer_r, y0=cy - outer_r,
        x1=cx + outer_r, y1=cy + outer_r,
        xref=xref, yref=yref,
        line=dict(color="rgba(255,255,255,0.35)", width=1.5),
        fillcolor="rgba(14,17,23,0.6)",
    )

    # Cardinal tick marks (short lines at N/E/S/W) and labels
    cardinals = {"N": 0, "E": 90, "S": 180, "W": 270}
    for label, deg in cardinals.items():
        rad = np.radians(deg)
        # Tick from outer_r-8 to outer_r
        ix = cx + (outer_r - 8) * np.sin(rad)
        iy = cy + (outer_r - 8) * np.cos(rad)
        ox = cx + outer_r * np.sin(rad)
        oy = cy + outer_r * np.cos(rad)
        fig.add_shape(
            type="line",
            x0=ix, y0=iy, x1=ox, y1=oy,
            xref=xref, yref=yref,
            line=dict(color="white", width=2),
        )
        # Label just inside the tick
        lx = cx + (outer_r - 18) * np.sin(rad)
        ly = cy + (outer_r - 18) * np.cos(rad)
        fig.add_annotation(
            x=lx, y=ly,
            xref=xref, yref=yref,
            text=label,
            showarrow=False,
            font=dict(size=9, color="white"),
        )

    # Wind needle — points toward (blowing direction)
    toward_deg = (wind_direction_deg + 180.0) % 360.0
    toward_rad = np.radians(toward_deg)
    arrow_len = outer_r - 22

    tip_x = cx + arrow_len * np.sin(toward_rad)
    tip_y = cy + arrow_len * np.cos(toward_rad)

    fig.add_annotation(
        x=tip_x, y=tip_y,
        ax=cx, ay=cy,
        xref=xref, yref=yref,
        axref=xref, ayref=yref,
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor="deepskyblue",
    )

    # Speed + direction label below the rose
    fig.add_annotation(
        x=cx, y=cy - outer_r - 12,
        xref=xref, yref=yref,
        text=f"{wind_speed:.1f} m/s | {wind_direction_deg:.0f}\u00b0",
        showarrow=False,
        font=dict(size=9, color="deepskyblue"),
    )


def create_score_profile(
    recommendations: List[dict],
) -> go.Figure:
    """Create a bar chart of the top recommended waypoints and their scores."""
    if not recommendations:
        fig = go.Figure()
        fig.add_annotation(text="No recommendations available", showarrow=False)
        return fig

    labels = [f"#{i+1} ({r['x']:.0f}, {r['y']:.0f})" for i, r in enumerate(recommendations)]
    scores = [r["score"] for r in recommendations]
    probs = [r["detection_prob"] for r in recommendations]
    concs = [r["concentration_ppm"] for r in recommendations]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=scores,
            name="Tasking Score",
            marker_color="gold",
            hovertemplate=(
                "Score: %{y:.4f}<br>"
                "P(detect): %{customdata[0]:.3f}<br>"
                "Conc: %{customdata[1]:.1f} ppm<extra></extra>"
            ),
            customdata=list(zip(probs, concs)),
        )
    )

    fig.update_layout(
        title="Top Recommended Waypoints",
        xaxis_title="Waypoint",
        yaxis_title="Tasking Score",
        template="plotly_dark",
        height=300,
    )

    return fig

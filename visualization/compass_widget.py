"""
Compass Widget — sidebar SVG compass showing wind direction and speed.

Rendered via st.components.v1.html(); display-only (sliders remain the input).
"""

import math


def compass_html(wind_direction_deg: float, wind_speed: float, size: int = 180) -> str:
    """
    Return an HTML string containing an SVG compass.

    The needle arrow points in the direction the wind is blowing TOWARD
    (i.e. meteorological direction + 180).

    Args:
        wind_direction_deg: Meteorological wind direction (degrees, where wind comes FROM).
        wind_speed: Wind speed in m/s.
        size: Pixel width/height of the compass.

    Returns:
        HTML string with embedded SVG.
    """
    cx = cy = size / 2
    r = size / 2 - 12  # outer circle radius
    toward_deg = (wind_direction_deg + 180.0) % 360.0

    # Build tick marks every 10 degrees
    ticks_svg = []
    for deg in range(0, 360, 10):
        rad = math.radians(deg)
        is_cardinal = deg % 90 == 0
        inner = r - (12 if is_cardinal else 6)
        outer = r
        x1 = cx + inner * math.sin(rad)
        y1 = cy - inner * math.cos(rad)
        x2 = cx + outer * math.sin(rad)
        y2 = cy - outer * math.cos(rad)
        width = 2 if is_cardinal else 1
        color = "#ffffff" if is_cardinal else "rgba(255,255,255,0.4)"
        ticks_svg.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"/>'
        )

    # Cardinal labels
    labels = {"N": 0, "E": 90, "S": 180, "W": 270}
    labels_svg = []
    for label, deg in labels.items():
        rad = math.radians(deg)
        lr = r - 22
        lx = cx + lr * math.sin(rad)
        ly = cy - lr * math.cos(rad)
        labels_svg.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" fill="#ffffff" font-size="13" '
            f'font-weight="bold" font-family="sans-serif">{label}</text>'
        )

    # Wind needle (toward direction)
    needle_rad = math.radians(toward_deg)
    tip_r = r - 30
    tail_r = 12
    tip_x = cx + tip_r * math.sin(needle_rad)
    tip_y = cy - tip_r * math.cos(needle_rad)
    tail_x = cx - tail_r * math.sin(needle_rad)
    tail_y = cy + tail_r * math.cos(needle_rad)

    # Arrowhead wings
    wing_offset = 12
    wing_back = 18
    perp_rad = needle_rad + math.pi / 2
    back_rad = needle_rad + math.pi
    base_x = tip_x + wing_back * math.sin(back_rad)
    base_y = tip_y - wing_back * math.cos(back_rad)
    w1x = base_x + wing_offset * math.sin(perp_rad)
    w1y = base_y - wing_offset * math.cos(perp_rad)
    w2x = base_x - wing_offset * math.sin(perp_rad)
    w2y = base_y + wing_offset * math.cos(perp_rad)

    needle_svg = (
        f'<line x1="{tail_x:.1f}" y1="{tail_y:.1f}" x2="{tip_x:.1f}" y2="{tip_y:.1f}" '
        f'stroke="deepskyblue" stroke-width="3" stroke-linecap="round"/>'
        f'<polygon points="{tip_x:.1f},{tip_y:.1f} {w1x:.1f},{w1y:.1f} {w2x:.1f},{w2y:.1f}" '
        f'fill="deepskyblue"/>'
    )

    # Center speed readout
    center_svg = (
        f'<circle cx="{cx}" cy="{cy}" r="18" fill="#1a1a2e" stroke="deepskyblue" stroke-width="1.5"/>'
        f'<text x="{cx}" y="{cy - 3}" text-anchor="middle" dominant-baseline="central" '
        f'fill="deepskyblue" font-size="11" font-weight="bold" font-family="sans-serif">'
        f'{wind_speed:.1f}</text>'
        f'<text x="{cx}" y="{cy + 10}" text-anchor="middle" dominant-baseline="central" '
        f'fill="rgba(255,255,255,0.6)" font-size="8" font-family="sans-serif">m/s</text>'
    )

    # "FROM" direction label below compass
    from_label = (
        f'<text x="{cx}" y="{size - 1}" text-anchor="middle" '
        f'fill="rgba(255,255,255,0.5)" font-size="10" font-family="sans-serif">'
        f'From {wind_direction_deg:.0f}\u00b0</text>'
    )

    svg = (
        f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{size}" height="{size}" rx="8" fill="#0e1117"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="1.5"/>'
        f'{"".join(ticks_svg)}'
        f'{"".join(labels_svg)}'
        f'{needle_svg}'
        f'{center_svg}'
        f'{from_label}'
        f'</svg>'
    )

    return f'<div style="display:flex;justify-content:center;">{svg}</div>'

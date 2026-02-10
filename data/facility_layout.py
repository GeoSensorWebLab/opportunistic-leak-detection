"""
Facility Layout for the Methane Leak Opportunistic Tasking System.

Defines a midstream gas processing / compressor station complex
that provides realistic context for the 5 leak source locations.
"""

from typing import List, Dict, Any


def get_facility_layout() -> Dict[str, Any]:
    """
    Return the full facility layout as a dict of element lists.

    The facility is a ~700m x 600m midstream gas processing complex
    centered roughly at the origin of the 1 km simulation grid.

    Returns:
        Dict with keys: 'buildings', 'fence', 'roads', 'pipe_racks', 'equipment_pads'.
    """
    return {
        "buildings": _buildings(),
        "fence": _fence(),
        "roads": _roads(),
        "pipe_racks": _pipe_racks(),
        "equipment_pads": _equipment_pads(),
    }


def _buildings() -> List[dict]:
    """Three main buildings in the facility."""
    return [
        {
            "name": "Compressor House",
            "cx": 80.0,
            "cy": -100.0,
            "width": 60.0,
            "height": 40.0,
            "color": "rgba(100,100,100,0.85)",
        },
        {
            "name": "Control Room",
            "cx": 0.0,
            "cy": 50.0,
            "width": 40.0,
            "height": 30.0,
            "color": "rgba(70,130,180,0.85)",
        },
        {
            "name": "MCC / Electrical",
            "cx": 60.0,
            "cy": 50.0,
            "width": 30.0,
            "height": 25.0,
            "color": "rgba(180,130,70,0.85)",
        },
    ]


def _fence() -> List[List[float]]:
    """Site boundary as a closed polyline (list of [x, y] vertices)."""
    return [
        [-350.0, 200.0],
        [-350.0, -380.0],
        [380.0, -380.0],
        [380.0, 200.0],
        [200.0, 200.0],
        [200.0, 250.0],
        [-200.0, 250.0],
        [-200.0, 200.0],
        [-350.0, 200.0],  # close polygon
    ]


def _roads() -> List[dict]:
    """Three internal roads."""
    return [
        {
            "name": "Main Access Road",
            "points": [
                [0.0, -380.0],   # south gate
                [0.0, -270.0],
                [0.0, -160.0],
                [0.0, 0.0],
                [0.0, 120.0],
                [0.0, 250.0],    # north gate
            ],
            "color": "rgba(160,160,160,0.5)",
            "width": 6,
        },
        {
            "name": "Tank Farm Road",
            "points": [
                [0.0, -270.0],
                [-60.0, -270.0],
                [-180.0, -270.0],
                [-300.0, -270.0],
            ],
            "color": "rgba(160,160,160,0.5)",
            "width": 5,
        },
        {
            "name": "Well Pad Access",
            "points": [
                [0.0, 120.0],
                [-100.0, 120.0],
                [-200.0, 130.0],
                [-300.0, 140.0],
            ],
            "color": "rgba(160,160,160,0.5)",
            "width": 5,
        },
    ]


def _pipe_racks() -> List[dict]:
    """Two main pipe racks connecting facility zones."""
    return [
        {
            "name": "Main Pipe Rack (E-W)",
            "points": [
                [-300.0, -30.0],
                [-150.0, -30.0],
                [0.0, -30.0],
                [150.0, -30.0],
                [300.0, -30.0],
            ],
            "color": "rgba(255,165,0,0.7)",
            "width": 3,
        },
        {
            "name": "Tank Farm Header",
            "points": [
                [-110.0, -200.0],
                [-110.0, -270.0],
                [-110.0, -340.0],
            ],
            "color": "rgba(255,165,0,0.7)",
            "width": 3,
        },
    ]


def _equipment_pads() -> List[dict]:
    """Four major equipment pads / zones."""
    return [
        {
            "name": "Well Pad",
            "cx": -260.0,
            "cy": 140.0,
            "width": 120.0,
            "height": 80.0,
            "color": "rgba(34,139,34,0.15)",
        },
        {
            "name": "Compressor Pad",
            "cx": 75.0,
            "cy": -140.0,
            "width": 140.0,
            "height": 100.0,
            "color": "rgba(70,130,180,0.15)",
        },
        {
            "name": "Tank Containment Berm",
            "cx": -110.0,
            "cy": -270.0,
            "width": 160.0,
            "height": 120.0,
            "color": "rgba(178,34,34,0.15)",
        },
        {
            "name": "Valve / Metering Station",
            "cx": 280.0,
            "cy": -45.0,
            "width": 100.0,
            "height": 80.0,
            "color": "rgba(218,165,32,0.15)",
        },
    ]

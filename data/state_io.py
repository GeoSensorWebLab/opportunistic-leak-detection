"""
State serialization / deserialization for Bayesian belief maps.

Saves and restores belief state as compressed NPZ archives with
JSON-encoded measurements for roundtrip persistence.
"""

import io
import json
import numpy as np
from typing import Any, Dict, List, Optional

from models.measurement import Measurement


def serialize_state(
    belief: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    measurements: Optional[List[Measurement]] = None,
    entropy_history: Optional[List[float]] = None,
    metadata: Optional[dict] = None,
) -> bytes:
    """Serialize belief state to a compressed NPZ archive.

    Args:
        belief: 2D belief map array.
        grid_x: 2D meshgrid X coordinates.
        grid_y: 2D meshgrid Y coordinates.
        measurements: Optional list of Measurement objects.
        entropy_history: Optional list of entropy values.
        metadata: Optional dict of extra metadata.

    Returns:
        Bytes of the compressed NPZ archive.
    """
    save_dict: Dict[str, Any] = {
        "belief": belief,
        "grid_x": grid_x,
        "grid_y": grid_y,
    }

    if entropy_history:
        save_dict["entropy_history"] = np.array(entropy_history)

    # Encode measurements as JSON string stored in a numpy char array
    if measurements:
        meas_dicts = [
            {
                "x": m.x,
                "y": m.y,
                "concentration_ppm": m.concentration_ppm,
                "detected": m.detected,
                "wind_speed": m.wind_speed,
                "wind_direction_deg": m.wind_direction_deg,
                "stability_class": m.stability_class,
            }
            for m in measurements
        ]
        meas_json = json.dumps(meas_dicts)
        save_dict["measurements_json"] = np.array(meas_json)

    if metadata:
        meta_json = json.dumps(metadata, default=str)
        save_dict["metadata_json"] = np.array(meta_json)

    buf = io.BytesIO()
    np.savez_compressed(buf, **save_dict)
    buf.seek(0)
    return buf.read()


def deserialize_state(data: bytes) -> dict:
    """Deserialize belief state from NPZ bytes.

    Args:
        data: Bytes of a compressed NPZ archive.

    Returns:
        Dict with keys:
            'belief': 2D numpy array
            'grid_x': 2D numpy array
            'grid_y': 2D numpy array
            'measurements': List[Measurement] (may be empty)
            'entropy_history': List[float] (may be empty)
            'metadata': dict (may be empty)
    """
    buf = io.BytesIO(data)
    npz = np.load(buf, allow_pickle=False)

    result: Dict[str, Any] = {
        "belief": npz["belief"],
        "grid_x": npz["grid_x"],
        "grid_y": npz["grid_y"],
        "measurements": [],
        "entropy_history": [],
        "metadata": {},
    }

    if "entropy_history" in npz:
        result["entropy_history"] = npz["entropy_history"].tolist()

    if "measurements_json" in npz:
        meas_json = str(npz["measurements_json"])
        meas_dicts = json.loads(meas_json)
        result["measurements"] = [
            Measurement(**md) for md in meas_dicts
        ]

    if "metadata_json" in npz:
        meta_json = str(npz["metadata_json"])
        result["metadata"] = json.loads(meta_json)

    return result

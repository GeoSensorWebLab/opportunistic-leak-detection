"""
Tasking Optimizer.

Given a worker's baseline path and a detection probability map,
recommends optimal waypoints that maximize detection probability
while minimizing deviation from the planned route.

Score = DetectionProbability / (PathDeviationCost + epsilon)
"""

import numpy as np
import streamlit as st
from typing import List, Optional, Tuple
from scipy.spatial.distance import cdist

from config import (
    DEVIATION_EPSILON,
    MAX_DEVIATION_M,
    TOP_K_RECOMMENDATIONS,
    MIN_WAYPOINT_SEPARATION_M,
    CLUSTER_FRACTION,
)


def compute_path_deviation(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    baseline_path: np.ndarray,
) -> np.ndarray:
    """
    Compute the minimum distance from each grid cell to the baseline path.

    For each grid point, finds the closest point on the baseline path
    (approximated as distance to nearest path waypoint).

    Args:
        grid_x, grid_y: 2D meshgrid arrays (meters).
        baseline_path: (N, 2) array of [x, y] waypoints defining the worker's route.

    Returns:
        deviation: 2D array (same shape as grid_x) of minimum distances in meters.
    """
    # Flatten grid to (M, 2) for distance computation
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Compute pairwise distances between grid points and path waypoints
    distances = cdist(grid_points, baseline_path, metric="euclidean")

    # Minimum distance to any waypoint on the path
    min_dist = distances.min(axis=1)

    return min_dist.reshape(grid_x.shape)


@st.cache_data(max_entries=4)
def cached_path_deviation(
    grid_size: float,
    resolution: float,
    baseline_path_key: Tuple[Tuple[float, float], ...],
) -> np.ndarray:
    """
    Cached wrapper around compute_path_deviation.

    The deviation grid depends only on the grid geometry and the baseline
    path — it is completely independent of wind conditions.  Caching it
    avoids the expensive cdist call on every wind change.

    Args:
        grid_size: Site extent in meters.
        resolution: Grid cell size in meters.
        baseline_path_key: Hashable tuple-of-tuples of [x, y] waypoints.

    Returns:
        deviation: 2D array of minimum distances (same shape as the grid).
    """
    from optimization.opportunity_map import create_grid

    X, Y = create_grid(grid_size, resolution)
    baseline_path = np.array(baseline_path_key)
    return compute_path_deviation(X, Y, baseline_path)


def compute_tasking_scores(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    detection_prob: np.ndarray,
    baseline_path: np.ndarray,
    epsilon: float = DEVIATION_EPSILON,
    max_deviation: float = MAX_DEVIATION_M,
    precomputed_deviation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the tasking score for each grid cell.

    Score = DetectionProbability / (PathDeviation + epsilon)

    Cells beyond max_deviation are scored 0 (not worth visiting).

    Args:
        grid_x, grid_y: 2D meshgrid arrays.
        detection_prob: 2D array of detection probabilities [0, 1].
        baseline_path: (N, 2) array of worker's planned waypoints.
        epsilon: Small constant to prevent division by zero (meters).
        max_deviation: Maximum allowable deviation from path (meters).
        precomputed_deviation: Optional pre-cached deviation grid to skip cdist.

    Returns:
        scores: 2D array of tasking scores (same shape as grid_x).
    """
    if precomputed_deviation is not None:
        deviation = precomputed_deviation
    else:
        deviation = compute_path_deviation(grid_x, grid_y, baseline_path)

    scores = detection_prob / (deviation + epsilon)

    # Zero out cells that are too far from the path
    scores[deviation > max_deviation] = 0.0

    return scores


def recommend_waypoints(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    scores: np.ndarray,
    detection_prob: np.ndarray,
    concentration_ppm: np.ndarray,
    top_k: int = TOP_K_RECOMMENDATIONS,
    min_separation: float = MIN_WAYPOINT_SEPARATION_M,
) -> List[dict]:
    """
    Extract the top-K recommended waypoints from the score map.

    Uses non-maximum suppression to ensure recommendations are spatially
    spread out (at least min_separation meters apart).

    Args:
        grid_x, grid_y: 2D meshgrid arrays.
        scores: 2D tasking score array.
        detection_prob: 2D detection probability array.
        concentration_ppm: 2D concentration array (for reporting).
        top_k: Number of waypoints to return.
        min_separation: Minimum distance between recommendations (meters).

    Returns:
        List of dicts with keys: 'x', 'y', 'score', 'detection_prob', 'concentration_ppm'.
    """
    # Flatten and sort by score descending
    flat_scores = scores.ravel()
    sorted_indices = np.argsort(flat_scores)[::-1]

    recommendations = []
    selected_coords = []

    for idx in sorted_indices:
        if len(recommendations) >= top_k:
            break

        if flat_scores[idx] <= 0:
            break

        x = grid_x.ravel()[idx]
        y = grid_y.ravel()[idx]

        # Non-maximum suppression: skip if too close to an already-selected point
        too_close = False
        for sx, sy in selected_coords:
            if np.hypot(x - sx, y - sy) < min_separation:
                too_close = True
                break
        if too_close:
            continue

        selected_coords.append((x, y))
        recommendations.append(
            {
                "x": float(x),
                "y": float(y),
                "score": float(flat_scores[idx]),
                "detection_prob": float(detection_prob.ravel()[idx]),
                "concentration_ppm": float(concentration_ppm.ravel()[idx]),
            }
        )

    return recommendations


def _project_onto_path(
    point: np.ndarray,
    path: np.ndarray,
) -> Tuple[int, float, np.ndarray, float]:
    """
    Project a point onto the nearest segment of a polyline path.

    Returns:
        (segment_index, t_parameter, closest_point, cumulative_distance_along_path)
    """
    best_dist = float("inf")
    best_seg = 0
    best_t = 0.0
    best_closest = path[0].copy()

    for i in range(len(path) - 1):
        seg_vec = path[i + 1] - path[i]
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue
        t = np.clip(np.dot(point - path[i], seg_vec) / (seg_len ** 2), 0, 1)
        closest = path[i] + t * seg_vec
        dist = np.linalg.norm(point - closest)
        if dist < best_dist:
            best_dist = dist
            best_seg = i
            best_t = t
            best_closest = closest

    # Cumulative distance along the path up to the projection point
    cum = 0.0
    for i in range(best_seg):
        cum += np.linalg.norm(path[i + 1] - path[i])
    cum += best_t * np.linalg.norm(path[best_seg + 1] - path[best_seg])

    return best_seg, best_t, best_closest, cum


def _tour_length(
    coords: List[np.ndarray],
    entry: np.ndarray,
    exit_pt: np.ndarray,
) -> float:
    """Total distance: entry -> coords[0] -> ... -> coords[-1] -> exit_pt."""
    d = np.linalg.norm(coords[0] - entry)
    for i in range(len(coords) - 1):
        d += np.linalg.norm(coords[i + 1] - coords[i])
    d += np.linalg.norm(coords[-1] - exit_pt)
    return d


def _nearest_neighbor_order(
    coords: List[np.ndarray],
    entry: np.ndarray,
) -> List[np.ndarray]:
    """Order waypoints by nearest-neighbor starting from *entry*."""
    remaining = list(range(len(coords)))
    order: List[int] = []
    current = entry

    while remaining:
        dists = [np.linalg.norm(coords[i] - current) for i in remaining]
        best = remaining[int(np.argmin(dists))]
        order.append(best)
        current = coords[best]
        remaining.remove(best)

    return [coords[i] for i in order]


def _two_opt(
    coords: List[np.ndarray],
    entry: np.ndarray,
    exit_pt: np.ndarray,
) -> List[np.ndarray]:
    """Improve a waypoint tour with 2-opt edge swaps."""
    if len(coords) <= 2:
        return coords

    best = list(coords)
    improved = True
    while improved:
        improved = False
        best_dist = _tour_length(best, entry, exit_pt)
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                d = _tour_length(candidate, entry, exit_pt)
                if d < best_dist - 1e-6:
                    best = candidate
                    best_dist = d
                    improved = True
    return best


def build_optimized_path(
    baseline_path: np.ndarray,
    waypoints: List[dict],
) -> np.ndarray:
    """
    Build an efficient walking route that follows the baseline path and takes
    optimised detours to visit every recommended waypoint.

    Algorithm:
      1. Project each waypoint onto the baseline to find its walking-order
         position (cumulative distance along the route).
      2. Sort waypoints by that position so they are visited in the order the
         worker naturally encounters them.
      3. Cluster consecutive waypoints whose baseline projections are close
         together — these can be chained into a single detour instead of
         separate out-and-back trips.
      4. Within each cluster, find the best visit order using
         nearest-neighbour seeding + 2-opt local search.
      5. Splice each cluster detour into the baseline at the correct point.

    Args:
        baseline_path: (N, 2) array of original waypoints.
        waypoints: List of recommendation dicts with 'x', 'y' keys.

    Returns:
        optimized_path: (M, 2) array — a connected route the worker can follow.
    """
    if not waypoints:
        return baseline_path.copy()

    n_base = len(baseline_path)

    # --- Step 1: project every waypoint onto the baseline -----------------
    projections = []
    for wp in waypoints:
        coord = np.array([wp["x"], wp["y"]])
        seg_idx, t, proj_pt, dist_along = _project_onto_path(coord, baseline_path)
        projections.append({
            "coord": coord,
            "seg_idx": seg_idx,
            "t": t,
            "proj": proj_pt,
            "dist_along": dist_along,
        })

    # --- Step 2: sort by walking-order position along the baseline --------
    projections.sort(key=lambda p: p["dist_along"])

    # --- Step 3: cluster nearby waypoints ---------------------------------
    # Total baseline length for a relative merge threshold
    total_len = sum(
        np.linalg.norm(baseline_path[i + 1] - baseline_path[i])
        for i in range(n_base - 1)
    )
    merge_thresh = total_len * CLUSTER_FRACTION

    clusters: List[List[dict]] = [[projections[0]]]
    for p in projections[1:]:
        if p["dist_along"] - clusters[-1][-1]["dist_along"] < merge_thresh:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    # --- Steps 4 & 5: optimise each cluster and splice into baseline ------
    # Build a list of (insert_after_baseline_index, [ordered coords])
    # sorted by baseline position so we can splice them in one pass.
    detours: List[Tuple[int, List[np.ndarray]]] = []

    for cluster in clusters:
        depart_seg = cluster[0]["seg_idx"]
        rejoin_seg = min(cluster[-1]["seg_idx"] + 1, n_base - 1)

        entry = baseline_path[depart_seg]
        exit_pt = baseline_path[rejoin_seg]

        coords = [p["coord"] for p in cluster]
        if len(coords) == 1:
            ordered = coords
        else:
            ordered = _nearest_neighbor_order(coords, entry)
            ordered = _two_opt(ordered, entry, exit_pt)

        detours.append((depart_seg + 1, ordered))

    # Splice detours into the baseline (reverse order to keep indices stable)
    path = list(baseline_path)
    detours.sort(key=lambda d: d[0], reverse=True)
    for insert_idx, coords in detours:
        for c in reversed(coords):
            path.insert(insert_idx, c)

    return np.array(path)

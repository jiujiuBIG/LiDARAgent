from collections import Counter
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

TREE_INSTANCE_DIMENSION = "lidaragent_tree_id"
HEIGHT_DIMENSION = "lidaragent_hag"
HEIGHT_ABOVE_GROUND_FIELDS = (HEIGHT_DIMENSION, "height_above_ground", "heightaboveground")
BREAST_HEIGHT_MIN = 1.2
BREAST_HEIGHT_MAX = 1.4
STEM_SECTION_MIN_M = 0.8
CROWN_HEIGHT_FRACTION = 0.4
STEM_ASSIGNMENT_RADIUS_M = 0.55
TREETOP_HEIGHT_TOLERANCE_M = 1.5
CROWN_NEIGHBOUR_RADIUS_M = 1.0


def las_height_above_ground(las) -> np.ndarray | None:
    dimensions = {name.lower(): name for name in las.point_format.dimension_names}
    for name in HEIGHT_ABOVE_GROUND_FIELDS:
        if name in dimensions:
            return np.asarray(las[dimensions[name]], dtype=np.float64)
    return None


def label_traced_instances(
    cloud: np.ndarray,
    owners: np.ndarray,
    origin: np.ndarray,
    horizontal_m: float,
    vertical_m: float,
    traced_tops: dict,
    treetops: dict,
    stem_base_m: float = 1.5,
) -> np.ndarray:
    """Assign points only for traces with an unambiguous link to a final treetop."""
    labels = np.zeros(len(cloud), dtype=np.int32)
    if not owners.size or not traced_tops or not treetops:
        return labels

    tree_ids = np.array(list(treetops), dtype=np.int32)
    tops = np.asarray(list(treetops.values()), dtype=float)
    tree_index = cKDTree(tops[:, :2])
    matches = {}
    for owner_id, top in traced_tops.items():
        if top is None:
            continue
        x, y, height = top
        nearby = tree_index.query_ball_point((x, y), r=2.0)
        candidates = [
            index for index in nearby
            if abs(tops[index, 2] - height) <= TREETOP_HEIGHT_TOLERANCE_M
        ]
        if len(candidates) == 1:
            matches[owner_id] = int(tree_ids[candidates[0]])
    counts = Counter(matches.values())
    owner_labels = np.zeros(max(matches, default=-1) + 2, dtype=np.int32)
    for owner_id, tree_id in matches.items():
        if counts[tree_id] == 1:
            owner_labels[owner_id + 1] = tree_id
    if not owner_labels.any():
        return labels

    selected = np.flatnonzero(np.isfinite(cloud).all(axis=1) & (cloud[:, 2] >= STEM_SECTION_MIN_M))
    col, row = np.floor((cloud[selected, :2] - origin) / horizontal_m).astype(np.int64).T
    height = cloud[selected, 2]
    levels = np.floor(height / vertical_m).astype(np.int64)
    rows, cols, slices = owners.shape
    in_grid = (row >= 0) & (row < rows) & (col >= 0) & (col < cols) & (levels < slices)
    point_owners = np.full(len(selected), -1, dtype=np.int32)
    traced = in_grid & (height >= stem_base_m)
    point_owners[traced] = owners[row[traced], col[traced], levels[traced]]

    # Include the local cylinder fit's lower stem section only in the same XY
    # voxel when the first two trace layers support one owner.
    base_level = int(np.floor(stem_base_m / vertical_m))
    below = in_grid & (height < stem_base_m)
    if below.any() and base_level < slices:
        anchors = owners[row[below], col[below], base_level:base_level + 2]
        candidate = anchors.max(axis=1)
        agrees = ((anchors < 0) | (anchors == candidate[:, None])).all(axis=1)
        point_owners[below] = np.where(agrees, candidate, -1)

    matched = (point_owners >= 0) & (point_owners + 1 < len(owner_labels))
    labels[selected[matched]] = owner_labels[point_owners[matched] + 1]
    return labels


def _sample_crown_ids(xy, crown_ids, transform):
    inverse = ~transform
    x, y = xy.T
    col, row = inverse * (x, y)
    col, row = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
    rows, cols = crown_ids.shape
    inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    ids = np.zeros(len(xy), dtype=np.int32)
    ids[inside] = crown_ids[row[inside], col[inside]]
    return ids


def _resolve_overlapping_crowns(cloud, selected, labels, crown_ceiling, stem_top_m):
    """Limit crown/trace claims to the apex and use neighbouring 3D canopy evidence."""
    point_labels = labels[selected]
    heights = cloud[selected, 2]
    rejected = (point_labels > 0) & (heights > crown_ceiling[point_labels])
    if not rejected.any():
        return

    points = selected[rejected]
    labels[points] = 0
    queries = cloud[points]
    radius = CROWN_NEIGHBOUR_RADIUS_M
    lower, upper = queries.min(axis=0) - radius, queries.max(axis=0) + radius
    candidates = selected[(point_labels > 0) & ~rejected & (heights >= stem_top_m)]
    for axis in range(3):
        values = cloud[candidates, axis]
        candidates = candidates[(values >= lower[axis]) & (values <= upper[axis])]
    if not len(candidates):
        return
    distance, neighbours = cKDTree(cloud[candidates]).query(queries, distance_upper_bound=radius)
    near = np.isfinite(distance)
    owners = labels[candidates[neighbours[near]]]
    compatible = queries[near, 2] <= crown_ceiling[owners]
    labels[points[near][compatible]] = owners[compatible]


def label_crown_and_stem_instances(
    cloud, crown_ids, transform, treetops, min_canopy_height_m, *,
    stem_positions=(), traced_labels=None, stem_top_m=6.5,
):
    """Join elevated crown returns and verified stem returns under the same IDs."""
    labels = np.zeros(len(cloud), dtype=np.int32) if traced_labels is None else traced_labels.copy()
    if not treetops:
        return labels
    selected = np.flatnonzero(np.isfinite(cloud).all(axis=1) & (cloud[:, 2] >= STEM_SECTION_MIN_M))
    xy, heights = cloud[selected, :2], cloud[selected, 2]
    crowns = _sample_crown_ids(xy, crown_ids, transform)
    crown_floor = np.full(max(treetops) + 1, np.inf)
    crown_ceiling = np.full_like(crown_floor, np.inf)
    for tree_id, (_, _, height) in treetops.items():
        crown_floor[tree_id] = max(min_canopy_height_m, CROWN_HEIGHT_FRACTION * height)
        crown_ceiling[tree_id] = height + TREETOP_HEIGHT_TOLERANCE_M
    canopy = (crowns > 0) & (crowns < len(crown_floor))
    canopy[canopy] &= (heights[canopy] > BREAST_HEIGHT_MAX) & (
        heights[canopy] >= crown_floor[crowns[canopy]]
    )
    free = labels[selected] == 0
    labels[selected[canopy & free]] = crowns[canopy & free]

    if len(stem_positions):
        anchors = np.asarray(stem_positions, dtype=float)
        anchor_ids = _sample_crown_ids(anchors, crown_ids, transform)
        counts = Counter(anchor_ids)
        # A crown with competing stems cannot identify a unique lower trunk.
        eligible = np.array([tree_id in treetops and counts[tree_id] == 1 for tree_id in anchor_ids])
        low = np.flatnonzero(free & ~canopy & (heights <= stem_top_m))
        if len(low):
            distances, neighbours = cKDTree(anchors).query(
                xy[low], k=2, distance_upper_bound=STEM_ASSIGNMENT_RADIUS_M,
            )
            unique = np.isfinite(distances[:, 0]) & ~np.isfinite(distances[:, 1])
            points, nearest = low[unique], neighbours[unique, 0]
            consistent = eligible[nearest] & ((crowns[points] == 0) | (crowns[points] == anchor_ids[nearest]))
            labels[selected[points[consistent]]] = anchor_ids[nearest[consistent]]
    _resolve_overlapping_crowns(cloud, selected, labels, crown_ceiling, stem_top_m)
    return labels


def write_instance_cloud(
    source: Path, target: Path, height_above_ground, labels, *,
    method="stem_trace_and_unique_treetop_match",
) -> dict:
    """Copy the original LAS point record and append separate prediction fields."""
    las = laspy.read(source)
    heights = np.asarray(height_above_ground, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32).copy()
    if heights.shape != (len(las.points),) or labels.shape != heights.shape:
        raise ValueError("单木编号和离地高度必须与源点云逐点对齐。")
    existing = {name.lower() for name in las.point_format.dimension_names}
    if existing.intersection((TREE_INSTANCE_DIMENSION, HEIGHT_DIMENSION)):
        raise ValueError("输入已包含 LiDARAgent 编号或离地高度字段。请使用原始点云重新分割。")
    labels[(np.asarray(las.classification) == 2) | ~np.isfinite(heights)] = 0
    las.add_extra_dims([
        laspy.ExtraBytesParams(name=TREE_INSTANCE_DIMENSION, type=np.int32),
        laspy.ExtraBytesParams(name=HEIGHT_DIMENSION, type=np.float64),
    ])
    las[TREE_INSTANCE_DIMENSION] = labels
    las[HEIGHT_DIMENSION] = heights
    las.write(target)
    assigned = labels > 0
    slice_points = assigned & (heights >= BREAST_HEIGHT_MIN) & (heights <= BREAST_HEIGHT_MAX)
    return {
        "filename": target.name,
        "method": method,
        "labelled_points": int(np.count_nonzero(assigned)),
        "unassigned_points": int(np.count_nonzero(~assigned)),
        "labelled_trees": int(len(np.unique(labels[assigned]))),
        "breast_height_labelled_points": int(np.count_nonzero(slice_points)),
    }

from collections import Counter, defaultdict
import csv
from pathlib import Path

import laspy
import numpy as np
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

TREE_INSTANCE_DIMENSION = "lidaragent_tree_id"
HEIGHT_DIMENSION = "lidaragent_hag"
HEIGHT_ABOVE_GROUND_FIELDS = (HEIGHT_DIMENSION, "height_above_ground", "heightaboveground")
BREAST_HEIGHT_MIN = 1.2
BREAST_HEIGHT_MAX = 1.4
STEM_SECTION_MIN_M = 0.8
STEM_SECTION_MAX_M = 1.8
CROWN_HEIGHT_FRACTION = 0.4
STEM_ASSIGNMENT_RADIUS_M = 0.55
TREETOP_HEIGHT_TOLERANCE_M = 1.5
CROWN_NEIGHBOUR_RADIUS_M = 1.0
MIN_SEPARATED_CROWN_PIXELS = 2


def las_height_above_ground(las) -> np.ndarray | None:
    dimensions = {name.lower(): name for name in las.point_format.dimension_names}
    for name in HEIGHT_ABOVE_GROUND_FIELDS:
        if name in dimensions:
            return np.asarray(las[dimensions[name]], dtype=np.float64)
    return None


def compute_height_above_ground(las, *, dtm_cell_m: float = 2.0) -> tuple[np.ndarray, str]:
    """Return per-point height above ground (metres) and the method used.

    Priority:
    1. Existing explicit HAG field (`lidaragent_hag` / `HeightAboveGround`).
    2. IDW interpolation from ASPRS ground points (`classification == 2`).
    3. Coarse local-minimum DTM grid (`dtm_cell_m`) IDW fallback when no ground
       classification is present.
    """
    existing = las_height_above_ground(las)
    if existing is not None and np.isfinite(existing).any():
        return existing.astype(np.float64, copy=False), "existing_field"

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    n = len(z)
    if n == 0:
        return np.empty(0, dtype=np.float64), "empty"

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    hag = np.full(n, np.nan, dtype=np.float64)
    if not finite.any():
        return hag, "non_finite"

    xy_all = np.column_stack((x[finite], y[finite]))
    z_all = z[finite]

    has_cls = "classification" in las.point_format.dimension_names
    ground_mask = finite & (np.asarray(las.classification) == 2) if has_cls else np.zeros(n, dtype=bool)
    n_ground = int(np.count_nonzero(ground_mask))

    if n_ground >= 3:
        g_xy = np.column_stack((x[ground_mask], y[ground_mask]))
        g_z = z[ground_mask]
        k = min(8, n_ground)
        dists, idxs = cKDTree(g_xy).query(xy_all, k=k)
        if k == 1:
            z_ground = g_z[idxs]
        else:
            weights = 1.0 / np.maximum(dists, 1e-4)
            z_ground = np.sum(weights * g_z[idxs], axis=1) / np.sum(weights, axis=1)
        hag[finite] = z_all - z_ground
        return hag, "ground_class_idw"

    # Fallback: estimate coarse DTM from 5th-percentile Z per dtm_cell_m grid cell
    min_xy = xy_all.min(axis=0)
    cell_ij = np.floor((xy_all - min_xy) / float(dtm_cell_m)).astype(np.int64)
    order = np.lexsort((cell_ij[:, 1], cell_ij[:, 0]))
    sorted_ij = cell_ij[order]
    sorted_z = z_all[order]
    change = np.empty(len(order), dtype=bool)
    change[0] = True
    change[1:] = (sorted_ij[1:, 0] != sorted_ij[:-1, 0]) | (sorted_ij[1:, 1] != sorted_ij[:-1, 1])
    starts = np.flatnonzero(change)
    stops = np.r_[starts[1:], len(order)]

    cell_centers = np.empty((len(starts), 2), dtype=np.float64)
    cell_z = np.empty(len(starts), dtype=np.float64)
    for idx_c, (s, e) in enumerate(zip(starts, stops)):
        cell_centers[idx_c] = min_xy + (sorted_ij[s] + 0.5) * float(dtm_cell_m)
        cell_z[idx_c] = float(np.percentile(sorted_z[s:e], 5))

    k = min(4, len(cell_z))
    dists, idxs = cKDTree(cell_centers).query(xy_all, k=k)
    if k == 1:
        z_ground = cell_z[idxs]
    else:
        weights = 1.0 / np.maximum(dists, 1e-4)
        z_ground = np.sum(weights * cell_z[idxs], axis=1) / np.sum(weights, axis=1)
    hag[finite] = z_all - z_ground
    return hag, "local_grid_dtm"


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


def select_separated_crown_tops(stem_positions, traced_tops, crown_ids, chm, transform, treetops):
    """Recover a visible lower crown only when another stem explains its upper crown."""
    if not len(stem_positions) or not traced_tops or not treetops:
        return []

    anchors = np.asarray(stem_positions, dtype=float)
    owners = _sample_crown_ids(anchors, crown_ids, transform)
    caps_by_crown = defaultdict(list)
    for stem_id, cap in traced_tops.items():
        owner = owners[stem_id]
        if owner in treetops and cap is not None and np.isfinite(cap).all():
            caps_by_crown[owner].append((stem_id, np.asarray(cap, dtype=float)))

    all_tops = np.asarray(list(treetops.values()), dtype=float)
    inverse = ~transform
    accepted = []
    for owner, caps in caps_by_crown.items():
        if len(caps) < 2:
            continue
        incumbent = np.asarray(treetops[owner], dtype=float)
        dominant, upper = min(caps, key=lambda item: np.linalg.norm(item[1] - incumbent))
        if np.linalg.norm(upper[:2] - incumbent[:2]) > 2.0 or abs(upper[2] - incumbent[2]) > 3.0:
            continue
        for stem_id, cap in caps:
            if stem_id == dominant or cap[2] < 10.0 or upper[2] - cap[2] < 2.5:
                continue
            if np.linalg.norm(upper[:2] - cap[:2]) < 0.75:
                continue
            if np.any(
                (np.linalg.norm(all_tops[:, :2] - cap[:2], axis=1) < 0.75)
                & (np.abs(all_tops[:, 2] - cap[2]) <= TREETOP_HEIGHT_TOLERANCE_M)
            ):
                continue
            col, row = np.floor(inverse * tuple(cap[:2])).astype(int)
            if not (0 <= row < chm.shape[0] and 0 <= col < chm.shape[1]):
                continue
            if crown_ids[row, col] == owner and abs(chm[row, col] - cap[2]) <= 2.0:
                accepted.append(tuple(float(value) for value in cap))
    return sorted(accepted, key=lambda point: (-point[2], point[0], point[1]))


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


def _link_shared_stems(anchors, candidate_stems, targets):
    """Keep stem-to-apex links that every maximum one-to-one matching requires."""
    assigned = np.zeros(len(anchors), dtype=np.int32)
    if not len(candidate_stems) or not targets:
        return assigned

    tree_ids = np.asarray(list(targets), dtype=np.int32)
    tops = np.asarray(list(targets.values()))[:, :2]
    distances = cKDTree(anchors[candidate_stems]).sparse_distance_matrix(cKDTree(tops), 2.0, output_type="coo_matrix")
    distances.data.fill(1)
    graph = distances.tocsr()
    matching = maximum_bipartite_matching(graph, perm_type="column")
    maximum = np.count_nonzero(matching >= 0)
    for row in np.flatnonzero(matching >= 0):
        edge = graph.indptr[row] + np.flatnonzero(
            graph.indices[graph.indptr[row]:graph.indptr[row + 1]] == matching[row]
        )[0]
        graph.data[edge] = 0
        without_link = graph.copy()
        without_link.eliminate_zeros()
        alternative = maximum_bipartite_matching(without_link, perm_type="column")
        if np.count_nonzero(alternative >= 0) < maximum:
            assigned[candidate_stems[row]] = tree_ids[matching[row]]
        graph.data[edge] = 1
    return assigned


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
        low = np.flatnonzero(~canopy & (heights <= stem_top_m))
        if len(low):
            distances, neighbours = cKDTree(anchors).query(
                xy[low], k=2, distance_upper_bound=STEM_ASSIGNMENT_RADIUS_M,
            )
            unique = np.isfinite(distances[:, 0]) & ~np.isfinite(distances[:, 1])
            points, nearest = low[unique], neighbours[unique, 0]
            consistent = eligible[nearest] & ((crowns[points] == 0) | (crowns[points] == anchor_ids[nearest]))
            direct = consistent & free[points]
            labels[selected[points[direct]]] = anchor_ids[nearest[direct]]

            in_section = heights[points] <= STEM_SECTION_MAX_M
            at_breast = in_section & (heights[points] >= BREAST_HEIGHT_MIN) & (heights[points] <= BREAST_HEIGHT_MAX)
            point_ids = labels[selected[points]]
            shared = np.array([tree_id in treetops and counts[tree_id] > 1 for tree_id in anchor_ids])
            occupied_stems = np.unique(nearest[in_section & (point_ids > 0)])
            missing_stems = np.setdiff1d(np.unique(nearest[at_breast & (point_ids == 0) & shared[nearest]]), occupied_stems)
            if len(missing_stems):
                chest_ids = set(labels[selected[(heights >= BREAST_HEIGHT_MIN) & (heights <= BREAST_HEIGHT_MAX)]])
                canopy_ids = set(labels[selected[heights > stem_top_m]])
                missing_tops = {i: top for i, top in treetops.items() if i in canopy_ids and i not in chest_ids}
                recovered = _link_shared_stems(anchors, missing_stems, missing_tops)[nearest]
                compatible = (crowns[points] == 0) | (crowns[points] == anchor_ids[nearest]) | (crowns[points] == recovered)
                add = in_section & (point_ids == 0) & (recovered > 0) & compatible
                labels[selected[points[add]]] = recovered[add]
    _resolve_overlapping_crowns(cloud, selected, labels, crown_ceiling, stem_top_m)
    return labels


def recover_occluded_instances(cloud, labels, crown_ids, transform, chm, stems, caps, treetops):
    """Split point-only lower crowns with an independent stem and an upper owner."""
    if not len(stems) or not treetops:
        return labels, {}
    anchors = np.asarray(stems, dtype=float)
    stem_crowns = _sample_crown_ids(anchors, crown_ids, transform)
    final_tops = np.asarray(list(treetops.values()))
    inverse = ~transform
    candidates = []
    for stem_id, cap in caps.items():
        if cap is None:
            continue
        x, y, height = cap
        parent = int(stem_crowns[stem_id])
        if parent not in treetops or not 10.0 <= height <= treetops[parent][2] - 3.0:
            continue
        col, row = np.floor(inverse * (x, y)).astype(int)
        if not (0 <= row < chm.shape[0] and 0 <= col < chm.shape[1]):
            continue
        if crown_ids[row, col] != parent or chm[row, col] < height + 3.0:
            continue
        if np.any(
            (np.linalg.norm(final_tops[:, :2] - (x, y), axis=1) <= 2.0)
            & (np.abs(final_tops[:, 2] - height) <= TREETOP_HEIGHT_TOLERANCE_M)
        ):
            continue
        upper = treetops[parent]
        if any(
            other is not None and other[2] > height + 3.0
            and np.linalg.norm(np.asarray(other[:2]) - upper[:2]) <= 2.0
            for other in caps.values()
        ):
            candidates.append((stem_id, parent, cap))
    if not candidates:
        return labels, {}

    finite = np.flatnonzero(np.isfinite(cloud).all(axis=1))
    nearby = cKDTree(cloud[finite, :2])
    stems_index = cKDTree(anchors)
    proposals = []
    claims = np.zeros(len(labels), dtype=np.int32)
    for stem_id, parent, (x, y, height) in candidates:
        indices = finite[nearby.query_ball_point((x, y), r=1.25)]
        indices = indices[(labels[indices] == parent) | (labels[indices] == 0)]
        heights = cloud[indices, 2]
        crown = (heights >= max(6.5, 0.35 * height)) & (heights <= height + TREETOP_HEIGHT_TOLERANCE_M)
        lower = np.flatnonzero((heights >= STEM_SECTION_MIN_M) & (heights < 6.5))
        distances, owners = stems_index.query(
            cloud[indices[lower], :2], k=2, distance_upper_bound=STEM_ASSIGNMENT_RADIUS_M,
        )
        stem = lower[(owners[:, 0] == stem_id) & ~np.isfinite(distances[:, 1])]
        if np.count_nonzero(crown) >= 100 and np.count_nonzero(
            (heights[stem] >= BREAST_HEIGHT_MIN) & (heights[stem] <= BREAST_HEIGHT_MAX)
        ) >= 12:
            points = indices[np.r_[np.flatnonzero(crown), stem]]
            claims[points] += 1
            proposals.append((parent, points))

    recovered = labels.copy()
    parents = {}
    for parent, points in proposals:
        points = points[claims[points] == 1]
        heights = cloud[points, 2]
        if np.count_nonzero(heights >= 6.5) < 100 or np.count_nonzero(
            (heights >= BREAST_HEIGHT_MIN) & (heights <= BREAST_HEIGHT_MAX)
        ) < 12:
            continue
        tree_id = max(treetops) + len(parents) + 1
        recovered[points] = tree_id
        parents[tree_id] = parent
    return recovered, parents


def write_instance_cloud(
    source: Path, target: Path, height_above_ground, labels, *,
    method="stem_trace_and_unique_treetop_match",
    summary_path=None, occluded_parent_ids=None,
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
    metadata = {
        "filename": target.name,
        "method": method,
        "labelled_points": int(np.count_nonzero(assigned)),
        "unassigned_points": int(np.count_nonzero(~assigned)),
        "labelled_trees": int(len(np.unique(labels[assigned]))),
        "breast_height_labelled_points": int(np.count_nonzero(slice_points)),
    }
    if summary_path is not None:
        parents = occluded_parent_ids or {}
        tree_ids, counts = np.unique(labels[assigned], return_counts=True)
        max_heights = np.full(int(labels.max(initial=0)) + 1, -np.inf)
        np.maximum.at(max_heights, labels[assigned], heights[assigned])
        breast_counts = np.bincount(labels[slice_points], minlength=len(max_heights))
        with summary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("tree_id", "point_count", "height_m", "breast_height_points", "parent_2d_crown_id"))
            for tree_id, count in zip(tree_ids, counts):
                writer.writerow((tree_id, count, max_heights[tree_id], breast_counts[tree_id], parents.get(tree_id)))
        metadata["summary_filename"] = summary_path.name
        metadata["occluded_tree_count"] = sum(int(tree_id) in parents for tree_id in tree_ids)
    return metadata

import math
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

import laspy
import numpy as np
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from scipy.spatial import cKDTree

mcp = FastMCP("PointCloudNormals")
MIN_NEIGHBORS = 12
IN_MEMORY_POINT_LIMIT = 2_000_000
DEFAULT_TILE_SIZE = 16.0
WRITE_CHUNK_SIZE = 250_000
NORMAL_DIMENSIONS = (
    ("normal_x", np.float32),
    ("normal_y", np.float32),
    ("normal_z", np.float32),
    ("normal_valid", np.uint8),
)


def estimate_normals(
    xyz: np.ndarray,
    radius: float,
    max_nn: int,
    orient_up: bool,
    query_indices=None,
    workers: int = -1,
):
    queries = xyz if query_indices is None else xyz[query_indices]
    normals = np.zeros((len(queries), 3), dtype=np.float32)
    valid = np.zeros(len(queries), dtype=bool)
    shortage = np.ones(len(queries), dtype=bool)
    if len(xyz) <= MIN_NEIGHBORS:
        return normals, valid, shortage

    tree = cKDTree(xyz)
    k = min(max_nn, len(xyz) - 1) + 1
    batch_size = min(16384, max(1024, 524288 // k))
    query_k = min(k + 1, len(xyz))
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        query = queries[start:end]
        distances, indices = tree.query(query, k=query_k, workers=workers)
        if k < len(xyz):
            ties = (distances[:, k] == distances[:, k - 1]) & (distances[:, k] <= radius)
            for row in np.flatnonzero(ties):
                neighbors = np.asarray(
                    tree.query_ball_point(query[row], np.nextafter(distances[row, k], np.inf)),
                    dtype=np.intp,
                )
                d2 = ((xyz[neighbors] - query[row]) ** 2).sum(axis=1)
                # Distinct coordinates are lexicographically ordered in both input paths.
                choice = np.lexsort((neighbors, d2))[:k]
                indices[row, :k] = neighbors[choice]
                distances[row, :k] = np.sqrt(d2[choice])
        distances, indices = distances[:, :k], indices[:, :k]
        # The first neighbor is the query point itself. Each row of xyz is unique.
        inside = distances[:, 1:] <= radius
        counts = inside.sum(axis=1)
        enough = counts >= MIN_NEIGHBORS
        shortage[start:end] = ~enough
        mask = inside[enough, :, None]
        neighbors = xyz[indices[enough, 1:]]
        totals = np.where(mask, neighbors, 0).sum(axis=1)
        counts = counts[enough, None]
        center = totals / counts
        centered = np.where(mask, neighbors - center[:, None, :], 0)
        covariance = centered.transpose(0, 2, 1) @ centered / counts[:, :, None]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Require two directions of spread and a distinct least-variance direction.
        planar = (eigenvalues[:, 1] > 0.05 * eigenvalues[:, 2]) & (
            eigenvalues[:, 0] <= 0.25 * eigenvalues[:, 1]
        )
        selected = start + np.flatnonzero(enough)[planar]
        vectors = eigenvectors[planar, :, 0]
        if orient_up:
            vectors[vectors[:, 2] < 0] *= -1
        normals[selected] = vectors
        valid[selected] = True
    return normals, valid, shortage


def fit_stem_circle(xy: np.ndarray, min_sectors: int):
    if len(xy) < 20:
        return None
    center = xy.mean(axis=0)
    local = xy - center
    design = np.column_stack((2 * local, np.ones(len(local))))
    solution, _, rank, _ = np.linalg.lstsq(design, (local * local).sum(axis=1), rcond=None)
    if rank < 3:
        return None
    center += solution[:2]
    radial = xy - center
    distances = np.linalg.norm(radial, axis=1)
    radius = distances.mean()
    residual = np.sqrt(np.mean((distances - radius) ** 2))
    angles = np.mod(np.arctan2(radial[:, 1], radial[:, 0]), 2 * np.pi)
    sector_count = np.unique(np.floor(angles * 6 / np.pi)).size
    if not (0.045 <= radius <= 0.60 and residual <= min(0.04, 0.28 * radius)):
        return None
    if sector_count < min_sectors:
        return None
    return center


def fit_stem_axis(xyz: np.ndarray):
    if len(xyz) < 60 or fit_stem_circle(xyz[:, :2], min_sectors=6) is None:
        return None
    lower_mask = xyz[:, 2] < np.median(xyz[:, 2])
    lower = xyz[lower_mask]
    upper = xyz[~lower_mask]
    lo = fit_stem_circle(lower[:, :2], min_sectors=5)
    hi = fit_stem_circle(upper[:, :2], min_sectors=5)
    if lo is None or hi is None:
        return None
    origin = np.r_[lo, lower[:, 2].mean()]
    axis = np.r_[hi, upper[:, 2].mean()] - origin
    if axis[2] < 0.12 or np.linalg.norm(axis[:2]) > axis[2] * math.tan(math.radians(18)):
        return None
    return origin, axis / np.linalg.norm(axis)


def stem_label_fields(header, label_schema):
    dimensions = {name.lower(): name for name in header.point_format.dimension_names}
    class_field = "classification" if label_schema == "forinstance" else "label"
    if "treeid" not in dimensions or class_field not in dimensions:
        return None, f"缺少 {class_field} 或 treeID 字段，已提供基础法向量。"
    crs = header.parse_crs()
    if crs and (crs.is_geographic or any(
        not math.isclose(axis.unit_conversion_factor, 1.0) for axis in crs.axis_info
    )):
        return None, "树干朝外需要米制坐标，已提供基础法向量。"
    return (dimensions[class_field], dimensions["treeid"]), ""


def orient_stem_tree(xyz, normals, valid):
    oriented = np.zeros(len(xyz), dtype=bool)
    if not len(xyz):
        return oriented
    z = xyz[:, 2]
    height_bands = np.floor((z - z.min()) / 0.8)
    order = np.argsort(height_bands, kind="stable")
    band_edges = np.flatnonzero(np.diff(height_bands[order])) + 1
    for rows in np.split(order, band_edges):
        axis = fit_stem_axis(np.unique(xyz[rows], axis=0))
        if axis is None:
            continue
        origin, direction = axis
        targets = rows[valid[rows]]
        vectors = xyz[targets] - origin
        radial = vectors - (vectors @ direction)[:, None] * direction
        lengths = np.linalg.norm(radial, axis=1)
        dot = np.sum(normals[targets] * radial, axis=1)
        usable = (lengths > 0) & (np.abs(dot) >= lengths * math.cos(math.radians(45)))
        oriented[targets[usable]] = True
        normals[targets[usable & (dot < 0)]] *= -1
    return oriented


def orient_stem_normals(las, xyz, inverse, normals, valid, label_schema):
    oriented = np.zeros(len(inverse), dtype=bool)
    fields, note = stem_label_fields(las.header, label_schema)
    if not fields:
        return oriented, note
    class_field, tree_field = fields
    tree_id = np.asarray(las[tree_field])
    classes = np.asarray(las[class_field])
    candidates = np.flatnonzero((classes == 4) & (tree_id > 0))
    groups = candidates[np.argsort(tree_id[candidates], kind="stable")]
    tree_edges = np.flatnonzero(np.diff(tree_id[groups])) + 1
    for tree_rows in np.split(groups, tree_edges):
        tree_normals = normals[tree_rows].copy()
        oriented[tree_rows] = orient_stem_tree(xyz[inverse[tree_rows]], tree_normals, valid[tree_rows])
        normals[tree_rows] = tree_normals
    note = "" if oriented.any() else "找不到可可靠定向的树干点，已提供基础法向量。"
    return oriented, note


def write_tiled_output(source, output_file, header, dimensions, results, orientation):
    header = deepcopy(header)
    header.add_extra_dims(
        [laspy.ExtraBytesParams(name=name, type=dtype) for name, dtype in dimensions]
    )
    counts = {"valid_count": 0, "neighbor_shortage_count": 0, "oriented_count": 0}
    offset = 0
    stream = output_file.open("xb")
    try:
        with stream, laspy.open(source) as reader, laspy.open(
            stream, mode="w", header=header, do_compress=False, closefd=False,
        ) as writer:
            for points in reader.chunk_iterator(WRITE_CHUNK_SIZE):
                data = np.array(results[offset:offset + len(points)], copy=True)
                result_points = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)
                # Copy packed records to preserve bit fields and existing extra dimensions.
                for name in points.array.dtype.names:
                    result_points.array[name] = points.array[name]
                for axis, name in enumerate(("normal_x", "normal_y", "normal_z")):
                    result_points[name] = data["normals"][:, axis]
                result_points["normal_valid"] = data["valid"]
                if orientation != "none":
                    directed = data["oriented"]
                    if orientation == "up":
                        directed = data["valid"] & (data["normals"][:, 2] > 1e-6)
                    result_points["orientation_valid"] = directed
                    counts["oriented_count"] += int(directed.sum())
                writer.write_points(result_points)
                counts["valid_count"] += int(data["valid"].sum())
                counts["neighbor_shortage_count"] += int(data["shortage"].sum())
                offset += len(points)
            if reader.header.evlrs:
                writer.write_evlrs(reader.header.evlrs)
    except Exception:
        output_file.unlink()
        raise
    return counts


def process_tiled_file(
    source, output_file, header, dimensions, radius, max_nn, orientation, label_schema, tile_size,
):
    if __package__:
        from .normal_tiles import estimate_tiled_normals
    else:
        from normal_tiles import estimate_tiled_normals

    fields, note = stem_label_fields(header, label_schema) if orientation == "stem_outward" else (None, "")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="normal_tiles_", dir=output_file.parent) as folder:
        results = estimate_tiled_normals(
            source, Path(folder), radius, max_nn, orientation == "up", tile_size,
            partial(estimate_normals, workers=1), fields, orient_stem_tree,
        )
        try:
            counts = write_tiled_output(source, output_file, header, dimensions, results, orientation)
        finally:
            if isinstance(results, np.memmap):
                results._mmap.close()
    n = header.point_count
    message = f"法向量分块估计完成，共 {n} 点，其中 {counts['valid_count']} 点有效。"
    if orientation == "none":
        counts.pop("oriented_count")
    else:
        if orientation == "stem_outward":
            if not note and not counts["oriented_count"]:
                note = "找不到可可靠定向的树干点，已提供基础法向量。"
            counts["label_schema"] = label_schema
        message += note + f"其中 {counts['oriented_count']} 点已按要求定向。"
    return {
        "status": "success",
        "output_path": str(output_file),
        "message": message,
        "data": {
            "point_count": n, "invalid_count": n - counts["valid_count"], **counts,
            "radius": radius, "max_nn": max_nn, "orientation": orientation,
            "processing": "tiled", "tile_size": tile_size,
        },
    }


@mcp.tool(name="estimate_point_cloud_normals")
async def estimate_point_cloud_normals(
    file_path: str,
    radius: Annotated[float, Field(gt=0, description="Search radius in the input XYZ coordinate units.")],
    max_nn: Annotated[int, Field(ge=12, description="Maximum number of distinct neighbors.")] = 32,
    orientation: Literal["none", "up", "stem_outward"] = "none",
    output_path: str = "./",
    label_schema: Literal["forinstance", "nibio_mls"] | None = None,
    tile_size: Annotated[float, Field(gt=0, description="Optional XY tile width in input coordinate units.")] | None = None,
) -> dict:
    """Estimate per-point normals in LAS or LAZ with local PCA and write a new LAS.

    Basic normals are the default. Stem outward orientation is optional and
    uses metric coordinates and the specified label schema with tree IDs.
    """
    try:
        if type(radius) not in (int, float) or not 0 < radius < math.inf:
            raise ValueError("搜索半径必须为有限的正数。")
        if type(max_nn) is not int or max_nn < MIN_NEIGHBORS:
            raise ValueError("最大邻居数必须是不小于 12 的整数。")
        if tile_size is not None and (
            type(tile_size) not in (int, float) or not 0 < tile_size < math.inf
        ):
            raise ValueError("分块宽度必须为有限的正数。")
        if orientation not in ("none", "up", "stem_outward"):
            raise ValueError("请选择基础法向量、朝上（+Z）或树干朝外。")
        if orientation == "stem_outward" and label_schema not in ("forinstance", "nibio_mls"):
            raise ValueError("树干朝外需要 label_schema，请指定 forinstance 或 nibio_mls。")
        source = Path(file_path)
        if source.suffix.lower() not in (".las", ".laz"):
            raise ValueError("请选择 LAS 或 LAZ 文件。")
        out_dir = Path(output_path).resolve()
        output_file = out_dir / f"{source.stem}_normals.las"
        if output_file.exists():
            raise FileExistsError(f"output already exists: {output_file}")
        extra_dimensions = NORMAL_DIMENSIONS
        if orientation != "none":
            extra_dimensions += (("orientation_valid", np.uint8),)
        with laspy.open(source) as reader:
            header = reader.header
        dimensions = set(header.point_format.dimension_names)
        if any(name in dimensions for name, _ in extra_dimensions):
            raise ValueError("input already contains normal dimensions")
        if tile_size is not None or header.point_count >= IN_MEMORY_POINT_LIMIT:
            width = tile_size if tile_size is not None else max(DEFAULT_TILE_SIZE, radius)
            return process_tiled_file(
                source, output_file, header, extra_dimensions,
                radius, max_nn, orientation, label_schema, width,
            )

        las = laspy.read(source)
        raw = np.column_stack((las.X, las.Y, las.Z))
        distinct, inverse = np.unique(raw, axis=0, return_inverse=True)
        xyz = distinct * las.header.scales
        normals, valid, shortage = estimate_normals(xyz, radius, max_nn, orientation == "up")
        valid_points = valid[inverse]
        point_normals = normals[inverse]
        n = len(las.points)
        valid_count = int(valid_points.sum())
        message = f"法向量估计完成，共 {n} 点，其中 {valid_count} 点有效。"
        direction_data = {}
        if orientation == "up":
            oriented = valid_points & (point_normals[:, 2] > 1e-6)
        elif orientation == "stem_outward":
            oriented, note = orient_stem_normals(
                las, xyz, inverse, point_normals, valid_points, label_schema,
            )
            message += note
            direction_data["label_schema"] = label_schema
        if orientation != "none":
            direction_data["oriented_count"] = int(oriented.sum())
            message += f"其中 {direction_data['oriented_count']} 点已按要求定向。"
        las.add_extra_dims(
            [laspy.ExtraBytesParams(name=name, type=dtype) for name, dtype in extra_dimensions]
        )
        las.normal_x, las.normal_y, las.normal_z = point_normals.T
        las.normal_valid = valid_points.astype(np.uint8)
        if orientation != "none":
            las.orientation_valid = oriented.astype(np.uint8)
        out_dir.mkdir(parents=True, exist_ok=True)
        stream = output_file.open("xb")
        try:
            with stream:
                las.write(stream, do_compress=False)
        except Exception:
            output_file.unlink()
            raise
        return {
            "status": "success",
            "output_path": str(output_file),
            "message": message,
            "data": {
                "point_count": n,
                "valid_count": valid_count,
                "invalid_count": n - valid_count,
                "radius": radius,
                "max_nn": max_nn,
                "orientation": orientation,
                "neighbor_shortage_count": int(shortage[inverse].sum()),
                **direction_data,
            },
        }
    except Exception as exc:
        return {"status": "error", "output_path": None, "message": str(exc), "data": None}


if __name__ == "__main__":
    mcp.run(transport="stdio")

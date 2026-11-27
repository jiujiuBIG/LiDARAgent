"""Disk-backed XY tiling for point-cloud normal estimation."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import math
import os
from pathlib import Path

import laspy
import numpy as np

RESULT_DTYPE = np.dtype([
    ("normals", np.float32, (3,)),
    ("valid", np.bool_),
    ("shortage", np.bool_),
    ("oriented", np.bool_),
])
ROW_DTYPE = np.dtype([("row", np.int64), ("xyz", np.int32, (3,))])
READ_CHUNK_SIZE = 250_000


def spool_points(reader, scratch, scales, tile_size, stem_fields):
    tiles = {}
    stems = {}
    offset = 0
    for chunk in reader.chunk_iterator(READ_CHUNK_SIZE):
        records = np.empty(len(chunk), dtype=ROW_DTYPE)
        records["row"] = np.arange(offset, offset + len(chunk))
        records["xyz"] = np.column_stack((chunk.X, chunk.Y, chunk.Z))
        offset += len(chunk)

        xy_tile = np.floor((records["xyz"][:, :2] * scales[:2]) / tile_size)
        if not np.all(np.abs(xy_tile) < 2**63):
            raise ValueError("分块宽度太小，无法为点云建立空间索引。")
        xy_tile = xy_tile.astype(np.int64)
        order = np.lexsort((xy_tile[:, 1], xy_tile[:, 0]))
        edges = np.flatnonzero(np.any(np.diff(xy_tile[order], axis=0), axis=1)) + 1
        for rows in np.split(order, edges):
            tx, ty = (int(v) for v in xy_tile[rows[0]])
            path = tiles.setdefault((tx, ty), scratch / f"tile_{tx}_{ty}.bin")
            with path.open("ab") as stream:
                records[rows].tofile(stream)

        if stem_fields:
            class_field, tree_field = stem_fields
            classes = np.asarray(chunk[class_field])
            tree_id = np.asarray(chunk[tree_field])
            stem_rows = np.flatnonzero((classes == 4) & (tree_id > 0))
            if stem_rows.size:
                order = stem_rows[np.argsort(tree_id[stem_rows], kind="stable")]
                edges = np.flatnonzero(tree_id[order[1:]] != tree_id[order[:-1]]) + 1
                for rows in np.split(order, edges):
                    tid = int(tree_id[rows[0]])
                    path = stems.setdefault(tid, scratch / f"stem_{tid}.bin")
                    with path.open("ab") as stream:
                        records[rows].tofile(stream)
    if offset != reader.header.point_count:
        raise ValueError("LAS 实际点数与文件头不一致。")
    return tiles, stems


def halo_coordinates(key, tiles, scales, radius, tile_size, reach):
    tx, ty = key
    lower = np.nextafter(np.array([tx, ty], dtype=float) * tile_size - radius, -np.inf)
    upper = np.nextafter(np.array([tx + 1, ty + 1], dtype=float) * tile_size + radius, np.inf)
    if (2 * reach + 1) ** 2 <= len(tiles):
        nearby = (
            (x, y)
            for x in range(tx - reach, tx + reach + 1)
            for y in range(ty - reach, ty + reach + 1)
        )
    else:
        nearby = (other for other in tiles if max(abs(tx - other[0]), abs(ty - other[1])) <= reach)
    for other in nearby:
        path = tiles.get(other)
        if other == key or path is None:
            continue
        raw = np.fromfile(path, dtype=ROW_DTYPE)["xyz"]
        xy = raw[:, :2] * scales[:2]
        inside = np.all((xy >= lower) & (xy <= upper), axis=1)
        if inside.any():
            yield raw[inside]


def estimate_tiled_normals(
    source, scratch, radius, max_nn, orient_up, tile_size,
    estimate, stem_fields=None, orient_tree=None,
):
    """Compute normals using complete radius halos, with results keyed by input row."""
    scratch = Path(scratch)
    with laspy.open(source) as reader:
        scales = reader.header.scales.copy()
        count = reader.header.point_count
        if count == 0:
            return np.empty(0, dtype=RESULT_DTYPE)
        tiles, stems = spool_points(reader, scratch, scales, tile_size, stem_fields)

    result = np.memmap(scratch / "normals.dat", mode="w+", dtype=RESULT_DTYPE, shape=(count,))
    result["oriented"] = False
    try:
        # Exact reach: floor(radius / tile_size) + 1 covers exact-radius boundary cases
        # without reading an unnecessary extra ring of tiles when radius < tile_size.
        reach = int(math.floor(radius / tile_size)) + 1
        max_workers = max(1, min(os.cpu_count() or 1, 8, len(tiles)))

        def _process_tile(key, path):
            core = np.fromfile(path, dtype=ROW_DTYPE)
            parts = [core["xyz"], *halo_coordinates(key, tiles, scales, radius, tile_size, reach)]
            unique, inverse = np.unique(np.concatenate(parts), axis=0, return_inverse=True)
            queries, remap = np.unique(inverse.reshape(-1)[:len(core)], return_inverse=True)
            normals, valid, shortage = estimate(unique * scales, radius, max_nn, orient_up, queries)
            return core["row"], normals[remap], valid[remap], shortage[remap]

        if max_workers == 1:
            for key, path in tiles.items():
                rows, t_normals, t_valid, t_shortage = _process_tile(key, path)
                result["normals"][rows] = t_normals
                result["valid"][rows] = t_valid
                result["shortage"][rows] = t_shortage
        else:
            tile_iter = iter(tiles.items())
            max_in_flight = max_workers * 2
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                pending = set()
                for _ in range(min(max_in_flight, len(tiles))):
                    key, path = next(tile_iter)
                    pending.add(pool.submit(_process_tile, key, path))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        rows, t_normals, t_valid, t_shortage = fut.result()
                        result["normals"][rows] = t_normals
                        result["valid"][rows] = t_valid
                        result["shortage"][rows] = t_shortage
                        try:
                            key, path = next(tile_iter)
                            pending.add(pool.submit(_process_tile, key, path))
                        except StopIteration:
                            pass

        # Tree files collect all XY tiles so each height band uses a common stem axis.
        for path in stems.values():
            points = np.fromfile(path, dtype=ROW_DTYPE)
            rows = points["row"]
            normals = result["normals"][rows]
            directed = orient_tree(points["xyz"] * scales, normals, result["valid"][rows])
            result["normals"][rows] = normals
            result["oriented"][rows] = directed
        result.flush()
        return result
    except BaseException:
        result._mmap.close()
        raise

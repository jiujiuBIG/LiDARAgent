"""SegmentAnyTree remote GPU 3D individual tree segmentation MCP tool.

Executes the hardened Docker image (`segment-any-tree:fixed` with official
`PointGroup-PAPER.pt` weights and default `radius=8 m`) on the remote GPU host
via SSH/SCP + WSL 2, maps `PredInstance` and `PredSemantic` back to 100% of the
original input points via `source_index`, and writes standard `treeID` outputs
compatible with downstream LiDARAgent ForestParams tools.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import Annotated, Any
from uuid import uuid4

import laspy
import numpy as np
from mcp.server.fastmcp import FastMCP
from pydantic import Field

if __package__:
    from .point_instances import (
        BREAST_HEIGHT_MAX,
        BREAST_HEIGHT_MIN,
        HEIGHT_DIMENSION,
        TREE_INSTANCE_DIMENSION,
        compute_height_above_ground,
    )
else:
    from point_instances import (
        BREAST_HEIGHT_MAX,
        BREAST_HEIGHT_MIN,
        HEIGHT_DIMENSION,
        TREE_INSTANCE_DIMENSION,
        compute_height_above_ground,
    )

mcp = FastMCP("SegmentAnyTreeSegmentation")

DEFAULT_REMOTE_HOST = os.environ.get("SAT_REMOTE_HOST", "admin@100.67.193.74")
DEFAULT_WSL_DISTRO = os.environ.get("SAT_WSL_DISTRO", "Ubuntu-24.04")
DEFAULT_DOCKER_IMAGE = os.environ.get("SAT_DOCKER_IMAGE", "segment-any-tree:fixed")
DEFAULT_REMOTE_WORK_ROOT = os.environ.get("SAT_REMOTE_WORK_ROOT", "/home/lidar/sat_work/mcp_jobs")
DEFAULT_REMOTE_WIN_STAGE = os.environ.get("SAT_REMOTE_WIN_STAGE", "E:/LiDARAgent_deep_trees_20261123/sat_mcp_stage")
DEFAULT_REMOTE_WSL_STAGE = os.environ.get("SAT_REMOTE_WSL_STAGE", "/mnt/e/LiDARAgent_deep_trees_20261123/sat_mcp_stage")

MAX_FLOAT32_EXACT_INT = 2**24  # 16,777,216 (SegmentAnyTree round-trips source_index via float32 PLY)
LOW_DENSITY_WARNING_PTS_M2 = 15.0
LOWRAM_THRESHOLD_POINTS = 5_000_000

LOWRAM_UTM2LOCAL_PY = """\
import argparse, json, os, laspy, numpy as np
from plyfile import PlyData, PlyElement

def process_las(in_path, out_path, json_path):
    las = laspy.read(in_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    mins = [float(x.min()), float(y.min()), float(z.min())]
    with open(json_path, "w") as f:
        json.dump(mins, f)
    has_src = "source_index" in list(las.point_format.dimension_names)
    if has_src:
        arr = np.empty(len(x), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("source_index", "<f4")])
        arr["source_index"] = np.asarray(las["source_index"], dtype=np.float32)
    else:
        arr = np.empty(len(x), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    arr["x"] = (x - mins[0]).astype(np.float32)
    arr["y"] = (y - mins[1]).astype(np.float32)
    arr["z"] = (z - mins[2]).astype(np.float32)
    del las, x, y, z
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input_folder", required=True)
    p.add_argument("-o", "--output_folder", required=True)
    a = p.parse_args()
    os.makedirs(a.output_folder, exist_ok=True)
    for fn in sorted(os.listdir(a.input_folder)):
        if fn.endswith((".las", ".laz")):
            base = os.path.splitext(fn)[0]
            process_las(
                os.path.join(a.input_folder, fn),
                os.path.join(a.output_folder, f"{base}_out.ply"),
                os.path.join(a.output_folder, f"{base}_out_min_values.json"),
            )
"""


def _stage_point_cloud(
    source_path: Path,
    staged_las_path: Path,
    voxel_dedup_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare a local-origin LAS with uint32 `source_index` for exact round-trip mapping.

    Returns `(inverse_indices, stats_dict)` where `inverse_indices` maps each original
    input point to its index in the staged LAS.
    """
    cloud = laspy.read(source_path)
    n_orig = len(cloud.points)
    if n_orig == 0:
        raise ValueError("输入点云为空，无法执行 SegmentAnyTree 分割。")

    x = np.asarray(cloud.x, dtype=np.float64)
    y = np.asarray(cloud.y, dtype=np.float64)
    z = np.asarray(cloud.z, dtype=np.float64)

    extent_x = float(x.max() - x.min())
    extent_y = float(y.max() - y.min())
    extent_z = float(z.max() - z.min())
    area_m2 = max(extent_x * extent_y, 1e-6)
    density_pts_m2 = float(n_orig / area_m2)

    origin = np.array([x.min(), y.min(), z.min()], dtype=np.float64)
    effective_voxel = float(voxel_dedup_m)

    if effective_voxel > 0.0:
        ix = np.maximum(0, np.floor((x - origin[0]) / effective_voxel).astype(np.int64))
        iy = np.maximum(0, np.floor((y - origin[1]) / effective_voxel).astype(np.int64))
        iz = np.maximum(0, np.floor((z - origin[2]) / effective_voxel).astype(np.int64))
        keys = (ix.astype(np.uint64) << 42) | (iy.astype(np.uint64) << 21) | iz.astype(np.uint64)
        uniq_keys, inverse = np.unique(keys, return_inverse=True)
        mask = np.uint64((1 << 21) - 1)
        vx = ((uniq_keys >> 42) & mask).astype(np.float64) * effective_voxel
        vy = ((uniq_keys >> 21) & mask).astype(np.float64) * effective_voxel
        vz = (uniq_keys & mask).astype(np.float64) * effective_voxel
        scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
        distinct = np.column_stack((
            np.round(vx / scales[0]),
            np.round(vy / scales[1]),
            np.round(vz / scales[2]),
        )).astype(np.int64)
    else:
        xyz_int = np.column_stack((cloud.X, cloud.Y, cloud.Z)).astype(np.int64)
        raw_origin = xyz_int.min(axis=0)
        distinct, inverse = np.unique(xyz_int - raw_origin, axis=0, return_inverse=True)
        scales = np.array(cloud.header.scales, dtype=np.float64)
        if len(distinct) > MAX_FLOAT32_EXACT_INT:
            # Automatically switch to 5 cm voxel deduplication (4x finer than SAT's 20 cm grid)
            effective_voxel = 0.05
            ix = np.maximum(0, np.floor((x - origin[0]) / effective_voxel).astype(np.int64))
            iy = np.maximum(0, np.floor((y - origin[1]) / effective_voxel).astype(np.int64))
            iz = np.maximum(0, np.floor((z - origin[2]) / effective_voxel).astype(np.int64))
            keys = (ix.astype(np.uint64) << 42) | (iy.astype(np.uint64) << 21) | iz.astype(np.uint64)
            uniq_keys, inverse = np.unique(keys, return_inverse=True)
            mask = np.uint64((1 << 21) - 1)
            vx = ((uniq_keys >> 42) & mask).astype(np.float64) * effective_voxel
            vy = ((uniq_keys >> 21) & mask).astype(np.float64) * effective_voxel
            vz = (uniq_keys & mask).astype(np.float64) * effective_voxel
            scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
            distinct = np.column_stack((
                np.round(vx / scales[0]),
                np.round(vy / scales[1]),
                np.round(vz / scales[2]),
            )).astype(np.int64)

    n_staged = len(distinct)
    if n_staged > MAX_FLOAT32_EXACT_INT:
        raise ValueError(
            f"去重后点数仍为 {n_staged:,}（超过 float32 精确索引上限 {MAX_FLOAT32_EXACT_INT:,}），"
            "请先裁剪点云或增大 voxel_dedup_m（如 0.10 或 0.20）。"
        )

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = scales
    staged = laspy.LasData(header)
    staged.X, staged.Y, staged.Z = distinct.T
    staged.add_extra_dim(laspy.ExtraBytesParams(name="source_index", type="u4"))
    staged.source_index = np.arange(n_staged, dtype=np.uint32)
    staged_las_path.parent.mkdir(parents=True, exist_ok=True)
    staged.write(staged_las_path)

    stats = {
        "source_points": int(n_orig),
        "staged_points": int(n_staged),
        "voxel_dedup_m": float(effective_voxel),
        "extent_x_m": round(extent_x, 2),
        "extent_y_m": round(extent_y, 2),
        "extent_z_m": round(extent_z, 2),
        "bbox_area_m2": round(area_m2, 2),
        "bbox_density_pts_m2": round(density_pts_m2, 2),
    }
    return inverse.reshape(-1).astype(np.int64), stats


def _remap_and_export(
    source_path: Path,
    sat_output_path: Path,
    inverse: np.ndarray,
    n_staged: int,
    out_las_path: Path,
    tree_csv_path: Path,
    min_instance_points: int = 0,
) -> dict[str, Any]:
    """Map SegmentAnyTree output `PredInstance`/`PredSemantic` back to original points and write LAS + CSV."""
    sat_cloud = laspy.read(sat_output_path)
    dims = set(sat_cloud.point_format.dimension_names)
    if "source_index" not in dims or "PredInstance" not in dims:
        raise ValueError("SegmentAnyTree 输出缺少必需字段 source_index 或 PredInstance。")

    src_idx = np.asarray(sat_cloud["source_index"], dtype=np.int64)
    if len(src_idx) != n_staged or not np.array_equal(np.sort(src_idx), np.arange(n_staged)):
        raise ValueError("SegmentAnyTree 输出的 source_index 未与输入点一一对应。")

    order = np.argsort(src_idx)
    pred_inst_staged = np.asarray(sat_cloud["PredInstance"], dtype=np.int64)[order]
    pred_sem_staged = (
        np.asarray(sat_cloud["PredSemantic"], dtype=np.int64)[order]
        if "PredSemantic" in dims
        else (pred_inst_staged > 0).astype(np.int64)
    )

    pred_inst_orig = pred_inst_staged[inverse].astype(np.int32)
    pred_sem_orig = pred_sem_staged[inverse].astype(np.uint8)

    # Read original cloud, compute true HAG, and zero out ground / non-finite points
    orig_cloud = laspy.read(source_path)
    hag, hag_method = compute_height_above_ground(orig_cloud)
    invalid_mask = ~np.isfinite(hag)
    if "classification" in orig_cloud.point_format.dimension_names:
        invalid_mask |= np.asarray(orig_cloud.classification) == 2
    pred_inst_orig[invalid_mask] = 0

    # Filter small fragments and renumber consecutive instance IDs (1..K, 0=non-tree/ground)
    uniq_ids, counts = np.unique(pred_inst_orig[pred_inst_orig > 0], return_counts=True)
    kept_ids = uniq_ids[counts >= int(min_instance_points)]
    filtered_count = int(len(uniq_ids) - len(kept_ids))

    clean_tree_id = np.zeros(len(pred_inst_orig), dtype=np.int32)
    if len(kept_ids):
        keep_mask = np.isin(pred_inst_orig, kept_ids)
        clean_tree_id[keep_mask] = (np.searchsorted(kept_ids, pred_inst_orig[keep_mask]) + 1).astype(np.int32)

    # Write output LAS preserving original coordinates, header, CRS, and any existing ground-truth treeID
    existing_dims = set(orig_cloud.point_format.dimension_names)
    existing_lower = {name.lower() for name in existing_dims}
    dims_to_write: list[tuple[str, str]] = [
        (TREE_INSTANCE_DIMENSION, "i4"),
        (HEIGHT_DIMENSION, "f8"),
        ("PredInstance", "i4"),
        ("PredSemantic", "u1"),
    ]
    write_compat_treeid = "treeid" not in existing_lower
    if write_compat_treeid:
        dims_to_write.append(("treeID", "i4"))
    for dim_name, dim_type in dims_to_write:
        if dim_name not in existing_dims:
            orig_cloud.add_extra_dim(laspy.ExtraBytesParams(name=dim_name, type=dim_type))
    orig_cloud[TREE_INSTANCE_DIMENSION] = clean_tree_id
    orig_cloud[HEIGHT_DIMENSION] = hag
    orig_cloud["PredInstance"] = clean_tree_id
    orig_cloud["PredSemantic"] = pred_sem_orig
    if write_compat_treeid:
        orig_cloud["treeID"] = clean_tree_id
    orig_cloud.write(out_las_path)

    # Per-tree summary CSV using true per-point HAG (accurate on sloped terrain)
    x = np.asarray(orig_cloud.x, dtype=np.float64)
    y = np.asarray(orig_cloud.y, dtype=np.float64)
    z = np.asarray(orig_cloud.z, dtype=np.float64)
    bh_mask_all = (clean_tree_id > 0) & (hag >= BREAST_HEIGHT_MIN) & (hag <= BREAST_HEIGHT_MAX)

    heights: list[float] = []
    crown_diams: list[float] = []
    trees_with_bh = 0
    with tree_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "treeID", "point_count", "breast_height_points",
            "height_m", "crown_diameter_m", "centroid_x", "centroid_y", "max_z",
        ])
        for tid in range(1, len(kept_ids) + 1):
            m = clean_tree_id == tid
            cnt = int(m.sum())
            bh_cnt = int(np.count_nonzero(m & bh_mask_all))
            if bh_cnt > 0:
                trees_with_bh += 1
            tx, ty, tz, th = x[m], y[m], z[m], hag[m]
            h = float(max(0.0, np.nanmax(th))) if len(th) else 0.0
            diam = float(math.hypot(tx.max() - tx.min(), ty.max() - ty.min()))
            heights.append(h)
            crown_diams.append(diam)
            writer.writerow([
                tid,
                cnt,
                bh_cnt,
                round(h, 3),
                round(diam, 3),
                round(float(tx.mean()), 3),
                round(float(ty.mean()), 3),
                round(float(tz.max()), 3),
            ])

    return {
        "tree_count": int(len(kept_ids)),
        "filtered_small_instances": filtered_count,
        "tree_points": int((clean_tree_id > 0).sum()),
        "nontree_points": int((clean_tree_id == 0).sum()),
        "instance_field": TREE_INSTANCE_DIMENSION,
        "height_field": HEIGHT_DIMENSION,
        "hag_method": hag_method,
        "breast_height_labelled_points": int(np.count_nonzero(bh_mask_all)),
        "trees_with_breast_height_points": trees_with_bh,
        "height_min_m": round(min(heights, default=0.0), 2),
        "height_max_m": round(max(heights, default=0.0), 2),
        "height_mean_m": round(float(np.mean(heights)) if heights else 0.0, 2),
        "crown_diameter_mean_m": round(float(np.mean(crown_diams)) if crown_diams else 0.0, 2),
    }


def _run_remote_segmentanytree(
    local_staged_las: Path,
    local_output_laz: Path,
    *,
    staged_points: int,
    remote_host: str,
    wsl_distro: str,
    docker_image: str,
    timeout_sec: int = 1800,
) -> dict[str, Any]:
    """Upload staged LAS to remote host, execute SegmentAnyTree GPU Docker container, download LAZ."""
    job_id = f"mcp_{uuid4().hex[:12]}"
    win_stage_dir = f"{DEFAULT_REMOTE_WIN_STAGE}/{job_id}"
    wsl_stage_dir = f"{DEFAULT_REMOTE_WSL_STAGE}/{job_id}"
    wsl_work_dir = f"{DEFAULT_REMOTE_WORK_ROOT}/{job_id}"
    ssh_opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]

    # 1. Create remote staging folder & upload staged LAS via SCP
    subprocess.run(
        ["ssh", *ssh_opts, remote_host, f"wsl -d {wsl_distro} -- mkdir -p {wsl_stage_dir}"],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    subprocess.run(
        ["scp", *ssh_opts, str(local_staged_las), f"{remote_host}:{win_stage_dir}/plot.las"],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )

    use_lowram = staged_points >= LOWRAM_THRESHOLD_POINTS
    lowram_setup = ""
    lowram_mount = ""
    if use_lowram:
        lowram_setup = (
            f"cat << 'PYEOF' > {wsl_work_dir}/utm2local_lowram.py\n"
            f"{LOWRAM_UTM2LOCAL_PY}\n"
            "PYEOF\n"
        )
        lowram_mount = (
            f"-v {wsl_work_dir}/utm2local_lowram.py:"
            "/home/nibio/mutable-outside-world/nibio_inference/pipeline_utm2local_parallel.py:ro"
        )

    remote_script = f"""set -euo pipefail
mkdir -p "{wsl_work_dir}/in" "{wsl_work_dir}/out"
cp "{wsl_stage_dir}/plot.las" "{wsl_work_dir}/in/plot.las"
{lowram_setup}
t0=$(date +%s)
docker run --rm --gpus all \\
  --entrypoint bash \\
  -e OMP_NUM_THREADS=8 \\
  {lowram_mount} \\
  -v "{wsl_work_dir}/in:/work/in:ro" \\
  -v "{wsl_work_dir}/out:/work/out" \\
  "{docker_image}" \\
  run_inference.sh /work/in /work/out true \\
  > "{wsl_work_dir}/console.log" 2>&1
t1=$(date +%s)
out_laz=$(find "{wsl_work_dir}/out/final_results" -maxdepth 1 -type f \\( -name "*.laz" -o -name "*.las" \\) | head -n1)
if [ -z "$out_laz" ]; then
  echo "SAT_NO_OUTPUT" >&2
  tail -n 30 "{wsl_work_dir}/console.log" >&2
  exit 1
fi
cp "$out_laz" "{wsl_stage_dir}/plot_out.laz"
cp "{wsl_work_dir}/console.log" "{wsl_stage_dir}/console.log"
echo "ELAPSED_SEC=$((t1 - t0))"
rm -rf "{wsl_work_dir}"
"""

    try:
        # Upload and execute remote runner script via WSL root (handles Docker cleanup cleanly)
        with TemporaryFile() as script:
            script.write(remote_script.encode("utf-8"))
            script.seek(0)
            subprocess.run(
                ["ssh", *ssh_opts, remote_host, f"wsl -d {wsl_distro} -u root -- dd of=/tmp/{job_id}.sh status=none"],
                stdin=script,
                check=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        proc = subprocess.run(
            [
                "ssh",
                *ssh_opts,
                remote_host,
                f"wsl -d {wsl_distro} -u root -- bash /tmp/{job_id}.sh && wsl -d {wsl_distro} -u root -- rm -f /tmp/{job_id}.sh",
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        # Download segmented LAZ
        subprocess.run(
            ["scp", *ssh_opts, f"{remote_host}:{win_stage_dir}/plot_out.laz", str(local_output_laz)],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
    finally:
        # Best-effort cleanup of remote staging directory
        subprocess.run(
            ["ssh", *ssh_opts, remote_host, f"wsl -d {wsl_distro} -u root -- rm -rf {wsl_stage_dir} {wsl_work_dir} /tmp/{job_id}.sh"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )

    remote_elapsed = None
    for line in proc.stdout.splitlines():
        if line.startswith("ELAPSED_SEC="):
            remote_elapsed = int(line.split("=", 1)[1].strip())
    return {
        "job_id": job_id,
        "remote_host": remote_host,
        "docker_image": docker_image,
        "used_lowram_utm2local": use_lowram,
        "remote_inference_sec": remote_elapsed,
    }


@mcp.tool(name="segment_trees_segmentanytree")
async def segment_trees_segmentanytree(
    file_path: str,
    min_instance_points: Annotated[
        int,
        Field(ge=0, description="Minimum points per predicted tree instance (0 keeps all instances)."),
    ] = 0,
    voxel_dedup_m: Annotated[
        float,
        Field(
            ge=0,
            description="Optional voxel deduplication resolution in metres (0 enables automatic 0.05m dedup only when unique points exceed 16.7M).",
        ),
    ] = 0.0,
    output_path: str = "./",
) -> dict:
    """Segment individual 3D trees from LiDAR point clouds using SegmentAnyTree on remote GPU.

    Recommended primary 3D individual tree segmentation tool for medium-to-high density
    UAV, MLS, TLS, and high-density ALS point clouds (>=15 pts/m²). For ultra-sparse ALS
    (<15 pts/m²), prefer segment_trees_pycrown or segment_trees_watershed.
    """
    try:
        src = Path(file_path)
        if not src.is_file() or src.suffix.lower() not in (".las", ".laz"):
            raise ValueError("请提供有效的 LAS 或 LAZ 点云文件。")
        if type(min_instance_points) is not int or min_instance_points < 0:
            raise ValueError("min_instance_points 必须是不小于 0 的整数。")
        if type(voxel_dedup_m) not in (int, float) or not math.isfinite(voxel_dedup_m) or voxel_dedup_m < 0:
            raise ValueError("voxel_dedup_m 必须是不小于 0 的有限数字（米）。")

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_file = out_dir / "segmentanytree_results.zip"
        t_start = time.time()

        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            staged_las = tmp_dir / "staged_plot.las"
            sat_out_laz = tmp_dir / "sat_plot_out.laz"

            inverse, stage_stats = _stage_point_cloud(src, staged_las, voxel_dedup_m=float(voxel_dedup_m))
            remote_meta = _run_remote_segmentanytree(
                staged_las,
                sat_out_laz,
                staged_points=stage_stats["staged_points"],
                remote_host=DEFAULT_REMOTE_HOST,
                wsl_distro=DEFAULT_WSL_DISTRO,
                docker_image=DEFAULT_DOCKER_IMAGE,
            )

            out_cloud_name = f"{src.stem}_segmentanytree.las"
            out_cloud_path = out_dir / out_cloud_name
            tree_csv_path = out_dir / "tree_summary.csv"
            export_stats = _remap_and_export(
                src,
                sat_out_laz,
                inverse,
                stage_stats["staged_points"],
                out_cloud_path,
                tree_csv_path,
                min_instance_points=int(min_instance_points),
            )

            density_warning = None
            if stage_stats["bbox_density_pts_m2"] < LOW_DENSITY_WARNING_PTS_M2:
                density_warning = (
                    f"输入点云密度较低（{stage_stats['bbox_density_pts_m2']:.2f} 点/m² < "
                    f"{LOW_DENSITY_WARNING_PTS_M2:.0f} 点/m²），极稀疏机载场景建议优先使用 "
                    "segment_trees_pycrown 或 segment_trees_watershed。"
                )

            summary_data = {
                "algorithm": "SegmentAnyTree",
                "model_weights": "PointGroup-PAPER.pt (official)",
                "input_file": str(src),
                "segmented_cloud": str(out_cloud_path),
                "tree_summary_csv": str(tree_csv_path),
                "min_instance_points": int(min_instance_points),
                "total_elapsed_sec": round(time.time() - t_start, 2),
                "density_warning": density_warning,
                **stage_stats,
                **export_stats,
                **remote_meta,
            }
            summary_json_path = out_dir / "summary.json"
            summary_json_path.write_text(
                json.dumps(summary_data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(out_cloud_path, arcname=out_cloud_path.name)
                zf.write(tree_csv_path, arcname=tree_csv_path.name)
                zf.write(summary_json_path, arcname=summary_json_path.name)

        message = (
            f"SegmentAnyTree 3D 单木分割完成，共分割出 {summary_data['tree_count']} 棵单木"
            f"（点云密度 {summary_data['bbox_density_pts_m2']:.1f} 点/m²，耗时 {summary_data['total_elapsed_sec']:.1f} s）。"
            f"平均树高 {summary_data['height_mean_m']:.2f} m（最高 {summary_data['height_max_m']:.2f} m），"
            f"平均冠幅直径 {summary_data['crown_diameter_mean_m']:.2f} m，输出已写入标准 treeID 字段。"
        )
        if density_warning:
            message += f" 注意：{density_warning}"

        return {
            "status": "success",
            "output_path": str(zip_file),
            "message": message,
            "data": summary_data,
        }
    except Exception as exc:
        return {"status": "error", "output_path": None, "message": str(exc), "data": None}


if __name__ == "__main__":
    mcp.run(transport="stdio")

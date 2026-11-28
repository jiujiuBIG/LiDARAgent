"""TreeLearn 3D deep-learning individual tree segmentation MCP tool.

Executes the verified official TreeLearn pipeline (`ecker-lab/TreeLearn`,
commit `fd240ce7caa4c444fe3418aca454dc578bc557d4`) on the remote CUDA GPU host
(`admin@100.67.193.74`, Windows 11 + RTX 3070 Ti), supporting both official
released weights (`default` and `small`) with `tau_min=75` by default.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
import time
import uuid
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

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

mcp = FastMCP("TreeLearnSegmentation")

DEFAULT_REMOTE_HOST = os.getenv("TREELEARN_REMOTE_HOST", "admin@100.67.193.74")
DEFAULT_REMOTE_LAB = os.getenv("TREELEARN_REMOTE_LAB", "E:/LiDARAgent_deep_trees_20261123")
DEFAULT_REMOTE_UPSTREAM = os.getenv(
    "TREELEARN_REMOTE_UPSTREAM",
    f"{DEFAULT_REMOTE_LAB}/TreeLearn-fd240ce7caa4c444fe3418aca454dc578bc557d4",
)
DEFAULT_REMOTE_PYTHON = os.getenv(
    "TREELEARN_REMOTE_PYTHON",
    f"{DEFAULT_REMOTE_LAB}/.venv/Scripts/python.exe",
)
DEFAULT_SSH_TIMEOUT_SEC = int(os.getenv("TREELEARN_SSH_TIMEOUT_SEC", "1800"))

WEIGHT_FILES: dict[str, str] = {
    "default": f"{DEFAULT_REMOTE_LAB}/weights/model_weights_20241213_5df2f928.pth",
    "small": f"{DEFAULT_REMOTE_LAB}/weights/model_weights_with_small_20241213.pth",
}

MAX_STAGED_POINTS = 2**24
AUTO_VOXEL_DEDUP_M = 0.05
LOW_DENSITY_WARNING_PTS_M2 = 15.0


def _encode_ps(script: str) -> str:
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _run_remote_ps(remote_host: str, script: str, timeout: int) -> subprocess.CompletedProcess[str]:
    encoded = _encode_ps(script)
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            remote_host,
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _stage_point_cloud(
    source_path: Path,
    staged_las_path: Path,
    *,
    voxel_dedup_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deduplicate and shift input LAS/LAZ to local coordinates for TreeLearn.

    Returns ``(inverse, stats)`` where ``inverse`` maps each original point index
    to its row in ``staged_las_path``.
    """
    cloud = laspy.read(source_path)
    n_source = len(cloud.points)
    if n_source == 0:
        raise ValueError(f"输入点云为空: {source_path}")

    scales = np.asarray(cloud.header.scales, dtype=np.float64)
    xyz_int = np.column_stack((cloud.X, cloud.Y, cloud.Z)).astype(np.int64)
    raw_origin = xyz_int.min(axis=0)
    local_int = xyz_int - raw_origin
    extent_m = local_int.max(axis=0) * scales
    bbox_area_m2 = float(max(extent_m[0], 0.1) * max(extent_m[1], 0.1))
    density_pts_m2 = float(n_source / bbox_area_m2)

    effective_voxel_m = float(voxel_dedup_m)
    if effective_voxel_m > 0.0:
        local_m = local_int * scales
        voxel_keys = np.floor(local_m / effective_voxel_m).astype(np.int64)
        _, first_idx, inverse = np.unique(
            voxel_keys, axis=0, return_index=True, return_inverse=True
        )
        distinct_int = local_int[first_idx]
    else:
        distinct_int, inverse = np.unique(local_int, axis=0, return_inverse=True)
        if len(distinct_int) > MAX_STAGED_POINTS:
            effective_voxel_m = AUTO_VOXEL_DEDUP_M
            local_m = local_int * scales
            voxel_keys = np.floor(local_m / effective_voxel_m).astype(np.int64)
            _, first_idx, inverse = np.unique(
                voxel_keys, axis=0, return_index=True, return_inverse=True
            )
            distinct_int = local_int[first_idx]

    inverse = inverse.reshape(-1)
    n_staged = len(distinct_int)
    if n_staged > MAX_STAGED_POINTS:
        raise ValueError(
            f"去重后点数 ({n_staged:,}) 仍超过单次推理上限 ({MAX_STAGED_POINTS:,})，"
            "请增大 voxel_dedup_m（例如 0.10）或先对点云分块后再分割。"
        )
    if distinct_int.max() > np.iinfo(np.int32).max:
        raise ValueError("点云空间跨度超出 LAS int32 坐标范围，请先裁切为较小的样地。")

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = scales.copy()
    staged = laspy.LasData(header)
    staged.X = distinct_int[:, 0].astype(np.int32)
    staged.Y = distinct_int[:, 1].astype(np.int32)
    staged.Z = distinct_int[:, 2].astype(np.int32)
    staged.write(staged_las_path)

    stats = {
        "source_points": int(n_source),
        "staged_points": int(n_staged),
        "voxel_dedup_m": round(effective_voxel_m, 4),
        "extent_x_m": round(float(extent_m[0]), 2),
        "extent_y_m": round(float(extent_m[1]), 2),
        "extent_z_m": round(float(extent_m[2]), 2),
        "bbox_area_m2": round(bbox_area_m2, 2),
        "bbox_density_pts_m2": round(density_pts_m2, 2),
    }
    return inverse, stats


def _run_remote_treelearn(
    local_staged_las: Path,
    local_output_npz: Path,
    *,
    weight_variant: str = "default",
    tau_min: int = 75,
    remote_host: str = DEFAULT_REMOTE_HOST,
    remote_lab: str = DEFAULT_REMOTE_LAB,
    remote_upstream: str = DEFAULT_REMOTE_UPSTREAM,
    remote_python: str = DEFAULT_REMOTE_PYTHON,
    timeout_sec: int = DEFAULT_SSH_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Transfer staged LAS to remote Windows GPU host, run TreeLearn, retrieve NPZ."""
    weight_path = WEIGHT_FILES.get(weight_variant)
    if not weight_path:
        raise ValueError(f"不支持的权重类型: {weight_variant!r}（仅支持 'default' 或 'small'）")

    job_id = f"tl_mcp_{uuid.uuid4().hex[:12]}"
    remote_job_dir = f"{remote_lab}/mcp_jobs/{job_id}"
    remote_forest_dir = f"{remote_job_dir}/forest"
    remote_input_las = f"{remote_forest_dir}/plot.las"
    remote_output_npz = f"{remote_job_dir}/results/full_forest/plot.npz"

    try:
        # 1. Create remote job directory
        mkdir_res = _run_remote_ps(
            remote_host,
            f"$ErrorActionPreference = 'Stop'; New-Item -ItemType Directory -Force -Path '{remote_forest_dir}' | Out-Null",
            timeout=60,
        )
        if mkdir_res.returncode != 0:
            raise RuntimeError(
                f"远端创建作业目录失败 (code={mkdir_res.returncode}): {mkdir_res.stderr or mkdir_res.stdout}"
            )

        # 2. SCP staged LAS to remote Windows host
        scp_up = subprocess.run(
            [
                "scp",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                str(local_staged_las),
                f"{remote_host}:{remote_input_las}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if scp_up.returncode != 0:
            raise RuntimeError(
                f"上传点云至远端 GPU 主机失败 (code={scp_up.returncode}): {scp_up.stderr or scp_up.stdout}"
            )

        # 3. Generate pipeline.yaml and execute TreeLearn on remote GPU
        run_ps = f"""
$ErrorActionPreference = 'Stop'
$upstream = '{remote_upstream}'
$py = '{remote_python}'
$jobDir = '{remote_job_dir}'
$weight = '{weight_path}'
$forest = '{remote_input_las}'
$tauMin = {int(tau_min)}

$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONUTF8 = '1'

$cfgScript = @"
from pathlib import Path
import yaml
src = Path(r'$upstream') / 'configs' / 'pipeline' / 'pipeline.yaml'
cfg = yaml.safe_load(src.read_text(encoding='utf-8'))
cfg['forest_path'] = Path(r'$forest').resolve().as_posix()
cfg['pretrain'] = Path(r'$weight').resolve().as_posix()
cfg['dataloader']['num_workers'] = 0
cfg['save_cfg'].update(save_formats=['npz'], save_treewise=False, save_pointwise=False, return_type='original')
cfg.setdefault('grouping', {{}})['tau_min'] = $tauMin
out = Path(r'$jobDir') / 'pipeline.yaml'
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
"@

& $py -B -c $cfgScript
if ($LASTEXITCODE -ne 0) {{ throw "TreeLearn config generation failed (exit $LASTEXITCODE)" }}

$t0 = Get-Date
Set-Location -LiteralPath $upstream
$ErrorActionPreference = 'Continue'
& $py -B -u -c "import runpy,torch; torch.manual_seed(42); torch.cuda.set_device(0); runpy.run_path('tools/pipeline/pipeline.py', run_name='__main__')" --config "$jobDir/pipeline.yaml" *> "$jobDir/console.log"
$rc = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
$elapsed = [int]((Get-Date) - $t0).TotalSeconds
if ($rc -ne 0) {{
    $tail = if (Test-Path "$jobDir/console.log") {{ (Get-Content "$jobDir/console.log" -Tail 40) -join "`n" }} else {{ "" }}
    throw "TreeLearn inference failed (exit $rc):`n$tail"
}}
if (-not (Test-Path '{remote_output_npz}')) {{
    throw "TreeLearn finished without producing plot.npz"
}}
Write-Output "TL_ELAPSED_SEC=$elapsed"
"""
        run_res = _run_remote_ps(remote_host, run_ps, timeout=timeout_sec)
        if run_res.returncode != 0:
            tail = (run_res.stderr or run_res.stdout or "")[-2000:]
            if "Found array with 0 sample(s)" in tail and "HDBSCAN" in tail:
                raise RuntimeError(
                    "TreeLearn 未检测到满足垂直树干条件的候选种子点（常见于纯冠层顶部扫描或缺少树干点的点云），"
                    "请改用 segment_trees_segmentanytree。"
                )
            raise RuntimeError(f"远端 TreeLearn GPU 推理失败 (code={run_res.returncode}):\n{tail}")

        remote_sec = None
        for line in (run_res.stdout or "").splitlines():
            if line.startswith("TL_ELAPSED_SEC="):
                try:
                    remote_sec = int(line.split("=", 1)[1].strip())
                except ValueError:
                    pass

        # 4. SCP result NPZ back to local
        scp_down = subprocess.run(
            [
                "scp",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                f"{remote_host}:{remote_output_npz}",
                str(local_output_npz),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if scp_down.returncode != 0 or not local_output_npz.is_file():
            raise RuntimeError(
                f"下载 TreeLearn 推理结果失败 (code={scp_down.returncode}): {scp_down.stderr or scp_down.stdout}"
            )

        return {
            "job_id": job_id,
            "remote_host": remote_host,
            "weight_variant": weight_variant,
            "weight_file": Path(weight_path).name,
            "tau_min": int(tau_min),
            "remote_inference_sec": remote_sec,
        }
    finally:
        # Best-effort cleanup of remote job directory
        try:
            _run_remote_ps(
                remote_host,
                f"Remove-Item -LiteralPath '{remote_job_dir}' -Recurse -Force -ErrorAction SilentlyContinue",
                timeout=30,
            )
        except Exception:
            pass


def _remap_and_export(
    source_path: Path,
    staged_las_path: Path,
    tl_output_npz: Path,
    inverse: np.ndarray,
    out_las_path: Path,
    out_csv_path: Path,
    *,
    min_instance_points: int = 0,
) -> dict[str, Any]:
    """Verify NPZ alignment, remap TreeLearn labels to 100% original points, and export."""
    staged = laspy.read(staged_las_path)
    staged_xyz = np.column_stack(
        (np.asarray(staged.x), np.asarray(staged.y), np.asarray(staged.z))
    )

    with np.load(tl_output_npz) as npz:
        returned_xyz = np.asarray(npz["points"])
        raw_labels = np.asarray(npz["labels"])

    if returned_xyz.shape != staged_xyz.shape or not np.allclose(
        returned_xyz, staged_xyz, rtol=0, atol=0.001
    ):
        raise ValueError("TreeLearn 输出点坐标与输入暂存点云未对齐")
    if (
        raw_labels.shape != (len(staged_xyz),)
        or not np.isfinite(raw_labels).all()
        or not np.array_equal(raw_labels, raw_labels.astype(np.int64))
        or np.any(raw_labels < 0)
    ):
        raise ValueError("TreeLearn 输出包含非法实例标签")

    full_inst_raw = raw_labels.astype(np.int64)[inverse]

    # Read original cloud, compute true HAG, and zero out ground / non-finite points
    src = laspy.read(source_path)
    hag, hag_method = compute_height_above_ground(src)
    invalid_mask = ~np.isfinite(hag)
    if "classification" in src.point_format.dimension_names:
        invalid_mask |= np.asarray(src.classification) == 2
    full_inst_raw[invalid_mask] = 0

    # Filter small instances & renumber to consecutive 1..N (0 = non-tree/ground)
    pos_mask = full_inst_raw > 0
    unique_ids, counts = np.unique(full_inst_raw[pos_mask], return_counts=True)
    kept_ids = [
        int(tid)
        for tid, cnt in zip(unique_ids, counts)
        if int(cnt) >= int(min_instance_points)
    ]
    filtered_small_count = int(len(unique_ids) - len(kept_ids))

    remap_lut: dict[int, int] = {old_id: new_id for new_id, old_id in enumerate(kept_ids, start=1)}
    tree_ids = np.zeros(len(full_inst_raw), dtype=np.int32)
    if kept_ids:
        max_old = int(unique_ids.max())
        lut_arr = np.zeros(max_old + 1, dtype=np.int32)
        for old_id, new_id in remap_lut.items():
            lut_arr[old_id] = new_id
        tree_ids[pos_mask] = lut_arr[full_inst_raw[pos_mask]]

    # Write output LAS preserving original coordinates, CRS, and any existing ground-truth treeID
    existing_dims = set(src.point_format.dimension_names)
    existing_lower = {name.lower() for name in existing_dims}
    dims_to_write: list[tuple[str, str]] = [
        (TREE_INSTANCE_DIMENSION, "i4"),
        (HEIGHT_DIMENSION, "f8"),
    ]
    write_compat_treeid = "treeid" not in existing_lower
    if write_compat_treeid:
        dims_to_write.append(("treeID", "i4"))
    for dim_name, dim_type in dims_to_write:
        if dim_name not in existing_dims:
            src.add_extra_dim(laspy.ExtraBytesParams(name=dim_name, type=dim_type))
    src[TREE_INSTANCE_DIMENSION] = tree_ids
    src[HEIGHT_DIMENSION] = hag
    if write_compat_treeid:
        src["treeID"] = tree_ids
    src.write(out_las_path)

    # Compute per-tree summary statistics using true per-point HAG (accurate on slopes)
    x = np.asarray(src.x, dtype=np.float64)
    y = np.asarray(src.y, dtype=np.float64)
    z = np.asarray(src.z, dtype=np.float64)
    bh_mask_all = (tree_ids > 0) & (hag >= BREAST_HEIGHT_MIN) & (hag <= BREAST_HEIGHT_MAX)

    tree_rows: list[dict[str, Any]] = []
    heights: list[float] = []
    crown_diams: list[float] = []
    trees_with_bh = 0
    for new_id in range(1, len(kept_ids) + 1):
        mask = tree_ids == new_id
        bh_cnt = int(np.count_nonzero(mask & bh_mask_all))
        if bh_cnt > 0:
            trees_with_bh += 1
        tx, ty, tz, th = x[mask], y[mask], z[mask], hag[mask]
        z_top = float(tz.max())
        z_min = float(tz.min())
        height_m = float(max(0.0, np.nanmax(th))) if len(th) else 0.0
        span_x = float(tx.max() - tx.min())
        span_y = float(ty.max() - ty.min())
        crown_diam_m = 0.5 * (span_x + span_y)
        top_idx = int(np.argmax(tz))
        heights.append(height_m)
        crown_diams.append(crown_diam_m)
        tree_rows.append(
            {
                "treeID": new_id,
                "point_count": int(mask.sum()),
                "breast_height_points": bh_cnt,
                "x_top": round(float(tx[top_idx]), 3),
                "y_top": round(float(ty[top_idx]), 3),
                "z_top": round(z_top, 3),
                "z_min": round(z_min, 3),
                "height_m": round(height_m, 2),
                "crown_span_x_m": round(span_x, 2),
                "crown_span_y_m": round(span_y, 2),
                "crown_diameter_m": round(crown_diam_m, 2),
            }
        )

    with out_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "treeID",
                "point_count",
                "breast_height_points",
                "x_top",
                "y_top",
                "z_top",
                "z_min",
                "height_m",
                "crown_span_x_m",
                "crown_span_y_m",
                "crown_diameter_m",
            ],
        )
        writer.writeheader()
        writer.writerows(tree_rows)

    tree_points = int(np.count_nonzero(tree_ids > 0))
    return {
        "tree_count": len(kept_ids),
        "filtered_small_instances": filtered_small_count,
        "tree_points": tree_points,
        "nontree_points": int(len(tree_ids) - tree_points),
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


@mcp.tool(name="segment_trees_treelearn")
async def segment_trees_treelearn(
    file_path: Annotated[
        str,
        Field(description="输入森林点云文件路径（.las 或 .laz）。适合温带树干清晰、高密度 UAV/MLS/TLS 林分。"),
    ],
    output_path: Annotated[
        str,
        Field(description="输出目录路径，结果将打包保存为 treelearn_results.zip。"),
    ] = "./results",
    weight_variant: Annotated[
        str,
        Field(
            description="官方权重选择：'default'（默认，高精度、误检少，适合常规成熟林）或 'small'（增强 <10 m 低矮小树检出，误检略增）。"
        ),
    ] = "default",
    tau_min: Annotated[
        int,
        Field(description="HDBSCAN 最小成簇体素数（默认 75，经验证较上游 50 精度更高，必须 >= 2）。"),
    ] = 75,
    min_instance_points: Annotated[
        int,
        Field(description="可选：过滤点数少于该阈值的微小碎片实例（0 表示保留全部预测实例）。"),
    ] = 0,
    voxel_dedup_m: Annotated[
        float,
        Field(description="可选体素去重分辨率（米）。0 表示仅做精确重复点去重；点数超过 2^24 时自动启用 0.05 m 体素去重并完整回填原始点。"),
    ] = 0.0,
) -> dict[str, Any]:
    """基于 TreeLearn 官方 3D 深度学习模型执行单木实例分割（远程 RTX 3070 Ti GPU 推理）。

    适用场景：温带针叶/阔叶林中树干结构清晰的高密度 UAV/MLS/TLS 点云，精度高、边界干净；
    若需更强的跨平台泛化能力（含林下复杂结构），推荐 `segment_trees_segmentanytree`；
    若点云稀疏（< 15 点/m²，如低密度 ALS），推荐 `segment_trees_pycrown` 或 `segment_trees_watershed`。

    输出：包含带标准 `treeID` 字段的分割点云（*_treelearn.las）、单木汇总表（tree_summary.csv）
    和元数据（summary.json）的 ZIP 压缩包。
    """
    t0 = time.time()
    try:
        if weight_variant not in WEIGHT_FILES:
            raise ValueError(f"weight_variant 必须为 'default' 或 'small'，收到: {weight_variant!r}")
        if int(tau_min) < 2:
            raise ValueError("tau_min 必须 >= 2")
        if int(min_instance_points) < 0:
            raise ValueError("min_instance_points 不能为负数")
        if float(voxel_dedup_m) < 0:
            raise ValueError("voxel_dedup_m 不能为负数")

        source = Path(file_path).resolve()
        if not source.is_file():
            raise ValueError(f"输入文件不存在: {file_path}")
        if source.suffix.lower() not in (".las", ".laz"):
            raise ValueError("TreeLearn 仅支持 .las 或 .laz 格式的 3D 点云文件。")

        out_dir = Path(output_path).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_las = out_dir / f"{source.stem}_treelearn.las"
        out_csv = out_dir / "tree_summary.csv"
        out_json = out_dir / "summary.json"
        out_zip = out_dir / "treelearn_results.zip"

        with TemporaryDirectory(prefix="tl_mcp_") as tmp_str:
            tmp_dir = Path(tmp_str)
            staged_las = tmp_dir / "plot.las"
            tl_out_npz = tmp_dir / "plot.npz"

            inverse, stage_stats = _stage_point_cloud(
                source, staged_las, voxel_dedup_m=float(voxel_dedup_m)
            )
            remote_stats = _run_remote_treelearn(
                staged_las,
                tl_out_npz,
                weight_variant=weight_variant,
                tau_min=int(tau_min),
            )
            export_stats = _remap_and_export(
                source,
                staged_las,
                tl_out_npz,
                inverse,
                out_las,
                out_csv,
                min_instance_points=int(min_instance_points),
            )

        elapsed_sec = round(time.time() - t0, 2)
        density_warning = None
        if stage_stats["bbox_density_pts_m2"] < LOW_DENSITY_WARNING_PTS_M2:
            density_warning = (
                f"检测到点云密度偏低（{stage_stats['bbox_density_pts_m2']:.1f} 点/m² < "
                f"{LOW_DENSITY_WARNING_PTS_M2:.0f} 点/m²），TreeLearn 3D 分割在稀疏点云上容易漏检，"
                "建议改用 segment_trees_pycrown 或 segment_trees_watershed。"
            )

        summary_data: dict[str, Any] = {
            "algorithm": "TreeLearn",
            "model_commit": "fd240ce7caa4c444fe3418aca454dc578bc557d4",
            "input_file": str(source),
            "segmented_cloud": str(out_las),
            "tree_summary_csv": str(out_csv),
            "min_instance_points": int(min_instance_points),
            "total_elapsed_sec": elapsed_sec,
            "density_warning": density_warning,
            **stage_stats,
            **export_stats,
            **remote_stats,
        }
        out_json.write_text(
            json.dumps(summary_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_las, arcname=out_las.name)
            zf.write(out_csv, arcname=out_csv.name)
            zf.write(out_json, arcname=out_json.name)

        message = (
            f"TreeLearn（{weight_variant} 权重，tau_min={int(tau_min)}）3D 单木分割完成，"
            f"共分割出 {export_stats['tree_count']} 棵单木"
            f"（点云密度 {stage_stats['bbox_density_pts_m2']:.1f} 点/m²，耗时 {elapsed_sec:.1f} s）。"
            f"平均树高 {export_stats['height_mean_m']:.2f} m（最高 {export_stats['height_max_m']:.2f} m），"
            f"平均冠幅直径 {export_stats['crown_diameter_mean_m']:.2f} m，输出已写入标准 treeID 字段。"
        )
        if density_warning:
            message += f"\n注意：{density_warning}"

        return {
            "status": "success",
            "output_path": str(out_zip),
            "message": message,
            "data": summary_data,
        }
    except Exception as exc:
        return {
            "status": "error",
            "output_path": None,
            "message": f"TreeLearn 单木分割失败: {exc}",
            "data": None,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")

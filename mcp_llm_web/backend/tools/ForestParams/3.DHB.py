import os
import sys
import subprocess

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import pandas as pd
import numpy as np
import laspy
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import HEIGHT_ABOVE_GROUND_FIELDS, read_point_cloud
from tools.Segmentation.point_instances import BREAST_HEIGHT_MAX, BREAST_HEIGHT_MIN
from tools.Preprocessing.utils import clean_point_cloud
from tools.Preprocessing.write import dataframe_to_las
from scipy.optimize import least_squares


def has_treeid(file_path: str) -> bool:
    """检查点云文件是否包含 treeid 或 PredInstance 字段"""
    try:
        las = laspy.read(file_path)
        return (
            hasattr(las, "treeid")
            or hasattr(las, "PredInstance")
            or hasattr(las, "treeID")
            or hasattr(las, "TreeID")
        )
    except:
        return False


def run_tree_segmentation(file_path: str, output_path: str) -> str:
    """运行单木分割，返回带有 treeid 的点云文件路径"""
    import torch

    print(f"🔍 检测到点云缺少 treeid，正在自动执行单木分割...")

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 读取点云
    df = read_point_cloud(file_path)
    df_clean = clean_point_cloud(df)

    points = df_clean[["x", "y", "z"]].values.astype(np.float32)
    save_dict = {"points": points}
    if "classification" in df_clean.columns:
        save_dict["classification"] = df_clean["classification"].values

    pth_path = os.path.join(output_path, f"{base_name}.pth")
    torch.save(save_dict, pth_path)

    # 运行语义分割
    semantic_pth = os.path.join(output_path, f"{base_name}_semantic.pth")
    cmd = [
        "/home/may/software/anaconda/install/envs/pg/bin/python",
        "/mnt/Huxiaomei2/May/Huxiaomei/Graduation/H/Graduation/code/Superpoint_HeadMap_Transformer/test_semantic_mcp.py",
        pth_path,
        semantic_pth,
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    # 生成 CHM
    chm_path = os.path.join(output_path, f"{base_name}_CHM.npy")
    cmd = [
        "/home/may/software/anaconda/install/envs/yolo/bin/python",
        "/mnt/Huxiaomei2/May/Huxiaomei/yolo_tree/crop_2d_bbox_to_pointcloud_prepare.py",
        semantic_pth,
        chm_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    # YOLO 检测
    yolo_dir = os.path.join(output_path, f"{base_name}_yolo")
    cmd = [
        "/home/may/software/anaconda/install/envs/yolo/bin/python",
        "/mnt/Huxiaomei2/May/Huxiaomei/yolo_tree/crop_2d_bbox_to_pointcloud_mcp.py",
        chm_path,
        yolo_dir,
        semantic_pth,
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    # 实例分割
    bboxfiles = [f for f in os.listdir(yolo_dir) if f.endswith(".laz")]
    instance_dir = os.path.join(output_path, f"{base_name}_yolo_instance")
    os.makedirs(instance_dir, exist_ok=True)

    pc_all = []
    cl_all = []

    for i, file in enumerate(bboxfiles):
        input_path = os.path.join(yolo_dir, file)
        output_file = file.replace(".laz", "_instance.laz")
        output_instance = os.path.join(instance_dir, output_file)

        cmd = [
            "/home/may/software/anaconda/install/envs/pg/bin/python",
            "/mnt/Huxiaomei2/May/Huxiaomei/Graduation/H/Graduation/code/Superpoint_HeadMap_Transformer/test_instance_mcp.py",
            input_path,
            output_instance,
        ]
        subprocess.run(cmd, capture_output=True, text=True)

        try:
            las = laspy.read(output_instance)
            xyz = np.vstack([las.x, las.y, las.z]).T
            cls = las.instance_pred
            index = cls == 1
            if np.sum(index) > 0:
                xyz = xyz[index]
                cls = cls[index] * (i + 1)
                pc_all.append(xyz)
                cl_all.append(cls)
        except:
            continue

    if len(pc_all) == 0:
        raise RuntimeError("未能检测到任何树木")

    # 转移 treeid
    source_pc_all = np.vstack(pc_all)
    source_cl_all = np.hstack(cl_all)

    target_pth = torch.load(semantic_pth, weights_only=False)
    target_points = target_pth["points"]
    target_classification = target_pth["semantic_pred"]

    output_las = os.path.join(output_path, f"{base_name}_instance.las")

    from scipy.spatial import cKDTree

    tree = cKDTree(target_points)
    _, indices = tree.query(source_pc_all)
    treeid = np.zeros(len(target_points), dtype=np.uint8)
    treeid[indices] = source_cl_all

    las = laspy.read(file_path)
    las.treeid = treeid
    las.write(output_las)

    print(f"✅ 单木分割完成，检测到 {len(np.unique(source_cl_all))} 棵树")
    return output_las


def fit_circle_least_squares(xy: np.ndarray):
    n = xy.shape[0]
    if n < 3:
        return None, None, None, None, n

    x = xy[:, 0]
    y = xy[:, 1]

    mx = x.mean()
    my = y.mean()
    X = x - mx
    Y = y - my
    Z = X**2 + Y**2

    A = np.column_stack((2 * X, 2 * Y, np.ones_like(X)))
    b = Z

    try:
        params, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None, None, n
    if rank < 3:
        return None, None, None, None, n

    a, b_param, c = params
    cx_local = a
    cy_local = b_param
    r_sq = cx_local**2 + cy_local**2 + c
    if r_sq <= 0:
        return None, None, None, None, n

    r = np.sqrt(r_sq)
    cx = cx_local + mx
    cy = cy_local + my

    radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residuals = radii - r

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((radii - radii.mean()) ** 2)
    r2 = 1.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    return cx, cy, r, r2, n


MIN_DBH_POINTS = 20
DBH_INLIER_TOLERANCE_M = 0.03
DBH_MIN_RADIUS_M = 0.04
DBH_MAX_RADIUS_M = 5.0
BREAST_HEIGHT_M = 1.3
STEM_BAND_HALF_HEIGHT_M = 0.5
STEM_BAND_INLIER_TOLERANCE_M = 0.04
MIN_STEM_SIDE_POINTS = 8
MAX_STEM_EVIDENCE_GAP_M = 0.55


def fit_stem_section(xy: np.ndarray):
    """Fit directly to returns, retaining partial arcs and rejecting weak circles."""
    origin = xy.mean(axis=0)
    points = xy - origin
    rng = np.random.default_rng(0)
    sample = points if len(points) <= 1024 else points[rng.choice(len(points), 1024, replace=False)]
    triples = sample[rng.integers(len(sample), size=(256, 3))]
    a = triples[:, 1] - triples[:, 0]
    b = triples[:, 2] - triples[:, 0]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    stable = np.abs(cross) > 1e-4
    a, b, cross = a[stable], b[stable], cross[stable]
    aa, bb = (a * a).sum(axis=1), (b * b).sum(axis=1)
    offsets = np.column_stack((aa * b[:, 1] - bb * a[:, 1], bb * a[:, 0] - aa * b[:, 0]))
    offsets /= 2 * cross[:, None]
    centres = offsets + triples[stable, 0]
    radii = np.linalg.norm(offsets, axis=1)
    cx, cy, radius, _, _ = fit_circle_least_squares(sample)
    if radius is not None:
        centres = np.vstack((centres, [cx, cy]))
        radii = np.append(radii, radius)
    valid = (radii >= DBH_MIN_RADIUS_M) & (radii <= DBH_MAX_RADIUS_M)
    centres, radii = centres[valid], radii[valid]
    if not len(radii):
        return None

    residuals = np.abs(np.linalg.norm(sample[None, :, :] - centres[:, None, :], axis=2) - radii[:, None])
    # Cap the loss so distant returns do not dominate the candidate selection.
    best = np.argmin(np.minimum(residuals, DBH_INLIER_TOLERANCE_M).mean(axis=1))
    start = np.r_[centres[best], radii[best]]

    def radial_error(params, data):
        return np.linalg.norm(data - params[:2], axis=1) - params[2]

    minimum_support = max(12, int(np.ceil(len(points) * 0.55)))
    consensus = np.abs(radial_error(start, points)) <= DBH_INLIER_TOLERANCE_M
    if np.count_nonzero(consensus) < minimum_support:
        return None
    fit = least_squares(
        radial_error, start, args=(points[consensus],), loss="soft_l1", f_scale=0.01,
        bounds=([-np.inf, -np.inf, DBH_MIN_RADIUS_M], [np.inf, np.inf, DBH_MAX_RADIUS_M]),
    )
    errors = radial_error(fit.x, points)
    support = np.abs(errors) <= DBH_INLIER_TOLERANCE_M
    count = int(np.count_nonzero(support))
    if not fit.success or count < minimum_support:
        return None

    relative = points[support] - fit.x[:2]
    angles = np.sort(np.arctan2(relative[:, 1], relative[:, 0]))
    arc = 2 * np.pi - np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max()
    if arc < np.pi / 2:
        return None

    _, _, _, score, _ = fit_circle_least_squares(points[support])
    return {
        "circle_id": 0, "n_points": count,
        "cx": float(fit.x[0] + origin[0]), "cy": float(fit.x[1] + origin[1]),
        "radius": float(fit.x[2]), "DBH": float(2 * fit.x[2]), "r2": score, "vote": np.nan,
        "inlier_fraction": count / len(points),
        "fit_rmse_m": float(np.sqrt(np.mean(errors[support] ** 2))),
        "arc_degrees": float(np.degrees(arc)),
    }


def fit_dbh_sections(xyz, treeid):
    clip = (xyz[:, 2] >= BREAST_HEIGHT_MIN) & (xyz[:, 2] <= BREAST_HEIGHT_MAX)
    xy, ids = xyz[clip, :2], treeid[clip]
    order = np.argsort(ids, kind="stable")
    ids, starts = np.unique(ids[order], return_index=True)
    results = []
    for tree_id, start, stop in zip(ids, starts, np.r_[starts[1:], len(order)]):
        if tree_id <= 0 or stop - start < MIN_DBH_POINTS:
            continue
        circle = fit_stem_section(xy[order[start:stop]])
        if circle is not None:
            results.append({"treeid": tree_id, **circle})
    return pd.DataFrame(results)


def fit_bracketed_stem(xyz: np.ndarray):
    """Estimate a local cylinder through the reference breast height."""
    below = xyz[:, 2] < BREAST_HEIGHT_MIN
    above = xyz[:, 2] > BREAST_HEIGHT_MAX
    if min(np.count_nonzero(below), np.count_nonzero(above)) < MIN_STEM_SIDE_POINTS:
        return None

    origin = np.median(xyz[:, :2], axis=0)
    xy = xyz[:, :2] - origin
    height = xyz[:, 2] - BREAST_HEIGHT_M
    slope = (np.median(xy[above], axis=0) - np.median(xy[below], axis=0))
    slope /= np.median(height[above]) - np.median(height[below])
    slope = np.clip(slope, -0.6, 0.6)

    def radial_error(params):
        slope = params[3:]
        local = xy - params[:2] - height[:, None] * slope
        distance_sq = (local * local).sum(axis=1) - (local @ slope)**2 / (1 + slope @ slope)
        return np.sqrt(np.maximum(distance_sq, 0)) - params[2]

    tolerance = STEM_BAND_INLIER_TOLERANCE_M
    bounds = ([-np.inf, -np.inf, DBH_MIN_RADIUS_M, -0.65, -0.65],
              [np.inf, np.inf, DBH_MAX_RADIUS_M, 0.65, 0.65])
    candidates = []
    for axis in (slope, np.zeros(2)):
        flattened = xy - height[:, None] * axis
        section = fit_stem_section(flattened)
        if section:
            cx, cy, radius = section["cx"], section["cy"], section["radius"]
        else:
            cx, cy, radius, _, _ = fit_circle_least_squares(flattened)
        if radius is None:
            continue
        start = np.r_[cx, cy, np.clip(radius, DBH_MIN_RADIUS_M, DBH_MAX_RADIUS_M), axis]
        fit = least_squares(radial_error, start, loss="cauchy", f_scale=0.02, bounds=bounds)
        if not fit.success:
            continue

        errors = radial_error(fit.x)
        support = np.abs(errors) <= tolerance
        count = int(np.count_nonzero(support))
        lower_support, upper_support = support & below, support & above
        supported_sides = min(np.count_nonzero(lower_support), np.count_nonzero(upper_support))
        side_residual = max(np.median(np.abs(errors[group])) for group in (below, above))
        if count < 0.55 * len(xyz) or supported_sides < MIN_STEM_SIDE_POINTS or side_residual > tolerance:
            continue
        if height[upper_support].min() - height[lower_support].max() > MAX_STEM_EVIDENCE_GAP_M:
            continue
        unleaned = xy[support] - fit.x[:2] - height[support, None] * fit.x[3:]
        angles = np.sort(np.arctan2(unleaned[:, 1], unleaned[:, 0]))
        arc = 2 * np.pi - np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max()
        if arc < np.pi / 2:
            continue
        rmse = float(np.sqrt(np.mean(errors[support] ** 2)))
        score = float(np.minimum(np.abs(errors), tolerance).mean())
        candidates.append((score, {
            "circle_id": 0, "n_points": count,
            "cx": float(fit.x[0] + origin[0]), "cy": float(fit.x[1] + origin[1]),
            "radius": float(fit.x[2]), "DBH": float(2 * fit.x[2]), "r2": np.nan, "vote": np.nan,
            "inlier_fraction": count / len(xyz), "fit_rmse_m": rmse,
            "arc_degrees": float(np.degrees(arc)), "fit_method": "bracketed_stem_cylinder",
        }))
    return min(candidates, key=lambda entry: entry[0])[1] if candidates else None


def fit_missing_dbh_from_stem_band(xyz, tree_ids, measured_ids):
    in_band = np.isfinite(xyz).all(axis=1) & (np.abs(xyz[:, 2] - BREAST_HEIGHT_M) <= STEM_BAND_HALF_HEIGHT_M)
    selected = np.flatnonzero(in_band & ~np.isin(tree_ids, measured_ids))
    selected = selected[np.argsort(tree_ids[selected], kind="stable")]
    ids, starts = np.unique(tree_ids[selected], return_index=True)
    results = []
    for tree_id, start, stop in zip(ids, starts, np.r_[starts[1:], len(selected)]):
        fit = fit_bracketed_stem(xyz[selected[start:stop]])
        if fit is not None:
            results.append({"treeid": tree_id, **fit})
    return pd.DataFrame(results)


# Initialize MCP server
mcp = FastMCP("ForestDBHExtractor")


@mcp.tool(
    name="extract_forest_dbh",
    description=(
        "Compute the Diameter at Breast Height (DBH) for each tree in a forest point cloud. "
        "Provide an instance-labelled LAS/LAZ with an explicit height-above-ground field in metres. "
        "Set z_is_normalized=true only when the input Z is confirmed to be height above ground in metres. "
        "Fits a 1.2–1.4 m trunk section, with a local cylinder fallback if observations bracket 1.3 m. "
        "The CSV identifies the method used for each measurement."
    ),
)
async def extract_forest_dbh(
    file_path: str, output_path: str = "./", z_is_normalized: bool = False
) -> dict:
    """
    Compute DBH from instance-labelled points with height above ground in metres.
    """

    # =============== 1️⃣ Read point cloud ===============
    try:
        data = read_point_cloud(file_path, is_treeid=True)
        data = clean_point_cloud(data)
    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"读取点云文件失败: {e}",
            "data": None,
        }

    if "treeid" not in data:
        return {"status": "error", "output_path": None, "message": "请提供带单木编号的 LAS 或 LAZ。", "data": None}
    height_field = next((field for field in HEIGHT_ABOVE_GROUND_FIELDS if field in data), None)
    if height_field is None and z_is_normalized is not True:
        return {
            "status": "error", "output_path": None,
            "message": (
                "缺少离地高度字段。请提供 HeightAboveGround。"
                "如果输入 Z 已按地面归一化，可在消息开头写「确认 Z 已按地面归一化」，"
                "或直接调用工具时设置 z_is_normalized=true。"
            ),
            "data": None,
        }
    height_reference = height_field or "normalized_z"
    points = data[data["treeid"] > 0]
    if points.empty:
        return {"status": "error", "output_path": None, "message": "点云没有可用于胸径测量的单木标签。", "data": None}

    xyz = points[["x", "y", height_field or "z"]].to_numpy(dtype=np.float64)
    tree_ids = points["treeid"].to_numpy()
    valid_height = np.isfinite(xyz).all(axis=1)
    in_slice = valid_height & (xyz[:, 2] >= BREAST_HEIGHT_MIN) & (xyz[:, 2] <= BREAST_HEIGHT_MAX)
    per_tree = pd.DataFrame({
        "treeid": tree_ids, "valid_height_points": valid_height, "breast_height_points": in_slice,
    }).groupby("treeid", as_index=False).sum()

    eligible = per_tree.loc[per_tree["breast_height_points"] >= MIN_DBH_POINTS, "treeid"]
    fit_points = in_slice & np.isin(tree_ids, eligible.to_numpy())
    results = pd.DataFrame(columns=["treeid", "DBH", "fit_method"])
    if fit_points.any():
        fitted = fit_dbh_sections(xyz=xyz[fit_points], treeid=tree_ids[fit_points])
        if not fitted.empty:
            results = fitted.assign(fit_method="breast_height_section_circle")
    fallback = fit_missing_dbh_from_stem_band(xyz, tree_ids, results["treeid"].to_numpy())
    if not fallback.empty:
        results = pd.concat([results, fallback], ignore_index=True) if not results.empty else fallback
    per_tree = per_tree.merge(results, how="left", on="treeid")
    measured = pd.to_numeric(per_tree["DBH"], errors="coerce").where(lambda values: np.isfinite(values) & (values > 0))
    per_tree["DBH"] = measured
    per_tree["height_reference"] = height_reference
    per_tree["measurement_status"] = np.select(
        [measured.notna(), per_tree["valid_height_points"].eq(0), per_tree["breast_height_points"].eq(0),
         per_tree["breast_height_points"].lt(MIN_DBH_POINTS)],
        ["measured", "missing_height_above_ground", "no_breast_height_points", "insufficient_breast_height_points"],
        default="fit_failed",
    )
    os.makedirs(output_path, exist_ok=True)
    output_csv = os.path.join(output_path, "forest_dbh_results.csv")
    per_tree.to_csv(output_csv, index=False)

    diameters = measured.dropna()
    missed = len(per_tree) - len(diameters)
    summary = {
        "tree_count": len(diameters), "detected_trees": len(diameters), "unmeasured_tree_count": missed,
        "height_reference": height_reference,
        "measurement_status_counts": per_tree["measurement_status"].value_counts().to_dict(),
        "fit_method_counts": per_tree["fit_method"].value_counts().to_dict(),
    }
    if not len(diameters):
        return {
            "status": "warning", "output_path": output_csv,
            "message": "胸高截面及邻近树干点无法支持胸径拟合。CSV 已逐树记录原因。", "data": summary,
        }
    summary.update(dbh_mean=round(float(diameters.mean()), 2), dbh_max=round(float(diameters.max()), 2),
                   dbh_min=round(float(diameters.min()), 2))
    return {
        "status": "success", "output_path": output_csv,
        "message": f"已测量 {len(diameters)} 棵树的胸径，平均 {diameters.mean():.2f} 米。{missed} 棵未测得，原因见 CSV。",
        "data": summary,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

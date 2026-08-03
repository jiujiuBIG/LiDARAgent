import numpy as np
import pandas as pd
import laspy
import subprocess
import torch
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.utils import clean_point_cloud
import os


def has_treeid(file_path: str) -> bool:
    """检查点云文件是否包含 treeid 或 PredInstance 字段"""
    try:
        las = laspy.read(file_path)
        return (
            hasattr(las, "treeid")
            or hasattr(las, "treeID")
            or hasattr(las, "TreeID")
            or hasattr(las, "PredInstance")
        )
    except:
        return False


def run_tree_segmentation(file_path: str, output_path: str) -> str:
    """运行单木分割，返回带有 treeid 的点云文件路径"""
    print(f"检测点云缺少 treeid，自动执行单木分割...")

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    df = read_point_cloud(file_path)
    df_clean = clean_point_cloud(df)

    points = df_clean[["x", "y", "z"]].values.astype(np.float32)
    save_dict = {"points": points}
    if "classification" in df_clean.columns:
        save_dict["classification"] = df_clean["classification"].values

    pth_path = os.path.join(output_path, f"{base_name}.pth")
    torch.save(save_dict, pth_path)

    semantic_pth = os.path.join(output_path, f"{base_name}_semantic.pth")
    cmd = [
        "/home/may/software/anaconda/install/envs/pg/bin/python",
        "/mnt/Huxiaomei2/May/Huxiaomei/Graduation/H/Graduation/code/Superpoint_HeadMap_Transformer/test_semantic_mcp.py",
        pth_path,
        semantic_pth,
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    chm_path = os.path.join(output_path, f"{base_name}_CHM.npy")
    cmd = [
        "/home/may/software/anaconda/install/envs/yolo/bin/python",
        "/mnt/Huxiaomei2/May/Huxiaomei/yolo_tree/crop_pointcloud_by_2dbbox.py",
        semantic_pth,
        chm_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: crop_pointcloud_by_2dbbox.py failed: {result.stderr}")

    yolo_dir = os.path.join(output_path, f"{base_name}_yolo")
    cmd = [
        "/home/may/software/anaconda/install/envs/yolo/bin/python",
        "/mnt/Huxiaomei2/May/Huxiaomei/yolo_tree/extract_pointcloud_by_bbox.py",
        chm_path,
        yolo_dir,
        semantic_pth,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: extract_pointcloud_by_bbox.py failed: {result.stderr}")

    if not os.path.exists(yolo_dir):
        raise RuntimeError(f"YOLO输出目录不存在: {yolo_dir}。YOLO处理步骤可能失败了，请检查前面的步骤。")

    bboxfiles = [f for f in os.listdir(yolo_dir) if f.endswith(".laz")]
    if len(bboxfiles) == 0:
        raise RuntimeError(f"YOLO未检测到任何树木边界框文件。输入点云可能不包含树木。")
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

    from scipy.spatial import cKDTree

    source_pc_all = np.vstack(pc_all)
    source_cl_all = np.hstack(cl_all)

    target_pth = torch.load(semantic_pth, weights_only=False)
    target_points = target_pth["points"]
    target_classification = target_pth["semantic_pred"]

    output_las = os.path.join(output_path, f"{base_name}_instance.las")

    tree = cKDTree(target_points)
    _, indices = tree.query(source_pc_all)
    treeid = np.zeros(len(target_points), dtype=np.uint8)
    treeid[indices] = source_cl_all

    las = laspy.read(file_path)
    las.treeid = treeid
    las.write(output_las)

    print(f"单木分割完成，检测到 {len(np.unique(source_cl_all))} 棵树")
    return output_las


# -------------------------------------------------------------
# 基础高度指标
# -------------------------------------------------------------
def calc_height_metrics(df):
    z = df["z"].values
    metrics = {
        "Q0": np.percentile(z, 0),
        "Q1": np.percentile(z, 1),
        "Q5": np.percentile(z, 5),
        "Q25": np.percentile(z, 25),
        "Q50": np.percentile(z, 50),
        "Q75": np.percentile(z, 75),
        "Q95": np.percentile(z, 95),
        "Q99": np.percentile(z, 99),
        "Q100": np.percentile(z, 100),
        "MH": np.mean(z),
        "MCH": np.mean(z[z > 2]) if np.any(z > 2) else 0,
    }
    return metrics


# -------------------------------------------------------------
# LAI (Leaf Area Index)
# -------------------------------------------------------------
def calc_LAI_metrics(df):
    df = df[df["classification"] != 0]
    if df.empty:
        return np.nan
    voxel_size = 1.0
    df["x_voxel"] = (df["x"] / voxel_size).astype(int)
    df["y_voxel"] = (df["y"] / voxel_size).astype(int)
    df["z_voxel"] = (df["z"] / voxel_size).astype(int)
    voxel_counts = (
        df.groupby(["x_voxel", "y_voxel", "z_voxel"]).size().reset_index(name="count")
    )
    voxel_counts["VAD"] = voxel_counts["count"] / (voxel_size**3)
    VAI = (
        voxel_counts.groupby(["x_voxel", "y_voxel"])["VAD"]
        .sum()
        .reset_index(name="VAI")
    )
    LAI = VAI["VAI"].mean()
    return LAI


# -------------------------------------------------------------
# DGF (Gap Fraction)
# -------------------------------------------------------------
def compute_DGF(df, voxel_xy=1.0, canopy_threshold=2.0):
    df_veg = df[df["classification"] != 0]
    if df_veg.empty:
        return np.nan
    df_veg["x_cell"] = (df_veg["x"] / voxel_xy).astype(int)
    df_veg["y_cell"] = (df_veg["y"] / voxel_xy).astype(int)
    all_cells = (
        df.groupby([(df["x"] / voxel_xy).astype(int), (df["y"] / voxel_xy).astype(int)])
        .size()
        .reset_index()
    )
    all_cells.columns = ["x_cell", "y_cell", "count"]
    canopy_cells = df_veg[df_veg["z"] > canopy_threshold][
        ["x_cell", "y_cell"]
    ].drop_duplicates()
    Ngaps_1m2 = len(all_cells) - len(canopy_cells)
    Ncells_1m2 = len(all_cells)
    DGF = Ngaps_1m2 / Ncells_1m2 if Ncells_1m2 > 0 else np.nan
    return DGF


# -------------------------------------------------------------
# GFP (Gap Fraction Profile)
# -------------------------------------------------------------
def compute_GFP(df, dz=1.0):
    df_veg = df[df["classification"] != 0]
    if df_veg.empty:
        return np.nan
    max_height = df["z"].max()
    z_bins = np.arange(0, max_height + dz, dz)
    df_veg["z_bin"] = pd.cut(df_veg["z"], bins=z_bins, labels=z_bins[:-1])
    layer_counts = df_veg.groupby("z_bin").size().reset_index(name="count")
    total = layer_counts["count"].sum()
    gap_profile = 1 - (layer_counts["count"] / total)
    return gap_profile.mean()  # 取平均值表示整体gap特征


# -------------------------------------------------------------
# Exterior Complexity
# -------------------------------------------------------------
def compute_exterior_complexity(df, res=1.0):
    x, y, z = df["x"].values, df["y"].values, df["z"].values
    if len(z) < 5:
        return {"TopRugosity": np.nan, "RumpleIndex": np.nan}
    # CHM
    x_min, y_min = x.min(), y.min()
    ix = ((x - x_min) / res).astype(int)
    iy = ((y - y_min) / res).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    chm = np.full((ny, nx), np.nan)
    for i, j, h in zip(ix, iy, z):
        if np.isnan(chm[j, i]) or h > chm[j, i]:
            chm[j, i] = h
    chm_valid = chm[~np.isnan(chm)]
    top_rugosity = np.std(chm_valid) if len(chm_valid) > 1 else np.nan

    # Rumple Index
    try:
        tri = Delaunay(np.column_stack([x, y]))
        surface_area = 0.0
        for simplex in tri.simplices:
            pts = np.column_stack([x, y, z])[simplex]
            a = np.linalg.norm(pts[0] - pts[1])
            b = np.linalg.norm(pts[1] - pts[2])
            c = np.linalg.norm(pts[2] - pts[0])
            s = 0.5 * (a + b + c)
            surface_area += np.sqrt(s * (s - a) * (s - b) * (s - c))
        projected_area = MultiPoint(np.column_stack([x, y])).convex_hull.area
        rumple_index = surface_area / projected_area if projected_area > 0 else np.nan
    except:
        rumple_index = np.nan
    return {"TopRugosity": top_rugosity, "RumpleIndex": rumple_index}


# -------------------------------------------------------------
# Interior Complexity
# -------------------------------------------------------------
def compute_interior_complexity(df, grid_size=3.0, voxel_height=1.0):
    z = df["z"].values
    if len(z) < 5:
        return dict.fromkeys(
            ["GiniCoeff", "FHD", "SDH", "CVH", "MSDH", "SDSDH"], np.nan
        )

    # Gini
    z_sorted = np.sort(z)
    n = len(z_sorted)
    mean_z = np.mean(z_sorted)
    GiniCoeff = (np.sum((2 * np.arange(1, n + 1) - n - 1) * z_sorted)) / (n**2 * mean_z)

    # FHD
    bins = np.arange(z.min(), z.max() + voxel_height, voxel_height)
    counts, _ = np.histogram(z, bins=bins)
    p = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    p = p[p > 0]
    FHD = -np.sum(p * np.log(p))

    # SDH / CVH
    SDH = np.std(z)
    MH = np.mean(z)
    CVH = SDH / MH if MH > 0 else np.nan

    # MSDH / SDSDH
    x, y = df["x"].values, df["y"].values
    x_min, y_min = x.min(), y.min()
    ix = ((x - x_min) / grid_size).astype(int)
    iy = ((y - y_min) / grid_size).astype(int)
    grid_dict = {}
    for i, j, h in zip(ix, iy, z):
        grid_dict.setdefault((i, j), []).append(h)
    sd_list = [np.std(v) for v in grid_dict.values() if len(v) > 1]
    MSDH = np.mean(sd_list) if len(sd_list) > 0 else np.nan
    SDSDH = np.std(sd_list) if len(sd_list) > 0 else np.nan

    return {
        "GiniCoeff": GiniCoeff,
        "FHD": FHD,
        "SDH": SDH,
        "CVH": CVH,
        "MSDH": MSDH,
        "SDSDH": SDSDH,
    }


# -------------------------------------------------------------
# Ancillary
# -------------------------------------------------------------
def compute_ancillary_metrics(df):
    x, y = df["x"].values, df["y"].values
    n = len(x)
    if n < 3:
        return {"Area": np.nan, "PointDensity": np.nan}
    area = (x.max() - x.min()) * (y.max() - y.min())
    if area <= 0:
        area = np.nan
    pdensity = n / area if area and not np.isnan(area) else np.nan
    return {"Area": area, "PointDensity": pdensity}


# -------------------------------------------------------------
# 主函数：整合所有指标
# -------------------------------------------------------------
def compute_all_metrics(df):
    df = df.copy()
    
    # 过滤非植被点（classification != 0 表示植被/树木）
    # 如果 classification 全是0，则使用 treeid 来区分
    if "treeid" in df.columns and df["treeid"].max() > 0:
        # 使用 treeid 分组
        df = df[df["treeid"] > 0]
        group_col = "treeid"
    elif "classification" in df.columns:
        # 使用 classification 分组
        df = df[df["classification"] != 0]
        group_col = "classification"
    else:
        # 没有分类信息，将整个点云作为一棵树
        df = df.copy()
        group_col = None
    
    all_results = []
    
    if group_col:
        for treeid, group in df.groupby(group_col):
            res = {"treeid": treeid}
            res.update(calc_height_metrics(group))
            res["LAI"] = calc_LAI_metrics(group)
            res["DGF"] = compute_DGF(group)
            res["GFP_mean"] = compute_GFP(group)
            res.update(compute_exterior_complexity(group))
            res.update(compute_interior_complexity(group))
            res.update(compute_ancillary_metrics(group))
            all_results.append(res)
    else:
        # 整个点云作为一棵树
        res = {"treeid": 1}
        res.update(calc_height_metrics(df))
        res["LAI"] = calc_LAI_metrics(df)
        res["DGF"] = compute_DGF(df)
        res["GFP_mean"] = compute_GFP(df)
        res.update(compute_exterior_complexity(df))
        res.update(compute_interior_complexity(df))
        res.update(compute_ancillary_metrics(df))
        all_results.append(res)
    
    return pd.DataFrame(all_results)


# -------------------------------------------------------------
# MCP TOOL: 计算 + 保存
# -------------------------------------------------------------
mcp = FastMCP("PointCloudDiversityAnalyzer")


@mcp.tool(
    name="compute_diversity_index",
    description="Compute and save 3D forest diversity metrics from a classified point cloud.",
)
async def compute_3d_fill_index(
    file_path: str, output_path: str = "./", output_name: str = "diversity_metrics.csv"
) -> str:
    """
    Compute forest 3D structural diversity metrics from point cloud.
    Automatically performs tree segmentation if treeid is missing.
    """

    # 检查是否需要单木分割
    if not has_treeid(file_path):
        print("点云缺少 treeid，自动执行单木分割...")
        try:
            segment_output = os.path.join(output_path, "segmentation")
            file_path = run_tree_segmentation(file_path, segment_output)
            print(f"单木分割完成，使用分割结果: {file_path}")
        except Exception as e:
            raise RuntimeError(f"自动单木分割失败: {e}")

    try:
        df = read_point_cloud(file_path)
        df = clean_point_cloud(df)
    except Exception as e:
        raise RuntimeError(f"Failed to read point cloud {file_path}: {e}")

    if not {"x", "y", "z", "classification"}.issubset(df.columns):
        raise ValueError(
            "Point cloud must include x, y, z, and classification columns."
        )

    print(f"Computing forest diversity metrics for {len(df)} points ...")
    results = compute_all_metrics(df)

    os.makedirs(output_path, exist_ok=True)
    output_csv = os.path.join(output_path, output_name)
    results.to_csv(output_csv, index=False)

    avg_height = results["MH"].mean()
    avg_density = results["PointDensity"].mean()

    summary = (
        f"✅ 3D forest diversity metrics computed successfully.\n"
        f"📊 Output saved to: {output_csv}\n"
        f"🌲 Number of trees analyzed: {len(results)}\n"
        f"🌿 Mean tree height: {avg_height:.2f} m\n"
        f"🌾 Mean point density: {avg_density:.2f} pts/m²"
    )
    return summary


if __name__ == "__main__":
    mcp.run(transport="stdio")

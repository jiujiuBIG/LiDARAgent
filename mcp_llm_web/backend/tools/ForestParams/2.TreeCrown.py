import os
import sys
import subprocess
import torch

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import pandas as pd
import numpy as np
import laspy
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.utils import clean_point_cloud, get_xyz_array
import open3d as o3d
from scipy.spatial import ConvexHull
from tqdm import tqdm
import cv2


def has_treeid(file_path: str) -> bool:
    """检查点云文件是否包含 treeid 或 PredInstance 字段"""
    try:
        las = laspy.read(file_path)
        return hasattr(las, 'treeid') or hasattr(las, 'PredInstance')
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
        '/home/may/software/anaconda/install/envs/pg/bin/python',
        "/mnt/Huxiaomei2/May/Huxiaomei/Graduation/H/Graduation/code/Superpoint_HeadMap_Transformer/test_semantic_mcp.py",
        pth_path, semantic_pth
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    
    chm_path = os.path.join(output_path, f"{base_name}_CHM.npy")
    cmd = [
        '/home/may/software/anaconda/install/envs/yolo/bin/python',
        "/mnt/Huxiaomei2/May/Huxiaomei/yolo_tree/crop_2d_bbox_to_pointcloud_prepare.py",
        semantic_pth, chm_path
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    
    yolo_dir = os.path.join(output_path, f"{base_name}_yolo")
    cmd = [
        '/home/may/software/anaconda/install/envs/yolo/bin/python',
        "/mnt/Huxiaomei2/May/Huxiaomei/yolo_tree/crop_2d_bbox_to_pointcloud_mcp.py",
        chm_path, yolo_dir, semantic_pth
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    
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
            '/home/may/software/anaconda/install/envs/pg/bin/python',
            "/mnt/Huxiaomei2/May/Huxiaomei/Graduation/H/Graduation/code/Superpoint_HeadMap_Transformer/test_instance_mcp.py",
            input_path, output_instance
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


def denoise_point_cloud(
    points_array: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0
) -> np.ndarray:
    if len(points_array) < nb_neighbors:
        return points_array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    denoised_points = np.asarray(cl.points)
    return denoised_points


def calculate_crown_ellipse_dimensions(points_3d: np.ndarray) -> dict:
    xy_points = points_3d[:, :2]
    if xy_points.shape[0] < 2:
        return {"major_axis_length": 0.0, "minor_axis_length": 0.0}
    hull = ConvexHull(xy_points)
    hull_points = xy_points[hull.vertices]
    hull_points_cv = hull_points.astype(np.float32)
    if len(hull_points_cv) >= 5:
        (center_x, center_y), (major_axis, minor_axis), angle = cv2.fitEllipse(
            hull_points_cv
        )
        return {
            "major_axis_length": major_axis / 2.0,
            "minor_axis_length": minor_axis / 2.0,
            "center_x": center_x,
            "center_y": center_y,
            "orientation": np.radians(angle),
            "method": "opencv_convex_hull",
            "convex_hull_area": hull.area,
            "convex_hull_points": len(hull_points),
        }
    return {"major_axis_length": 0.0, "minor_axis_length": 0.0}


mcp = FastMCP("TreeCrownAnalyzer")


@mcp.tool(
    name="extract_tree_crown",
    description=(
        "Compute tree crown (canopy) dimensions for each individual tree in a forest point cloud. "
        "Uses convex hull and ellipse fitting to calculate crown width, length, and orientation. "
        "If the user does not provide the file_path, ask them to clarify it. "
        "If the user does not specify an output path, results will be saved in the current directory."
    ),
)
async def extract_tree_crown(file_path: str, output_path: str = "./") -> dict:
    """
    Read a classified forest point cloud (.las / .laz) and compute crown-related statistics for each tree.
    Automatically performs tree segmentation if treeid is missing.

    Parameters
    ----------
    file_path : str
        Path to the input point cloud file (LAS or LAZ format).
    output_path : str, optional
        Directory to save the generated CSV results. Default is the current folder.

    Returns
    -------
    dict: {
        "status": "success" | "error",

    # 检查是否需要单木分割
    if not has_treeid(file_path):
        print("点云缺少 treeid，自动执行单木分割...")
        try:
            segment_output = os.path.join(output_path, "segmentation")
            file_path = run_tree_segmentation(file_path, segment_output)
            print(f"单木分割完成，使用分割结果: {file_path}")
        except Exception as e:
            return {
                "status": "error",
                "output_path": None,
                "message": f"自动单木分割失败: {e}",
                "data": None
            }
        "output_path": str | None,
        "message": str,
        "data": dict | None
    }
    """
    try:
        datasets = read_point_cloud(file_path, is_treeid=True)
        datasets = clean_point_cloud(datasets)
    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"读取点云文件失败: {e}",
            "data": None,
        }

    if "treeid" not in datasets.columns:
        return {
            "status": "error",
            "output_path": None,
            "message": "点云缺少 'treeid' 字段，无法进行单木分割分析",
            "data": None,
        }

    if "z" not in datasets.columns:
        return {
            "status": "error",
            "output_path": None,
            "message": "点云缺少 'z' 坐标",
            "data": None,
        }

    valid_mask = datasets["treeid"] != 0
    datasets = datasets[valid_mask]
    xyz = datasets[["x", "y", "z"]].values
    treeid = datasets["treeid"].values

    results = []
    for id in tqdm(np.unique(treeid)):
        if id == 0:
            continue
        index = treeid == id
        clipxyz = xyz[index]
        denoise_points = denoise_point_cloud(clipxyz)
        index = clipxyz[:, -1] >= np.percentile(clipxyz[:, -1], 30)
        clipxyz = clipxyz[index]
        if len(clipxyz) <= 3000:
            continue
        res = calculate_crown_ellipse_dimensions(denoise_points)
        try:
            results.append(
                {
                    "treeid": int(id),
                    "major_axis_length": res["major_axis_length"],
                    "minor_axis_length": res["minor_axis_length"],
                    "center_x": res["center_x"],
                    "center_y": res["center_y"],
                    "points_length": len(denoise_points),
                }
            )
        except Exception:
            continue

    os.makedirs(output_path, exist_ok=True)
    output_csv = os.path.join(output_path, "TreeCrown.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    if not df.empty:
        mean_major = df["major_axis_length"].mean()
        mean_minor = df["minor_axis_length"].mean()
        max_major = df["major_axis_length"].max()
        min_major = df["major_axis_length"].min()
        var_major = df["major_axis_length"].var()
    else:
        mean_major = mean_minor = max_major = min_major = var_major = 0.0

    return {
        "status": "success",
        "output_path": output_csv,
        "message": f"成功提取 {len(df)} 棵树的冠幅参数。平均冠幅长度: {mean_major:.2f}m，最大: {max_major:.2f}m，最小: {min_major:.2f}m。",
        "data": {
            "tree_count": len(df),
            "crown_mean": round(mean_major, 2),
            "crown_max": round(max_major, 2),
            "crown_min": round(min_major, 2),
            "crown_variance": round(var_major, 2),
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

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
from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.utils import clean_point_cloud, get_xyz_array
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError
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
    import torch

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
    if xy_points.shape[0] < 5:
        return {"major_axis_length": 0.0, "minor_axis_length": 0.0}

    # Keep metre-scale crown geometry precise when OpenCV converts to float32.
    origin = xy_points.mean(axis=0)
    local_xy = xy_points - origin
    hull = ConvexHull(local_xy)
    if len(hull.vertices) < 5:
        return {"major_axis_length": 0.0, "minor_axis_length": 0.0}

    center, (width, height), angle = cv2.fitEllipse(local_xy[hull.vertices].astype(np.float32))
    center_x, center_y = origin + center
    major_axis, minor_axis = max(width, height), min(width, height)
    major_angle = angle + (90.0 if height > width else 0.0)
    return {
        "major_axis_length": major_axis,
        "minor_axis_length": minor_axis,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "orientation": float(np.radians(major_angle % 180.0)),
        "method": "opencv_convex_hull",
        "convex_hull_area": float(hull.volume),  # SciPy's 2D area is stored as volume.
        "convex_hull_points": len(hull.vertices),
    }


mcp = FastMCP("TreeCrownAnalyzer")


@mcp.tool(
    name="extract_tree_crown",
    description=(
        "Compute tree crown (canopy) dimensions for each individual tree in a forest point cloud. "
        "Fits an ellipse to the crown in XY and reports full major and minor axis lengths in metres. "
        "If the user does not provide the file_path, ask them to clarify it. "
        "If the user does not specify an output path, results will be saved in the current directory."
    ),
)
async def extract_tree_crown(file_path: str, output_path: str = "./") -> dict:
    """Export crown statistics from an instance-labelled LAS or LAZ."""
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

    valid_mask = datasets["treeid"] > 0
    datasets = datasets[valid_mask]
    xyz = datasets[["x", "y", "z"]].values
    treeid = datasets["treeid"].values
    tree_ids = np.unique(treeid)

    results = []
    for tree_id in tqdm(tree_ids):
        index = treeid == tree_id
        points = denoise_point_cloud(xyz[index])
        if not len(points):
            continue
        crown_points = points[points[:, 2] >= np.percentile(points[:, 2], 30)]
        if len(crown_points) <= 3000:
            continue
        try:
            res = calculate_crown_ellipse_dimensions(crown_points)
        except (QhullError, cv2.error):
            continue
        if "center_x" not in res:
            continue
        results.append({
            "treeid": int(tree_id),
            "major_axis_length": res["major_axis_length"],
            "minor_axis_length": res["minor_axis_length"],
            "center_x": res["center_x"],
            "center_y": res["center_y"],
            "points_length": len(crown_points),
        })

    os.makedirs(output_path, exist_ok=True)
    output_csv = os.path.join(output_path, "TreeCrown.csv")
    df = pd.DataFrame(results, columns=[
        "treeid", "major_axis_length", "minor_axis_length", "center_x", "center_y", "points_length",
    ])
    df.to_csv(output_csv, index=False)

    skipped = len(tree_ids) - len(df)
    if df.empty:
        return {
            "status": "warning",
            "output_path": output_csv,
            "message": "未拟合到冠幅。单木筛选后需要超过 3,000 点，且树冠投影须支持椭圆拟合。",
            "data": {"tree_count": 0, "unmeasured_tree_count": skipped},
        }

    mean_major = float(df["major_axis_length"].mean())
    max_major = float(df["major_axis_length"].max())
    min_major = float(df["major_axis_length"].min())
    var_major = float(df["major_axis_length"].var()) if len(df) > 1 else 0.0

    return {
        "status": "success",
        "output_path": output_csv,
        "message": f"已拟合 {len(df)} 棵树的冠幅，{skipped} 棵因点数不足或几何拟合失败跳过。平均冠幅长度为 {mean_major:.2f} 米。",
        "data": {
            "tree_count": len(df),
            "unmeasured_tree_count": skipped,
            "crown_mean": round(mean_major, 2),
            "crown_max": round(max_major, 2),
            "crown_min": round(min_major, 2),
            "crown_variance": round(var_major, 2),
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

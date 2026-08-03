import os
import sys
import subprocess
import laspy
import torch

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.utils import clean_point_cloud


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


mcp = FastMCP("TreeHeightAnalyzer")


@mcp.tool(
    name="extract_tree_height",
    description=(
        "Compute tree height metrics for each individual tree in a forest point cloud. "
        "If the user does not provide the file_path, ask them to clarify it"
        "If the user does not specify an output path, results will be saved in the current directory."
    ),
)
async def extract_tree_height(file_path: str, output_path: str = "./") -> dict:
    """
    Read a classified forest point cloud and compute height-related statistics for each tree.
    Automatically performs tree segmentation if treeid is missing.
    """
    try:
        datasets = read_point_cloud(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read point cloud file: {file_path}. Error: {e}")

    # Clean and validate data
    df_clean = clean_point_cloud(datasets)

    # Check for treeid
    if "treeid" not in df_clean.columns:
        print(
            "Warning: No 'treeid' field found. Computing height statistics for all points instead."
        )
        # Compute overall height stats
        z_values = df_clean["z"].values
        max_h = np.max(z_values)
        min_h = np.min(z_values)
        mean_h = np.mean(z_values)

        os.makedirs(output_path, exist_ok=True)
        output_csv = os.path.join(output_path, "HeightStats.csv")
        pd.DataFrame(
            {
                "metric": ["max_height", "min_height", "mean_height"],
                "value": [float(max_h), float(min_h), float(mean_h)],
            }
        ).to_csv(output_csv, index=False)

        summary = (
            f"Successfully computed height statistics.\n"
            f"Results saved to: {output_csv}\n"
            f"Maximum height: {max_h:.2f} m\n"
            f"Minimum height: {min_h:.2f} m\n"
            f"Average height: {mean_h:.2f} m\n"
            f"Note: No treeid field found. Computing statistics for all points."
        )
        return summary

    # Check mandatory fields
    if "z" not in df_clean.columns:
        raise ValueError("The point cloud is missing required 'z' attribute.")

    # Remove background points (treeid == 0)
    valid_mask = df_clean["treeid"] != 0
    datasets = df_clean[valid_mask]

    if len(datasets) == 0:
        raise ValueError("No valid tree points found after filtering (treeid == 0).")

    # Compute per-tree height metrics
    results = []
    for tid, group in datasets.groupby("treeid"):
        z_values = group["z"].values
        max_h = np.max(z_values)
        min_h = np.min(z_values)
        h95 = np.percentile(z_values, 95)
        h05 = np.percentile(z_values, 5)

        # if h95 - h05 < 1:
        #     continue

        results.append(
            {
                "treeid": int(tid),
                "max_height": float(max_h),
                "min_height": float(min_h),
                "95th_height": float(h95),
                "05th_height": float(h05),
                "height_range": float(h95 - h05),
                "n_points": len(z_values),
            }
        )

    result_df = pd.DataFrame(results).sort_values("treeid")

    os.makedirs(output_path, exist_ok=True)
    output_csv = os.path.join(output_path, "TreeHeight.csv")
    result_df.to_csv(output_csv, index=False)

    if not result_df.empty:
        mean_height = float(result_df["height_range"].mean())
        var_height = float(result_df["height_range"].var())
        max_height = float(result_df["height_range"].max())
        min_height = float(result_df["height_range"].min())
    else:
        mean_height = var_height = max_height = min_height = 0.0

    output_summary = (
        f"Successfully extracted the height of each tree to {output_csv}.\n"
        f"The average tree height in this area is {mean_height:.2f} m, with a variance of {var_height:.2f}.\n"
        f"The tallest tree reaches {max_height:.2f} m, and the shortest tree is {min_height:.2f} m.\n"
        f"Number of trees analyzed: {len(result_df)}.\n"
        f"Please remember to check even if you just view it."
    )

    return output_summary


if __name__ == "__main__":
    mcp.run(transport="stdio")

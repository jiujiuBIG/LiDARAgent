import os
import sys
import subprocess
import laspy

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import HEIGHT_ABOVE_GROUND_FIELDS, read_point_cloud
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


mcp = FastMCP("TreeHeightAnalyzer")


@mcp.tool(
    name="extract_tree_height",
    description=(
        "Compute vertical metrics for each tree in an instance-labelled LAS/LAZ in metres. "
        "Report tree height only when the point cloud carries an explicit height-above-ground field."
    ),
)
async def extract_tree_height(file_path: str, output_path: str = "./") -> dict:
    """
    Export per-tree Z statistics and ground-relative height when available.
    """
    try:
        df_clean = clean_point_cloud(read_point_cloud(file_path))
    except Exception as e:
        return {"status": "error", "output_path": None, "message": f"读取点云失败: {e}", "data": None}

    if "treeid" not in df_clean:
        return {"status": "error", "output_path": None, "message": "请提供带单木编号的 LAS 或 LAZ。", "data": None}

    datasets = df_clean[df_clean["treeid"] > 0]
    if datasets.empty:
        return {"status": "error", "output_path": None, "message": "点云中没有有效的单木编号。", "data": None}

    ground_height_field = next(
        (name for name in HEIGHT_ABOVE_GROUND_FIELDS if name in datasets), None
    )
    results = []
    for tid, group in datasets.groupby("treeid"):
        z_values = group["z"].values
        max_h = np.max(z_values)
        min_h = np.min(z_values)
        h95 = np.percentile(z_values, 95)
        h05 = np.percentile(z_values, 5)

        record = {
            "treeid": int(tid),
            "max_height": float(max_h),
            "min_height": float(min_h),
            "95th_height": float(h95),
            "05th_height": float(h05),
            "height_range": float(h95 - h05),
            "n_points": len(z_values),
        }
        if ground_height_field is not None:
            relative_heights = group[ground_height_field].to_numpy()
            relative_heights = relative_heights[np.isfinite(relative_heights) & (relative_heights >= 0)]
            record["tree_height_m"] = float(relative_heights.max()) if len(relative_heights) else None
        results.append(record)

    result_df = pd.DataFrame(results).sort_values("treeid")

    os.makedirs(output_path, exist_ok=True)
    output_csv = os.path.join(output_path, "TreeHeight.csv")
    result_df.to_csv(output_csv, index=False)

    mean_span = float(result_df["height_range"].mean())
    data = {
        "tree_count": len(result_df),
        "vertical_span_mean_m": mean_span,
        "height_above_ground_field": ground_height_field,
    }
    message = f"已计算 {len(result_df)} 棵树的 Z 值统计，平均垂直跨度（P95−P05）为 {mean_span:.2f} 米。"
    if ground_height_field is not None:
        heights = result_df["tree_height_m"].dropna()
        data["tree_height_count"] = len(heights)
        if len(heights):
            data["tree_height_mean_m"] = float(heights.mean())
            message += f"有 {len(heights)} 棵树可计算离地树高，平均为 {heights.mean():.2f} 米。"
    else:
        message += "输入缺少离地高度字段，CSV 中的高程统计使用输入 Z 值。"

    return {"status": "success", "output_path": output_csv, "message": message, "data": data}


if __name__ == "__main__":
    mcp.run(transport="stdio")

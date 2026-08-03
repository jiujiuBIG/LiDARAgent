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
from tools.Preprocessing.utils import clean_point_cloud
from tools.Preprocessing.write import dataframe_to_las
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import torch


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
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
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


def process_one_file_hough(
    xyz, treeid, z_min=1.2, z_max=1.25, canny_sigma=2.0, hough_threshold=0.1
):
    # --------------------
    #  1. 选取胸高截面
    # --------------------
    clipindex = (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    clipxyz = xyz[clipindex][:, :2]
    cliptreeid = treeid[clipindex]

    results = []

    for id in np.unique(cliptreeid):
        if id == 0:
            continue
        index = cliptreeid == id
        pts = clipxyz[index]

        res = 0.01
        x_min, y_min = pts.min(axis=0)
        pts_px = ((pts - [x_min, y_min]) / res).astype(np.int32)
        px_w = pts_px[:, 0].max() + 3
        px_h = pts_px[:, 1].max() + 3

        img = np.zeros((px_h, px_w), dtype=np.uint8)
        img[pts_px[:, 1], pts_px[:, 0]] = 255

        edges = canny(img, sigma=canny_sigma)

        min_r = 10
        max_r = 500
        hough_radii = np.arange(min_r, max_r + 1, 1)

        hough_res = hough_circle(edges, hough_radii)
        accums, cx_list, cy_list, radii = hough_circle_peaks(
            hough_res,
            hough_radii,
            total_num_peaks=1,  # 最多检测10个树干
            normalize=True,
        )

        for idx, (cx_px, cy_px, r_px, vote) in enumerate(
            zip(cx_list, cy_list, radii, accums)
        ):
            if vote < hough_threshold:
                continue  # 弱圆直接丢弃

            cx = cx_px * res + x_min
            cy = cy_px * res + y_min
            r = r_px * res

            sel = (
                np.abs(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) - r)
                < 0.03
            )
            pts_circle = pts[sel]

            if pts_circle.shape[0] < 20:
                continue

            cx2, cy2, r2, score, n_used = fit_circle_least_squares(pts_circle)
            if cx2 is None:
                continue

            DBH = 2 * r2  # 胸径 = 直径

            results.append(
                {
                    "circle_id": idx,
                    "treeid": id,
                    "n_points": int(n_used),
                    "cx": float(cx2),
                    "cy": float(cy2),
                    "radius": float(r2),
                    "DBH": float(DBH),
                    "r2": float(score),
                    "vote": float(vote),
                }
            )

    return pd.DataFrame(results)


# Initialize MCP server
mcp = FastMCP("ForestDBHExtractor")


@mcp.tool(
    name="extract_forest_dbh",
    description=(
        "Compute the Diameter at Breast Height (DBH) for each tree in a forest point cloud. "
        "If the user does not provide the file_path, ask them to clarify it "
        "The function reads a classified point cloud where 'treeid' labels trees, "
        "extracts a 1.3 m ± 0.1 m slice from the ground for each tree, fits a circle to that slice, "
        "and saves DBH results as a CSV file."
    ),
)
async def extract_forest_dbh(file_path: str, output_path: str = "./") -> dict:
    """
    Read a classified forest point cloud and compute DBH for each tree using circle fitting.
    Automatically performs tree segmentation if treeid is missing.
    """

    # =============== 检查是否需要单木分割 ===============
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
                "data": None,
            }

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

    # Ensure data is in DataFrame format
    if isinstance(data, np.ndarray):
        ncols = data.shape[1]
        if ncols >= 4:
            cols = ["x", "y", "z", "treeid"]
        else:
            raise ValueError("Point cloud array does not contain enough columns.")
        df = pd.DataFrame(data, columns=cols)
    else:
        df = data

    # Check required columns
    if "treeid" not in df.columns or "z" not in df.columns:
        return {
            "status": "error",
            "output_path": None,
            "message": "点云缺少 'z' 或 'treeid' 字段",
            "data": None,
        }

    # Check treeid distribution and handle special cases
    treeid_counts = df["treeid"].value_counts()
    zero_count = treeid_counts.get(0, 0)
    non_zero_count = sum(v for k, v in treeid_counts.items() if k != 0)

    # If treeid=0 has most points (e.g., > 90%), treeid may not be proper segmentation
    # In this case, treat treeid=0 as valid tree points
    if zero_count > 0 and zero_count > non_zero_count * 9:
        # treeid=0 dominates, treat it as tree points (not ground)
        df = df[df["treeid"] == 0].copy()
    else:
        # Keep only valid tree points (treeid != 0)
        df = df[df["treeid"] != 0]

    # Check if there are any valid tree points
    if len(df) == 0:
        return {
            "status": "error",
            "output_path": None,
            "message": f"点云中没有有效的树点数据，请确保已进行单木分割",
            "data": None,
        }

    result_df = process_one_file_hough(
        xyz=df[["x", "y", "z"]].values,
        treeid=df["treeid"].values,
        z_min=4.8,
        z_max=5.8,
        canny_sigma=2.0,
        hough_threshold=0.2,
    )

    # Check if result is empty
    if result_df.empty:
        result_df.to_csv(
            output_csv := os.path.join(output_path, "forest_dbh_results.csv"),
            index=False,
        )
        return {
            "status": "warning",
            "output_path": output_csv,
            "message": f"未在点云中检测到树木。点云可能未进行单木分类或在1.3m高度处点密度不足",
            "data": {"detected_trees": 0},
        }

    # Check if 'vote' column exists
    if "vote" not in result_df.columns:
        result_df.to_csv(
            output_csv := os.path.join(output_path, "forest_dbh_results.csv"),
            index=False,
        )
        return {
            "status": "warning",
            "output_path": output_csv,
            "message": f"未检测到有效的树干圆，可能需要调整检测参数",
            "data": {"detected_trees": 0},
        }

    result_df.to_csv(
        output_csv := os.path.join(output_path, "forest_dbh_results.csv"), index=False
    )

    # Compute tree height statistics
    # index = result_df["vote"] >= 0.5
    DBH = result_df["DBH"].dropna()
    mean_height = float(DBH.mean()) if not DBH.empty else 0.0
    var_height = float(DBH.var()) if not DBH.empty else 0.0
    max_height = float(DBH.max()) if not DBH.empty else 0.0
    min_height = float(DBH.min()) if not DBH.empty else 0.0

    return {
        "status": "success",
        "output_path": output_csv,
        "message": f"成功提取 {len(DBH)} 棵树的胸径。平均胸径: {mean_height:.2f}米，最大: {max_height:.2f}米，最小: {min_height:.2f}米",
        "data": {
            "tree_count": len(DBH),
            "dbh_mean": round(mean_height, 2),
            "dbh_max": round(max_height, 2),
            "dbh_min": round(min_height, 2),
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.write import dataframe_to_las
from tools.Preprocessing.utils import clean_point_cloud, get_xyz_array
import open3d as o3d

mcp = FastMCP("PointCloudFilter")


@mcp.tool(
    name="statistical_filter",
    description=(
        "Remove statistical outliers from point cloud. "
        "Uses statistical outlier removal to filter noise based on local point density. "
        "If user does not provide file_path, ask them to clarify it. "
        "If the user does not specify an output path, results will be saved in the current directory."
    ),
)
async def statistical_filter(
    file_path: str,
    output_path: str = "./",
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> dict:
    """
    Remove outliers using statistical outlier removal method.

    Args:
        file_path (str): Path to input point cloud file
        output_path (str): Directory to save filtered point cloud
        nb_neighbors (int): Number of neighbors to analyze (default: 20)
        std_ratio (float): Standard deviation ratio threshold (default: 2.0)

    Returns:
        dict: {
            "status": "success" | "error",
            "output_path": str | None,
            "message": str,
            "data": dict | None
        }
    """
    try:
        df = read_point_cloud(file_path)
    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"读取点云文件失败: {e}",
            "data": None,
        }

    df_clean = clean_point_cloud(df)
    xyz = get_xyz_array(df_clean)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    filtered_pcd, indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    df_filtered = df_clean.iloc[indices].reset_index(drop=True)

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_path, f"{base_name}_filtered_stat.las")
    dataframe_to_las(df_filtered, output_file)

    removed_count = len(df_clean) - len(df_filtered)
    removed_ratio = removed_count / len(df_clean) * 100

    return {
        "status": "success",
        "output_path": output_file,
        "message": f"成功应用统计滤波。原始点: {len(df_clean)}，滤波后: {len(df_filtered)}，移除异常点: {removed_count} ({removed_ratio:.1f}%)",
        "data": {
            "original_points": len(df_clean),
            "filtered_points": len(df_filtered),
            "removed_points": removed_count,
            "removed_ratio": round(removed_ratio, 2),
            "method": "statistical",
            "nb_neighbors": nb_neighbors,
            "std_ratio": std_ratio,
        },
    }


@mcp.tool(
    name="radius_filter",
    description=(
        "Remove outliers from point cloud based on point density within a radius. "
        "Uses radius-based outlier removal to filter noise. "
        "If user does not provide file_path, ask them to clarify it. "
        "If user does not specify an output path, results will be saved in current directory."
    ),
)
async def radius_filter(
    file_path: str, output_path: str = "./", nb_points: int = 16, radius: float = 0.05
) -> dict:
    """
    Remove outliers using radius-based outlier removal.

    Args:
        file_path (str): Path to input point cloud file
        output_path (str): Directory to save filtered point cloud
        nb_points (int): Minimum number of points in radius (default: 16)
        radius (float): Search radius (default: 0.05)

    Returns:
        dict: {
            "status": "success" | "error",
            "output_path": str | None,
            "message": str,
            "data": dict | None
        }
    """
    try:
        df = read_point_cloud(file_path)
    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"读取点云文件失败: {e}",
            "data": None,
        }

    df_clean = clean_point_cloud(df)
    xyz = get_xyz_array(df_clean)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    filtered_pcd, indices = pcd.remove_radius_outlier(
        nb_points=nb_points, radius=radius
    )

    df_filtered = df_clean.iloc[indices].reset_index(drop=True)

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_path, f"{base_name}_filtered_radius.las")
    dataframe_to_las(df_filtered, output_file)

    removed_count = len(df_clean) - len(df_filtered)
    removed_ratio = removed_count / len(df_clean) * 100

    return {
        "status": "success",
        "output_path": output_file,
        "message": f"成功应用半径滤波。原始点: {len(df_clean)}，滤波后: {len(df_filtered)}，移除异常点: {removed_count} ({removed_ratio:.1f}%)",
        "data": {
            "original_points": len(df_clean),
            "filtered_points": len(df_filtered),
            "removed_points": removed_count,
            "removed_ratio": round(removed_ratio, 2),
            "method": "radius",
            "nb_points": nb_points,
            "radius": radius,
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

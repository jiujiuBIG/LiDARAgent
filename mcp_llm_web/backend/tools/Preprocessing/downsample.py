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
from tools.Preprocessing.utils import clean_point_cloud, get_xyz_array, get_bounding_box
import open3d as o3d

mcp = FastMCP("PointCloudDownsampler")


@mcp.tool(
    name="voxel_downsample",
    description=(
        "Downsample point cloud using voxel grid filter. "
        "Reduces point density by averaging points within each voxel. "
        "If user does not provide file_path, ask them to clarify it. "
        "If user does not specify an output path, results will be saved in current directory."
    ),
)
async def voxel_downsample(
    file_path: str, output_path: str = "./", voxel_size: float = 0.1
) -> dict:
    """
    Downsample point cloud using voxel grid filter.

    Args:
        file_path (str): Path to input point cloud file
        output_path (str): Directory to save downsampled point cloud
        voxel_size (float): Voxel size for downsampling (default: 0.1 meters)

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

    bbox_min, bbox_max = get_bounding_box(xyz)

    down_pcd, indices = pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )

    representative_indices = [int(idx[0]) for idx in indices if len(idx) > 0]

    df_downsampled = df_clean.iloc[representative_indices].reset_index(drop=True)

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_path, f"{base_name}_downsampled.las")
    dataframe_to_las(df_downsampled, output_file)

    reduction_ratio = (1 - len(df_downsampled) / len(df_clean)) * 100

    return {
        "status": "success",
        "output_path": output_file,
        "message": f"成功下采样。原始点: {len(df_clean)}，下采样后: {len(df_downsampled)}，压缩比: {reduction_ratio:.1f}%",
        "data": {
            "original_points": len(df_clean),
            "downsampled_points": len(df_downsampled),
            "reduction_ratio": round(reduction_ratio, 2),
            "voxel_size": voxel_size,
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

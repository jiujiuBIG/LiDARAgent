import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import laspy
import open3d as o3d
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.write import dataframe_to_las
from tools.Preprocessing.utils import clean_point_cloud

mcp = FastMCP("PointCloudConverter")


@mcp.tool(
    name="convert_point_cloud_format",
    description=(
        "Convert point cloud between different formats (LAS, LAZ, PLY, TXT, XYZ). "
        "Supports bidirectional conversion between common point cloud formats. "
        "If user does not provide input_path, ask them to clarify it. "
        "If user does not specify output_path, file will be saved in current directory."
    ),
)
async def convert_point_cloud_format(
    input_path: str, target_format: str, output_path: str = "./"
) -> dict:
    """
    Convert point cloud between formats.

    Args:
        input_path (str): Path to input point cloud file
        target_format (str): Target format (las, laz, ply, txt, xyz)
        output_path (str): Directory to save output file

    Returns:
        dict: {
            "status": "success" | "error",
            "output_path": str | None,
            "message": str,
            "data": dict | None
        }
    """
    target_format = target_format.lower()
    supported_formats = ["las", "laz", "ply", "txt", "xyz"]

    if target_format not in supported_formats:
        return {
            "status": "error",
            "output_path": None,
            "message": f"不支持的目标格式。支持的格式: {supported_formats}",
            "data": None,
        }

    input_ext = os.path.splitext(input_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    try:
        df = read_point_cloud(input_path)
    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"读取点云文件失败: {e}",
            "data": None,
        }

    df_clean = clean_point_cloud(df)

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{base_name}.{target_format}")

    if target_format in ["las", "laz"]:
        dataframe_to_las(df_clean, output_file)
    elif target_format == "ply":
        pcd = o3d.geometry.PointCloud()
        xyz = df_clean[["x", "y", "z"]].values.astype(np.float64)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud(output_file, pcd)
    elif target_format == "txt":
        sep = "\t"
        df_clean.to_csv(
            output_file, sep=sep, index=False, header=False, float_format="%.6f"
        )
    elif target_format == "xyz":
        df_clean.to_csv(
            output_file, sep=" ", index=False, header=False, float_format="%.6f"
        )

    return {
        "status": "success",
        "output_path": output_file,
        "message": f"成功将 {input_ext} 转换为 {target_format} 格式。转换 {len(df_clean)} 个点。",
        "data": {
            "input_format": input_ext.replace(".", ""),
            "output_format": target_format,
            "point_count": len(df_clean),
            "output_file": output_file,
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

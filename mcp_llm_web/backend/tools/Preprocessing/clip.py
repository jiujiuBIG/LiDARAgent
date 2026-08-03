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
from tools.Preprocessing.utils import clean_point_cloud

mcp = FastMCP("PointCloudClipper")


@mcp.tool(
    name="clip_by_bounds",
    description=(
        "Clip point cloud by bounding box boundaries. "
        "Extract points within specified x, y, z coordinate ranges. "
        "If user does not provide file_path, ask them to clarify it. "
        "If user does not specify an output path, results will be saved in current directory."
    ),
)
async def clip_by_bounds(
    file_path: str,
    output_path: str = "./",
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    z_min: float = None,
    z_max: float = None,
) -> dict:
    """
    Clip point cloud by bounding box.

    Args:
        file_path (str): Path to input point cloud file
        output_path (str): Directory to save clipped point cloud
        x_min, x_max: X coordinate range (optional)
        y_min, y_max: Y coordinate range (optional)
        z_min, z_max: Z coordinate range (optional)

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

    if all(v is None for v in [x_min, x_max, y_min, y_max, z_min, z_max]):
        return {
            "status": "error",
            "output_path": None,
            "message": "至少需要指定一个边界参数",
            "data": None,
        }

    df_clean = clean_point_cloud(df)

    x = df_clean["x"].values
    y = df_clean["y"].values
    z = df_clean["z"].values

    mask = pd.Series([True] * len(df_clean))
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    if y_min is not None:
        mask &= y >= y_min
    if y_max is not None:
        mask &= y <= y_max
    if z_min is not None:
        mask &= z >= z_min
    if z_max is not None:
        mask &= z <= z_max

    df_clipped = df_clean[mask].reset_index(drop=True)

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_path, f"{base_name}_clipped.las")
    dataframe_to_las(df_clipped, output_file)

    kept_ratio = len(df_clipped) / len(df_clean) * 100

    return {
        "status": "success",
        "output_path": output_file,
        "message": f"成功裁剪点云。原始点: {len(df_clean)}，裁剪后: {len(df_clipped)}，保留比例: {kept_ratio:.1f}%",
        "data": {
            "original_points": len(df_clean),
            "clipped_points": len(df_clipped),
            "kept_ratio": round(kept_ratio, 2),
            "bounding_box": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "z_min": z_min,
                "z_max": z_max,
            },
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

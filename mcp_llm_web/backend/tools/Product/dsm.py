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
from tools.Preprocessing.utils import clean_point_cloud, get_xyz_array
from scipy.interpolate import griddata
from skimage import io

mcp = FastMCP("DSMGenerator")


@mcp.tool(
    name="generate_dsm",
    description=(
        "Generate Digital Surface Model (DSM) from point cloud. "
        "DSM represents the top surface of all objects including vegetation, buildings, and terrain. "
        "If user does not provide file_path, ask them to clarify it. "
        "If user does not specify output_path, results will be saved in current directory."
    ),
)
async def generate_dsm(
    file_path: str, output_path: str = "./", resolution: float = 0.5
) -> dict:
    """
    Generate Digital Surface Model (DSM) from point cloud.

    Args:
        file_path (str): Path to input point cloud file
        output_path (str): Directory to save DSM image
        resolution (float): Grid resolution in meters (default: 0.5)

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
    pc = get_xyz_array(df_clean)

    min_x, min_y, _ = np.min(pc, axis=0)
    max_x, max_y, _ = np.max(pc, axis=0)

    x_grid = np.arange(min_x, max_x, resolution)
    y_grid = np.arange(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)

    dsm_grid = griddata(
        (pc[:, 0], pc[:, 1]), pc[:, 2], (X, Y), method="linear", fill_value=np.nan
    )

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_tif = os.path.join(output_path, f"{base_name}_DSM.tif")
    io.imsave(output_tif, dsm_grid)

    mean_height = np.nanmean(dsm_grid)
    max_height = np.nanmax(dsm_grid)
    min_height = np.nanmin(dsm_grid)

    return {
        "status": "success",
        "output_path": output_tif,
        "message": f"成功生成DSM。网格分辨率: {resolution}m。高程范围: {min_height:.2f} 到 {max_height:.2f}m，平均: {mean_height:.2f}m",
        "data": {
            "resolution": resolution,
            "height_min": round(min_height, 2),
            "height_max": round(max_height, 2),
            "height_mean": round(mean_height, 2),
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
filling_path = os.path.join(
    PROJECT_ROOT, "para", "lib", "Filling", "build", "lib.linux-x86_64-cpython-310"
)
if os.path.exists(filling_path):
    sys.path.insert(0, filling_path)
else:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "para", "lib", "Filling"))

import json
import pandas as pd
import numpy as np
import laspy
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.utils import clean_point_cloud
import filling_grid
from skimage import io
import open3d as o3d


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


def get_treeid_array(las) -> np.ndarray:
    """从las文件中提取treeid数组"""
    if hasattr(las, "treeid"):
        return np.array(las.treeid)
    elif hasattr(las, "PredInstance"):
        return np.array(las.PredInstance)
    elif hasattr(las, "treeID"):
        return np.array(las.treeID)
    elif hasattr(las, "TreeID"):
        return np.array(las.TreeID)
    else:
        return np.zeros(len(las.x), dtype=np.int64)


mcp = FastMCP("FillingIndexAnalyzer")


@mcp.tool(
    name="extract_filling_index",
    description=(
        "Compute the canopy filling index from point cloud data. "
        "The filling index measures the ratio of canopy volume to bounding box volume. "
        "If the user does not provide the file_path, ask them to clarify it. "
        "If the user does not specify an output path, results will be saved in the current directory."
    ),
)
async def extract_filling_index(file_path: str, output_path: str = "./") -> dict:
    """
    Compute the canopy filling index from a forest point cloud.

    Parameters
    ----------
    file_path : str
        Path to the input point cloud file (LAS or LAZ format).
    output_path : str, optional
        Directory to save the generated CSV and TIF results. Default is the current folder.

    Returns
    -------
    dict: {
        "status": "success" | "error",
        "output_path": str | None,
        "message": str,
        "data": dict | None
    }
    """
    try:
        las = laspy.read(file_path)
        offsets = las.header.offsets

        xyz = np.vstack([las.x - offsets[0], las.y - offsets[1], las.z]).T.astype(
            np.float64
        )

        treeid = get_treeid_array(las)

        min_x, min_y, min_z = map(float, np.min(xyz, axis=0))
        max_x, max_y, max_z = map(float, np.max(xyz, axis=0))

        mix_z = float(np.percentile(xyz[:, -1], 10))
        max_z = float(np.percentile(xyz[:, -1], 90))

        filling = filling_grid.compute_filling_image(
            xyz, min_x, max_x, min_y, max_y, mix_z, max_z, 0.2, 1.0
        )

        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        output_csv = os.path.join(output_path, f"{base_name}_filling.csv")
        output_tif = os.path.join(output_path, f"{base_name}_filling.tif")

        df = pd.DataFrame(filling)
        df.to_csv(output_csv, index=False)
        io.imsave(output_tif, filling)

        filling_mean = float(np.mean(filling))
        filling_max = float(np.max(filling))
        filling_min = float(np.min(filling))

        return {
            "status": "success",
            "output_path": output_csv,
            "message": f"成功计算填充指数。平均填充指数: {filling_mean:.4f}，最大: {filling_max:.4f}，最小: {filling_min:.4f}",
            "data": {
                "filling_mean": round(filling_mean, 4),
                "filling_max": round(filling_max, 4),
                "filling_min": round(filling_min, 4),
                "tif_output": output_tif,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"计算填充指数失败: {e}",
            "data": None,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")

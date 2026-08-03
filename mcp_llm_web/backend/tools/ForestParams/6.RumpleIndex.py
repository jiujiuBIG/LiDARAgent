import os
import sys

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
import rumple_index


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
        return np.zeros(len(las.x), dtype=np.int32)


mcp = FastMCP("RumpleIndexAnalyzer")


@mcp.tool(
    name="extract_rumple_index",
    description=(
        "Compute the rumple index from point cloud data. "
        "Rumple index measures the ratio of canopy surface area to ground projected area, "
        "indicating canopy complexity and three-dimensional structure. "
        "Higher values indicate more complex canopy surfaces. "
        "If the user does not provide the file_path, ask them to clarify it. "
        "If the user does not specify an output path, results will be saved in the current directory."
    ),
)
async def extract_rumple_index(
    file_path: str, output_path: str = "./", voxel_size: float = 0.1
) -> dict:
    """
    Compute the rumple index from a forest point cloud.

    Parameters
    ----------
    file_path : str
        Path to the input point cloud file (LAS or LAZ format).
    output_path : str, optional
        Directory to save the generated CSV results. Default is the current folder.
    voxel_size : float, optional
        Voxel size for rumple index calculation. Default is 0.1.

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

        xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        treeid = get_treeid_array(las)

        xyz[:, 0] -= np.min(xyz[:, 0])
        xyz[:, 1] -= np.min(xyz[:, 1])

        max_z = float(np.max(xyz[:, 2]))
        min_z = float(np.min(xyz[:, 2]))

        rumple_value = rumple_index.compute_rumple_index(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], treeid, voxel_size
        )

        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_csv = os.path.join(output_path, f"{base_name}_rumple_index.csv")

        rumple_df = pd.DataFrame({"rumple_index": [rumple_value]})
        rumple_df.to_csv(output_csv, index=False)

        return {
            "status": "success",
            "output_path": output_csv,
            "message": f"成功计算凹凸指数。Rumple Index: {float(rumple_value[0]):.4f}",
            "data": {
                "rumple_index": round(float(rumple_value[0]), 4),
                # "voxel_size": voxel_size,
                "height_range": f"{min_z:.2f}m - {max_z:.2f}m",
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"计算凹凸指数失败: {e}",
            "data": None,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")

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

import canopy_gaps


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


mcp = FastMCP("GapFractionAnalyzer")


@mcp.tool(
    name="extract_gap_fraction",
    description=(
        "Compute the canopy gap fraction from point cloud data. "
        "Gap fraction represents the proportion of open sky visible from a point beneath the canopy. "
        "It is an important indicator of canopy structure and forest density. "
        "If the user does not provide the file_path, ask them to clarify it. "
        "If the user does not specify an output path, results will be saved in the current directory."
    ),
)
async def extract_gap_fraction(file_path: str, output_path: str = "./") -> dict:
    """
    Compute the canopy gap fraction from a forest point cloud.

    Parameters
    ----------
    file_path : str
        Path to the input point cloud file (LAS or LAZ format).
    output_path : str, optional
        Directory to save the generated CSV results. Default is the current folder.

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

        max_z = float(np.max(xyz[:, 2]))
        min_z = float(np.min(xyz[:, 2]))

        gap_fraction = canopy_gaps.fraction_canopy_gaps(xyz[:, 0], xyz[:, 1], treeid)

        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_csv = os.path.join(output_path, f"{base_name}_gap_fraction.csv")

        gap_df = pd.DataFrame({"gap_fraction": [gap_fraction]})
        gap_df.to_csv(output_csv, index=False)

        return {
            "status": "success",
            "output_path": output_csv,
            "message": f"成功计算林冠空隙率。空隙率: {gap_fraction:.4f}",
            "data": {
                "gap_fraction": round(gap_fraction, 4),
                "canopy_cover": round(1.0 - gap_fraction, 4),
                "height_range": f"{min_z:.2f}m - {max_z:.2f}m",
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"计算林冠空隙率失败: {e}",
            "data": None,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")

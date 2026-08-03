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
import gini_voxel_module


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


mcp = FastMCP("GiniCoeffAnalyzer")


@mcp.tool(
    name="extract_gini_coeff",
    description=(
        "Compute the Gini coefficient from point cloud data. "
        "The Gini coefficient measures the inequality in the spatial distribution of points within the canopy. "
        "A higher Gini coefficient indicates more uneven point distribution (clumped canopy). "
        "A lower Gini coefficient indicates more uniform distribution. "
        "If the user does not provide the file_path, ask them to clarify it. "
        "If the user does not specify an output path, results will be saved in the current directory."
    ),
)
async def extract_gini_coeff(file_path: str, output_path: str = "./") -> dict:
    """
    Compute the Gini coefficient from a forest point cloud.

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

        gini_coeff = gini_voxel_module.gini_voxelized(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], treeid
        )

        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_csv = os.path.join(output_path, f"{base_name}_gini_coeff.csv")

        gini_df = pd.DataFrame({"gini_coeff": [gini_coeff]})
        gini_df.to_csv(output_csv, index=False)

        return {
            "status": "success",
            "output_path": output_csv,
            "message": f"成功计算基尼系数。基尼系数: {gini_coeff:.4f}",
            "data": {
                "gini_coeff": round(gini_coeff, 4),
                "interpretation": "0表示完全均匀分布，1表示完全不均匀分布"
                if gini_coeff > 0.5
                else "0表示完全均匀分布，1表示完全不均匀分布",
                "height_range": f"{min_z:.2f}m - {max_z:.2f}m",
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"计算基尼系数失败: {e}",
            "data": None,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")

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

import lai_calculator

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


def calculate_fhd(lad_values):
    """计算FHD (Foliar Height Diversity) - 基于Shannon多样性指数"""
    lad_values = np.array(lad_values)
    positive_lad = lad_values[lad_values > 0]

    if len(positive_lad) == 0:
        return np.nan

    relative_lad = positive_lad / np.sum(positive_lad)
    shannon = -np.sum(relative_lad * np.log(relative_lad))

    return shannon


mcp = FastMCP("FHDAnalyzer")


@mcp.tool(
    name="extract_fhd",
    description=(
        "Compute the Foliar Height Diversity (FHD) from point cloud data. "
        "FHD is a measure of vertical diversity in foliage distribution, "
        "based on the Shannon diversity index of the leaf area density profile. "
        "Higher values indicate more complex vertical canopy structure. "
        "If the user does not provide the file_path, ask them to clarify it. "
        "If the user does not specify an output path, results will be saved in the current directory."
    ),
)
async def extract_fhd(
    file_path: str, output_path: str = "./", voxel_size: float = 0.05
) -> dict:
    """
    Compute the Foliar Height Diversity (FHD) from a forest point cloud.

    Parameters
    ----------
    file_path : str
        Path to the input point cloud file (LAS or LAZ format).
    output_path : str, optional
        Directory to save the generated CSV results. Default is the current folder.
    voxel_size : float, optional
        Voxel size for LAI profile calculation. Default is 0.05.

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

        lai_profile = lai_calculator.compute_lai_profile(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], treeid, voxel_size
        )

        fhd_value = calculate_fhd(lai_profile)

        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_csv = os.path.join(output_path, f"{base_name}_fhd.csv")

        fhd_df = pd.DataFrame({"fhd": [fhd_value]})
        fhd_df.to_csv(output_csv, index=False)

        return {
            "status": "success",
            "output_path": output_csv,
            "message": f"成功计算叶面积高度分布多样性。FHD: {fhd_value:.4f}",
            "data": {
                "fhd": round(float(fhd_value), 4),
                "voxel_size": voxel_size,
                "height_range": f"{min_z:.2f}m - {max_z:.2f}m",
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"计算FHD失败: {e}",
            "data": None,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")

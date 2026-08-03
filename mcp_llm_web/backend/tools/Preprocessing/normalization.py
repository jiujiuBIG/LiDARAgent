import os
import sys
import json
import subprocess

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import laspy
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.write import dataframe_to_las
from tools.Preprocessing.utils import clean_point_cloud, get_xyz_array
import ground_raster

mcp = FastMCP("PointCloudNormalizer")

_CSF_SCRIPT = '''
import sys
import os
import numpy as np
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import CSF

def run_csf(xyz_data, cloth_resolution):
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cloth_resolution
    csf.params.iter_time = 500
    csf.params.class_threshold = 0.3
    csf.params.rigidness = 3
    
    csf.setPointCloud(xyz_data)
    ground_index = CSF.VecInt()
    non_ground_index = CSF.VecInt()
    csf.do_filtering(ground_index, non_ground_index)
    
    return np.array(ground_index).tolist(), np.array(non_ground_index).tolist()

if __name__ == "__main__":
    data = json.load(sys.stdin)
    ground_idx, non_ground_idx = run_csf(np.array(data["xyz"]), data["cloth_resolution"])
    result = {"ground": ground_idx, "non_ground": non_ground_idx}
    json.dump(result, sys.stderr)
'''


def _run_csf_in_subprocess(xyz, cloth_resolution):
    """在子进程中运行CSF，避免日志干扰MCP协议"""
    input_data = {
        "xyz": xyz.tolist(),
        "cloth_resolution": cloth_resolution
    }
    
    proc = subprocess.Popen(
        [sys.executable, '-c', _CSF_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = proc.communicate(json.dumps(input_data))
    
    if proc.returncode != 0:
        raise RuntimeError(f"CSF subprocess failed: {stderr}")
    
    result = json.loads(stderr)
    return np.array(result["ground"]), np.array(result["non_ground"])


@mcp.tool(
    name="normalize_point_cloud",
    description=(
        "Normalize point cloud heights relative to ground surface. "
        "Uses CSF algorithm to detect ground points and calculates height above ground for all points. "
        "If the user does not provide the file_path, ask them to clarify it. "
        "If the user does not specify an output path, results will be saved in the current directory."
    ),
)
async def normalize_point_cloud(
    file_path: str, output_path: str = "./", cloth_resolution: float = 0.1
) -> dict:
    """
    Normalize point cloud heights relative to ground surface.

    Args:
        file_path (str): Path to input point cloud file (.las/.laz/.ply)
        output_path (str): Directory to save normalized point cloud
        cloth_resolution (float): CSF cloth resolution for ground filtering

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

    if "z" not in df_clean.columns:
        return {
            "status": "error",
            "output_path": None,
            "message": "点云缺少 'z' 坐标",
            "data": None,
        }

    xyz = get_xyz_array(df_clean)

    ground_index, non_ground_index = _run_csf_in_subprocess(xyz, cloth_resolution)

    ground = xyz[ground_index]
    nonground = xyz[non_ground_index]

    normalized = ground_raster.correct_point_cloud(xyz, ground)

    df_normalized = df_clean.copy()
    df_normalized["z"] = normalized[:, 2]

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_path, f"{base_name}_normalized.las")

    dataframe_to_las(df_normalized, output_file)

    height_min = round(df_normalized["z"].min(), 2)
    height_max = round(df_normalized["z"].max(), 2)

    return {
        "status": "success",
        "output_path": output_file,
        "message": f"成功归一化点云。地面点: {len(ground)}, 非地面点: {len(nonground)}。高度范围: {height_min} 到 {height_max} m。",
        "data": {
            "ground_points": len(ground),
            "non_ground_points": len(nonground),
            "height_min": height_min,
            "height_max": height_max,
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

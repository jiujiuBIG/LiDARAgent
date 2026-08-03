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
from tools.Preprocessing.utils import clean_point_cloud
import CSF

mcp = FastMCP("GroundFilter")

_CSF_SCRIPT = '''
import sys
import os
import numpy as np
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import CSF

def run_csf(xyz_data, cloth_resolution, class_threshold):
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cloth_resolution
    csf.params.iter_time = 500
    csf.params.class_threshold = class_threshold
    csf.params.rigidness = 3
    
    csf.setPointCloud(xyz_data)
    ground_index = CSF.VecInt()
    non_ground_index = CSF.VecInt()
    csf.do_filtering(ground_index, non_ground_index)
    
    return np.array(ground_index).tolist(), np.array(non_ground_index).tolist()

if __name__ == "__main__":
    data = json.load(sys.stdin)
    ground_idx, non_ground_idx = run_csf(
        np.array(data["xyz"]), data["cloth_resolution"], data["class_threshold"]
    )
    result = {"ground": ground_idx, "non_ground": non_ground_idx}
    json.dump(result, sys.stderr)
'''


def _run_csf_in_subprocess(xyz, cloth_resolution, class_threshold):
    input_data = {
        "xyz": xyz.tolist(),
        "cloth_resolution": cloth_resolution,
        "class_threshold": class_threshold
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
    name="extract_ground_points",
    description=(
        "Extract ground and non-ground points from point cloud using CSF (Cloth Simulation Filter) algorithm. "
        "Separates ground from vegetation, buildings, and other objects. "
        "If user does not provide file_path, ask them to clarify it. "
        "If user does not specify an output path, results will be saved in current directory."
    ),
)
async def extract_ground_points(
    file_path: str,
    output_path: str = "./",
    cloth_resolution: float = 0.1,
    class_threshold: float = 0.3,
) -> dict:
    """
    Extract ground and non-ground points from point cloud using CSF algorithm.

    Args:
        file_path (str): Path to input point cloud file
        output_path (str): Directory to save filtered point clouds
        cloth_resolution (float): Cloth resolution for CSF algorithm (default: 0.1)
        class_threshold (float): Classification threshold (default: 0.3)

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
        df = clean_point_cloud(df)
    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "message": f"读取点云文件失败: {e}",
            "data": None,
        }

    if "z" not in df.columns:
        return {
            "status": "error",
            "output_path": None,
            "message": "点云缺少 'z' 坐标",
            "data": None,
        }

    xyz = df[["x", "y", "z"]].values

    ground_index, non_ground_index = _run_csf_in_subprocess(xyz, cloth_resolution, class_threshold)

    ground_points = xyz[ground_index]
    non_ground_points = xyz[non_ground_index]

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    ground_df = df.iloc[ground_index].copy()
    non_ground_df = df.iloc[non_ground_index].copy()

    ground_df["classification"] = 2
    non_ground_df["classification"] = 1

    ground_output = os.path.join(output_path, f"{base_name}_ground.las")
    non_ground_output = os.path.join(output_path, f"{base_name}_nonground.las")

    dataframe_to_las(ground_df, ground_output)
    dataframe_to_las(non_ground_df, non_ground_output)

    ground_ratio = len(ground_points) / len(xyz) * 100

    return {
        "status": "success",
        "output_path": ground_output,
        "message": f"成功分离地面点与非地面点。地面点: {len(ground_points)}，非地面点: {len(non_ground_points)}，地面占比: {ground_ratio:.1f}%",
        "data": {
            "ground_points": len(ground_points),
            "non_ground_points": len(non_ground_points),
            "ground_ratio": round(ground_ratio, 2),
            "ground_file": ground_output,
            "non_ground_file": non_ground_output,
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

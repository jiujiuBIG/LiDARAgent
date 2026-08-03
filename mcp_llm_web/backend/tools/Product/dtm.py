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
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.utils import clean_point_cloud, get_xyz_array
from scipy.interpolate import griddata
from skimage import io
import CSF

mcp = FastMCP("DTMGenerator")

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
    
    return np.array(ground_index).tolist()

if __name__ == "__main__":
    data = json.load(sys.stdin)
    ground_idx = run_csf(np.array(data["xyz"]), data["cloth_resolution"])
    result = {"ground": ground_idx}
    json.dump(result, sys.stderr)
'''


def _run_csf_in_subprocess(xyz, cloth_resolution):
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
    return np.array(result["ground"])


@mcp.tool(
    name="generate_dtm",
    description=(
        "Generate Digital Terrain Model (DTM) from point cloud. "
        "DTM represents bare earth surface after removing vegetation and objects using ground filtering. "
        "If user does not provide file_path, ask them to clarify it. "
        "If user does not specify output_path, results will be saved in current directory."
    ),
)
async def generate_dtm(
    file_path: str,
    output_path: str = "./",
    resolution: float = 0.5,
    cloth_resolution: float = 0.1,
) -> dict:
    """
    Generate Digital Terrain Model (DTM) from point cloud.

    Args:
        file_path (str): Path to input point cloud file
        output_path (str): Directory to save DTM image
        resolution (float): Grid resolution in meters (default: 0.5)
        cloth_resolution (float): CSF cloth resolution for ground filtering (default: 0.1)

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

    ground_index = _run_csf_in_subprocess(pc, cloth_resolution)

    ground = pc[ground_index]

    if len(ground) == 0:
        return {
            "status": "error",
            "output_path": None,
            "message": "未检测到地面点，点云可能不包含地面数据",
            "data": None,
        }

    min_x, min_y, _ = np.min(pc, axis=0)
    max_x, max_y, _ = np.max(pc, axis=0)

    x_grid = np.arange(min_x, max_x, resolution)
    y_grid = np.arange(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)

    dtm_grid = griddata(
        (ground[:, 0], ground[:, 1]),
        ground[:, 2],
        (X, Y),
        method="linear",
        fill_value=np.nan,
    )

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_tif = os.path.join(output_path, f"{base_name}_DTM.tif")
    io.imsave(output_tif, dtm_grid)

    mean_height = np.nanmean(dtm_grid)
    max_height = np.nanmax(dtm_grid)
    min_height = np.nanmin(dtm_grid)

    return {
        "status": "success",
        "output_path": output_tif,
        "message": f"成功生成DTM。网格分辨率: {resolution}m。地面高程范围: {min_height:.2f} 到 {max_height:.2f}m，平均: {mean_height:.2f}m",
        "data": {
            "resolution": resolution,
            "height_min": round(min_height, 2),
            "height_max": round(max_height, 2),
            "height_mean": round(mean_height, 2),
            "ground_points": len(ground),
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

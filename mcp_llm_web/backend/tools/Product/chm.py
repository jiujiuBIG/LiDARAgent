import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import read_point_cloud
from tools.Preprocessing.utils import clean_point_cloud, get_xyz_array
from scipy.interpolate import griddata
from skimage import io

mcp = FastMCP("CHMGenerator")

_CSF_SCRIPT = """
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
    ground_idx, non_ground_idx = run_csf(
        np.array(data["xyz"]), 
        data["cloth_resolution"]
    )
    result = {"ground": ground_idx, "non_ground": non_ground_idx}
    json.dump(result, sys.stderr)
"""


def _run_csf_in_subprocess(xyz, cloth_resolution):
    input_data = {"xyz": xyz.tolist(), "cloth_resolution": cloth_resolution}

    proc = subprocess.Popen(
        [sys.executable, "-c", _CSF_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = proc.communicate(json.dumps(input_data))

    if proc.returncode != 0:
        raise RuntimeError(f"CSF subprocess failed: {stderr}")

    result = json.loads(stderr)
    return np.array(result["ground"]), np.array(result["non_ground"])


@mcp.tool(
    name="generate_chm",
    description=(
        "Generate Canopy Height Model (CHM) from point cloud. "
        "CHM represents vegetation height above ground, calculated as CHM = DSM - DTM. "
        "If user does not provide file_path, ask them to clarify it. "
        "If user does not specify output_path, results will be saved in current directory."
    ),
)
async def generate_chm(
    file_path: str,
    output_path: str = "./",
    resolution: float = 0.5,
    cloth_resolution: float = 0.1,
) -> dict:
    """
    Generate Canopy Height Model (CHM) from point cloud.

    Args:
        file_path (str): Path to input point cloud file
        output_path (str): Directory to save CHM image
        resolution (float): Grid resolution in meters (default: 0.1)
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

    min_x, min_y, _ = np.min(pc, axis=0)
    max_x, max_y, _ = np.max(pc, axis=0)
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0:
        return {
            "status": "error",
            "output_path": None,
            "message": "网格分辨率必须为正数。",
            "data": None,
        }

    x_cells = int(np.ceil((max_x - min_x) / resolution)) + 1
    y_cells = int(np.ceil((max_y - min_y) / resolution)) + 1

    row_indices = np.clip(((max_y - pc[:, 1]) / resolution).astype(int), 0, y_cells - 1)
    col_indices = np.clip(((pc[:, 0] - min_x) / resolution).astype(int), 0, x_cells - 1)

    dsm_grid = np.full((y_cells, x_cells), -np.inf, dtype=np.float32)

    for i in range(len(pc)):
        r, c = row_indices[i], col_indices[i]
        dsm_grid[r, c] = max(dsm_grid[r, c], pc[i, 2])

    ground_index, non_ground_index = _run_csf_in_subprocess(pc, cloth_resolution)

    ground = pc[np.array(ground_index)]

    if len(ground) == 0:
        return {
            "status": "error",
            "output_path": None,
            "message": "未检测到地面点，点云可能不包含地面数据",
            "data": None,
        }
    
    row_indices = np.clip(((max_y - ground[:, 1]) / resolution).astype(int), 0, y_cells - 1)
    col_indices = np.clip(((ground[:, 0] - min_x) / resolution).astype(int), 0, x_cells - 1)

    dtm_grid = np.full((y_cells, x_cells), 99999, dtype=np.float32)
    for i in range(len(ground)):
        r, c = row_indices[i], col_indices[i]
        dtm_grid[r, c] = min(dtm_grid[r, c], ground[i, 2])
    
    chm_grid = dsm_grid - dtm_grid
    chm_grid[chm_grid < 0] = 0
    chm_grid[np.isnan(chm_grid)] = 0
    chm_grid[chm_grid == -np.inf] = 0
    chm_grid[chm_grid == 99999] = 0

    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_tif = os.path.join(output_path, f"{base_name}_CHM.tif")
    transform = from_origin(float(min_x), float(max_y), resolution, resolution)
    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=chm_grid.shape[0],
        width=chm_grid.shape[1],
        count=1,
        dtype="float32",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(chm_grid.astype(np.float32), 1)

    mean_height = np.nanmean(chm_grid)
    max_height = np.nanmax(chm_grid)
    min_height = np.nanmin(chm_grid[chm_grid > 0]) if np.any(chm_grid > 0) else 0.0

    return {
        "status": "success",
        "output_path": output_tif,
        "message": f"成功生成CHM。网格分辨率: {resolution}m。冠层高度范围: {min_height:.2f} 到 {max_height:.2f}m，平均高度: {mean_height:.2f}m。",
        "data": {
            "resolution": resolution,
            "height_min": round(min_height, 2),
            "height_max": round(max_height, 2),
            "height_mean": round(mean_height, 2),
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

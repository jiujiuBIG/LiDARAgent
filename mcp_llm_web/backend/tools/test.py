import os
import sys
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))  # 当前文件所在项目根目录
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP
from tools.readpc import read_point_cloud
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

def compute_3d_fill_index(file_path: str, output_path: str = "./", pillar_size: float = 1.0, voxel_size: float = 0.5) -> str:
    """
    Compute 3D spatial filling index of a point cloud.
    
    Args:
        file_path (str): Path to point cloud file
        output_path (str): Folder to save output image
        pillar_size (float): Size of each pillar in meters
        voxel_size (float): Size of each voxel in meters

    Returns:
        str: Path to the generated colored TIFF
    """

    # =================== 1️⃣ Read point cloud ===================
    try:
        df = read_point_cloud(file_path, is_treeid=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read point cloud {file_path}: {e}")

    if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
        raise ValueError("Point cloud missing x, y, or z coordinates")

    points = df[['x', 'y', 'z']].values

    # =================== 2️⃣ Define pillar grid ===================
    x_min, x_max = points[:,0].min() - 20, points[:,0].max() + 20
    y_min, y_max = points[:,1].min() - 20, points[:,1].max() + 20
    z_min, z_max = points[:,2].min(), points[:,2].max()

    H = int(np.ceil((y_max - y_min) / pillar_size))
    W = int(np.ceil((x_max - x_min) / pillar_size))
    Z = int(np.ceil((z_max - z_min) / voxel_size))

    print(f"Pillars grid: H={H}, W={W}, Voxels per pillar={Z}")

    # =================== 3️⃣ Compute occupancy ===================
    # Map points to pillar indices
    pillar_x = np.floor((points[:,0] - x_min) / pillar_size).astype(int)
    pillar_y = np.floor((points[:,1] - y_min) / pillar_size).astype(int)
    voxel_z = np.floor((points[:,2] - z_min) / voxel_size).astype(int)

    occupancy_map = np.zeros((H, W), dtype=float)
    count_map = np.zeros((H, W), dtype=int)

    # Count filled voxels per pillar
    for x_idx, y_idx, z_idx in zip(pillar_x, pillar_y, voxel_z):
        count_map[y_idx, x_idx] |= 1 << z_idx  # Use bitmask to mark voxel occupied

    # Compute fill ratio per pillar
    occupancy_map = np.zeros_like(count_map, dtype=float)
    for i in range(H):
        for j in range(W):
            mask = count_map[i,j]
            n_filled_voxels = bin(mask).count('1')
            occupancy_map[i,j] = n_filled_voxels / Z

    # =================== 4️⃣ Save as TIFF ===================
    os.makedirs(output_path, exist_ok=True)
    output_tif = os.path.join(output_path, "3D_fill_index.tif")

    transform = from_origin(x_min, y_max, pillar_size, pillar_size)  # top-left corner
    with rasterio.open(
        output_tif,
        'w',
        driver='GTiff',
        height=H,
        width=W,
        count=1,
        dtype=np.float32,
        transform=transform,
        nodata=0
    ) as dst:dst.write(occupancy_map, 1)

    # =================== 6️⃣ MCP JSON output ===================
    print(json.dumps({
        "status": "success",
        "output_tif": output_tif,
        "summary": {
            "pillars_H": H,
            "pillars_W": W
        }
    }, indent=2), flush=True)

    return output_tif

if __name__== '__main__':
    compute_3d_fill_index(r'/mnt/Huxiaomei2/output/Segmentation_leafon.txt',r'/mnt/Huxiaomei2/output')
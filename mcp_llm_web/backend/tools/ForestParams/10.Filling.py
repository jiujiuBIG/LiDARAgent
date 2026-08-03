import os
import sys

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mcp.server.fastmcp import FastMCP
from tools.Preprocessing.read import read_point_cloud
import rasterio
from rasterio.transform import from_origin
import filling_grid
from skimage import io

# Initialize MCP server
mcp = FastMCP("PointCloudFillingAnalyzer")


@mcp.tool(
    name='compute_3d_fill_index',
    description=(
        "Compute 3D point cloud spatial filling index based on voxel occupancy. "
        "Generates a colored 2D image (red-to-green) representing filling rates for each pillar."
    ),
)
async def compute_3d_fill_index(file_path: str, output_path: str = "./", pillar_size: float = 0.5, voxel_size: float = 0.1) -> dict:
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
        df = read_point_cloud(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read point cloud {file_path}: {e}")

    if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
        raise ValueError("Point cloud missing x, y, or z coordinates")

    points = df[['x', 'y', 'z']].values

    # =================== 2️⃣ Define pillar grid ===================
    x_min, x_max = points[:,0].min(), points[:,0].max()
    y_min, y_max = points[:,1].min(), points[:,1].max()
    z_min, z_max = points[:,2].min()+0.5, points[:,2].max()-0.5

    filling = filling_grid.compute_filling_image(
            points, x_min, x_max, y_min, y_max, z_min, z_max,
            voxel_size = voxel_size,  
            agg_xy = pillar_size    
        ) 
    
    filling = pd.DataFrame(filling)
    output_tif = os.path.join(output_path,'Filling.tif')
    filling.to_csv(os.path.join(output_path,'Filling.csv'), index=False)
    io.imsave(output_tif, filling.values)
    print('Filling10_with.tif')

    index = filling == 0
    filling[index] = np.nan
    mean_filling = np.nanmean(filling)*100
    var_filing = np.nanvar(filling)*100
    max_filling = np.nanmax(filling)*100
    min_fillling = np.nanmin(filling)*100

    output_summary = (
        f"Successfully extracted 3D spatial filling index to the specified path {output_tif}.\n"
        f"The average 3D spatial filling index in this area is {mean_filling:.2f} m, with a variance of {var_filing:.2f}.\n"
        f"The highest 3D spatial filling index in this area is {max_filling:.2f} m, and the lowest is {min_fillling:.2f} m.\n"
        f"Please remember to check even if you just view it."
    )

    return output_summary


if __name__ == "__main__":
    mcp.run(transport="stdio")

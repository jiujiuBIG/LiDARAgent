import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from mcp.server.fastmcp import FastMCP
from tools.Product.raster_grid import load_raster_source, save_raster_product, surface_grid

mcp = FastMCP("DSMGenerator")


@mcp.tool(
    name="generate_dsm",
    description=(
        "Generate Digital Surface Model (DSM) from point cloud. "
        "DSM represents the top surface of all objects including vegetation, buildings, and terrain. "
        "GeoTIFF output uses a metre-based projected CRS from LAS/LAZ or a known input_crs. "
        "If user does not specify output_path, results will be saved in current directory."
    ),
)
async def generate_dsm(
    file_path: str,
    output_path: str = "./",
    resolution: float = 0.5,
    input_crs: str | None = None,
) -> dict:
    """Generate a north-up DSM GeoTIFF from a metre-based point cloud."""
    try:
        source = load_raster_source(file_path, resolution, input_crs)
        return save_raster_product(source, surface_grid(source), output_path, "DSM")
    except Exception as exc:
        return {"status": "error", "output_path": None, "message": f"生成 DSM 失败：{exc}", "data": None}


if __name__ == "__main__":
    mcp.run(transport="stdio")

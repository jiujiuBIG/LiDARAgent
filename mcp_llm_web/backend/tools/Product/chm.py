import os
import sys
import json
import subprocess
import numpy as np
from mcp.server.fastmcp import FastMCP
from tools.Product.raster_grid import load_raster_source, save_raster_product, surface_grid, terrain_grid

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
        "GeoTIFF output uses a metre-based projected CRS from LAS/LAZ or a known input_crs. "
        "If user does not specify output_path, results will be saved in current directory."
    ),
)
async def generate_chm(
    file_path: str,
    output_path: str = "./",
    resolution: float = 0.5,
    cloth_resolution: float = 0.1,
    input_crs: str | None = None,
) -> dict:
    """Generate a north-up CHM GeoTIFF from a metre-based point cloud."""
    try:
        source = load_raster_source(file_path, resolution, input_crs)
        ground = source.points[source.ground_mask]
        if not len(ground):
            ground_indices = np.asarray(_run_csf_in_subprocess(source.points, cloth_resolution)[0], dtype=np.intp)
            ground = source.points[ground_indices]
        terrain = terrain_grid(source, ground)
        canopy = np.maximum(surface_grid(source) - terrain, 0.0)
        return save_raster_product(source, canopy, output_path, "CHM")
    except Exception as exc:
        return {"status": "error", "output_path": None, "message": f"生成 CHM 失败：{exc}", "data": None}


if __name__ == "__main__":
    mcp.run(transport="stdio")

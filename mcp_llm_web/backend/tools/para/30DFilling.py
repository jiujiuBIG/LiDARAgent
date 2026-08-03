import numpy as np
import pandas as pd
import os
import rasterio
from rasterio.transform import from_origin
from typing import Union, Dict, Any

def compute_3d_fill_index_with_treeid(
    xyz: np.ndarray,
    treeid: np.ndarray,
    pillar_size: float = 1.0,
    voxel_size: float = 0.5,
    output_path: str = "./"
) -> Dict[str, Any]:

    # =================== 1️⃣ Define the bounds of the grid ===================
    x_min, x_max = xyz[:, 0].min() - 20, xyz[:, 0].max() + 20
    y_min, y_max = xyz[:, 1].min() - 20, xyz[:, 1].max() + 20
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()

    H = int(np.ceil((y_max - y_min) / pillar_size))
    W = int(np.ceil((x_max - x_min) / pillar_size))
    Z = int(np.ceil((z_max - z_min) / voxel_size))

    print(f"Pillars grid: H={H}, W={W}, Voxels per pillar={Z}")

    # =================== 2️⃣ Map points to pillar indices ===================
    pillar_x = np.floor((xyz[:, 0] - x_min) / pillar_size).astype(int)
    pillar_y = np.floor((xyz[:, 1] - y_min) / pillar_size).astype(int)
    voxel_z = np.floor((xyz[:, 2] - z_min) / voxel_size).astype(int)

    # =================== 3️⃣ Create a map for occupancy ===================
    count_map = np.zeros((H, W), dtype=int)
    
    # Use bitmask to mark voxels occupied for each pillar
    for x_idx, y_idx, z_idx in zip(pillar_x, pillar_y, voxel_z):
        count_map[y_idx, x_idx] |= 1 << z_idx  # Mark voxel as occupied using bitmask

    # Compute the fill ratio per pillar
    occupancy_map = np.zeros_like(count_map, dtype=float)
    for i in range(H):
        for j in range(W):
            mask = count_map[i, j]
            n_filled_voxels = bin(mask).count('1')
            occupancy_map[i, j] = n_filled_voxels / Z  # Normalized fill ratio per pillar

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
    ) as dst:
        dst.write(occupancy_map, 1)

    # =================== 5️⃣ Calculate Statistics ===================
    index = occupancy_map != 0
    occupancy_map[index] = np.nan  # Set non-filled pillars to NaN

    mean_filling = np.nanmean(occupancy_map)
    var_filling = np.nanvar(occupancy_map)
    max_filling = np.nanmax(occupancy_map)
    min_filling = np.nanmin(occupancy_map)

    # 计算各树的填充指数
    tree_fill_stats = []
    for tree in np.unique(treeid):
        # 过滤出当前树的点
        tree_mask = treeid == tree
        tree_points = xyz[tree_mask]

        # 获取当前树的柱子坐标
        pillar_x_tree = np.floor((tree_points[:, 0] - x_min) / pillar_size).astype(int)
        pillar_y_tree = np.floor((tree_points[:, 1] - y_min) / pillar_size).astype(int)
        voxel_z_tree = np.floor((tree_points[:, 2] - z_min) / voxel_size).astype(int)

        # 统计每棵树的柱子内填充情况
        tree_count_map = np.zeros((H, W), dtype=int)
        for x_idx, y_idx, z_idx in zip(pillar_x_tree, pillar_y_tree, voxel_z_tree):
            tree_count_map[y_idx, x_idx] |= 1 << z_idx

        # 计算每个柱子的填充比例
        tree_occupancy_map = np.zeros_like(tree_count_map, dtype=float)
        for i in range(H):
            for j in range(W):
                mask = tree_count_map[i, j]
                n_filled_voxels = bin(mask).count('1')
                tree_occupancy_map[i, j] = n_filled_voxels / Z

        # 计算树的填充指数（平均值）
        mean_tree_fill = np.nanmean(tree_occupancy_map)

        tree_fill_stats.append({
            "treeid": tree,
            "mean_fill": mean_tree_fill,
            "n_voxels": np.count_nonzero(tree_occupancy_map),
        })

    # =================== 6️⃣ Prepare Output Summary ===================
    output_summary = {
        "file_path": output_tif,
        "global_stats": {
            "mean_filling": mean_filling,
            "var_filling": var_filling,
            "max_filling": max_filling,
            "min_filling": min_filling,
        },
        "tree_stats": tree_fill_stats
    }

    return output_summary

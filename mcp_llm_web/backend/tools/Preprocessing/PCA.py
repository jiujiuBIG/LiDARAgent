import laspy
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R_scipy
from pathlib import Path
import open3d as o3d

def flatten_las_by_pca(all_coords, instance_ids, ground_id: int = 0):

    ground_indices = (instance_ids == ground_id)
    
    if np.sum(ground_indices) < 3:
        print("❌ 错误：找到的地面点（PredInstance=0）少于 3 个，无法进行 PCA。")
        return None

    ground_coords = all_coords[ground_indices]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ground_coords)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    denoised_ground_coords = np.asarray(pcd.select_by_index(ind).points)
    ground_coords = denoised_ground_coords

    mean_g = ground_coords.mean(axis=0)
    pca = PCA(n_components=3)
    pca.fit(ground_coords - mean_g)
    sorted_indices = np.argsort(pca.explained_variance_)
    normal_vector = pca.components_[sorted_indices[0]] 
    
    target_z = np.array([0, 0, 1])

    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    r = R_scipy.align_vectors([target_z], [normal_vector])
    rotation_matrix = r[0].as_matrix()
    all_coords_centered = all_coords - mean_g
    all_coords_rotated = all_coords_centered @ rotation_matrix.T 
    all_coords_rotated += mean_g
    
    return all_coords_rotated

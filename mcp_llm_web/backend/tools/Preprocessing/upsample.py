import numpy as np
from sklearn.neighbors import NearestNeighbors

def upsample_point_cloud_vectorized(pc, factor=2, k=5):
    """
    Fast point cloud upsampling
    Parameters:
    pc (numpy.ndarray): Input point cloud of shape (N, 3)
    factor (int): Upsampling factor (e.g., 2 doubles the points)
    k (int): Number of nearest neighbors for interpolation
    Returns:
    numpy.ndarray: Upsampled point cloud
    """
    
    N = pc.shape[0]
    new_N = N * factor
    
    # Find k nearest neighbors for each point
    knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pc)
    distances, indices = knn.kneighbors(pc)
    
    # Number of new points to generate
    num_new_points = new_N - N
    
    # Randomly select base points for new points (vectorized)
    base_indices = np.random.randint(0, N, num_new_points)
    neighbor_choices = np.random.randint(1, k, num_new_points)
    neighbor_indices = indices[base_indices, neighbor_choices]
    alpha = np.random.rand(num_new_points, 1)  # Keep shape for broadcasting
    new_points = (1 - alpha) * pc[base_indices] + alpha * pc[neighbor_indices]
    
    # Combine original points with new points
    upsampled_pc = np.vstack([pc, new_points])
    
    return upsampled_pc
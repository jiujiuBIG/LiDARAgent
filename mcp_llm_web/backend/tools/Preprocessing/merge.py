import numpy as np
def combine_point_clouds(pc1, pc2):
    """
    Specialized function for combining point clouds (2D arrays with 3 columns).
    Parameters:
    pc1 (numpy.ndarray): First point cloud of shape (N1, 3)
    pc2 (numpy.ndarray): Second point cloud of shape (N2, 3)
    Returns:
    numpy.ndarray: Combined point cloud of shape (N1+N2, 3)
    """
    return np.vstack([pc1, pc2])
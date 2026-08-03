import numpy as np
import pandas as pd


def clean_point_cloud(df):
    """
    Clean and validate point cloud DataFrame.

    This function ensures:
    1. x, y, z columns exist and are numeric
    2. Remove rows with NaN values
    3. Convert to float64 for compatibility with Open3D, CSF, etc.

    Args:
        df: pandas DataFrame with point cloud data

    Returns:
        df_clean: Cleaned DataFrame with valid float64 x, y, z columns

    Raises:
        ValueError: If no valid points remain after cleaning
    """
    # Check required columns
    for col in ["x", "y", "z"]:
        if col not in df.columns:
            raise ValueError(f"Point cloud missing required column: {col}")

    # Make a copy to avoid modifying original
    df_clean = df.copy()

    # Convert to numeric, coerce errors to NaN
    for col in ["x", "y", "z"]:
        if df_clean[col].dtype == "object":
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Remove rows with NaN in x, y, or z
    df_clean = df_clean.dropna(subset=["x", "y", "z"]).copy()

    # Convert to float64
    for col in ["x", "y", "z"]:
        df_clean[col] = df_clean[col].astype(np.float64)

    # Validate
    if len(df_clean) == 0:
        raise ValueError("No valid points in point cloud after cleaning")

    return df_clean


def get_xyz_array(df):
    """
    Extract x, y, z values as float64 numpy array.

    Args:
        df: pandas DataFrame with point cloud data

    Returns:
        xyz: numpy array of shape (n, 3) with float64 dtype
    """
    df_clean = clean_point_cloud(df)
    return df_clean[["x", "y", "z"]].values.astype(np.float64)


def get_bounding_box(xyz):
    """
    Calculate bounding box with proper float64 dtype.

    Args:
        xyz: numpy array of shape (n, 3)

    Returns:
        bbox_min: numpy array of shape (3,) with min values
        bbox_max: numpy array of shape (3,) with max values
    """
    return (
        np.asarray(xyz.min(axis=0), dtype=np.float64),
        np.asarray(xyz.max(axis=0), dtype=np.float64),
    )

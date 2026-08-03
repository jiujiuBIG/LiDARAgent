import os
import sys

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)
import numpy as np
import open3d as o3d
import laspy
import pandas as pd


def _ensure_float64(series):
    """Ensure pandas Series is float64, converting from object if needed."""
    if series.dtype == "object":
        series = pd.to_numeric(series, errors="coerce")
    return series.astype(np.float64)


def read_point_cloud(file_path, is_treeid=True):
    ext = os.path.splitext(file_path)[1].lower()

    # ---------- 1. PLY ----------
    if ext == ".ply":
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        data = pd.DataFrame(points, columns=["x", "y", "z"])
        data["x"] = _ensure_float64(data["x"])
        data["y"] = _ensure_float64(data["y"])
        data["z"] = _ensure_float64(data["z"])
        return data

    # ---------- 2. LAS / LAZ ----------
    elif ext in [".las", ".laz"]:
        las = laspy.read(file_path)
        df = pd.DataFrame()

        # Convert x, y, z to float64 (handle ScaledArrayView)
        x_arr = np.array(las.x) if not isinstance(las.x, np.ndarray) else las.x
        y_arr = np.array(las.y) if not isinstance(las.y, np.ndarray) else las.y
        z_arr = np.array(las.z) if not isinstance(las.z, np.ndarray) else las.z
        df["x"] = _ensure_float64(pd.Series(x_arr))
        df["y"] = _ensure_float64(pd.Series(y_arr))
        df["z"] = _ensure_float64(pd.Series(z_arr))

        for dimension in las.point_format.dimensions:
            attr_name = dimension.name
            if attr_name not in ["X", "Y", "Z"]:
                attr_value = getattr(las, attr_name)
                # Convert SubFieldView to numpy array
                if not isinstance(attr_value, np.ndarray):
                    attr_value = np.array(attr_value)
                attr_series = pd.Series(attr_value)
                # Convert to float64 for numeric columns
                if attr_name.lower() in [
                    "intensity",
                    "return_number",
                    "number_of_returns",
                    "scan_direction_flag",
                    "edge_of_flight_line",
                    "classification",
                    "synthetic",
                    "key_point",
                    "withheld",
                    "scan_angle_rank",
                    "user_data",
                    "point_source_id",
                    "red",
                    "green",
                    "blue",
                ]:
                    df[attr_name.lower()] = attr_series.astype(np.float64)
                else:
                    df[attr_name.lower()] = _ensure_float64(attr_series)

        return df

    # ---------- 3. TXT ----------
    elif ext == ".txt":
        with open(file_path, "r") as f:
            first_line = f.readline()
        if "," in first_line:
            sep = ","
        elif "\t" in first_line:
            sep = "\t"
        else:
            sep = " "

        df = pd.read_csv(file_path, sep=sep)

        # Ensure x, y, z are float64
        for col in ["x", "y", "z"]:
            if col in df.columns:
                df[col] = _ensure_float64(df[col])

        return df

    # ---------- 4. BIN ----------
    elif ext == ".bin":
        data = np.fromfile(file_path, dtype=np.float32)

        for i in range(2, 10):
            if len(data) % i == 0:
                data = data.reshape(len(data) // i, i)
                break

        columns = ["x", "y", "z"] + ["c" + str(i) for i in range(i - 3)]
        data = pd.DataFrame(data, columns=columns)
        data["x"] = data["x"].astype(np.float64)
        data["y"] = data["y"].astype(np.float64)
        data["z"] = data["z"].astype(np.float64)
        return data

    # ---------- 5. Others ----------
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")

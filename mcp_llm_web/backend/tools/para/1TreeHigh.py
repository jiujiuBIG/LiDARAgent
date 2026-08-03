import numpy as np
import pandas as pd
from typing import Union, Dict, Any

def extract_tree_heights(xyz_or_df: Union[np.ndarray, pd.DataFrame],
                         cols: list = None) -> Dict[str, Any]:

    # Convert input to DataFrame
    if isinstance(xyz_or_df, np.ndarray):
        if xyz_or_df.ndim != 2 or xyz_or_df.shape[1] < 4:
            raise ValueError("numpy input must be shape (N, >=4) with columns x,y,z,treeid (or use cols to specify).")
        if cols is None:
            df = pd.DataFrame(xyz_or_df[:, :4], columns=["x","y","z","treeid"])
        else:
            df = pd.DataFrame(xyz_or_df[:, :4], columns=cols[:4])
    elif isinstance(xyz_or_df, pd.DataFrame):
        df = xyz_or_df.copy()
        if cols is not None:
            # Rename first 4 columns to x,y,z,treeid as specified
            if len(cols) < 4:
                raise ValueError("cols must have at least 4 names when provided for DataFrame.")
            rename_map = {df.columns[i]: cols[i] for i in range(4)}
            df = df.rename(columns=rename_map)
    else:
        raise ValueError("Input must be a numpy.ndarray or pandas.DataFrame.")
    
    # Ensure required columns exist
    if not {"z", "treeid"}.issubset(df.columns):
        raise ValueError("Data must contain 'z' and 'treeid' columns (or provide cols to map them).")
    
    # Filter out background / unlabeled points (treeid == 0)
    df = df[df["treeid"] != 0]
    if df.empty:
        stats = {"count": 0, "mean": np.nan, "var": np.nan, "min": np.nan, "max": np.nan, "p95": np.nan, "p5": np.nan}
        return {"per_tree": [], "stats": stats}
    
    # Group by treeid and compute per-tree height = max(z) - min(z)
    grouped = df.groupby("treeid")["z"]
    per_tree_list = []
    heights = []
    for tid, zs in grouped:
        zs_arr = zs.values
        n_pts = int(zs_arr.size)
        if n_pts == 0:
            continue
        height = float(np.max(zs_arr) - np.min(zs_arr))
        per_tree_list.append({"treeid": int(tid), "height": height, "n_points": n_pts})
        heights.append(height)
    
    heights = np.array(heights, dtype=float)
    
    # Regional statistics over the per-tree heights
    stats = {
        "count": int(heights.size),
        "mean": float(np.nanmean(heights)) if heights.size > 0 else float("nan"),
        "var": float(np.nanvar(heights, ddof=0)) if heights.size > 0 else float("nan"),
        "min": float(np.nanmin(heights)) if heights.size > 0 else float("nan"),
        "max": float(np.nanmax(heights)) if heights.size > 0 else float("nan"),
        "p95": float(np.nanpercentile(heights, 95)) if heights.size > 0 else float("nan"),
        "p5": float(np.nanpercentile(heights, 5)) if heights.size > 0 else float("nan")
    }

    return {"per_tree": per_tree_list, "stats": stats}

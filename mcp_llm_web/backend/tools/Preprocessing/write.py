import laspy
import numpy as np
import pandas as pd
from pathlib import Path

def dataframe_to_las(df, output_path, scale=(0.01, 0.01, 0.01), offset=(0, 0, 0)):
    """
    Convert pandas DataFrame to LAS/LAZ file with all attributes.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with point cloud data
    output_path (str): Output file path (.las or .laz extension)
    scale (tuple): Scale factors for x, y, z (default: 0.01, 0.01, 0.01)
    offset (tuple): Offset for x, y, z (default: 0, 0, 0)
    
    Returns:
    str: Path to saved file
    """

    header = laspy.LasHeader(version="1.4", point_format=6)
    header.scales = np.array(scale)
    header.offsets = np.array(offset)
    las = laspy.LasData(header)
    
    las.x = df['x'].values
    las.y = df['y'].values
    las.z = df['z'].values

    # Map common attributes
    attribute_mapping = {
        'intensity': 'intensity',
        'classification': 'classification',
        'return_number': 'return_number',
        'number_of_returns': 'number_of_returns',
        'scan_angle_rank': 'scan_angle_rank',
        'scan_angle': 'scan_angle',
        'user_data': 'user_data',
        'point_source_id': 'point_source_id',
        'gps_time': 'gps_time',
        'red': 'red',
        'green': 'green',
        'blue': 'blue',
    }
    
    for df_col, las_attr in attribute_mapping.items():
        if df_col in df.columns:
            try:
                setattr(las, las_attr, df[df_col].values)
            except Exception as e:
                print(f"Warning: Could not set {las_attr}: {e}")

    
    # Add any remaining columns as extra dimensions
    extra_columns = [col for col in df.columns 
                     if col not in ['x', 'y', 'z'] and 
                     col not in attribute_mapping.keys()]
    
    for col in extra_columns:
        try:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name=col,
                type=np.float32 if df[col].dtype in [np.float32, np.float64] else np.int32
            ))
            setattr(las, col, df[col].values)
        except Exception as e:
            print(f"Warning: Could not add extra dimension {col}: {e}")
    
    # Save to file
    las.write(output_path)
    print(f"Saved to: {output_path}")
    return output_path


def dataframe_to_txt(df, output_path, sep=',', index=False, header=False):
    """
    Convert DataFrame to TXT file with simple space/tab separated format.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    output_path (str): Output TXT file path
    sep (str): Separator (default: space)
    index (bool): Whether to write row index (default: False)
    header (bool): Whether to write column names (default: False)
    
    Returns:
    str: Path to saved file
    """
    df.to_csv(output_path, sep=sep, index=index, header=header, float_format='%.6f')
    print(f"Saved to: {output_path}")
    return output_path

def dataframe_to_ply(df, output_path, vertex_only=True):
    """
    Convert DataFrame to PLY file (Polygon File Format).
    Parameters:
    df (pandas.DataFrame): Input DataFrame with point cloud data
    output_path (str): Output PLY file path
    vertex_only (bool): If True, creates vertex-only PLY (default: True)
    Returns:
    str: Path to saved file
    """
    
    # Open file for writing
    with open(output_path, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(df)}\n")
        
        # Write property definitions
        for col in df.columns:
            # Determine data type for PLY
            dtype = df[col].dtype
            if dtype in [np.int8, np.int16, np.int32, np.int64]:
                ply_type = "int"
            elif dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
                ply_type = "uint"
            elif dtype in [np.float32, np.float64]:
                ply_type = "float"
            else:
                ply_type = "float"  # Default to float
            
            f.write(f"property {ply_type} {col}\n")
        
        if not vertex_only:
            f.write(f"element face 0\n")
            f.write("property list uchar int vertex_indices\n")
        
        f.write("end_header\n")
        
        # Write vertex data
        for _, row in df.iterrows():
            line = " ".join([f"{row[col]:.6f}" if isinstance(row[col], (float, np.floating)) 
                            else f"{int(row[col])}" 
                            for col in df.columns])
            f.write(line + "\n")
    
    print(f"Saved PLY to: {output_path}")
    return {
        "status": "success",
        "output_path": output_path,
        "message": f"成功导出点云文件。格式: {output_format}，点数: {len(df)}",
        "data": {
            "format": output_format,
            "point_count": len(df)
        }
    }

import lai_calculator
import os
import pandas as pd
from glob import glob
import numpy as np
import laspy
from tqdm import tqdm


def calculate_fhd(lad_values, evenness=False):
    # 过滤掉LAD为0的层
    positive_lad = lad_values[lad_values > 0]
    
    print(f"总层数: {len(lad_values)}")
    print(f"有植被层数: {len(positive_lad)}")
    print(f"LAD范围: {positive_lad.min():.4f} - {positive_lad.max():.4f}")
    
    if len(positive_lad) == 0:
        return np.nan
    
    # 计算相对密度（转换为比例）
    relative_lad = positive_lad / np.sum(positive_lad)
    
    print(f"相对密度范围: {relative_lad.min():.6f} - {relative_lad.max():.6f}")
    
    # 计算Shannon多样性指数
    shannon = -np.sum(relative_lad * np.log(relative_lad))
    
    if evenness:
        # 计算Shannon均匀度
        n_layers = len(positive_lad)
        if n_layers > 1:
            evenness_value = shannon / np.log(n_layers)
            return evenness_value
        else:
            return 1.0
    else:
        return shannon

datatphroot = r'/mnt/Huxiaomei/Graduate/Segmentanytree/bucket_out_folder'

files = []
    
for _ in range(2):
    datatphroot = os.path.join(datatphroot,'*')
    # if os.path.exists(os.path.join(datatphroot,'final_results')):
    file = glob(os.path.join(datatphroot,'final_results','Normal.laz'))
    files.extend(file)

files = np.sort(files)
files = files[200:]
# files = [r'/mnt/Huxiaomei2/Graduate/Segmentanytree/bucket_out_folder/L000003/final_results/Normal_clipdata.las']

for name in tqdm(files):
    spth = ('/').join(name.split('/')[:-1])

    if os.path.exists(os.path.join(spth, "Normal_clipdata.las")):
        name = os.path.join(spth, "Normal_clipdata.las")
        print(name)

    os.makedirs(os.path.join(spth, "Params2026"), exist_ok=True)
    print(os.path.join(spth, "Params2026"))

    # if os.path.exists(os.path.join(spth, "Params2026/LAIProfile5.csv")):
    #     continue

    las = laspy.read(name)

    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    treeid = np.array(las["PredInstance"], dtype=np.int32)
    max_z = np.max(xyz[:, 2])
    min_z = np.min(xyz[:, 2])

    lai_profile = lai_calculator.compute_lai_profile(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        treeid,
        voxel_size=0.02,
        # voxel_size=0.5,
        # z_layer_height=1.0,
        # k_coefficient=0.5,
        # min_z=min_z,
        # max_z=max_z,
    )

    fhd = calculate_fhd(lai_profile)
    fhd = pd.DataFrame({'FHD': [fhd]})
    print(fhd)
    fhd.to_csv(os.path.join(spth, "FHD.csv"), index=False)






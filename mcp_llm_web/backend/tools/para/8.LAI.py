import lai_calculator
import os
import pandas as pd
from glob import glob
import numpy as np
import laspy
from tqdm import tqdm

datatphroot = r'/mnt/Huxiaomei/Graduate/Segmentanytree/bucket_out_folder'

# changename = ['L000021','L000025','L000051','L000052','L000053','L000112']

files = []
# for cn in changename:
#     file = glob(os.path.join(datatphroot,cn,'*','final_results','Normal.laz'))
#     files.extend(file)
#     file = glob(os.path.join(datatphroot,cn,'final_results','Normal.laz'))
#     files.extend(file)
    
for _ in range(2):
    datatphroot = os.path.join(datatphroot,'*')
    # if os.path.exists(os.path.join(datatphroot,'final_results')):
    file = glob(os.path.join(datatphroot,'final_results','Normal.laz'))
    files.extend(file)

files = np.sort(files)
files = files[43:56]

for name in tqdm(files):
    spth = ('/').join(name.split('/')[:-1])

    os.makedirs(os.path.join(spth, "Params2026"), exist_ok=True)
    print(os.path.join(spth, "Params2026"))

    # if os.path.exists(os.path.join(spth, "Params2026/LAIProfile.csv")):
    #     continue

    las = laspy.read(name)

    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    treeid = np.array(las["PredInstance"], dtype=np.int32)
    max_z = np.max(xyz[:, 2])
    min_z = np.min(xyz[:, 2])

    # lai = lai_calculator.compute_total_lai(
    #     xyz[:, 0],
    #     xyz[:, 1],
    #     xyz[:, 2],
    #     treeid
    #     # mad_k=3.0,      # 可调：越大越宽松
    #     # min_points=50,   # 树点少于5个就不算
    # )

    lai_profile = lai_calculator.compute_lai_profile(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        treeid, 0.05
        # voxel_size=0.5,
        # z_layer_height=1.0,
        # k_coefficient=0.5,
        # min_z=min_z,
        # max_z=max_z,
    )

    print(lai_profile)

    lai_profile = pd.DataFrame(lai_profile)
    lai_profile.to_csv(os.path.join(spth, "Params2026/LAIProfile.csv"), index=False)






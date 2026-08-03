import os
import pandas as pd
from glob import glob
import numpy as np
import laspy
import shutil
from tqdm import tqdm
import gini_voxel_module

datatphroot = r'/mnt/Huxiaomei2/Graduate/Segmentanytree/bucket_out_folder'

# changename = ['L000021','L000025','L000051','L000052','L000053','L000112']

# files = []
# for cn in changename:
#     file = glob(os.path.join(datatphroot,cn,'*','final_results','Normal.laz'))
#     files.extend(file)
#     file = glob(os.path.join(datatphroot,cn,'final_results','Normal.laz'))
#     files.extend(file)

files = []
for _ in range(2):
    datatphroot = os.path.join(datatphroot,'*')
    # if os.path.exists(os.path.join(datatphroot,'final_results')):
    file = glob(os.path.join(datatphroot,'final_results','Normal.laz'))
    files.extend(file)

files = np.sort(files)
files = files[200:]

for name in tqdm(files):
    spth = ('/').join(name.split('/')[:-1])

    os.makedirs(os.path.join(spth, "Params2026Ours"), exist_ok=True)
    print(os.path.join(spth, "Params2026Ours"))

    if os.path.exists(os.path.join(spth, "Params2026Ours/gini.csv")):
        continue

    # if os.path.exists(os.path.join(spth, "Params2026/gp.csv")):
    #     shutil.copy(os.path.join(spth, "Params2026/gp.csv"), os.path.join(spth, "Params2026Ours/gp.csv"))
    #     continue

    las = laspy.read(name)

    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    treeid = np.array(las["PredInstance"], dtype=np.int32)
    max_z = np.max(xyz[:, 2])
    min_z = np.min(xyz[:, 2])

    gp = gini_voxel_module.gini_voxelized(xyz[:,0],xyz[:,1],xyz[:,2], treeid)

    gp = pd.DataFrame({'gap_fraction': [gp]})
    print(gp)
    gp.to_csv(os.path.join(spth, "Params2026Ours/gini.csv"), index=False)






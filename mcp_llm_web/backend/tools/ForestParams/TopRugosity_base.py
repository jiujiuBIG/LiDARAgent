import os
import pandas as pd
from glob import glob
import numpy as np
import laspy
import shutil
from tqdm import tqdm
# import rumple_index
import top_rugosity

datatphroot = r'/mnt/Huxiaomei/Graduate/Segmentanytree/bucket_out_folder'

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
files = files[:50][::-1]

for name in tqdm(files):
    spth = ('/').join(name.split('/')[:-1])

    if os.path.exists(os.path.join(spth, "Normal_clipdata.las")):
        name = os.path.join(spth, "Normal_clipdata.las")
        print(name)

    os.makedirs(os.path.join(spth, "Params2026"), exist_ok=True)
    print(os.path.join(spth, "Params2026"))

    # if os.path.exists(os.path.join(spth, "Params2026Ours/rumple.csv")):
    #     continue

    # if os.path.exists(os.path.join(spth, "Params2026/gp.csv")):
    #     shutil.copy(os.path.join(spth, "Params2026/gp.csv"), os.path.join(spth, "Params2026Ours/gp.csv"))
    #     continue

    las = laspy.read(name)

    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    treeid = np.array(las["PredInstance"], dtype=np.int32)
    max_z = np.max(xyz[:, 2])
    min_z = np.min(xyz[:, 2])

    gp = top_rugosity.compute_top_rugosity(xyz[:,0],xyz[:,1],xyz[:,2], treeid, 0.05)

    gp = pd.DataFrame({'gap_fraction': [gp]})
    print(gp)
    gp.to_csv(os.path.join(spth, "Params2026/TopRugosity.csv"), index=False)






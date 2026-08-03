import numpy as np
import ground_raster
import laspy
import os
from glob import glob
from tools.utils.PCA import flatten_las_by_pca
from tools.utils.Ground import CSF_filter

def compute_normal_for_point_clouds(origindata, instanceid):
    if len(instanceid) == 0:
        instanceid = np.zeros(len(origindata))
    PCA_org = flatten_las_by_pca(origindata, instanceid, ground_id=0)
    ground, nonground = CSF_filter(PCA_org)
    orgindata = ground_raster.correct_point_cloud(PCA_org, ground)
    return orgindata
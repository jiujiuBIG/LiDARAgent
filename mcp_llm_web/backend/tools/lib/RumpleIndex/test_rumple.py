"""
cd到本目录并python test_rumple.py，结果如下：
读取LAZ文件...
点云总数: 140889246
地面点: 84480762, 植被点: 56408484
计算 Rumple Index...

==================================================
冠层表面积: 3930.73 平方米
投影面积: 2170.00 平方米
Rumple Index: 1.8114
==================================================
"""
import rumple_index
import laspy
import numpy as np
import os

laz_file_path = "/mnt/Huxiaomei2/Graduate/Segmentanytree/bucket_out_folder/L000009/5巴州若羌县阿拉干_5号样地激光雷达数据/final_results/Normal.laz"

if not os.path.exists(laz_file_path):
    print(f"错误：文件未找到：{laz_file_path}")
    exit()

print("读取LAZ文件...")
las = laspy.read(laz_file_path)
print(f"点云总数: {len(las.x)}")

# 提取坐标
x = las.x
y = las.y
z = las.z

# 生成 treeid：假设分类0为地面点，其他为植被
if hasattr(las, 'classification'):
    classifications = las.classification
    treeid = np.ones(len(x), dtype=np.int32)
    treeid[classifications == 0] = 0
    ground_count = np.sum(treeid == 0)
    veg_count = len(x) - ground_count
    print(f"地面点: {ground_count}, 植被点: {veg_count}")
else:
    print("警告：无分类字段，将所有点视为植被点")
    treeid = np.ones(len(x), dtype=np.int32)

# 调用C++函数
print("计算 Rumple Index...")
# 返回三个值：rumple, surface_area, projection_area
rumple, surf_area, proj_area = rumple_index.compute_rumple_index(x, y, z, treeid)

print("\n" + "="*50)
print(f"冠层表面积: {surf_area:.2f} 平方米")
print(f"投影面积: {proj_area:.2f} 平方米")
print(f"Rumple Index: {rumple:.4f}")
print("="*50)
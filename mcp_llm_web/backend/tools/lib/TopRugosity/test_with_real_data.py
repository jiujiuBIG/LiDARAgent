"""
cd到该目录下，python test_with_real_data.py，结果如下
=== DEBUG: 脚本开始执行 ===
正在读取LAZ文件...
文件读取成功！
点云总数: 140889246
根据分类字段识别出地面点 84480762 个
非地面点 56408484 个
正在调用C++模块计算Top Rugosity（使用95%分位数）...

========================================
Top Rugosity (基于95%分位数) = 3.5076 米
========================================
"""
print("=== DEBUG: 脚本开始执行 ===")
import top_rugosity  # 这是你编译好的C++模块
import laspy
import numpy as np
import os

# --- 配置区：请修改为你的数据路径 ---
laz_file_path = "/mnt/Huxiaomei2/Graduate/Segmentanytree/bucket_out_folder/L000009/5巴州若羌县阿拉干_5号样地激光雷达数据/final_results/Normal.laz"
# --- 配置结束 ---

# 1. 检查文件是否存在
if not os.path.exists(laz_file_path):
    print(f"错误：文件未找到，请检查路径：\n{laz_file_path}")
    exit()

print("正在读取LAZ文件...")
try:
    # 使用laspy读取LAZ文件
    las = laspy.read(laz_file_path)
    print("文件读取成功！")

    # 2. 提取坐标数据
    # 注意：laspy直接读取的x, y, z可能是缩放后的整数，用 .array 获取原始数组
    x = las.x
    y = las.y
    z = las.z
    print(f"点云总数: {len(x)}")

    # 3. 准备 treeid 数组
    # 你的C++代码约定：treeid == 0 为地面点，非0为植被点。
    # 在标准LAZ数据中，地面点的分类值(Classification)通常为0。
    # 我们需要根据分类字段创建一个数组：地面点赋值为0，其他点赋值为1（或其他非0值）。
    # 首先检查数据是否有分类信息
    if hasattr(las, 'classification'):
        classifications = las.classification
        treeid = np.ones(len(x), dtype=np.int32)
        treeid[classifications == 0] = 0   # 关键修改：地面点分类为0
        ground_point_count = np.sum(treeid == 0)
        print(f"根据分类字段识别出地面点 {ground_point_count} 个")
        print(f"非地面点 {len(x) - ground_point_count} 个")
    else:
        # 警告并处理...
        pass

    # 调用C++函数
    print("正在调用C++模块计算Top Rugosity（使用95%分位数）...")
    result_tuple = top_rugosity.compute_top_rugosity(x, y, z, treeid)
    rugosity_value = result_tuple[0]

    print("\n" + "="*40)
    print(f"Top Rugosity (基于95%分位数) = {rugosity_value:.4f} 米")
    print("="*40)

except Exception as e:
    print(f"处理过程中出现错误：{e}")
    # 可以打印更详细的错误信息来帮助调试
    import traceback
    traceback.print_exc()

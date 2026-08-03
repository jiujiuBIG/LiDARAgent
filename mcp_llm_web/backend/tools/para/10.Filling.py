import os
from glob import glob
import laspy
import numpy as np
import pandas as pd
import filling_grid
from skimage import io
import open3d as o3d
from tqdm import tqdm
import concurrent.futures
from functools import partial

# 原始数据处理函数，封装为可并行调用的形式
def process_single_file(file_path, voxel_size=0.01, show_progress=False):
    """
    处理单个LAZ文件
    file_path: LAZ文件路径
    voxel_size: 点云降采样体素大小（设为None跳过降采样）
    show_progress: 是否在单个文件处理时显示进度
    """
    try:
        savepth = ('/').join(str(file_path).split('/')[:-1])
        output_file = os.path.join(savepth, 'Filling_with_0201_20_10.csv')
        
        # 跳过已处理的文件（可选）
        # if os.path.exists(output_file):
        #     return f"Skipped (exists): {file_path}"
        
        # 读取点云
        las = laspy.read(file_path)
        offsets = las.header.offsets
        
        # 构建坐标数组
        xyz = np.vstack([
            las.x - offsets[0],
            las.y - offsets[1],
            las.z
        ]).T.astype(np.float64)
        
        # 点云降采样（可选，可显著加速后续计算）
        # if voxel_size is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.04)
        xyz = np.asarray(downsampled_pcd.points)
        #     if show_progress:
        #         print(f"降采样: {len(las.x)} -> {len(xyz)} 点")
        
        # 提取树ID（如果存在）
        treeid = np.array(las["PredInstance"], dtype=np.int64)
        
        # 计算边界
        min_x, min_y, min_z = map(float, np.min(xyz, axis=0))
        max_x, max_y, max_z = map(float, np.max(xyz, axis=0))
        
        # 使用分位数调整Z范围（排除异常值）
        mix_z = float(np.percentile(xyz[:, -1], 10))
        max_z = float(np.percentile(xyz[:, -1], 90))
        # max_z = 40.0  # 固定最大高度，避免过高
        
        # 计算填充网格
        filling = filling_grid.compute_filling_image(
            xyz, min_x, max_x, min_y, max_y, mix_z, max_z, 0.2, 1.0
        )
        
        # 保存结果
        df = pd.DataFrame(filling)
        df.to_csv(output_file, index=False)
        io.imsave(
            os.path.join(savepth, 'Filling_with_0201_20_10.tif'), 
            filling
        )
        
        return f"Success: {file_path}"
        
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

# 主函数 - 并行处理
def main():
    datatph = r'/mnt/Huxiaomei2/Graduate/Segmentanytree/bucket_out_folder'
    
    # 查找所有LAZ文件
    files = []
    for _ in range(2):
        datatph = os.path.join(datatph, '*')
        file_list = glob(os.path.join(datatph, 'final_results', 'Normal.laz'))
        files.extend(file_list)
    
    files = np.sort(files)[::-1]
    
    print(f"找到 {len(files)} 个文件待处理")
    
    # 配置并行参数
    max_workers = min(os.cpu_count(), len(files), 8)  # 限制最大并行数
    print(f"使用 {max_workers} 个并行进程")
    
    # 创建处理函数的部分应用（固定参数）
    process_func = partial(
        process_single_file, 
        voxel_size=None,  # 可调整为None跳过降采样
        show_progress=False
    )
    
    # 并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_func, f): f for f in files}
        
        # 使用tqdm显示总体进度
        results = []
        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            total=len(files),
            desc="处理文件中"
        ):
            result = future.result()
            results.append(result)
            # 实时显示处理状态（可选）
            # if "Error" in result:
            #     print(f"警告: {result}")
    
    # 输出统计信息
    success_count = sum(1 for r in results if "Success" in r)
    error_count = sum(1 for r in results if "Error" in r)
    skipped_count = sum(1 for r in results if "Skipped" in r)
    
    print(f"\n处理完成！")
    print(f"成功: {success_count}, 错误: {error_count}, 跳过: {skipped_count}")
    
    # 输出错误详情（如果有）
    if error_count > 0:
        print("\n错误详情:")
        for r in results:
            if "Error" in r:
                print(f"  {r}")

if __name__ == "__main__":
    main()
// gini_voxel_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <unordered_map>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

namespace py = pybind11;

// hash for pair<int,int>
struct pair_hash {
    std::size_t operator()(const std::pair<int,int>& p) const noexcept {
        return (static_cast<std::size_t>(p.first) << 32) ^ static_cast<std::size_t>(p.second);
    }
};

// hash for tuple<int,int,int>
struct tuple3_hash {
    std::size_t operator()(const std::tuple<int,int,int>& t) const noexcept {
        std::size_t a = static_cast<std::size_t>(std::get<0>(t));
        std::size_t b = static_cast<std::size_t>(std::get<1>(t));
        std::size_t c = static_cast<std::size_t>(std::get<2>(t));
        std::size_t res = a;
        res = (res * 73856093) ^ (b * 19349663) ^ (c * 83492791);
        return res;
    }
};

/**
 * 计算基尼系数（Gini coefficient）
 * 输入：非负数值向量
 * 输出：基尼系数，空向量返回NaN
 */
double gini_from_vector(std::vector<double> &vals) {
    size_t n = vals.size();
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();

    double sum = 0.0;
    for (double v : vals) sum += v;
    if (sum == 0.0) return 0.0;

    std::sort(vals.begin(), vals.end()); // 升序排序
    
    // 计算 Σ i * x_i (i从1开始)
    long double accum = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        accum += (long double)(i + 1) * vals[i];
    }
    
    long double numerator = 2.0L * accum;
    long double denom = (long double)n * (long double)sum;
    long double G = numerator / denom - ((long double)(n + 1) / (long double)n);
    
    // 处理数值误差
    if (G < 0 && G > -1e-14) G = 0.0;
    return static_cast<double>(G);
}

/**
 * 计算点云的体素化基尼系数
 * 输入的点云假设已经经过高度归一化（地面为0）
 * 
 * 算法步骤：
 * 1) 过滤treeid != 0的点（非地面点）
 * 2) XY平面网格化：对每个小格网保留最大Z值（冠层顶）
 * 3) 3D体素化：将冠层点体素化，计算每个体素的平均高度
 * 4) 收集体素平均高度，计算基尼系数
 * 
 * @param x_array X坐标数组
 * @param y_array Y坐标数组  
 * @param z_array Z坐标数组（已归一化）
 * @param treeid_array 点分类数组（0=地面/非树木）
 * @param small_xy XY网格大小（米），默认0.1米
 * @param voxel_size 体素大小（米），默认0.5米
 * @return 基尼系数（0-1）
 */
double gini_voxelized(py::array_t<double> x_array,
                      py::array_t<double> y_array,
                      py::array_t<double> z_array,
                      py::array_t<int> treeid_array,
                      double small_xy = 0.1,
                      double voxel_size = 0.5)
{
    // 参数有效性检查
    if (small_xy <= 0.0) small_xy = 0.1;
    if (voxel_size <= 0.0) voxel_size = 0.5;

    // 获取数组视图
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto z = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();
    
    size_t N = x_array.shape(0);
    if (N == 0) return std::numeric_limits<double>::quiet_NaN();

    // 步骤1: XY平面网格化，记录每个网格的最大Z值（冠层顶）
    std::unordered_map<std::pair<int,int>, double, pair_hash> xy_max_z;
    xy_max_z.reserve(N / 4 + 16);

    for (size_t i = 0; i < N; ++i) {
        // 跳过地面点（treeid == 0）
        if (tid(i) == 0) continue;
        
        // 计算XY网格坐标
        int gx = static_cast<int>(std::floor(x(i) / small_xy));
        int gy = static_cast<int>(std::floor(y(i) / small_xy));
        
        auto key = std::make_pair(gx, gy);
        auto it = xy_max_z.find(key);
        
        if (it == xy_max_z.end()) {
            xy_max_z.emplace(key, z(i));
        } else {
            if (z(i) > it->second) it->second = z(i);
        }
    }

    // 如果没有树木点，返回NaN
    if (xy_max_z.empty()) return std::numeric_limits<double>::quiet_NaN();

    // 步骤2: 3D体素化冠层点，计算每个体素的平均高度
    std::unordered_map<std::tuple<int,int,int>, std::pair<double,int>, tuple3_hash> voxel_accum;
    voxel_accum.reserve(xy_max_z.size() * 2);

    for (const auto &kv : xy_max_z) {
        int gx = kv.first.first;
        int gy = kv.first.second;
        double canopy_z = kv.second;
        
        // 计算网格中心坐标（代表冠层点的位置）
        double px = (gx + 0.5) * small_xy;
        double py_ = (gy + 0.5) * small_xy;
        
        // 计算体素坐标
        int vx = static_cast<int>(std::floor(px / voxel_size));
        int vy = static_cast<int>(std::floor(py_ / voxel_size));
        int vz = static_cast<int>(std::floor(canopy_z / voxel_size));
        
        auto vkey = std::make_tuple(vx, vy, vz);
        auto it = voxel_accum.find(vkey);
        
        if (it == voxel_accum.end()) {
            voxel_accum.emplace(vkey, std::make_pair(canopy_z, 1));
        } else {
            it->second.first += canopy_z;
            it->second.second += 1;
        }
    }

    if (voxel_accum.empty()) return std::numeric_limits<double>::quiet_NaN();

    // 步骤3: 计算每个体素的平均高度
    std::vector<double> voxel_means;
    voxel_means.reserve(voxel_accum.size());
    
    for (const auto &kv : voxel_accum) {
        double sumz = kv.second.first;
        int cnt = kv.second.second;
        voxel_means.push_back(sumz / static_cast<double>(cnt));
    }

    // 步骤4: 计算体素平均高度的基尼系数
    double G = gini_from_vector(voxel_means);
    return G;
}

// Python模块定义
PYBIND11_MODULE(gini_voxel_module, m) {
    m.doc() = "计算归一化点云的体素化基尼系数（输入点云需已归一化）";
    m.def("gini_voxelized", &gini_voxelized,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("z_array"),
          py::arg("treeid_array"),
          py::arg("small_xy") = 0.1,
          py::arg("voxel_size") = 0.5,
          "计算点云的体素化基尼系数\n\n"
          "参数:\n"
          "    x_array: X坐标数组\n"
          "    y_array: Y坐标数组\n"
          "    z_array: Z坐标数组（已归一化，地面为0）\n"
          "    treeid_array: 点分类数组（0=地面/非树木）\n"
          "    small_xy: XY网格大小（米），默认0.1\n"
          "    voxel_size: 体素大小（米），默认0.5\n"
          "返回:\n"
          "    基尼系数（0-1），无数据返回NaN");
}
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

// hash for tuple<int,int,int>
struct tuple3_hash {
    std::size_t operator()(const std::tuple<int,int,int>& t) const noexcept {
        std::size_t a = static_cast<std::size_t>(std::get<0>(t));
        std::size_t b = static_cast<std::size_t>(std::get<1>(t));
        std::size_t c = static_cast<std::size_t>(std::get<2>(t));
        // 使用不同的质数混合哈希值，减少冲突
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
 * 2) 直接对点云进行3D体素化，体素大小可自定义（默认0.05米）
 * 3) 计算每个体素的平均高度
 * 4) 收集体素平均高度，计算基尼系数
 * 
 * @param x_array X坐标数组
 * @param y_array Y坐标数组  
 * @param z_array Z坐标数组（已归一化，地面为0）
 * @param treeid_array 点分类数组（0=地面/非树木）
 * @param voxel_size 体素大小（米），默认0.05米
 * @return 基尼系数（0-1），无数据返回NaN
 */
double gini_voxelized(py::array_t<double> x_array,
                             py::array_t<double> y_array,
                             py::array_t<double> z_array,
                             py::array_t<int> treeid_array,
                             double voxel_size = 0.05)
{
    // 参数有效性检查
    if (voxel_size <= 0.0) voxel_size = 0.05;

    // 获取数组视图
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto z = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();
    
    size_t N = x_array.shape(0);
    if (N == 0) return std::numeric_limits<double>::quiet_NaN();

    // 直接进行3D体素化，计算每个体素的平均高度
    std::unordered_map<std::tuple<int,int,int>, std::pair<double,int>, tuple3_hash> voxel_accum;
    
    // 预分配空间以提升性能（估计约30%的点是非地面点）
    voxel_accum.reserve(N / 3 + 16);

    for (size_t i = 0; i < N; ++i) {
        // 跳过地面点（treeid == 0）
        if (tid(i) == 0) continue;
        
        // 计算体素坐标
        int vx = static_cast<int>(std::floor(x(i) / voxel_size));
        int vy = static_cast<int>(std::floor(y(i) / voxel_size));
        int vz = static_cast<int>(std::floor(z(i) / voxel_size));
        
        auto vkey = std::make_tuple(vx, vy, vz);
        auto it = voxel_accum.find(vkey);
        
        if (it == voxel_accum.end()) {
            voxel_accum.emplace(vkey, std::make_pair(z(i), 1));
        } else {
            it->second.first += z(i);
            it->second.second += 1;
        }
    }

    // 如果没有树木点，返回NaN
    if (voxel_accum.empty()) return std::numeric_limits<double>::quiet_NaN();

    // 计算每个体素的平均高度
    std::vector<double> voxel_means;
    voxel_means.reserve(voxel_accum.size());
    
    for (const auto &kv : voxel_accum) {
        double sumz = kv.second.first;
        int cnt = kv.second.second;
        voxel_means.push_back(sumz / static_cast<double>(cnt));
    }

    // 计算体素平均高度的基尼系数
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
          py::arg("voxel_size") = 0.05,
          "计算点云的体素化基尼系数（直接体素化）\n\n"
          "参数:\n"
          "    x_array: X坐标数组\n"
          "    y_array: Y坐标数组\n"
          "    z_array: Z坐标数组（已归一化，地面为0）\n"
          "    treeid_array: 点分类数组（0=地面/非树木）\n"
          "    voxel_size: 体素大小（米），默认0.05\n"
          "返回:\n"
          "    基尼系数（0-1），无数据返回NaN");
    
    // 保留原函数以便对比（可选）
    #ifdef KEEP_ORIGINAL
    m.def("gini_voxelized", &gini_voxelized,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("z_array"),
          py::arg("treeid_array"),
          py::arg("small_xy") = 0.1,
          py::arg("voxel_size") = 0.5,
          "计算点云的体素化基尼系数（两步法：XY网格化+体素化）");
    #endif
}
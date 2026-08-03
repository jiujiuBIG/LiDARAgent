#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// -------------------- 自定义 pair<int,int> hash --------------------
struct pair_hash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// -------------------- 点云抽稀函数 --------------------
/**
 * 对点云进行格网抽稀，每个small_cell_size格子内只保留一个点
 * 返回抽稀后的点索引列表
 */
std::vector<size_t> thin_point_cloud(py::array_t<double> x_array,
                                      py::array_t<double> y_array,
                                      py::array_t<int> treeid_array,
                                      double small_cell_size) {
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto treeid = treeid_array.unchecked<1>();
    size_t N = x_array.shape(0);
    
    std::unordered_map<std::pair<int,int>, size_t, pair_hash> grid_map;
    std::vector<size_t> kept_indices;
    
    for (size_t i = 0; i < N; i++) {
        // 跳过地面点
        if (treeid(i) == 0) continue;
        
        // 计算小格网坐标
        int gx = static_cast<int>(std::floor(x(i) / small_cell_size));
        int gy = static_cast<int>(std::floor(y(i) / small_cell_size));
        auto key = std::make_pair(gx, gy);
        
        // 如果这个格子还没有点，保留当前点
        if (grid_map.find(key) == grid_map.end()) {
            grid_map[key] = i;
            kept_indices.push_back(i);
        }
    }
    
    return kept_indices;
}

// -------------------- 计算空隙占比 --------------------
/**
 * 计算点云的冠层空隙占比
 * 步骤：
 * 1. 先用 small_thin_size (0.05m) 对点云进行抽稀
 * 2. 再用 cell_size (0.2m) 统计空隙占比
 */
double fraction_canopy_gaps(py::array_t<double> x_array,
                            py::array_t<double> y_array,
                            py::array_t<int> treeid_array,
                            double thin_size = 0.05,    // 抽稀格网大小
                            double cell_size = 0.2) {   // 统计格网大小
                            
    // 参数检查
    if (thin_size <= 0.0) thin_size = 0.05;
    if (cell_size <= 0.0) cell_size = 0.2;
    if (cell_size < thin_size) {
        // 警告：统计格网应该大于等于抽稀格网
        // 这里不做处理，但用户需要注意
    }
    
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto treeid = treeid_array.unchecked<1>();
    size_t N = x_array.shape(0);
    if (N == 0) return std::numeric_limits<double>::quiet_NaN();
    
    // 步骤1：点云抽稀
    std::vector<size_t> kept_indices = thin_point_cloud(x_array, y_array, treeid_array, thin_size);
    
    if (kept_indices.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // 步骤2：计算抽稀后点云的范围
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    
    for (size_t idx : kept_indices) {
        double xi = x(idx);
        double yi = y(idx);
        if (xi < min_x) min_x = xi;
        if (xi > max_x) max_x = xi;
        if (yi < min_y) min_y = yi;
        if (yi > max_y) max_y = yi;
    }
    
    // 步骤3：用cell_size统计被占用的格子
    std::unordered_map<std::pair<int,int>, bool, pair_hash> occupied;
    
    for (size_t idx : kept_indices) {
        int gx = static_cast<int>(std::floor((x(idx) - min_x) / cell_size));
        int gy = static_cast<int>(std::floor((y(idx) - min_y) / cell_size));
        occupied[{gx, gy}] = true;
    }
    
    // 步骤4：计算总格子数和空隙占比
    int nx = static_cast<int>(std::ceil((max_x - min_x) / cell_size)) + 1;
    int ny = static_cast<int>(std::ceil((max_y - min_y) / cell_size)) + 1;
    
    // 防止整数溢出
    if (nx <= 0 || ny <= 0 || nx > 1000000 || ny > 1000000) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    long long total_cells = static_cast<long long>(nx) * static_cast<long long>(ny);
    long long occupied_cells = static_cast<long long>(occupied.size());
    long long gap_cells = total_cells - occupied_cells;
    
    return static_cast<double>(gap_cells) / static_cast<double>(total_cells);
}

// -------------------- 简化版：只抽稀不统计（调试用） --------------------
py::array_t<int> thin_only(py::array_t<double> x_array,
                            py::array_t<double> y_array,
                            py::array_t<int> treeid_array,
                            double thin_size = 0.05) {
    std::vector<size_t> kept = thin_point_cloud(x_array, y_array, treeid_array, thin_size);
    
    // 创建结果数组
    auto result = py::array_t<int>(kept.size());
    auto r = result.mutable_unchecked<1>();
    for (size_t i = 0; i < kept.size(); i++) {
        r(i) = static_cast<int>(kept[i]);
    }
    return result;
}

// -------------------- Python 模块 --------------------
PYBIND11_MODULE(canopy_gaps, m) {
    m.doc() = R"pbdoc(
        Canopy gap fraction calculation with point cloud thinning
        
        Step 1: Thin point cloud to 0.05m grid (keep one point per cell)
        Step 2: Calculate gap fraction using 0.2m grid cells
    )pbdoc";
    
    m.def("fraction_canopy_gaps", &fraction_canopy_gaps,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("treeid_array"),
          py::arg("thin_size") = 0.05,
          py::arg("cell_size") = 0.2,
          R"pbdoc(
          计算冠层空隙占比
          
          参数:
              x_array: X坐标数组
              y_array: Y坐标数组
              treeid_array: 点分类数组 (0=地面点，跳过)
              thin_size: 抽稀格网大小（米），默认0.05
              cell_size: 统计格网大小（米），默认0.2
          
          返回:
              空隙占比 (0-1)，无数据返回NaN
          )pbdoc");
    
    m.def("thin_only", &thin_only,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("treeid_array"),
          py::arg("thin_size") = 0.05,
          "仅进行抽稀，返回保留点的索引（调试用）");
}
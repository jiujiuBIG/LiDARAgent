#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>

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

// -------------------- 计算空隙占比（改进版） --------------------
double fraction_canopy_gaps(py::array_t<double> x_array,
                            py::array_t<double> y_array,
                            py::array_t<int> treeid_array,
                            double thin_size = 0.05,    // 抽稀格网大小
                            double cell_size = 0.2) {   // 统计格网大小
                            
    // 参数检查
    if (thin_size <= 0.0) thin_size = 0.05;
    if (cell_size <= 0.0) cell_size = 0.2;
    
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto treeid = treeid_array.unchecked<1>();
    size_t N = x_array.shape(0);
    
    // 输入验证
    if (N == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
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
        min_x = std::min(min_x, xi);
        max_x = std::max(max_x, xi);
        min_y = std::min(min_y, yi);
        max_y = std::max(max_y, yi);
    }
    
    // 添加小偏移量避免边界问题
    double eps = 1e-6;
    min_x -= eps;
    min_y -= eps;
    max_x += eps;
    max_y += eps;
    
    // 步骤3：计算格网数量和占用情况
    std::unordered_map<std::pair<int,int>, bool, pair_hash> occupied;
    
    // 计算总格子数（使用double避免溢出）
    double width = max_x - min_x;
    double height = max_y - min_y;
    
    // 处理极端情况：点云范围太小
    if (width < cell_size * 0.1 || height < cell_size * 0.1) {
        // 如果点云范围太小，至少保证1x1的格网
        occupied[{0, 0}] = true;
        return 0.0;  // 没有空隙
    }
    
    int nx = static_cast<int>(std::ceil(width / cell_size));
    int ny = static_cast<int>(std::ceil(height / cell_size));
    
    // 防止格网数量过大或为负
    if (nx <= 0) nx = 1;
    if (ny <= 0) ny = 1;
    
    // 安全检查：防止格网数量过大导致内存问题
    const int MAX_GRID_SIZE = 1000000;
    if (nx > MAX_GRID_SIZE || ny > MAX_GRID_SIZE) {
        // 如果格网太大，可能是参数问题，返回NaN
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // 标记被占用的格子
    for (size_t idx : kept_indices) {
        int gx = static_cast<int>(std::floor((x(idx) - min_x) / cell_size));
        int gy = static_cast<int>(std::floor((y(idx) - min_y) / cell_size));
        
        // 边界检查
        if (gx >= 0 && gx < nx && gy >= 0 && gy < ny) {
            occupied[{gx, gy}] = true;
        }
    }
    
    // 步骤4：计算总格子数和空隙占比
    long long total_cells = static_cast<long long>(nx) * static_cast<long long>(ny);
    long long occupied_cells = static_cast<long long>(occupied.size());
    
    // 确保被占用的格子数不超过总格子数
    occupied_cells = std::min(occupied_cells, total_cells);
    
    double gap_fraction = 1.0 - static_cast<double>(occupied_cells) / static_cast<double>(total_cells);

    std::cout << "occupied_cells/total_cells = " << occupied_cells << "/" << total_cells 
              << " = " << (static_cast<double>(occupied_cells) / static_cast<double>(total_cells)) << std::endl;
    std::cout << "gap_fraction = 1 - " << (static_cast<double>(occupied_cells) / static_cast<double>(total_cells)) 
              << " = " << gap_fraction << std::endl;
    
    // 确保结果在[0,1]范围内
    if (gap_fraction < 0.0) gap_fraction = 0.0;
    if (gap_fraction > 1.0) gap_fraction = 1.0;
    
    return gap_fraction;
}

// -------------------- 带调试信息的版本 --------------------
py::tuple fraction_canopy_gaps_debug(py::array_t<double> x_array,
                                      py::array_t<double> y_array,
                                      py::array_t<int> treeid_array,
                                      double thin_size = 0.05,
                                      double cell_size = 0.2) {
    
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto treeid = treeid_array.unchecked<1>();
    size_t N = x_array.shape(0);
    
    if (N == 0) {
        return py::make_tuple(std::numeric_limits<double>::quiet_NaN(), 0, 0);
    }
    
    std::vector<size_t> kept_indices = thin_point_cloud(x_array, y_array, treeid_array, thin_size);
    
    if (kept_indices.empty()) {
        return py::make_tuple(std::numeric_limits<double>::quiet_NaN(), 0, 0);
    }
    
    // 计算范围
    double min_x = x(kept_indices[0]), max_x = x(kept_indices[0]);
    double min_y = y(kept_indices[0]), max_y = y(kept_indices[0]);
    
    for (size_t idx : kept_indices) {
        min_x = std::min(min_x, x(idx));
        max_x = std::max(max_x, x(idx));
        min_y = std::min(min_y, y(idx));
        max_y = std::max(max_y, y(idx));
    }
    
    double width = max_x - min_x + cell_size;  // 添加一个单元格的宽度
    double height = max_y - min_y + cell_size;
    
    int nx = static_cast<int>(std::ceil(width / cell_size));
    int ny = static_cast<int>(std::ceil(height / cell_size));
    
    if (nx <= 0) nx = 1;
    if (ny <= 0) ny = 1;
    
    long long total_cells = static_cast<long long>(nx) * static_cast<long long>(ny);
    
    std::unordered_map<std::pair<int,int>, bool, pair_hash> occupied;
    for (size_t idx : kept_indices) {
        int gx = static_cast<int>(std::floor((x(idx) - min_x) / cell_size));
        int gy = static_cast<int>(std::floor((y(idx) - min_y) / cell_size));
        occupied[{gx, gy}] = true;
    }
    
    long long occupied_cells = static_cast<long long>(occupied.size());
    double gap_fraction = 1.0 - static_cast<double>(occupied_cells) / static_cast<double>(total_cells);
    
    
    // 返回结果和调试信息
    return py::make_tuple(gap_fraction, total_cells, occupied_cells);
}

// -------------------- 简化版：只抽稀不统计 --------------------
py::array_t<int> thin_only(py::array_t<double> x_array,
                            py::array_t<double> y_array,
                            py::array_t<int> treeid_array,
                            double thin_size = 0.05) {
    std::vector<size_t> kept = thin_point_cloud(x_array, y_array, treeid_array, thin_size);
    
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
        
        Step 1: Thin point cloud to thin_size grid (keep one point per cell)
        Step 2: Calculate gap fraction using cell_size grid cells
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
    
    m.def("fraction_canopy_gaps_debug", &fraction_canopy_gaps_debug,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("treeid_array"),
          py::arg("thin_size") = 0.05,
          py::arg("cell_size") = 0.2,
          "带调试信息的版本，返回(gap_fraction, total_cells, occupied_cells)");
    
    m.def("thin_only", &thin_only,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("treeid_array"),
          py::arg("thin_size") = 0.05,
          "仅进行抽稀，返回保留点的索引（调试用）");
}
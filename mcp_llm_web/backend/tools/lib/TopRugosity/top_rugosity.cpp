#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>
#include <algorithm>

namespace py = pybind11;

// 简单的 pair<int,int> 哈希
struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// 主函数：计算Top Rugosity（使用95%分位数）- 简化版，假设输入已经是归一化高度
py::tuple compute_top_rugosity(
    py::array_t<double> x_array,
    py::array_t<double> y_array,
    py::array_t<double> z_array,
    py::array_t<int>    treeid_array,
    double cell_size = 1.0  // 网格大小参数
) {
    auto x   = x_array.unchecked<1>();
    auto y   = y_array.unchecked<1>();
    auto z   = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();

    const size_t N = x_array.shape(0);
    if (N == 0) {
        return py::make_tuple(py::float_(NAN));
    }

    // 1) 遍历所有非地面点（treeid != 0），按网格收集高度值
    std::unordered_map<std::pair<int,int>, std::vector<double>, PairHash> cell_heights;
    cell_heights.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        // 跳过地面点（treeid == 0）
        if (tid(i) == 0) continue;
        
        // 计算点所在的网格坐标
        int cx = static_cast<int>(std::floor(x(i) / cell_size));
        int cy = static_cast<int>(std::floor(y(i) / cell_size));
        auto key = std::make_pair(cx, cy);
        
        // 直接使用原始Z值（假设已经是归一化高度）
        cell_heights[key].push_back(z(i));
    }

    if (cell_heights.empty()) {
        return py::make_tuple(py::float_(NAN));
    }

    // 2) 对每个网格计算95%分位数
    std::vector<double> quantiles;
    quantiles.reserve(cell_heights.size());

    for (auto& kv : cell_heights) {
        std::vector<double>& heights = kv.second;
        if (heights.empty()) continue;
        
        // 排序
        std::sort(heights.begin(), heights.end());
        
        // 计算95%分位数索引（0-based）
        size_t idx = static_cast<size_t>(0.95 * (heights.size() - 1));
        quantiles.push_back(heights[idx]);
    }

    if (quantiles.empty()) {
        return py::make_tuple(py::float_(NAN));
    }

    // 3) 计算这些分位数的标准差
    double sum = 0.0, sum2 = 0.0;
    size_t M = quantiles.size();
    for (double q : quantiles) {
        sum += q;
        sum2 += q * q;
    }
    double mean = sum / M;
    double var = (sum2 / M) - mean * mean;
    if (var < 0.0) var = 0.0;
    double std = std::sqrt(var);

    return py::make_tuple(py::float_(std));
}

// 模块定义
PYBIND11_MODULE(top_rugosity, m) {
    m.doc() = "Top Rugosity computation module (using 95th percentile) - Simplified version for normalized points";
    m.def("compute_top_rugosity", &compute_top_rugosity,
          py::arg("x_array"), py::arg("y_array"), py::arg("z_array"), py::arg("treeid_array"),
          py::arg("cell_size") = 1.0,
          "Compute Top Rugosity: std of 95th percentile of height in each grid cell. "
          "Assumes input points are already normalized (treeid=0 points are ignored).");
}
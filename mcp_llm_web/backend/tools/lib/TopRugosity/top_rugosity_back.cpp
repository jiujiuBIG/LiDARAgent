#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>
#include <algorithm>   // 新增，用于 std::sort

namespace py = pybind11;

// 简单的 pair<int,int> 哈希（保持不变）
struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// 插值函数（保持不变）
double interpolate_ground(
    int gx, int gy,
    const std::unordered_map<std::pair<int,int>, double, PairHash>& ground_map,
    int max_radius = 5
) {
    auto it = ground_map.find({gx, gy});
    if (it != ground_map.end()) {
        return it->second;
    }

    for (int r = 1; r <= max_radius; ++r) {
        double sum = 0.0;
        int cnt = 0;
        for (int dx = -r; dx <= r; ++dx) {
            for (int dy = -r; dy <= r; ++dy) {
                if (dx == 0 && dy == 0) continue;
                auto it2 = ground_map.find({gx + dx, gy + dy});
                if (it2 != ground_map.end()) {
                    sum += it2->second;
                    cnt++;
                }
            }
        }
        if (cnt > 0) {
            return sum / cnt;
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

// 主函数：计算 Top Rugosity（使用95%分位数）
py::tuple compute_top_rugosity(
    py::array_t<double> x_array,
    py::array_t<double> y_array,
    py::array_t<double> z_array,
    py::array_t<int>    treeid_array,
    double ground_grid_size = 1.0,
    double cell1m_size      = 1.0
) {
    auto x   = x_array.unchecked<1>();
    auto y   = y_array.unchecked<1>();
    auto z   = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();

    const size_t N = x_array.shape(0);
    if (N == 0) {
        return py::make_tuple(py::float_(NAN));
    }

    // 1) 统计所有点所在的地面网格（用于插值）
    std::unordered_set<std::pair<int,int>, PairHash> all_cells;
    all_cells.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        int gx = static_cast<int>(std::floor(x(i) / ground_grid_size));
        int gy = static_cast<int>(std::floor(y(i) / ground_grid_size));
        all_cells.emplace(gx, gy);
    }

    // 2) 从 treeid==0 的点中取每个地面网格的最小z作为地面点
    std::unordered_map<std::pair<int,int>, double, PairHash> ground_min;
    ground_min.reserve(N / 4);
    for (size_t i = 0; i < N; ++i) {
        if (tid(i) != 0) continue;
        int gx = static_cast<int>(std::floor(x(i) / ground_grid_size));
        int gy = static_cast<int>(std::floor(y(i) / ground_grid_size));
        auto key = std::make_pair(gx, gy);
        auto it = ground_min.find(key);
        if (it == ground_min.end()) {
            ground_min[key] = z(i);
        } else if (z(i) < it->second) {
            it->second = z(i);
        }
    }

    // 3) 对所有有点的网格进行插值
    std::unordered_map<std::pair<int,int>, double, PairHash> ground_full;
    ground_full.reserve(all_cells.size());
    for (const auto& cell : all_cells) {
        int gx = cell.first;
        int gy = cell.second;
        double gz = interpolate_ground(gx, gy, ground_min, 5);
        ground_full[cell] = gz;
    }

    // 4) 遍历点，计算归一化高度，并按1m网格收集高度列表
    std::unordered_map<std::pair<int,int>, std::vector<double>, PairHash> cell1m_heights;
    for (size_t i = 0; i < N; ++i) {
        int gx = static_cast<int>(std::floor(x(i) / ground_grid_size));
        int gy = static_cast<int>(std::floor(y(i) / ground_grid_size));
        auto key = std::make_pair(gx, gy);
        auto itg = ground_full.find(key);
        if (itg == ground_full.end() || std::isnan(itg->second)) {
            continue;
        }
        double z_norm = z(i) - itg->second;

        int cx = static_cast<int>(std::floor(x(i) / cell1m_size));
        int cy = static_cast<int>(std::floor(y(i) / cell1m_size));
        auto ckey = std::make_pair(cx, cy);

        cell1m_heights[ckey].push_back(z_norm);
    }

    if (cell1m_heights.empty()) {
        return py::make_tuple(py::float_(NAN));
    }

    // 5) 对每个网格计算95%分位数
    std::vector<double> quantiles;
    quantiles.reserve(cell1m_heights.size());

    for (auto& kv : cell1m_heights) {
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

    // 6) 计算这些分位数的标准差
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
    m.doc() = "Top Rugosity computation module (using 95th percentile)";
    m.def("compute_top_rugosity", &compute_top_rugosity,
          py::arg("x_array"), py::arg("y_array"), py::arg("z_array"), py::arg("treeid_array"),
          py::arg("ground_grid_size") = 1.0,
          py::arg("cell1m_size") = 1.0,
          "Compute Top Rugosity: std of 95th percentile of normalized height in 1x1 m cells.");
}
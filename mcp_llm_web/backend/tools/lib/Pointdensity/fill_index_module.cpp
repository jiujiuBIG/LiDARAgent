#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <cmath>
#include <vector>
#include <numeric>  // std::accumulate

namespace py = pybind11;

// 自定义的哈希函数，用于 pair<int, int>
struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        auto h1 = std::hash<int>{}(p.first);  // 计算 p.first 的哈希值
        auto h2 = std::hash<int>{}(p.second); // 计算 p.second 的哈希值
        return h1 ^ (h2 << 1);  // 合并两个哈希值
    }
};

// 计算网格索引
std::pair<int, int> get_grid_index(double x, double y, double grid_size = 1.0) {
    int grid_x = static_cast<int>(std::floor(x / grid_size));
    int grid_y = static_cast<int>(std::floor(y / grid_size));
    return {grid_x, grid_y};
}

// 主函数：统计每个网格的点云数量，并计算平均点云数量
double compute_point_density(py::array_t<double> x_array, py::array_t<double> y_array, py::array_t<double> z_array, py::array_t<int> treeid_array, double grid_size = 1.0) {
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto z = z_array.unchecked<1>();
    auto treeid = treeid_array.unchecked<1>();

    const size_t N = x_array.shape(0);
    if (N == 0) {
        return 0.0;  
    }

    // 使用自定义的哈希函数
    std::unordered_map<std::pair<int, int>, size_t, PairHash> grid_count;

    // 遍历每个点，计算其所属的网格，并统计每个网格的点数量
    for (size_t i = 0; i < N; ++i) {
        int gx, gy;
        std::tie(gx, gy) = get_grid_index(x(i), y(i), grid_size);
        grid_count[{gx, gy}] += 1;
    }

    // 计算平均点云数量
    double total_count = std::accumulate(grid_count.begin(), grid_count.end(), 0.0, [](double sum, const std::pair<std::pair<int, int>, size_t>& entry) {
        return sum + entry.second;
    });

    // 计算所有网格的平均点云数量
    double average_density = total_count / grid_count.size();
    return average_density;
}

// 通过 pybind11 导出模块
PYBIND11_MODULE(fill_index_module, m) {
    m.def("compute_point_density", &compute_point_density, 
          py::arg("x_array"), py::arg("y_array"), py::arg("z_array"), py::arg("treeid_array"),
          py::arg("grid_size") = 1.0, "Compute average point density per grid.");
}

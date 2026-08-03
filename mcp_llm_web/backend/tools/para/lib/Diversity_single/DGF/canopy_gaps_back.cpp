#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <cmath>
#include <limits>

namespace py = pybind11;

// -------------------- 自定义 pair<int,int> hash --------------------
struct pair_hash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);  // 简单混合
    }
};

// -------------------- 计算 1m² 空隙占比 --------------------
double fraction_canopy_gaps(py::array_t<double> x_array,
                            py::array_t<double> y_array,
                            py::array_t<int> treeid_array,  // 添加treeid数组
                            double cell_size=1.0) {
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto treeid = treeid_array.unchecked<1>();  // 获取treeid数据
    size_t N = x_array.shape(0);
    if (N == 0) return std::numeric_limits<double>::quiet_NaN();

    // 使用 unordered_map 统计每个 1m² 栅格是否有点
    std::unordered_map<std::pair<int,int>, bool, pair_hash> occupied;

    for (size_t i = 0; i < N; i++) {
        if (treeid(i) == 0) continue;  // 如果是地面点，跳过

        int gx = static_cast<int>(std::floor(x(i) / cell_size));
        int gy = static_cast<int>(std::floor(y(i) / cell_size));
        occupied[{gx, gy}] = true;  // 标记该格子有点
    }

    // 计算总格子数
    double min_x = x(0), max_x = x(0);
    double min_y = y(0), max_y = y(0);
    for (size_t i = 1; i < N; i++) {
        if(x(i) < min_x) min_x = x(i);
        if(x(i) > max_x) max_x = x(i);
        if(y(i) < min_y) min_y = y(i);
        if(y(i) > max_y) max_y = y(i);
    }

    int nx = static_cast<int>(std::ceil((max_x - min_x) / cell_size));
    int ny = static_cast<int>(std::ceil((max_y - min_y) / cell_size));
    int total_cells = nx * ny;

    int occupied_cells = occupied.size();
    int gap_cells = total_cells - occupied_cells;

    return static_cast<double>(gap_cells) / total_cells;
}

// -------------------- Python 模块 --------------------
PYBIND11_MODULE(canopy_gaps, m) {
    m.doc() = "Fraction of 1m² canopy gaps";
    m.def("fraction_canopy_gaps", &fraction_canopy_gaps,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("treeid_array"),  // 新增treeid参数
          py::arg("cell_size") = 1.0);
}

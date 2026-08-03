#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>

namespace py = pybind11;

// 简单的 pair<int,int> 哈希
struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// 从已有 ground_map 中为指定 cell 插值地面高程
// 简单做法：按曼哈顿/棋盘距离扩圈找最近有值的邻居，取平均
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
    // 实在找不到
    return std::numeric_limits<double>::quiet_NaN();
}


// 主函数：地面归一化 + 统计
py::tuple compute_ground_normalized_stats(
    py::array_t<double> x_array,
    py::array_t<double> y_array,
    py::array_t<double> z_array,
    py::array_t<int>    treeid_array,
    double ground_grid_size = 1.0,  // 地面网格大小 (m)
    double cell3m_size      = 3.0   // 第二次 3×3m 网格尺寸
) {
    auto x   = x_array.unchecked<1>();
    auto y   = y_array.unchecked<1>();
    auto z   = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();

    const size_t N = x_array.shape(0);
    if (N == 0) {
        return py::make_tuple(
            py::float_(NAN),
            py::float_(NAN),
            py::float_(NAN),
            py::float_(NAN)
        );
    }

    // 1) 统计所有点所在的地面网格 (用于后面插值)
    std::unordered_set<std::pair<int,int>, PairHash> all_cells;
    all_cells.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        int gx = static_cast<int>(std::floor(x(i) / ground_grid_size));
        int gy = static_cast<int>(std::floor(y(i) / ground_grid_size));
        all_cells.emplace(gx, gy);
    }

    // 2) 在 treeid == 0 的点中，按地面网格取最小 z 作为地面点
    std::unordered_map<std::pair<int,int>, double, PairHash> ground_min;
    ground_min.reserve(N / 4 + 1);

    for (size_t i = 0; i < N; ++i) {
        if (tid(i) != 0) continue;  // 只使用 treeid==0 的点作为“非植被/地面候选”

        int gx = static_cast<int>(std::floor(x(i) / ground_grid_size));
        int gy = static_cast<int>(std::floor(y(i) / ground_grid_size));
        auto key = std::make_pair(gx, gy);

        auto it = ground_min.find(key);
        if (it == ground_min.end()) {
            ground_min[key] = z(i);
        } else {
            if (z(i) < it->second) {
                it->second = z(i);
            }
        }
    }

    // 3) 对所有有点的网格进行地面插值（如果该网格没有 ground_min）
    std::unordered_map<std::pair<int,int>, double, PairHash> ground_full;
    ground_full.reserve(all_cells.size());

    for (const auto& cell : all_cells) {
        int gx = cell.first;
        int gy = cell.second;

        double gz = interpolate_ground(gx, gy, ground_min, /*max_radius=*/5);
        ground_full[cell] = gz;  // 可以是 NaN，后面会判断
    }

    // 4) 对所有点做地面归一化 z_norm = z - ground_z(cell)
    double sum_z = 0.0;
    double sum_z2 = 0.0;
    size_t count_z = 0;

    // 为 3×3m 网格统计准备
    struct Stats {
        double sum = 0.0;
        double sum2 = 0.0;
        size_t n = 0;
    };
    std::unordered_map<std::pair<int,int>, Stats, PairHash> cell3_stats;

    for (size_t i = 0; i < N; ++i) {
        int gx = static_cast<int>(std::floor(x(i) / ground_grid_size));
        int gy = static_cast<int>(std::floor(y(i) / ground_grid_size));
        auto key = std::make_pair(gx, gy);

        auto itg = ground_full.find(key);
        if (itg == ground_full.end()) {
            // 没有插值结果，跳过
            continue;
        }
        double gz = itg->second;
        if (std::isnan(gz)) {
            // 插值失败，跳过
            continue;
        }

        double z_norm = z(i) - gz;

        // 全局 z_norm 统计
        sum_z  += z_norm;
        sum_z2 += z_norm * z_norm;
        count_z++;

        // 3×3m 网格统计（基于归一化 z）
        int cx = static_cast<int>(std::floor(x(i) / cell3m_size));
        int cy = static_cast<int>(std::floor(y(i) / cell3m_size));
        auto ckey = std::make_pair(cx, cy);

        auto &st = cell3_stats[ckey];
        st.sum  += z_norm;
        st.sum2 += z_norm * z_norm;
        st.n    += 1;
    }

    if (count_z == 0 || cell3_stats.empty()) {
        return py::make_tuple(
            py::float_(NAN),
            py::float_(NAN),
            py::float_(NAN),
            py::float_(NAN)
        );
    }

    // 5) 计算全局 z_norm 的均值和标准差
    double mean_z = sum_z / static_cast<double>(count_z);
    double var_z  = (sum_z2 / static_cast<double>(count_z)) - mean_z * mean_z;
    if (var_z < 0.0) var_z = 0.0;
    double std_z  = std::sqrt(var_z);

    // 6) 对每个 3×3m 网格，计算该网格内 z_norm 的 std，然后再对这些 std 做均值和 std
    std::vector<double> cell_std_list;
    cell_std_list.reserve(cell3_stats.size());

    for (const auto &kv : cell3_stats) {
        const Stats &st = kv.second;
        if (st.n < 2) continue;  // 至少需要两个点才能算 std

        double m = st.sum / static_cast<double>(st.n);
        double v = (st.sum2 / static_cast<double>(st.n)) - m * m;
        if (v < 0.0) v = 0.0;
        double s = std::sqrt(v);
        cell_std_list.push_back(s);
    }

    if (cell_std_list.empty()) {
        return py::make_tuple(
            py::float_(mean_z),
            py::float_(std_z),
            py::float_(NAN),
            py::float_(NAN)
        );
    }

    double sum_cs  = 0.0;
    double sum_cs2 = 0.0;
    size_t M = cell_std_list.size();
    for (double v : cell_std_list) {
        sum_cs  += v;
        sum_cs2 += v * v;
    }

    double mean_cell_std = sum_cs / static_cast<double>(M);
    double var_cell_std  = (sum_cs2 / static_cast<double>(M)) - mean_cell_std * mean_cell_std;
    if (var_cell_std < 0.0) var_cell_std = 0.0;
    double std_cell_std   = std::sqrt(var_cell_std);

    // 返回四个统计量：
    // 1) 归一化后 z 的全局均值
    // 2) 归一化后 z 的全局标准差
    // 3) 所有 3×3m 网格内 std 的平均值
    // 4) 所有 3×3m 网格内 std 的标准差
    return py::make_tuple(
        py::float_(mean_z),
        py::float_(std_z),
        py::float_(mean_cell_std),
        py::float_(std_cell_std)
    );
}

// 你在 PYBIND11_MODULE(...) 里把这个函数导出：
PYBIND11_MODULE(my_lidar_tools, m) {
    m.def("compute_ground_normalized_stats", &compute_ground_normalized_stats,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("z_array"),
          py::arg("treeid_array"),
          py::arg("ground_grid_size") = 1.0,
          py::arg("cell3m_size")      = 3.0,
          "Compute ground-normalized Z stats and 3x3m grid Z-std stats.");
}

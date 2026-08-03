#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // 添加此行，支持 STL 容器与 Python 类型之间的转换
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include <set>
#include <algorithm>  // For std::min, std::max

namespace py = pybind11;

// -------------------- Hash 函数 --------------------
// pair_hash 用于 std::pair<int, int>
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);  // 简单混合
    }
};

// tuple3_hash 用于 std::tuple<int, int, int>
struct tuple3_hash {
    std::size_t operator()(const std::tuple<int,int,int>& t) const {
        auto h1 = std::hash<int>{}(std::get<0>(t));
        auto h2 = std::hash<int>{}(std::get<1>(t));
        auto h3 = std::hash<int>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// -------------------- VAI 计算 --------------------
// 计算每一层的 VAI
std::vector<double> calculate_VAI_per_layer(py::array_t<double> x_array,
                                            py::array_t<double> y_array,
                                            py::array_t<double> z_array,
                                            py::array_t<int> treeid_array,
                                            double small_voxel_size=0.1,
                                            int large_voxel_ratio=5,
                                            double z_layer_height=1.0) {

    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto z = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();
    size_t N = x_array.shape(0);
    if (N == 0) return {};

    // -------------------- 计算地面点 --------------------
    // 使用 std::map 记录每个 voxel 中的最低地面点
    std::map<std::tuple<int, int>, double> ground_voxels;
    for (size_t i = 0; i < N; ++i) {
        if (tid(i) == 0) {  // 仅考虑地面点
            int xv = static_cast<int>(std::floor(x(i) / small_voxel_size));
            int yv = static_cast<int>(std::floor(y(i) / small_voxel_size));
            int zv = static_cast<int>(std::floor(z(i) / small_voxel_size));
            auto voxel = std::make_tuple(xv, yv);

            // 更新每个网格中的最小地面点
            if (ground_voxels.find(voxel) == ground_voxels.end() || ground_voxels[voxel] > z(i)) {
                ground_voxels[voxel] = z(i);
            }
        }
    }

    // -------------------- 插值处理 --------------------
    // 对没有地面点的网格进行插值
    std::map<std::tuple<int, int>, double> interpolated_ground_voxels = ground_voxels;

    // 对每个没有地面点的网格进行插值（邻近插值）
    for (const auto& kv : ground_voxels) {
        int xv = std::get<0>(kv.first);
        int yv = std::get<1>(kv.first);

        // 对四邻域进行插值：上下左右
        std::vector<double> neighbors;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue;  // 跳过中心点
                auto neighbor = std::make_tuple(xv + dx, yv + dy);
                if (ground_voxels.find(neighbor) != ground_voxels.end()) {
                    neighbors.push_back(ground_voxels[neighbor]);
                }
            }
        }

        // 如果邻近点有值，取平均
        if (!neighbors.empty()) {
            double interpolated_height = std::accumulate(neighbors.begin(), neighbors.end(), 0.0) / neighbors.size();
            interpolated_ground_voxels[{xv, yv}] = interpolated_height;
        }
    }

    // -------------------- 去地面化 --------------------
    // 使用地面点网格化计算去地面化后的点云
    std::vector<double> zp(N);
    for (size_t i = 0; i < N; ++i) {
        int xv = static_cast<int>(std::floor(x(i) / small_voxel_size));
        int yv = static_cast<int>(std::floor(y(i) / small_voxel_size));
        auto voxel = std::make_tuple(xv, yv);

        // 查找地面点并去地面化
        if (interpolated_ground_voxels.find(voxel) != interpolated_ground_voxels.end()) {
            double ground_height = interpolated_ground_voxels[voxel];
            zp[i] = z(i) - ground_height;  // 去地面化
        } else {
            zp[i] = z(i);  // 如果没有找到地面点，保持原高度
        }
    }

    // -------------------- Z轴分层 --------------------
    double z_min = *std::min_element(zp.begin(), zp.end());
    double z_max = *std::max_element(zp.begin(), zp.end());

    int z_layer_count = std::ceil((z_max - z_min) / z_layer_height);

    // -------------------- 小 voxel --------------------
    std::unordered_map<std::tuple<int,int,int>, bool, tuple3_hash> small_voxels;
    for (size_t i = 0; i < N; ++i) {
        if (tid(i) == 0) continue;  // 忽略地面
        int xv = static_cast<int>(std::floor(x(i) / small_voxel_size));
        int yv = static_cast<int>(std::floor(y(i) / small_voxel_size));
        int zv = static_cast<int>(std::floor(zp[i] / small_voxel_size));
        small_voxels[std::make_tuple(xv, yv, zv)] = true;
    }

    // 预先计算每个 Z 层的点云索引
    std::vector<std::vector<size_t>> z_layer_points(z_layer_count);
    for (size_t i = 0; i < N; ++i) {
        double z_value = zp[i];
        int layer_idx = std::min(static_cast<int>((z_value - z_min) / z_layer_height), z_layer_count - 1);
        z_layer_points[layer_idx].push_back(i);
    }

    // 存储每一层的 VAI 值
    std::vector<double> vai_per_layer;

    // 对每个 Z 层进行计算
    for (int layer = 0; layer < z_layer_count; ++layer) {
        double z_layer_min = z_min + layer * z_layer_height;
        double z_layer_max = z_layer_min + z_layer_height;

        // 计算该 Z 层内的 VAI
        double layer_VAI_sum = 0.0;
        size_t layer_voxel_count = 0;

        // 对该 Z 层的所有点进行处理
        std::unordered_map<std::pair<int, int>, double, pair_hash> large_voxel_VAI_layer;
        std::unordered_map<std::tuple<int, int, int>, bool, tuple3_hash> occupied_large_voxels;;
        for (size_t idx : z_layer_points[layer]) {
            double z_value = zp[idx];
            if (z_value >= z_layer_min && z_value < z_layer_max && tid(idx) != 0) {  // 确保在当前Z层且非地面
                int xv = static_cast<int>(std::floor(x(idx) / small_voxel_size));
                int yv = static_cast<int>(std::floor(y(idx) / small_voxel_size));
                int zv = static_cast<int>(std::floor(z_value / small_voxel_size));

                std::tuple<int, int, int> voxel_key = std::make_tuple(xv, yv, zv);

                // 计算大 voxel 累加
                int lx = xv / large_voxel_ratio;
                int ly = yv / large_voxel_ratio;
                if (occupied_large_voxels.find(voxel_key) == occupied_large_voxels.end()) {
                    large_voxel_VAI_layer[{lx, ly}] += 1.0;
                    occupied_large_voxels[voxel_key] = true;   // 记录该组合
                }
            }
        }

        // 计算该 Z 层所有大 voxel 的 VAI
        for (auto &kv : large_voxel_VAI_layer) {
            double voxel_VAI = kv.second / std::pow(large_voxel_ratio, 3);  // 计算每个大 voxel 的 VAI
            layer_VAI_sum += voxel_VAI;
            layer_voxel_count++;
        }

        // 对该 Z 层的 VAI 做平均
        if (layer_voxel_count > 0) {
            vai_per_layer.push_back(layer_VAI_sum / layer_voxel_count);  // 存储该层的 VAI
        }
    }

    return vai_per_layer;  // 返回所有 Z 层的 VAI
}

// -------------------- Python 模块 --------------------
PYBIND11_MODULE(vai_module, m) {
    m.doc() = "VAI calculation using 2-level voxelization and Z layer splitting, with ground removal and interpolation for empty voxels";
    m.def("calculate_VAI_per_layer", &calculate_VAI_per_layer,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("z_array"),
          py::arg("treeid_array"),
          py::arg("small_voxel_size")=0.1,
          py::arg("large_voxel_ratio")=5,
          py::arg("z_layer_height")=1.0);
}

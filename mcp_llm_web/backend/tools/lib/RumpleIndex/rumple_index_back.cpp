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

// 用于 pair<int,int> 的哈希（复用）
struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// 插值函数（与 TopRugosity 相同）
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

// 计算 Rumple Index 的主函数
py::tuple compute_rumple_index(
    py::array_t<double> x_array,
    py::array_t<double> y_array,
    py::array_t<double> z_array,
    py::array_t<int>    treeid_array,
    double ground_grid_size = 1.0,   // 地面网格大小（用于插值）
    double surface_grid_size = 1.0   // 冠层表面网格大小（用于表面积计算）
) {
    auto x   = x_array.unchecked<1>();
    auto y   = y_array.unchecked<1>();
    auto z   = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();

    const size_t N = x_array.shape(0);
    if (N == 0) {
        return py::make_tuple(py::float_(NAN), py::float_(NAN), py::float_(NAN));
    }

    // ---------- 地面归一化部分（与之前相同）----------
    std::unordered_set<std::pair<int,int>, PairHash> all_cells;
    all_cells.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        int gx = static_cast<int>(std::floor(x(i) / ground_grid_size));
        int gy = static_cast<int>(std::floor(y(i) / ground_grid_size));
        all_cells.emplace(gx, gy);
    }

    std::unordered_map<std::pair<int,int>, double, PairHash> ground_min;
    ground_min.reserve(N / 4);
    for (size_t i = 0; i < N; ++i) {
        if (tid(i) != 0) continue;   // 只用地面积点
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

    std::unordered_map<std::pair<int,int>, double, PairHash> ground_full;
    ground_full.reserve(all_cells.size());
    for (const auto& cell : all_cells) {
        int gx = cell.first;
        int gy = cell.second;
        double gz = interpolate_ground(gx, gy, ground_min, 5);
        ground_full[cell] = gz;
    }

    // ---------- 计算冠层表面网格高度（取每个网格内植被点的最大 z_norm）----------
    // 存储每个冠层网格的最高归一化高度，键为网格索引 (ix, iy)
    std::unordered_map<std::pair<int,int>, double, PairHash> canopy_max;
    for (size_t i = 0; i < N; ++i) {
        if (tid(i) == 0) continue;   // 忽略地面点（只取植被点）
        int gx = static_cast<int>(std::floor(x(i) / ground_grid_size));
        int gy = static_cast<int>(std::floor(y(i) / ground_grid_size));
        auto key = std::make_pair(gx, gy);
        auto itg = ground_full.find(key);
        if (itg == ground_full.end() || std::isnan(itg->second)) {
            continue;   // 无法获得地面高程，跳过
        }
        double z_norm = z(i) - itg->second;

        int cx = static_cast<int>(std::floor(x(i) / surface_grid_size));
        int cy = static_cast<int>(std::floor(y(i) / surface_grid_size));
        auto ckey = std::make_pair(cx, cy);

        auto it = canopy_max.find(ckey);
        if (it == canopy_max.end()) {
            canopy_max[ckey] = z_norm;
        } else if (z_norm > it->second) {
            it->second = z_norm;
        }
    }

    if (canopy_max.empty()) {
        return py::make_tuple(py::float_(NAN), py::float_(NAN), py::float_(NAN));
    }

    // ---------- 计算表面积（通过相邻三角形）----------
    double total_surface_area = 0.0;
    int valid_cells = 0;

    // 为了快速查找，将 canopy_max 放入一个 unordered_map 中，键不变
    // 我们直接使用 canopy_max 进行查找

    for (const auto& kv : canopy_max) {
        int ix = kv.first.first;
        int iy = kv.first.second;
        double h_center = kv.second;

        // 检查右邻网格 (ix+1, iy)
        auto it_right = canopy_max.find({ix+1, iy});
        // 检查上邻网格 (ix, iy+1)
        auto it_up = canopy_max.find({ix, iy+1});

        if (it_right != canopy_max.end() && it_up != canopy_max.end()) {
            // 两个邻网格都存在，可以形成两个三角形
            // 三角形1: (ix,iy) - (ix+1,iy) - (ix,iy+1)
            // 三角形2: (ix+1,iy) - (ix+1,iy+1) - (ix,iy+1)
            // 我们需要 (ix+1,iy+1) 的高度，即右上角网格
            auto it_right_up = canopy_max.find({ix+1, iy+1});
            if (it_right_up != canopy_max.end()) {
                // 计算三角形1的顶点坐标（以网格左下角为原点，但面积计算与x,y偏移无关，因为网格等距）
                // 将网格中心坐标设为 (ix * surface_grid_size, iy * surface_grid_size) 方便计算
                double x0 = ix * surface_grid_size;
                double y0 = iy * surface_grid_size;
                double x1 = (ix+1) * surface_grid_size;
                double y1 = iy * surface_grid_size;
                double x2 = ix * surface_grid_size;
                double y2 = (iy+1) * surface_grid_size;
                double x3 = (ix+1) * surface_grid_size;
                double y3 = (iy+1) * surface_grid_size;

                // 三角形1: A(x0,y0,h_center), B(x1,y1,it_right->second), C(x2,y2,it_up->second)
                double ABx = x1 - x0;
                double ABy = y1 - y0;
                double ABz = it_right->second - h_center;
                double ACx = x2 - x0;
                double ACy = y2 - y0;
                double ACz = it_up->second - h_center;
                // 叉积
                double nx1 = ABy * ACz - ABz * ACy;
                double ny1 = ABz * ACx - ABx * ACz;
                double nz1 = ABx * ACy - ABy * ACx;
                double area1 = 0.5 * std::sqrt(nx1*nx1 + ny1*ny1 + nz1*nz1);

                // 三角形2: B(x1,y1,it_right->second), D(x3,y3,it_right_up->second), C(x2,y2,it_up->second)
                double BDx = x3 - x1;
                double BDy = y3 - y1;
                double BDz = it_right_up->second - it_right->second;
                double BCx = x2 - x1;
                double BCy = y2 - y1;
                double BCz = it_up->second - it_right->second;
                double nx2 = BDy * BCz - BDz * BCy;
                double ny2 = BDz * BCx - BDx * BCz;
                double nz2 = BDx * BCy - BDy * BCx;
                double area2 = 0.5 * std::sqrt(nx2*nx2 + ny2*ny2 + nz2*nz2);

                total_surface_area += area1 + area2;
                valid_cells++;   // 这个网格有效（贡献了表面积）
            }
        }

    }

    if (valid_cells == 0) {
        return py::make_tuple(py::float_(NAN), py::float_(NAN), py::float_(NAN));
    }

    double projection_area = valid_cells * surface_grid_size * surface_grid_size;
    double rumple = total_surface_area / projection_area;

    return py::make_tuple(py::float_(rumple),
                          py::float_(total_surface_area),
                          py::float_(projection_area));
}

// 模块定义
PYBIND11_MODULE(rumple_index, m) {
    m.doc() = "Rumple Index computation module (surface area / projected area)";
    m.def("compute_rumple_index", &compute_rumple_index,
          py::arg("x_array"), py::arg("y_array"), py::arg("z_array"), py::arg("treeid_array"),
          py::arg("ground_grid_size") = 1.0,
          py::arg("surface_grid_size") = 1.0,
          "Compute Rumple Index: ratio of canopy surface area to projected ground area.");
}
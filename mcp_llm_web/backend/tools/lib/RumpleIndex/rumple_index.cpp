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

// 用于 pair<int,int> 的哈希
struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// 计算 Rumple Index 的主函数 - 简化版，假设输入已经是归一化高度
py::tuple compute_rumple_index(
    py::array_t<double> x_array,
    py::array_t<double> y_array,
    py::array_t<double> z_array,
    py::array_t<int>    treeid_array,
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

    // ---------- 计算冠层表面网格高度（取每个网格内植被点的最大高度）----------
    // 存储每个冠层网格的最高高度，键为网格索引 (ix, iy)
    std::unordered_map<std::pair<int,int>, double, PairHash> canopy_max;
    
    for (size_t i = 0; i < N; ++i) {
        if (tid(i) == 0) continue;   // 忽略地面点（只取植被点）
        
        // 直接使用原始Z值（假设已经是归一化高度）
        double height = z(i);
        
        int cx = static_cast<int>(std::floor(x(i) / surface_grid_size));
        int cy = static_cast<int>(std::floor(y(i) / surface_grid_size));
        auto ckey = std::make_pair(cx, cy);

        auto it = canopy_max.find(ckey);
        if (it == canopy_max.end()) {
            canopy_max[ckey] = height;
        } else if (height > it->second) {
            it->second = height;
        }
    }

    if (canopy_max.empty()) {
        return py::make_tuple(py::float_(NAN), py::float_(NAN), py::float_(NAN));
    }

    // ---------- 计算表面积（通过相邻三角形）----------
    double total_surface_area = 0.0;
    int valid_cells = 0;

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
                // 计算三角形1的顶点坐标
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
    m.doc() = "Rumple Index computation module - Simplified version for normalized points";
    m.def("compute_rumple_index", &compute_rumple_index,
          py::arg("x_array"), py::arg("y_array"), py::arg("z_array"), py::arg("treeid_array"),
          py::arg("surface_grid_size") = 1.0,
          "Compute Rumple Index: ratio of canopy surface area to projected ground area. "
          "Assumes input points are already normalized (treeid=0 points are ignored).");
}
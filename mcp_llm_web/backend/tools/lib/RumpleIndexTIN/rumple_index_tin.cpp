#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>
#include <algorithm>

// CGAL 头文件
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/Point_2.h>
#include <CGAL/Polygon_2.h>

namespace py = pybind11;

// 用于 pair<int,int> 的哈希（与之前相同）
struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// 插值函数（与之前相同）
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

// 主函数：基于TIN的Rumple Index
py::tuple compute_rumple_index_tin(
    py::array_t<double> x_array,
    py::array_t<double> y_array,
    py::array_t<double> z_array,
    py::array_t<int>    treeid_array,
    double ground_grid_size = 1.0,   // 地面网格大小
    double surface_grid_size = 1.0   // 冠层表面采样网格大小（与之前一致）
) {
    auto x   = x_array.unchecked<1>();
    auto y   = y_array.unchecked<1>();
    auto z   = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();

    const size_t N = x_array.shape(0);
    if (N == 0) {
        return py::make_tuple(py::float_(NAN), py::float_(NAN), py::float_(NAN));
    }

    // ---------- 地面归一化部分（与之前完全相同）----------
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

    // ---------- 提取冠层表面点（每个网格取最大z_norm）----------
    std::unordered_map<std::pair<int,int>, double, PairHash> canopy_max;
    for (size_t i = 0; i < N; ++i) {
        if (tid(i) == 0) continue;   // 忽略地面点
        int gx = static_cast<int>(std::floor(x(i) / ground_grid_size));
        int gy = static_cast<int>(std::floor(y(i) / ground_grid_size));
        auto key = std::make_pair(gx, gy);
        auto itg = ground_full.find(key);
        if (itg == ground_full.end() || std::isnan(itg->second)) {
            continue;
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

    // ---------- 准备表面点列表（用于TIN）----------
    // 使用CGAL内核
    using K = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point_2 = K::Point_2;
    using Point_3 = K::Point_3;   // 用于三维面积计算
    using Vector_3 = K::Vector_3;
    using Triangulation = CGAL::Delaunay_triangulation_2<K>;

    std::vector<Point_2> points_2d;
    std::vector<double> heights;   // 存储对应点的z_norm，索引与points_2d一致
    points_2d.reserve(canopy_max.size());
    heights.reserve(canopy_max.size());

    for (const auto& kv : canopy_max) {
        int ix = kv.first.first;
        int iy = kv.first.second;
        double h = kv.second;
        // 使用网格中心坐标（或左下角坐标），这里使用左下角坐标，不影响形状
        double px = ix * surface_grid_size;
        double py = iy * surface_grid_size;
        points_2d.emplace_back(px, py);
        heights.push_back(h);
    }

    // 构建Delaunay三角剖分
    Triangulation dt(points_2d.begin(), points_2d.end());

    // ---------- 计算总表面积和投影面积（凸包面积）----------
    double total_3d_area = 0.0;
    // 遍历所有有限面（三角形）
    for (auto fit = dt.finite_faces_begin(); fit != dt.finite_faces_end(); ++fit) {
        // 获取三个顶点的索引（在points_2d中的位置）
        // 注意：我们需要将顶点句柄转换为索引。可以通过映射，但这里直接获取坐标和高度。
        // 从顶点句柄获取点坐标
        Point_2 p1 = fit->vertex(0)->point();
        Point_2 p2 = fit->vertex(1)->point();
        Point_2 p3 = fit->vertex(2)->point();

        // 找到对应的z值：由于点坐标是唯一的（可能重复？网格点不会重复），我们可以通过坐标匹配。
        // 但效率较低，更好的做法是建立一个从坐标到高度的map。
        // 这里为了代码简洁，假设点坐标唯一且可以从points_2d中查找。由于可能有浮点误差，我们最好用map。
        // 改用map存储坐标->高度，坐标用double pair，但需精确相等。由于网格坐标是精确的（整数倍网格大小），可以直接用ix,iy构建key。
        // 其实我们可以直接使用之前canopy_max，但需要将ix,iy映射到高度。
        // 所以还是用之前的canopy_max map，通过ix,iy查找。为了快速，我们重新建立一个从网格索引到高度的map，键为std::pair<int,int>，就是canopy_max本身。
        // 现在通过网格索引来获取高度：需要从点的坐标反推网格索引。
        // 因为点坐标是ix*surface_grid_size, iy*surface_grid_size，所以ix = floor(px / surface_grid_size + 0.5)等。
        // 但更简单：在遍历三角面时，我们可以直接从canopy_max中查找，但需要将点坐标转为网格索引。
        auto coord_to_index = [surface_grid_size](double coord) -> int {
            return static_cast<int>(std::round(coord / surface_grid_size));
        };
        int ix1 = coord_to_index(p1.x());
        int iy1 = coord_to_index(p1.y());
        int ix2 = coord_to_index(p2.x());
        int iy2 = coord_to_index(p2.y());
        int ix3 = coord_to_index(p3.x());
        int iy3 = coord_to_index(p3.y());

        auto it1 = canopy_max.find({ix1, iy1});
        auto it2 = canopy_max.find({ix2, iy2});
        auto it3 = canopy_max.find({ix3, iy3});
        if (it1 == canopy_max.end() || it2 == canopy_max.end() || it3 == canopy_max.end()) {
            continue;  // 理论上不会发生
        }
        double z1 = it1->second;
        double z2 = it2->second;
        double z3 = it3->second;

        // 构造三维点
        Point_3 p3d1(p1.x(), p1.y(), z1);
        Point_3 p3d2(p2.x(), p2.y(), z2);
        Point_3 p3d3(p3.x(), p3.y(), z3);

        // 计算三维三角形面积：0.5 * |(p2-p1) x (p3-p1)|
        Vector_3 v1 = p3d2 - p3d1;
        Vector_3 v2 = p3d3 - p3d1;
        Vector_3 cross = CGAL::cross_product(v1, v2);
        double area_3d = 0.5 * std::sqrt(cross.squared_length());
        total_3d_area += area_3d;
    }

    // 计算投影面积（凸包面积）
    std::vector<Point_2> hull_points;
    CGAL::convex_hull_2(points_2d.begin(), points_2d.end(), std::back_inserter(hull_points));
    double projection_area = 0.0;
    if (hull_points.size() >= 3) {
        // 使用多边形面积公式
        for (size_t i = 0; i < hull_points.size(); ++i) {
            size_t j = (i + 1) % hull_points.size();
            projection_area += hull_points[i].x() * hull_points[j].y() - hull_points[j].x() * hull_points[i].y();
        }
        projection_area = std::abs(projection_area) / 2.0;
    } else {
        projection_area = 0.0;
    }

    if (projection_area <= 0.0 || total_3d_area <= 0.0) {
        return py::make_tuple(py::float_(NAN), py::float_(NAN), py::float_(NAN));
    }

    double rumple = total_3d_area / projection_area;

    return py::make_tuple(py::float_(rumple),
                          py::float_(total_3d_area),
                          py::float_(projection_area));
}

// 模块定义
PYBIND11_MODULE(rumple_index_tin, m) {
    m.doc() = "Rumple Index using TIN (CGAL)";
    m.def("compute_rumple_index_tin", &compute_rumple_index_tin,
          py::arg("x_array"), py::arg("y_array"), py::arg("z_array"), py::arg("treeid_array"),
          py::arg("ground_grid_size") = 1.0,
          py::arg("surface_grid_size") = 1.0,
          "Compute Rumple Index via TIN: 3D surface area / convex hull area.");
}
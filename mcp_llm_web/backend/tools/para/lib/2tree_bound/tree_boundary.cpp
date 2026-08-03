#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

// 简单二维点结构，记录原始索引
struct Pt
{
    double x;
    double y;
    std::size_t idx;  // 在原始数组中的索引
};

// 叉积： (OA x OB)
inline double cross(const Pt& O, const Pt& A, const Pt& B)
{
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

// 计算 2D 点集的凸包（Andrew 单调链算法），返回按顺序排列的 hull 顶点
std::vector<Pt> convex_hull(std::vector<Pt> pts)
{
    std::size_t n = pts.size();
    if (n == 0)
        return {};
    if (n == 1)
        return pts;

    // 按 x, y 排序
    std::sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& b) {
        if (a.x < b.x) return true;
        if (a.x > b.x) return false;
        return a.y < b.y;
    });

    std::vector<Pt> hull;
    hull.reserve(n * 2);

    // lower hull
    for (std::size_t i = 0; i < n; ++i)
    {
        while (hull.size() >= 2 &&
               cross(hull[hull.size() - 2], hull[hull.size() - 1], pts[i]) <= 0.0)
        {
            hull.pop_back();
        }
        hull.push_back(pts[i]);
    }

    // upper hull
    std::size_t lower_size = hull.size();
    for (std::size_t i = n; i-- > 0; )
    {
        while (hull.size() > lower_size &&
               cross(hull[hull.size() - 2], hull[hull.size() - 1], pts[i]) <= 0.0)
        {
            hull.pop_back();
        }
        hull.push_back(pts[i]);
    }

    // 最后一个点与第一个点重复，去掉
    if (!hull.empty())
        hull.pop_back();

    // 特殊情况：所有点共线，hull 只有两个端点
    return hull;
}

// 点到线段的最小距离平方
inline double distance2_point_segment(double px, double py,
                                      double ax, double ay,
                                      double bx, double by)
{
    double vx = bx - ax;
    double vy = by - ay;
    double wx = px - ax;
    double wy = py - ay;

    double c1 = vx * wx + vy * wy;
    if (c1 <= 0.0)
    {
        // 最近点是 A
        double dx = px - ax;
        double dy = py - ay;
        return dx * dx + dy * dy;
    }

    double c2 = vx * vx + vy * vy;
    if (c2 <= c1)
    {
        // 最近点是 B
        double dx = px - bx;
        double dy = py - by;
        return dx * dx + dy * dy;
    }

    double t = c1 / c2;
    double projx = ax + t * vx;
    double projy = ay + t * vy;
    double dx = px - projx;
    double dy = py - projy;
    return dx * dx + dy * dy;
}


// 核心函数：
// 输入：xyz (N,3) float64，treeid (N,) int64，margin
// 输出：保留点的索引 (K,) int64
py::array_t<long long> compute_keep_indices_convex_boundary(
    py::array_t<double,  py::array::c_style | py::array::forcecast> xyz,
    py::array_t<long long, py::array::c_style | py::array::forcecast> treeid,
    double margin
)
{
    if (xyz.ndim() != 2 || xyz.shape(1) != 3)
    {
        throw std::runtime_error("xyz 必须是 shape=(N,3) 的二维数组（float64）。");
    }
    if (treeid.ndim() != 1 || treeid.shape(0) != xyz.shape(0))
    {
        throw std::runtime_error("treeid 必须是一维数组，长度与 xyz 的 N 相同，且为 int64。");
    }
    if (margin < 0.0)
    {
        throw std::runtime_error("margin 必须 >= 0。");
    }

    const std::size_t N = static_cast<std::size_t>(xyz.shape(0));

    auto xyz_ = xyz.unchecked<2>();
    auto tid_ = treeid.unchecked<1>();

    // 1) 提取树点 (treeid != 0)
    std::vector<Pt> tree_pts;
    tree_pts.reserve(N);

    for (std::size_t i = 0; i < N; ++i)
    {
        if (tid_(i) != 0)
        {
            Pt p;
            p.x = xyz_(i, 0);
            p.y = xyz_(i, 1);
            p.idx = i;
            tree_pts.push_back(p);
        }
    }

    if (tree_pts.empty())
    {
        throw std::runtime_error("没有 treeid != 0 的树点，无法进行边界树删除。");
    }

    // 2) 计算树点的凸包
    std::vector<Pt> hull = convex_hull(tree_pts);
    if (hull.size() < 1)
    {
        throw std::runtime_error("凸包点数小于 1，数据异常。");
    }

    // 3) 预计算 margin^2
    const double margin2 = margin * margin;

    // 4) 找到“边界树”的 treeid：
    //    对每个树点，计算到凸包多边形边界的最小距离，
    //    若距离 <= margin，则认为该点在边界带内，该点所在 treeid 是边界树。

    std::unordered_set<long long> boundary_tree_ids;
    const std::size_t H = hull.size();

    for (const auto& p : tree_pts)
    {
        double px = p.x;
        double py = p.y;

        // 遍历凸包的每一条边，计算最小距离
        double min_dist2 = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < H; ++i)
        {
            const Pt& A = hull[i];
            const Pt& B = hull[(i + 1) % H];
            double d2 = distance2_point_segment(px, py, A.x, A.y, B.x, B.y);
            if (d2 < min_dist2)
                min_dist2 = d2;
        }

        if (min_dist2 <= margin2)
        {
            boundary_tree_ids.insert(tid_(p.idx));
        }
    }

    // 若没有边界树，则所有树都保留
    // 但仍然可能需要裁剪地面，这里按你需求，可以选择不裁剪，直接全部保留
    // 下面我们完整实现：删除边界树 + 用剩余树裁剪地面
    // ------------------------

    // 5) 计算 keep_tree_mask：treeid != 0 且 不在 boundary_tree_ids 中
    std::vector<char> keep_tree_mask(N, 0);
    bool any_keep_tree = false;
    for (std::size_t i = 0; i < N; ++i)
    {
        long long t = tid_(i);
        if (t != 0 && boundary_tree_ids.find(t) == boundary_tree_ids.end())
        {
            keep_tree_mask[i] = 1;
            any_keep_tree = true;
        }
    }

    if (!any_keep_tree)
    {
        throw std::runtime_error("删除边界树之后，没有剩余的树木。");
    }

    // 6) 根据剩余树计算 XY 范围，用于裁剪地面点
    double kept_min_x = 0.0, kept_max_x = 0.0, kept_min_y = 0.0, kept_max_y = 0.0;
    bool first_kept = true;

    for (std::size_t i = 0; i < N; ++i)
    {
        if (keep_tree_mask[i])
        {
            double x = xyz_(i, 0);
            double y = xyz_(i, 1);
            if (first_kept)
            {
                kept_min_x = kept_max_x = x;
                kept_min_y = kept_max_y = y;
                first_kept = false;
            }
            else
            {
                if (x < kept_min_x) kept_min_x = x;
                if (x > kept_max_x) kept_max_x = x;
                if (y < kept_min_y) kept_min_y = y;
                if (y > kept_max_y) kept_max_y = y;
            }
        }
    }

    if (first_kept)
    {
        throw std::runtime_error("没有找到剩余树点，无法计算 XY 范围。");
    }

    // 7) ground_keep_mask：treeid == 0 且 落在 [kept_min_x, kept_max_x] × [kept_min_y, kept_max_y]
    std::vector<char> final_keep_mask(N, 0);
    bool any_final_keep = false;
    for (std::size_t i = 0; i < N; ++i)
    {
        long long t = tid_(i);
        double x = xyz_(i, 0);
        double y = xyz_(i, 1);

        // 保留的树
        if (keep_tree_mask[i])
        {
            final_keep_mask[i] = 1;
            any_final_keep = true;
        }
        else if (t == 0)
        {
            // 地面点，按剩余树的 XY 范围裁剪
            if (x >= kept_min_x && x <= kept_max_x &&
                y >= kept_min_y && y <= kept_max_y)
            {
                final_keep_mask[i] = 1;
                any_final_keep = true;
            }
        }
    }

    if (!any_final_keep)
    {
        throw std::runtime_error("最终没有任何点被保留下来。");
    }

    // 8) 收集所有保留点的索引，返回为 numpy int64 数组
    std::vector<long long> keep_indices;
    keep_indices.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
    {
        if (final_keep_mask[i])
        {
            keep_indices.push_back(static_cast<long long>(i));
        }
    }

    py::array_t<long long> result(keep_indices.size());
    auto r = result.mutable_unchecked<1>();
    for (std::size_t i = 0; i < keep_indices.size(); ++i)
    {
        r(i) = keep_indices[i];
    }

    return result;
}


PYBIND11_MODULE(tree_boundary, m)
{
    m.doc() = "Convex-hull based boundary tree removal returning kept indices";

    m.def(
        "compute_keep_indices",
        &compute_keep_indices_convex_boundary,
        py::arg("xyz"),
        py::arg("treeid"),
        py::arg("margin") = 1.0,
        R"pbdoc(
        计算点云中“保留点”的索引（基于树点凸包的边界树剔除 + 按剩余树范围裁剪地面点）。

        参数
        ----
        xyz : (N, 3) float64 numpy array
            所有点的坐标 (x, y, z)。
        treeid : (N,) int64 numpy array
            每个点的树 ID。约定：0 为地面点，其它值为树木 ID。
        margin : float
            单位同 xyz（通常是米）。距离凸包边界 <= margin 的树视为“边界树”，整棵剔除。

        返回
        ----
        keep_indices : (K,) int64 numpy array
            保留的点在原始数组中的索引。
        )pbdoc"
    );
}

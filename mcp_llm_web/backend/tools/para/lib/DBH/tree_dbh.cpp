#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace py = pybind11;

// 计算中位数（原地修改）
double median_inplace(std::vector<double>& v)
{
    if (v.empty())
        return std::numeric_limits<double>::quiet_NaN();

    const size_t n = v.size();
    size_t mid = n / 2;

    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = v[mid];

    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
        m = 0.5 * (m + v[mid - 1]);
    }
    return m;
}

// 用简单最小二乘拟合圆：x^2 + y^2 + Ax + By + C = 0
// 返回是否成功，以及中心(cx, cy) 和半径 r
bool fit_circle_least_squares(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double& cx,
    double& cy,
    double& r
)
{
    const size_t n = xs.size();
    if (n < 3)
        return false;

    // 正规方程矩阵 3x3 和右边向量 3x1
    double Sxx = 0.0, Syy = 0.0, Sxy = 0.0;
    double Sx  = 0.0, Sy  = 0.0;
    double Sxz = 0.0, Syz = 0.0, Sz  = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double x = xs[i];
        double y = ys[i];
        double z = x * x + y * y;

        Sx  += x;
        Sy  += y;
        Sz  += z;
        Sxx += x * x;
        Syy += y * y;
        Sxy += x * y;
        Sxz += x * z;
        Syz += y * z;
    }

    // 构建矩阵和向量
    // [Sxx  Sxy  Sx] [A]   [Sxz]
    // [Sxy  Syy  Sy] [B] = [Syz]
    // [Sx   Sy   n ] [C]   [Sz ]
    double M[3][3] = {
        { Sxx, Sxy, Sx },
        { Sxy, Syy, Sy },
        { Sx,  Sy,  static_cast<double>(n) }
    };
    double b[3] = { Sxz, Syz, Sz };

    // 解 3x3 线性方程组，使用 Cramer's rule 或简单高斯消元
    // 这里用一个最简单的高斯消元（不做列主元，非常基础，如果数值条件不好可能会不稳定）
    int i, j, k;
    double A[3][4]; // augmented matrix 3x4

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j)
            A[i][j] = M[i][j];
        A[i][3] = b[i];
    }

    // 高斯消元
    for (i = 0; i < 3; ++i) {
        // pivot
        double piv = A[i][i];
        if (std::fabs(piv) < 1e-12)
            return false; // 奇异，失败

        for (j = i; j < 4; ++j)
            A[i][j] /= piv;

        for (k = 0; k < 3; ++k) {
            if (k == i) continue;
            double factor = A[k][i];
            for (j = i; j < 4; ++j) {
                A[k][j] -= factor * A[i][j];
            }
        }
    }

    double Acoef = A[0][3];
    double Bcoef = A[1][3];
    double Ccoef = A[2][3];

    cx = -Acoef / 2.0;
    cy = -Bcoef / 2.0;
    double term = cx * cx + cy * cy - Ccoef;
    if (term <= 0.0)
        return false;

    r = std::sqrt(term);
    return true;
}


///
///  主函数：计算每棵树的 DBH （胸径）
///
/// 输入：
///   xyz:   (N,3) float64
///   treeid:(N,)  int64
/// 参数：
///   z_min,z_max: 截取胸高层的 z 范围（默认 1.2~1.3）
///   mad_k:       半径的 MAD 去噪阈值倍数
///   min_points:  每棵树在胸高层至少多少点才计算 DBH
///
/// 返回：
///   dict:
///     "treeid":  int64 array
///     "dbh":     float64 array
///     "n_points":int64 array （拟合使用的点数）
///
py::dict compute_dbh_per_tree(
    py::array_t<double,  py::array::c_style | py::array::forcecast> xyz,
    py::array_t<long long, py::array::c_style | py::array::forcecast> treeid,
    double z_min = 1.2,
    double z_max = 1.3,
    double mad_k = 3.0,
    int min_points = 10
)
{
    if (xyz.ndim() != 2 || xyz.shape(1) != 3) {
        throw std::runtime_error("xyz must be a 2D array of shape (N,3).");
    }
    if (treeid.ndim() != 1 || treeid.shape(0) != xyz.shape(0)) {
        throw std::runtime_error("treeid must be a 1D array of length N.");
    }
    if (z_max <= z_min) {
        throw std::runtime_error("z_max must be > z_min.");
    }
    if (mad_k <= 0.0) {
        throw std::runtime_error("mad_k must be > 0.");
    }
    if (min_points < 3) {
        throw std::runtime_error("min_points must be >= 3.");
    }

    const ssize_t N = xyz.shape(0);
    auto xyz_ = xyz.unchecked<2>();
    auto tid_ = treeid.unchecked<1>();

    // 1) 先根据 z 范围筛选 + treeid != 0，按 treeid 分组
    std::unordered_map<long long, std::vector<size_t>> groups;
    groups.reserve(1024);

    for (ssize_t i = 0; i < N; ++i) {
        double z = xyz_(i, 2);
        if (z < z_min || z > z_max)
            continue;

        long long id = tid_(i);
        if (id == 0)
            continue;  // 忽略地面/无树 id

        groups[id].push_back(static_cast<size_t>(i));
    }

    // 2) 对每棵树拟合圆并计算 DBH
    std::vector<long long> out_ids;
    std::vector<double>    out_dbh;
    std::vector<long long> out_npts;

    out_ids.reserve(groups.size());
    out_dbh.reserve(groups.size());
    out_npts.reserve(groups.size());

    for (const auto& kv : groups) {
        long long id = kv.first;
        const auto& idxs = kv.second;

        if (idxs.size() < static_cast<size_t>(min_points)) {
            // 点太少，跳过
            continue;
        }

        // 2.1 提取该树在胸高层的 x,y
        std::vector<double> xs, ys;
        xs.reserve(idxs.size());
        ys.reserve(idxs.size());

        for (size_t idx : idxs) {
            xs.push_back(xyz_(idx, 0));
            ys.push_back(xyz_(idx, 1));
        }

        // 2.2 初始中心：均值
        double mx = 0.0, my = 0.0;
        for (size_t i = 0; i < xs.size(); ++i) {
            mx += xs[i];
            my += ys[i];
        }
        mx /= static_cast<double>(xs.size());
        my /= static_cast<double>(ys.size());

        // 2.3 计算每个点到中心的半径
        std::vector<double> radii;
        radii.reserve(xs.size());
        for (size_t i = 0; i < xs.size(); ++i) {
            double dx = xs[i] - mx;
            double dy = ys[i] - my;
            radii.push_back(std::sqrt(dx * dx + dy * dy));
        }

        // 2.4 用 MAD 过滤噪声点
        std::vector<double> temp_r = radii;
        double med_r = median_inplace(temp_r);
        if (std::isnan(med_r)) {
            continue;
        }

        std::vector<double> abs_dev;
        abs_dev.reserve(radii.size());
        for (double r : radii) {
            abs_dev.push_back(std::abs(r - med_r));
        }
        double mad = median_inplace(abs_dev);
        if (mad <= 0.0) {
            // 所有半径基本一样，直接用全部点
            // 用简单圆拟合
            double cx, cy, rr;
            bool ok = fit_circle_least_squares(xs, ys, cx, cy, rr);
            if (!ok || rr <= 0.0) {
                continue;
            }
            double dbh = 2.0 * rr;
            out_ids.push_back(id);
            out_dbh.push_back(dbh);
            out_npts.push_back(static_cast<long long>(xs.size()));
            continue;
        }

        const double sigma_approx = 1.4826 * mad;
        const double thresh = mad_k * sigma_approx;

        std::vector<double> xs_inlier, ys_inlier;
        xs_inlier.reserve(xs.size());
        ys_inlier.reserve(ys.size());

        for (size_t i = 0; i < radii.size(); ++i) {
            double dev = std::abs(radii[i] - med_r);
            if (dev <= thresh) {
                xs_inlier.push_back(xs[i]);
                ys_inlier.push_back(ys[i]);
            }
        }

        if (xs_inlier.size() < static_cast<size_t>(min_points)) {
            // 过滤后点太少
            continue;
        }

        // 2.5 拟合圆
        double cx, cy, rr;
        bool ok = fit_circle_least_squares(xs_inlier, ys_inlier, cx, cy, rr);
        if (!ok || rr <= 0.0) {
            continue;
        }

        double dbh = 2.0 * rr;

        out_ids.push_back(id);
        out_dbh.push_back(dbh);
        out_npts.push_back(static_cast<long long>(xs_inlier.size()));
    }

    // 3) 打包结果给 Python：dict{ "treeid": ..., "dbh": ..., "n_points": ... }
    py::array_t<long long> id_arr(out_ids.size());
    py::array_t<double>    dbh_arr(out_dbh.size());
    py::array_t<long long> npts_arr(out_npts.size());

    auto id_buf   = id_arr.mutable_unchecked<1>();
    auto dbh_buf  = dbh_arr.mutable_unchecked<1>();
    auto npts_buf = npts_arr.mutable_unchecked<1>();

    for (ssize_t i = 0; i < static_cast<ssize_t>(out_ids.size()); ++i) {
        id_buf(i)   = out_ids[static_cast<size_t>(i)];
        dbh_buf(i)  = out_dbh[static_cast<size_t>(i)];
        npts_buf(i) = out_npts[static_cast<size_t>(i)];
    }

    py::dict result;
    result["treeid"]   = id_arr;
    result["dbh"]      = dbh_arr;
    result["n_points"] = npts_arr;

    return result;
}


PYBIND11_MODULE(tree_dbh, m)
{
    m.doc() = "Compute DBH per tree from xyz + treeid using z-slice and circle fitting";

    m.def(
        "compute_dbh",
        &compute_dbh_per_tree,
        py::arg("xyz"),
        py::arg("treeid"),
        py::arg("z_min")      = 1.2,
        py::arg("z_max")      = 1.3,
        py::arg("mad_k")      = 3.0,
        py::arg("min_points") = 10,
        R"pbdoc(
        Compute diameter at breast height (DBH) for each treeid.

        1. Take only points with z in [z_min, z_max].
        2. Group by treeid != 0.
        3. For each tree:
           - Remove noisy points by MAD on radius from centroid.
           - Fit a circle in XY using least squares.
           - DBH = 2 * radius.

        Parameters
        ----------
        xyz : (N,3) float64 numpy array
            Point coordinates.
        treeid : (N,) int64 numpy array
            Tree id per point (0 will be ignored).
        z_min : float
            Lower bound of z slice (default 1.2).
        z_max : float
            Upper bound of z slice (default 1.3).
        mad_k : float
            MAD threshold multiplier for outlier removal on radius (default 3.0).
        min_points : int
            Minimum number of points per tree in the slice to compute DBH (default 10).

        Returns
        -------
        result : dict
            {
              "treeid":   int64 array,
              "dbh":      float64 array,
              "n_points": int64 array (number of inlier points used for fitting)
            }
        )pbdoc"
    );
}

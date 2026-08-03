#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

// 计算向量的中位数
double median_inplace(std::vector<double>& v)
{
    if (v.empty())
        return std::numeric_limits<double>::quiet_NaN();

    const size_t n = v.size();
    size_t mid = n / 2;

    // 找到中位数位置
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = v[mid];

    if (n % 2 == 0) {
        // 偶数个点，再找一个比 m 小的最大值
        std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
        m = 0.5 * (m + v[mid - 1]);
    }
    return m;
}


// 主函数：对每个 treeid 计算去除离群点后的树高
py::dict compute_tree_heights(
    py::array_t<double, py::array::c_style | py::array::forcecast> xyz,      // (N,3)
    py::array_t<long long, py::array::c_style | py::array::forcecast> treeid,// (N,)
    double mad_k = 3.0,   // MAD 过滤系数，越大保留越多点
    int min_points = 5     // 每棵树至少多少点才参与计算
)
{
    if (xyz.ndim() != 2 || xyz.shape(1) != 3) {
        throw std::runtime_error("xyz must be a 2D array of shape (N, 3).");
    }
    if (treeid.ndim() != 1 || treeid.shape(0) != xyz.shape(0)) {
        throw std::runtime_error("treeid must be a 1D array of length N (same as xyz.shape[0]).");
    }

    const ssize_t N = xyz.shape(0);
    auto xyz_ = xyz.unchecked<2>();
    auto tid_ = treeid.unchecked<1>();

    if (mad_k <= 0.0) {
        throw std::runtime_error("mad_k must be > 0.");
    }
    if (min_points < 2) {
        throw std::runtime_error("min_points must be >= 2.");
    }

    // 1) 按 treeid 分组：treeid -> 该树所有点的索引
    std::unordered_map<long long, std::vector<size_t>> groups;
    groups.reserve(1024);

    for (ssize_t i = 0; i < N; ++i) {
        long long id = tid_(i);
        // 通常 treeid == 0 是地面或无效，跳过
        if (id == 0)
            continue;
        groups[id].push_back(static_cast<size_t>(i));
    }

    if (groups.empty()) {
        throw std::runtime_error("No treeid != 0 found in treeid array.");
    }

    std::vector<long long> out_ids;
    std::vector<double> out_heights;

    out_ids.reserve(groups.size());
    out_heights.reserve(groups.size());

    // 2) 对每个 treeid 分别处理
    for (const auto& kv : groups) {
        long long id = kv.first;
        const std::vector<size_t>& idxs = kv.second;

        if (idxs.size() < static_cast<size_t>(min_points)) {
            // 点太少的树，跳过
            continue;
        }

        // 2.1 取出该树所有 Z 值
        std::vector<double> zvals;
        zvals.reserve(idxs.size());
        for (size_t idx : idxs) {
            zvals.push_back(xyz_(idx, 2));
        }

        // 2.2 计算 Z 的中位数
        std::vector<double> tmp_z = zvals;  // 拷贝一份用于 in-place median
        double med_z = median_inplace(tmp_z);
        if (std::isnan(med_z)) {
            continue;
        }

        // 2.3 计算绝对偏差 |z - median|
        std::vector<double> abs_dev;
        abs_dev.reserve(zvals.size());
        for (double z : zvals) {
            abs_dev.push_back(std::abs(z - med_z));
        }

        // 2.4 MAD：绝对偏差的中位数
        double mad = median_inplace(abs_dev);
        if (mad <= 0.0) {
            // 全部 z 相等或非常集中，认为没有离群点
            // 用全部点计算高度
            double min_z = zvals[0];
            double max_z = zvals[0];
            for (double z : zvals) {
                if (z < min_z) min_z = z;
                if (z > max_z) max_z = z;
            }
            double h = max_z - min_z;
            if (h > 0.0) {
                out_ids.push_back(id);
                out_heights.push_back(h);
            }
            continue;
        }

        // 2.5 使用 MAD 进行过滤：|z - median| <= mad_k * 1.4826 * MAD
        const double sigma_approx = 1.4826 * mad;
        const double thresh = mad_k * sigma_approx;

        double min_z = std::numeric_limits<double>::infinity();
        double max_z = -std::numeric_limits<double>::infinity();
        size_t inlier_count = 0;

        for (size_t i = 0; i < zvals.size(); ++i) {
            double z = zvals[i];
            double dz = std::abs(z - med_z);
            if (dz <= thresh) {
                // inlier
                ++inlier_count;
                if (z < min_z) min_z = z;
                if (z > max_z) max_z = z;
            }
        }

        if (inlier_count < 2) {
            // 过滤后点太少，跳过
            continue;
        }

        double height = max_z ;
        if (height > 0.0) {
            out_ids.push_back(id);
            out_heights.push_back(height);
        }
    }

    // 3) 转成 numpy 并打包成 dict，让 Python 端直接转 DataFrame
    py::array_t<long long> ids_arr(out_ids.size());
    py::array_t<double> heights_arr(out_heights.size());

    auto ids_buf = ids_arr.mutable_unchecked<1>();
    auto h_buf = heights_arr.mutable_unchecked<1>();

    for (ssize_t i = 0; i < static_cast<ssize_t>(out_ids.size()); ++i) {
        ids_buf(i) = out_ids[static_cast<size_t>(i)];
        h_buf(i) = out_heights[static_cast<size_t>(i)];
    }

    py::dict result;
    result["treeid"] = ids_arr;
    result["height"] = heights_arr;

    return result;
}


PYBIND11_MODULE(tree_height, m)
{
    m.doc() = "Compute per-tree height (maxZ-minZ) with outlier removal, from xyz + treeid";

    m.def(
        "compute_tree_heights",
        &compute_tree_heights,
        py::arg("xyz"),
        py::arg("treeid"),
        py::arg("mad_k") = 3.0,
        py::arg("min_points") = 5,
        R"pbdoc(
        Compute tree heights for each treeid using robust Z-based outlier removal.

        Parameters
        ----------
        xyz : (N, 3) float64 numpy array
            Point coordinates.
        treeid : (N,) int64 numpy array
            Tree id per point. treeid == 0 is ignored (e.g. ground).
        mad_k : float, optional
            Threshold multiplier for MAD-based outlier filtering. Default 3.0.
        min_points : int, optional
            Minimum number of points per tree to compute height. Default 5.

        Returns
        -------
        result : dict
            {
              "treeid": 1D numpy array of int64,
              "height": 1D numpy array of float64
            }
            You can directly wrap this as a pandas DataFrame.
        )pbdoc"
    );
}

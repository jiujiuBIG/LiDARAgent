#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;


// 把 (ix, iy, iz) 编码到一个 64-bit 整数，保证每个体素唯一
inline std::uint64_t encode_voxel(int ix, int iy, int iz)
{
    // 假设 ix, iy, iz 都 < 2^21  (~2.1e6)，对于常见样地尺寸足够
    return  ( (std::uint64_t)ix  << 42 )
          | ( (std::uint64_t)iy  << 21 )
          | ( (std::uint64_t)iz );
}


// 主函数：计算 Filling 图像
py::array_t<double> compute_filling_image(
    py::array_t<double, py::array::c_style | py::array::forcecast> xyz,
    double min_x,
    double max_x,
    double min_y,
    double max_y,
    double min_z,
    double max_z,
    double voxel_size = 0.1,   // 体素尺寸 (m)
    double agg_xy      = 0.5  // XY 聚合网格尺寸 (m)
)

{
    
    if (xyz.ndim() != 2 || xyz.shape(1) != 3) {
        throw std::runtime_error("xyz 必须是 shape=(N,3) 的二维 float64 数组。");
    }
    if (voxel_size <= 0.0) {
        throw std::runtime_error("voxel_size 必须 > 0。");
    }
    if (agg_xy <= 0.0) {
        throw std::runtime_error("agg_xy 必须 > 0。");
    }

    const std::size_t N = static_cast<std::size_t>(xyz.shape(0));
    auto xyz_ = xyz.unchecked<2>();

    if (N == 0) {
        throw std::runtime_error("xyz 为空，没有点。");
    }

    // 1) 计算 xyz 的 min / max
    // double min_x = std::numeric_limits<double>::infinity();
    // double max_x = -std::numeric_limits<double>::infinity();
    // double min_y = std::numeric_limits<double>::infinity();
    // double max_y = -std::numeric_limits<double>::infinity();
    // double min_z = std::numeric_limits<double>::infinity();
    // double max_z = -std::numeric_limits<double>::infinity();

    // for (std::size_t i = 0; i < N; ++i) {
    //     double x = xyz_(i, 0);
    //     double y = xyz_(i, 1);
    //     double z = xyz_(i, 2);
    //     if (x < min_x) min_x = x;
    //     if (x > max_x) max_x = x;
    //     if (y < min_y) min_y = y;
    //     if (y > max_y) max_y = y;
    //     if (z < min_z) min_z = z;
    //     if (z > max_z) max_z = z;
    // }

    // 防止 degenerate 情况
    // if (max_x <= min_x || max_y <= min_y || max_z <= min_z) {
    //     throw std::runtime_error("点云在某个维度上没有范围（max <= min）。");
    // }

    // 2) 体素网格的数量 (Nx, Ny, Nz)
    const int nx = static_cast<int>(std::ceil((max_x - min_x) / voxel_size));
    const int ny = static_cast<int>(std::ceil((max_y - min_y) / voxel_size));
    const int nz = static_cast<int>(std::ceil((max_z - min_z) / voxel_size));

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::runtime_error("体素数量非正，请检查数据或参数。");
    }

    // 3) 先把所有点映射到体素索引，并用 set 去重 (唯一体素)
    std::unordered_set<std::uint64_t> occupied_voxels;
    occupied_voxels.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        double x = xyz_(i, 0);
        double y = xyz_(i, 1);
        double z = xyz_(i, 2);

        int ix = static_cast<int>(std::floor((x - min_x) / voxel_size));
        int iy = static_cast<int>(std::floor((y - min_y) / voxel_size));
        int iz = static_cast<int>(std::floor((z - min_z) / voxel_size));

        // 边界保护（理论上不会超过 nx-1，但数值误差可能导致索引 == nx）
        if (ix < 0) ix = 0;
        if (ix >= nx) ix = nx - 1;
        if (iy < 0) iy = 0;
        if (iy >= ny) iy = ny - 1;
        if (iz < 0) iz = 0;
        if (iz >= nz) iz = nz - 1;

        std::uint64_t key = encode_voxel(ix, iy, iz);
        occupied_voxels.insert(key);
    }

    // 4) 0.5m 聚合在体素里的步长（即多少个 0.1m 体素组成一格）
    const double step_ratio = agg_xy / voxel_size;
    if (std::fabs(step_ratio - std::round(step_ratio)) > 1e-9) {
        throw std::runtime_error(
            "agg_xy / voxel_size 必须是整数，否则无法整齐划分。"
        );
    }
    const int step_xy = static_cast<int>(std::round(step_ratio));

    // 聚合后的 XY 网格数量
    const int agg_nx = (nx + step_xy - 1) / step_xy;  // 向上取整
    const int agg_ny = (ny + step_xy - 1) / step_xy;

    // 5) 为每个聚合格子统计：occupied_voxel_count / total_voxel_count
    std::vector<std::uint64_t> occ_counts(agg_nx * agg_ny, 0);
    std::vector<std::uint64_t> total_counts(agg_nx * agg_ny, 0);

    auto idx2d = [agg_nx](int ix, int iy) {
        return static_cast<std::size_t>(iy) * static_cast<std::size_t>(agg_nx)
             + static_cast<std::size_t>(ix);
    };

    // 5a) 先预计算每个聚合格子的 total voxel 数（含 z 方向）
    for (int by = 0; by < agg_ny; ++by) {
        // 这个块在 y 上包含的体素索引范围
        int y_start = by * step_xy;
        int y_end   = std::min((by + 1) * step_xy, ny);
        int ny_block = y_end - y_start;

        for (int bx = 0; bx < agg_nx; ++bx) {
            int x_start = bx * step_xy;
            int x_end   = std::min((bx + 1) * step_xy, nx);
            int nx_block = x_end - x_start;

            std::uint64_t total = static_cast<std::uint64_t>(nx_block)
                                * static_cast<std::uint64_t>(ny_block)
                                * static_cast<std::uint64_t>(nz);

            total_counts[idx2d(bx, by)] = total;
        }
    }

    // 5b) 遍历所有被占用的体素，把它们计入对应的聚合格子的 occ_counts
    for (const auto& key : occupied_voxels) {
        int iz = static_cast<int>( key & ((1u << 21) - 1) );
        int iy = static_cast<int>( (key >> 21) & ((1u << 21) - 1) );
        int ix = static_cast<int>( (key >> 42) & ((1u << 21) - 1) );
        (void)iz; // iz 目前没单独用到，但参与了 total voxel 的计算

        int bx = ix / step_xy;
        int by = iy / step_xy;

        if (bx < 0 || bx >= agg_nx || by < 0 || by >= agg_ny) {
            continue; // 理论上不会
        }

        occ_counts[idx2d(bx, by)]++;
    }

    // 6) 构建输出的 Filling 图像 (agg_ny, agg_nx)
    py::array_t<double> filling({agg_ny, agg_nx});
    auto fill_ = filling.mutable_unchecked<2>();

    for (int by = 0; by < agg_ny; ++by) {
        for (int bx = 0; bx < agg_nx; ++bx) {
            std::size_t idx = idx2d(bx, by);
            std::uint64_t total = total_counts[idx];
            std::uint64_t occ   = occ_counts[idx];

            double val = 0.0;
            if (total > 0) {
                val = static_cast<double>(occ) / static_cast<double>(total);
            }
            fill_(by, bx) = val;  // 第 0 维是 Y， 第 1 维是 X
        }
    }

    return filling;
}


PYBIND11_MODULE(filling_grid, m)
{
    m.doc() = "Compute 2D Filling image from xyz by 0.1m voxels and 0.5m XY blocks.";

    m.def(
        "compute_filling_image",
        &compute_filling_image,
        py::arg("xyz"),
        py::arg("min_x"),
        py::arg("max_x"),
        py::arg("min_y"),
        py::arg("max_y"),
        py::arg("min_z"),
        py::arg("max_z"),
        py::arg("voxel_size") = 0.1,
        py::arg("agg_xy")     = 0.5
    );
}

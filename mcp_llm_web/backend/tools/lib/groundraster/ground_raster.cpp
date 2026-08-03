// ground_raster.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <limits>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace py = pybind11;

struct Point3D
{
    double x;
    double y;
    double z;
};

inline int clampIndex(int v, int maxV)
{
    if (v < 0) return 0;
    if (v >= maxV) return maxV - 1;
    return v;
}

py::array_t<double> correct_point_cloud_with_ground_raster_numpy(
    py::array_t<double, py::array::c_style | py::array::forcecast> original_xyz,
    py::array_t<double, py::array::c_style | py::array::forcecast> ground_xyz,
    int gridX = 400,
    int gridY = 400)
{
    if (gridX <= 0 || gridY <= 0)
    {
        throw std::runtime_error("gridX and gridY must be positive.");
    }

    // --- 检查输入维度 ---
    if (original_xyz.ndim() != 2 || original_xyz.shape(1) != 3)
    {
        throw std::runtime_error("original_xyz must be a 2D array of shape (N, 3).");
    }
    if (ground_xyz.ndim() != 2 || ground_xyz.shape(1) != 3)
    {
        throw std::runtime_error("ground_xyz must be a 2D array of shape (M, 3).");
    }

    const ssize_t N = original_xyz.shape(0);
    const ssize_t M = ground_xyz.shape(0);

    if (N == 0)
    {
        throw std::runtime_error("original_xyz is empty.");
    }
    if (M == 0)
    {
        throw std::runtime_error("ground_xyz is empty.");
    }

    auto orig = original_xyz.unchecked<2>();
    auto grd  = ground_xyz.unchecked<2>();

    // --- 1. 计算原始点云 XY 范围 ---
    double minX = orig(0, 0);
    double maxX = orig(0, 0);
    double minY = orig(0, 1);
    double maxY = orig(0, 1);

    for (ssize_t i = 1; i < N; ++i)
    {
        double x = orig(i, 0);
        double y = orig(i, 1);
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }

    double rangeX = maxX - minX;
    double rangeY = maxY - minY;

    if (rangeX <= 0.0 || rangeY <= 0.0)
    {
        throw std::runtime_error("Invalid XY range: all points share the same X or Y.");
    }

    double cellSizeX = rangeX / static_cast<double>(gridX);
    double cellSizeY = rangeY / static_cast<double>(gridY);

    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const int gridSize = gridX * gridY;

    std::vector<double> sumZ(gridSize, 0.0);
    std::vector<int>    count(gridSize, 0);

    auto gridIndex = [gridX](int ix, int iy) {
        return iy * gridX + ix;
    };

    // --- 2. 把地面点映射到栅格，累加 Z ---
    for (ssize_t i = 0; i < M; ++i)
    {
        double x = grd(i, 0);
        double y = grd(i, 1);
        double z = grd(i, 2);

        int ix = static_cast<int>((x - minX) / cellSizeX);
        int iy = static_cast<int>((y - minY) / cellSizeY);
        ix = clampIndex(ix, gridX);
        iy = clampIndex(iy, gridY);

        int idx = gridIndex(ix, iy);
        sumZ[idx]   += z;
        count[idx]  += 1;
    }

    // --- 3. 构建地面 raster，先用均值填有数据的格子 ---
    std::vector<double> groundRaster(gridSize, NaN);
    for (int iy = 0; iy < gridY; ++iy)
    {
        for (int ix = 0; ix < gridX; ++ix)
        {
            int idx = gridIndex(ix, iy);
            if (count[idx] > 0)
            {
                groundRaster[idx] = sumZ[idx] / static_cast<double>(count[idx]);
            }
        }
    }

    // --- 4. 用 BFS 进行最近邻填补，把 NaN 都填满 ---
    std::vector<bool> visited(gridSize, false);
    std::queue<std::pair<int,int>> q;

    for (int iy = 0; iy < gridY; ++iy)
    {
        for (int ix = 0; ix < gridX; ++ix)
        {
            int idx = gridIndex(ix, iy);
            if (!std::isnan(groundRaster[idx]))
            {
                visited[idx] = true;
                q.emplace(ix, iy);
            }
        }
    }

    if (q.empty())
    {
        throw std::runtime_error("No valid ground cells: ground_xyz may not cover the XY range.");
    }

    const int dx[4] = { 1, -1,  0,  0 };
    const int dy[4] = { 0,  0,  1, -1 };

    while (!q.empty())
    {
        auto [cx, cy] = q.front();
        q.pop();
        int cidx = gridIndex(cx, cy);
        double cz = groundRaster[cidx];

        for (int k = 0; k < 4; ++k)
        {
            int nx = cx + dx[k];
            int ny = cy + dy[k];
            if (nx < 0 || nx >= gridX || ny < 0 || ny >= gridY)
                continue;

            int nidx = gridIndex(nx, ny);
            if (!visited[nidx])
            {
                groundRaster[nidx] = cz;
                visited[nidx] = true;
                q.emplace(nx, ny);
            }
        }
    }

    // --- 5. 创建输出 numpy 数组 (N, 3) ---
    py::array_t<double> output({N, static_cast<ssize_t>(3)});
    auto out = output.mutable_unchecked<2>();

    // --- 6. 对每个原始点做 Z 校正 ---
    for (ssize_t i = 0; i < N; ++i)
    {
        double x = orig(i, 0);
        double y = orig(i, 1);
        double z = orig(i, 2);

        int ix = static_cast<int>((x - minX) / cellSizeX);
        int iy = static_cast<int>((y - minY) / cellSizeY);
        ix = clampIndex(ix, gridX);
        iy = clampIndex(iy, gridY);

        int idx = gridIndex(ix, iy);
        double groundZ = groundRaster[idx];

        // 理论上 BFS 后不会 NaN，但保险
        if (!std::isnan(groundZ))
        {
            z = z - groundZ;
        }

        out(i, 0) = x;
        out(i, 1) = y;
        out(i, 2) = z;
    }

    return output;
}


PYBIND11_MODULE(ground_raster, m)
{
    m.doc() = "Ground raster based Z-correction for point clouds";

    m.def(
        "correct_point_cloud",
        &correct_point_cloud_with_ground_raster_numpy,
        py::arg("original_xyz"),
        py::arg("ground_xyz"),
        py::arg("gridX") = 400,
        py::arg("gridY") = 400,
        R"pbdoc(
        Correct point cloud Z by subtracting a ground raster built from ground points.

        Parameters
        ----------
        original_xyz : (N, 3) float64 numpy array
            Original point cloud (x, y, z).
        ground_xyz : (M, 3) float64 numpy array
            Ground points (x, y, z).
        gridX : int, optional
            Number of cells in X direction (default: 400).
        gridY : int, optional
            Number of cells in Y direction (default: 400).

        Returns
        -------
        corrected_xyz : (N, 3) float64 numpy array
            Corrected point cloud with z_new = z_old - ground_height(ix, iy).
        )pbdoc"
    );
}

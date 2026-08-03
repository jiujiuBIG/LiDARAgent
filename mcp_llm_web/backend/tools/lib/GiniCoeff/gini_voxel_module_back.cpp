// gini_voxel_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <unordered_map>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

namespace py = pybind11;

// hash for pair<int,int>
struct pair_hash {
    std::size_t operator()(const std::pair<int,int>& p) const noexcept {
        // 64-bit friendly mix
        return (static_cast<std::size_t>(p.first) << 32) ^ static_cast<std::size_t>(p.second);
    }
};

// hash for tuple<int,int,int>
struct tuple3_hash {
    std::size_t operator()(const std::tuple<int,int,int>& t) const noexcept {
        std::size_t a = static_cast<std::size_t>(std::get<0>(t));
        std::size_t b = static_cast<std::size_t>(std::get<1>(t));
        std::size_t c = static_cast<std::size_t>(std::get<2>(t));
        // simple mixing
        std::size_t res = a;
        res = (res * 73856093) ^ (b * 19349663) ^ (c * 83492791);
        return res;
    }
};

// Helper: compute Gini from non-negative values using O(n log n) formula
// Values should be >= 0. Returns NaN if empty.
double gini_from_vector(std::vector<double> &vals) {
    size_t n = vals.size();
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();

    // Remove non-positive? keep zeros but if sum==0 return 0
    double sum = 0.0;
    for (double v : vals) sum += v;
    if (sum == 0.0) return 0.0;

    std::sort(vals.begin(), vals.end()); // ascending
    // compute sum_i i * x_i (i starting from 1)
    long double accum = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        accum += (long double)(i + 1) * vals[i];
    }
    long double numerator = 2.0L * accum;
    long double denom = (long double)n * (long double)sum;
    long double G = numerator / denom - ( (long double)(n + 1) / (long double)n );
    // guard numerical issues
    if (G < 0 && G > -1e-14) G = 0.0;
    return static_cast<double>(G);
}


/*
 Pipeline:
 1) Filter points with treeid == 0
 2) Small XY grid: for each (gx,gy) keep max z (canopy)
 3) 3D voxelize those canopy points with voxel_size -> compute mean z per voxel
 4) Collect voxel means -> compute Gini
*/
double gini_voxelized(py::array_t<double> x_array,
                      py::array_t<double> y_array,
                      py::array_t<double> z_array,
                      py::array_t<int> treeid_array,
                      double small_xy = 0.1,    // small XY cell to get canopy max
                      double voxel_size = 0.5)  // 3D voxel size for averaging
{
    // bounds & checks
    if (small_xy <= 0.0) small_xy = 0.1;
    if (voxel_size <= 0.0) voxel_size = 0.5;

    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto z = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();
    size_t N = x_array.shape(0);
    if (N == 0) return std::numeric_limits<double>::quiet_NaN();

    // 1) XY small grid -> keep max z in each XY cell
    std::unordered_map<std::pair<int,int>, double, pair_hash> xy_max_z;
    xy_max_z.reserve(N / 4 + 16);

    for (size_t i = 0; i < N; ++i) {
        if (tid(i) == 0) continue; // skip ground / non-tree
        int gx = static_cast<int>(std::floor(x(i) / small_xy));
        int gy = static_cast<int>(std::floor(y(i) / small_xy));
        auto key = std::make_pair(gx, gy);
        auto it = xy_max_z.find(key);
        if (it == xy_max_z.end()) {
            xy_max_z.emplace(key, z(i));
        } else {
            if (z(i) > it->second) it->second = z(i);
        }
    }

    if (xy_max_z.empty()) return std::numeric_limits<double>::quiet_NaN();

    // 2) 3D voxelize the canopy points (the xy_max_z entries)
    // compute voxel coordinates for each canopy point then accumulate mean z
    std::unordered_map<std::tuple<int,int,int>, std::pair<double,int>, tuple3_hash> voxel_accum;
    voxel_accum.reserve(xy_max_z.size() * 2);

    for (const auto &kv : xy_max_z) {
        // get canopy point coords: convert back to coordinates of this small cell center
        // best to compute an approximate x,y coordinate (center of small cell)
        int gx = kv.first.first;
        int gy = kv.first.second;
        double canopy_z = kv.second;
        // compute approximate x,y position (center)
        double px = (gx + 0.5) * small_xy;
        double py_ = (gy + 0.5) * small_xy;
        // voxel indices
        int vx = static_cast<int>(std::floor(px / voxel_size));
        int vy = static_cast<int>(std::floor(py_ / voxel_size));
        int vz = static_cast<int>(std::floor(canopy_z / voxel_size));
        auto vkey = std::make_tuple(vx, vy, vz);
        auto it = voxel_accum.find(vkey);
        if (it == voxel_accum.end()) {
            voxel_accum.emplace(vkey, std::make_pair(canopy_z, 1));
        } else {
            it->second.first += canopy_z;
            it->second.second += 1;
        }
    }

    if (voxel_accum.empty()) return std::numeric_limits<double>::quiet_NaN();

    // 3) compute mean height per voxel
    std::vector<double> voxel_means;
    voxel_means.reserve(voxel_accum.size());
    for (const auto &kv : voxel_accum) {
        double sumz = kv.second.first;
        int cnt = kv.second.second;
        voxel_means.push_back(sumz / static_cast<double>(cnt));
    }

    // 4) compute Gini from voxel_means (efficient)
    double G = gini_from_vector(voxel_means);
    return G;
}

PYBIND11_MODULE(gini_voxel_module, m) {
    m.doc() = "Compute Gini coefficient of canopy heights using voxelization and XY max to avoid trunk points";
    m.def("gini_voxelized", &gini_voxelized,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("z_array"),
          py::arg("treeid_array"),
          py::arg("small_xy") = 0.1,
          py::arg("voxel_size") = 0.5);
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <limits>

namespace py = pybind11;

// ------------------- Key Structures -------------------
struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey &o) const { return x==o.x && y==o.y && z==o.z; }
};
struct HashVoxel {
    size_t operator()(VoxelKey const &k) const noexcept {
        return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y)<<1))>>1) ^ (std::hash<int>()(k.z)<<1);
    }
};

struct XYKey {
    int x, y;
    bool operator==(const XYKey &o) const { return x==o.x && y==o.y; }
};
struct HashXY {
    size_t operator()(XYKey const &k) const noexcept {
        return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y)<<1);
    }
};

// ------------------- LAD + FHD -------------------
py::tuple lad_fhd_canopy(py::array_t<double> x_array,
                         py::array_t<double> y_array,
                         py::array_t<double> z_array,
                         py::array_t<int> treeid_array,
                         double big_voxel_size,
                         double small_voxel_size,
                         double dz = 1.0,
                         double ground_threshold = 2.0)
{
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto z = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();
    size_t N = x_array.shape(0);
    if(N==0) return py::make_tuple(py::array_t<double>(), py::array_t<double>());

    // 1. 计算每个 xy 栅格地面最低点
    std::unordered_map<XYKey,double,HashXY> xy_ground_min;
    for(size_t i=0;i<N;i++){
        if(tid(i) != 0) continue; // 地面点
        int xv = static_cast<int>(std::floor(x(i)/big_voxel_size));
        int yv = static_cast<int>(std::floor(y(i)/big_voxel_size));
        XYKey key{xv,yv};
        auto it = xy_ground_min.find(key);
        if(it==xy_ground_min.end()) xy_ground_min[key] = z(i);
        else if(z(i) < it->second) it->second = z(i);
    }

    // 2. 最近邻插值函数
    auto get_ground_z = [&](int xv, int yv){
        XYKey key{xv,yv};
        if(xy_ground_min.count(key)) return xy_ground_min[key];
        // 最近邻搜索
        double min_dist = std::numeric_limits<double>::max();
        double z_nearest = 0.0;
        for(auto &kv: xy_ground_min){
            int dx = kv.first.x - xv;
            int dy = kv.first.y - yv;
            double dist = dx*dx + dy*dy;
            if(dist < min_dist){
                min_dist = dist;
                z_nearest = kv.second;
            }
        }
        return z_nearest;
    };

    // 3. 选取冠层点
    std::vector<double> cx, cy, cz;
    for(size_t i=0;i<N;i++){
        if(tid(i)==0) continue;
        int xv = static_cast<int>(std::floor(x(i)/big_voxel_size));
        int yv = static_cast<int>(std::floor(y(i)/big_voxel_size));
        double zmin = get_ground_z(xv,yv);
        if(z(i) >= zmin + ground_threshold){
            cx.push_back(x(i));
            cy.push_back(y(i));
            cz.push_back(z(i));
        }
    }
    if(cx.empty()) return py::make_tuple(py::array_t<double>(), py::array_t<double>());

    size_t M = cx.size();

    // 4. 小 voxel 占有
    std::unordered_set<VoxelKey, HashVoxel> small_voxels;
    for(size_t i=0;i<M;i++){
        int vx = static_cast<int>(std::floor(cx[i]/small_voxel_size));
        int vy = static_cast<int>(std::floor(cy[i]/small_voxel_size));
        int vz = static_cast<int>(std::floor(cz[i]/small_voxel_size));
        small_voxels.insert({vx,vy,vz});
    }

    // 5. 大 voxel LAD
    std::unordered_map<VoxelKey,int,HashVoxel> big_vox_count;
    for(auto &s: small_voxels){
        int bx = static_cast<int>(std::floor(s.x*small_voxel_size/big_voxel_size));
        int by = static_cast<int>(std::floor(s.y*small_voxel_size/big_voxel_size));
        int bz = static_cast<int>(std::floor(s.z*small_voxel_size/big_voxel_size));
        big_vox_count[{bx,by,bz}]++;
    }

    int small_per_big = static_cast<int>(std::pow(big_voxel_size/small_voxel_size,3));

    // 6. 高度层 LAD
    std::unordered_map<int,std::vector<double>> lad_layers;
    for(auto &kv: big_vox_count){
        int layer = static_cast<int>(std::floor(kv.first.z*big_voxel_size/dz));
        double lad = static_cast<double>(kv.second)/small_per_big;
        lad_layers[layer].push_back(lad);
    }

    // 7. 输出每层 LAD + FHD
    std::vector<double> lad_out, fhd_out;
    for(auto &kv: lad_layers){
        double sum=0;
        for(double v: kv.second) sum+=v;
        lad_out.insert(lad_out.end(), kv.second.begin(), kv.second.end());

        // FHD
        double total = sum;
        double fhd=0;
        if(total>0){
            for(double v: kv.second){
                double p=v/total;
                if(p>0) fhd -= p*std::log(p);
            }
        }
        fhd_out.push_back(fhd);
    }

    return py::make_tuple(py::array_t<double>(lad_out.size(), lad_out.data()),
                          py::array_t<double>(fhd_out.size(), fhd_out.data()));
}

// ------------------- Pybind11 Module -------------------
PYBIND11_MODULE(lad_fhd_canopy, m){
    m.doc() = "Compute canopy LAD + FHD per vertical layer";
    m.def("lad_fhd_canopy", &lad_fhd_canopy,
          py::arg("x_array"),
          py::arg("y_array"),
          py::arg("z_array"),
          py::arg("treeid_array"),
          py::arg("big_voxel_size"),
          py::arg("small_voxel_size"),
          py::arg("dz")=1.0,
          py::arg("ground_threshold")=2.0,
          "Compute LAD and FHD for canopy points per vertical layer");
}

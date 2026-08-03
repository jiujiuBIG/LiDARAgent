#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace py = pybind11;

// -------------------- 哈希函数 --------------------
struct tuple3_hash {
    std::size_t operator()(const std::tuple<int,int,int>& t) const {
        auto h1 = std::hash<int>{}(std::get<0>(t));
        auto h2 = std::hash<int>{}(std::get<1>(t));
        auto h3 = std::hash<int>{}(std::get<2>(t));
        // 更好的混合策略，减少碰撞
        return h1 ^ (h2 << 11) ^ (h3 << 22);
    }
};

// -------------------- LAI 计算 --------------------
/**
 * 计算点云的LAI垂直剖面
 * 
 * @param x_array X坐标数组
 * @param y_array Y坐标数组  
 * @param z_array Z坐标数组（已归一化，即离地高度）
 * @param treeid_array 点类别标识（0=地面点，>0=植被点）
 * @param voxel_size 体素大小（米），决定分辨率，默认0.5米
 * @param z_layer_height 垂直分层高度（米），默认1.0米
 * @param k_coefficient 消光系数，默认0.5（常见阔叶林）
 * @param min_z 最小高度（米），低于此高度的点忽略，默认0.0
 * @param max_z 最大高度（米），高于此高度的点忽略，默认100.0
 * @return 每个垂直层的平均LAI值
 */
std::vector<double> compute_lai_profile(
    py::array_t<double> x_array,
    py::array_t<double> y_array,
    py::array_t<double> z_array,
    py::array_t<int> treeid_array,
    double voxel_size = 0.05,
    double z_layer_height = 1.0,
    double k_coefficient = 0.5,
    double min_z = 0.0,
    double max_z = 100.0) {
    
    // 获取数组访问器
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto z = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();
    size_t n_points = x_array.shape(0);
    
    if (n_points == 0) {
        return {};
    }
    
    // -------------------- 步骤1: 体素化 --------------------
    // 记录被占据的体素（只考虑植被点，tid > 0）
    std::unordered_set<std::tuple<int,int,int>, tuple3_hash> occupied_voxels;
    
    // 同时记录每个点的z值用于后续统计（可选）
    std::vector<double> valid_heights;
    valid_heights.reserve(n_points);
    
    for (size_t i = 0; i < n_points; ++i) {
        if (tid(i) <= 0) continue;  // 跳过地面点和非植被点
        
        double z_val = z(i);
        // 高度范围过滤
        if (z_val < min_z || z_val > max_z) continue;
        
        valid_heights.push_back(z_val);
        
        // 计算体素坐标
        int vx = static_cast<int>(std::floor(x(i) / voxel_size));
        int vy = static_cast<int>(std::floor(y(i) / voxel_size));
        int vz = static_cast<int>(std::floor(z_val / voxel_size));

        auto voxel = std::make_tuple(vx, vy, vz);
        if (occupied_voxels.find(voxel) == occupied_voxels.end()) {
            occupied_voxels.insert(voxel);
        };
    }
    
    if (occupied_voxels.empty()) {
        return std::vector<double>(static_cast<size_t>(std::ceil(max_z / z_layer_height)), 0.0);
    }
    
    // -------------------- 步骤2: 确定体素网格范围 --------------------
    int vx_min = std::numeric_limits<int>::max();
    int vx_max = std::numeric_limits<int>::min();
    int vy_min = vx_min, vy_max = vx_max;
    int vz_min = vx_min, vz_max = vx_max;
    
    for (const auto& voxel : occupied_voxels) {
        vx_min = std::min(vx_min, std::get<0>(voxel));
        vx_max = std::max(vx_max, std::get<0>(voxel));
        vy_min = std::min(vy_min, std::get<1>(voxel));
        vy_max = std::max(vy_max, std::get<1>(voxel));
        vz_min = std::min(vz_min, std::get<2>(voxel));
        vz_max = std::max(vz_max, std::get<2>(voxel));
    }
    
    // 扩展一点范围，确保覆盖边界
    vx_min -= 1; vx_max += 1;
    vy_min -= 1; vy_max += 1;
    vz_min = std::max(0, vz_min - 1);  // 高度不能为负
    vz_max += 1;
    
    // -------------------- 步骤3: 计算垂直分层 --------------------
    // 基于点的实际高度确定分层数，而不是用max_z
    double z_actual_min = 0.0;
    double z_actual_max = 0.0;
    if (!valid_heights.empty()) {
        z_actual_min = *std::min_element(valid_heights.begin(), valid_heights.end());
        z_actual_max = *std::max_element(valid_heights.begin(), valid_heights.end());
    } else {
        return {};  // 没有有效点
    }
    
    int n_layers = static_cast<int>(std::ceil((z_actual_max - z_actual_min) / z_layer_height));
    n_layers = std::max(1, n_layers);
    
    std::vector<double> lai_per_layer(n_layers, 0.0);
    std::vector<int> valid_columns_per_layer(n_layers, 0);
    
    // -------------------- 步骤4: 对每个垂直层计算孔隙率并转换为LAI --------------------
    // 遍历每个水平网格柱（固定vx, vy）
    for (int vx = vx_min; vx <= vx_max; ++vx) {
        for (int vy = vy_min; vy <= vy_max; ++vy) {
            // 统计该柱内所有被占据的vz
            std::unordered_set<int> occupied_vz_in_column;
            
            for (int vz = vz_min; vz <= vz_max; ++vz) {
                auto voxel = std::make_tuple(vx, vy, vz);
                if (occupied_voxels.find(voxel) != occupied_voxels.end()) {
                    occupied_vz_in_column.insert(vz);
                }
            }
            
            if (occupied_vz_in_column.empty()) continue;
            
            // 对该柱的每个垂直层计算孔隙率
            for (int layer = 0; layer < n_layers; ++layer) {
                double layer_z_min = z_actual_min + layer * z_layer_height;
                double layer_z_max = layer_z_min + z_layer_height;
                
                // 该层对应的vz范围
                int layer_vz_min = static_cast<int>(std::floor(layer_z_min / voxel_size));
                int layer_vz_max = static_cast<int>(std::ceil(layer_z_max / voxel_size));
                
                // 确保不超出体素范围
                layer_vz_min = std::max(vz_min, layer_vz_min);
                layer_vz_max = std::min(vz_max, layer_vz_max);
                
                if (layer_vz_min >= layer_vz_max) continue;
                
                // 统计该层内被占据的体素数
                int occupied_in_layer = 0;
                int total_slices = layer_vz_max - layer_vz_min;
                
                for (int vz = layer_vz_min; vz < layer_vz_max; ++vz) {
                    if (occupied_vz_in_column.find(vz) != occupied_vz_in_column.end()) {
                        occupied_in_layer++;
                    }
                }
                
                // 计算孔隙率并转换为LAI
                if (total_slices > 0) {
                    double gap_fraction = 1.0 - static_cast<double>(occupied_in_layer) / total_slices;
                    
                    // 避免边界情况
                    if (gap_fraction > 0.01 && gap_fraction < 0.99) {
                        // 比尔-朗伯定律: LAI = -ln(P_gap) / (k * dz)
                        double lai = -std::log(gap_fraction) / (k_coefficient * z_layer_height);
                        lai_per_layer[layer] += lai;
                        valid_columns_per_layer[layer]++;
                    }
                }
            }
        }
    }
    
    // -------------------- 步骤5: 对每层取平均 --------------------
    for (int layer = 0; layer < n_layers; ++layer) {
        if (valid_columns_per_layer[layer] > 0) {
            lai_per_layer[layer] /= valid_columns_per_layer[layer];
        } else {
            lai_per_layer[layer] = 0.0;
        }
    }
    
    return lai_per_layer;
}

/**
 * 简化版本：只返回总LAI（各层之和）
 */
double compute_total_lai(
    py::array_t<double> x_array,
    py::array_t<double> y_array,
    py::array_t<double> z_array,
    py::array_t<int> treeid_array,
    double voxel_size = 0.05,
    double k_coefficient = 0.5,
    double min_z = 0.0,
    double max_z = 100.0) {
    
    auto profile = compute_lai_profile(
        x_array, y_array, z_array, treeid_array,
        voxel_size, 1.0, k_coefficient, min_z, max_z
    );
    
    double total_lai = 0.0;
    for (double val : profile) {
        total_lai += val;
    }
    return total_lai;
}

// -------------------- Python模块绑定 --------------------
PYBIND11_MODULE(lai_calculator, m) {
    m.doc() = "LAI calculation from normalized LiDAR point clouds using voxel-based gap fraction method";
    
    m.def("compute_lai_profile", &compute_lai_profile,
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("treeid"),
          py::arg("voxel_size") = 0.05,
          py::arg("z_layer_height") = 1.0,
          py::arg("k_coefficient") = 0.5,
          py::arg("min_z") = 0.0,
          py::arg("max_z") = 100.0,
          "计算垂直LAI剖面，返回每个高度层的平均LAI值");
    
    m.def("compute_total_lai", &compute_total_lai,
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("treeid"),
          py::arg("voxel_size") = 0.05,
          py::arg("k_coefficient") = 0.5,
          py::arg("min_z") = 0.0,
          py::arg("max_z") = 100.0,
          "计算总LAI（各层之和）");
}
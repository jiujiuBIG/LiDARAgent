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
#include <iostream>

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

// -------------------- LAI 计算（优化版：每个体素只计一次）--------------------
/**
 * 计算点云的LAI垂直剖面
 * 优化：每个体素无论有多少个点，只计为1次占据
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
 * @param debug 是否输出调试信息，默认false
 * @return 每个垂直层的平均LAI值
 */
std::vector<double> compute_lai_profile(
    py::array_t<double> x_array,
    py::array_t<double> y_array,
    py::array_t<double> z_array,
    py::array_t<int> treeid_array,
    double voxel_size = 0.5,
    double z_layer_height = 1.0,
    double k_coefficient = 0.5,
    double min_z = 0.0,
    double max_z = 100.0,
    bool debug = false) {
    
    // 获取数组访问器
    auto x = x_array.unchecked<1>();
    auto y = y_array.unchecked<1>();
    auto z = z_array.unchecked<1>();
    auto tid = treeid_array.unchecked<1>();
    size_t n_points = x_array.shape(0);
    
    if (n_points == 0) {
        return {};
    }
    
    if (debug) {
        py::print("总点数:", n_points);
    }
    
    // -------------------- 步骤1: 体素化（每个体素只计一次）--------------------
    // 使用unordered_set记录被占据的体素，自动去重
    std::unordered_set<std::tuple<int,int,int>, tuple3_hash> occupied_voxels;
    occupied_voxels.reserve(n_points / 5);  // 预分配空间，假设平均每个体素5个点
    
    // 记录所有有效点的z值（用于统计范围）
    double z_global_min = std::numeric_limits<double>::max();
    double z_global_max = std::numeric_limits<double>::lowest();
    size_t valid_point_count = 0;
    size_t voxel_count = 0;
    
    for (size_t i = 0; i < n_points; ++i) {
        if (tid(i) <= 0) continue;  // 跳过地面点和非植被点
        
        double z_val = z(i);
        // 高度范围过滤
        if (z_val < min_z || z_val > max_z) continue;
        
        // 更新全局高度范围
        z_global_min = std::min(z_global_min, z_val);
        z_global_max = std::max(z_global_max, z_val);
        valid_point_count++;
        
        // 计算体素坐标
        int vx = static_cast<int>(std::floor(x(i) / voxel_size));
        int vy = static_cast<int>(std::floor(y(i) / voxel_size));
        int vz = static_cast<int>(std::floor(z_val / voxel_size));
        
        auto voxel = std::make_tuple(vx, vy, vz);
        
        // 尝试插入，返回的second表示是否是新元素
        auto result = occupied_voxels.insert(voxel);
        if (result.second) {
            voxel_count++;
        }
    }
    
    if (debug) {
        py::print("有效植被点数:", valid_point_count);
        py::print("占据的体素数:", voxel_count);
        py::print("高度范围:", z_global_min, "-", z_global_max);
        py::print("点/体素比:", static_cast<double>(valid_point_count) / voxel_count);
    }
    
    if (occupied_voxels.empty()) {
        return std::vector<double>(static_cast<size_t>(std::ceil((max_z - min_z) / z_layer_height)), 0.0);
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
    int buffer = 1;
    vx_min -= buffer;
    vx_max += buffer;
    vy_min -= buffer;
    vy_max += buffer;
    vz_min = std::max(0, vz_min - buffer);
    vz_max += buffer;
    
    if (debug) {
        py::print("体素网格范围 X:", vx_min, "-", vx_max, 
                  "Y:", vy_min, "-", vy_max, 
                  "Z:", vz_min, "-", vz_max);
    }
    
    // -------------------- 步骤3: 计算垂直分层 --------------------
    // 使用实际点云高度范围
    double z_actual_min = z_global_min;
    double z_actual_max = z_global_max;
    
    int n_layers = static_cast<int>(std::ceil((z_actual_max - z_actual_min) / z_layer_height));
    n_layers = std::max(1, n_layers);
    
    if (debug) {
        py::print("垂直层数:", n_layers, "每层高度:", z_layer_height, "米");
    }
    
    std::vector<double> lai_per_layer(n_layers, 0.0);
    std::vector<int> valid_columns_per_layer(n_layers, 0);
    std::vector<int> total_columns_per_layer(n_layers, 0);
    
    // 预计算层边界
    std::vector<double> layer_bottoms(n_layers);
    std::vector<double> layer_tops(n_layers);
    for (int layer = 0; layer < n_layers; ++layer) {
        layer_bottoms[layer] = z_actual_min + layer * z_layer_height;
        layer_tops[layer] = layer_bottoms[layer] + z_layer_height;
    }
    
    // -------------------- 步骤4: 对每个垂直层计算孔隙率并转换为LAI --------------------
    // 遍历每个水平网格柱（固定vx, vy）
    for (int vx = vx_min; vx <= vx_max; ++vx) {
        for (int vy = vy_min; vy <= vy_max; ++vy) {
            // 创建该柱的体素占据查找表
            std::vector<bool> voxel_occupied_in_column(vz_max - vz_min + 1, false);
            bool column_has_vegetation = false;
            
            // 标记该柱内所有被占据的vz
            for (int vz = vz_min; vz <= vz_max; ++vz) {
                auto voxel = std::make_tuple(vx, vy, vz);
                if (occupied_voxels.find(voxel) != occupied_voxels.end()) {
                    voxel_occupied_in_column[vz - vz_min] = true;
                    column_has_vegetation = true;
                }
            }
            
            if (!column_has_vegetation) continue;
            
            // 对该柱的每个垂直层计算孔隙率
            for (int layer = 0; layer < n_layers; ++layer) {
                // 该层对应的vz范围
                int layer_vz_min = static_cast<int>(std::floor(layer_bottoms[layer] / voxel_size));
                int layer_vz_max = static_cast<int>(std::ceil(layer_tops[layer] / voxel_size));
                
                // 确保不超出体素范围
                layer_vz_min = std::max(vz_min, layer_vz_min);
                layer_vz_max = std::min(vz_max, layer_vz_max);
                
                int total_slices = layer_vz_max - layer_vz_min;
                if (total_slices <= 2) continue;  // 层太薄，跳过
                
                // 统计该层内被占据的体素数
                int occupied_in_layer = 0;
                for (int vz = layer_vz_min; vz < layer_vz_max; ++vz) {
                    if (voxel_occupied_in_column[vz - vz_min]) {
                        occupied_in_layer++;
                    }
                }
                
                // 只有当层内有体素时才记录该柱对该层有贡献
                total_columns_per_layer[layer]++;
                
                // 计算孔隙率并转换为LAI
                if (total_slices > 0) {
                    // 孔隙率 = 1 - (占据体素数 / 总可能体素数)
                    double gap_fraction = 1.0 - static_cast<double>(occupied_in_layer) / total_slices;
                    
                    // 限制孔隙率在合理范围内，避免数值问题
                    // 最小孔隙率0.01（即最大占据率0.99），最大孔隙率0.99
                    const double MIN_GAP = 0.01;
                    const double MAX_GAP = 0.99;
                    
                    if (gap_fraction < MIN_GAP) {
                        if (debug && layer == n_layers/2) {
                            py::print("  警告: 层", layer, "孔隙率过小:", gap_fraction, 
                                     "occupied:", occupied_in_layer, "total:", total_slices);
                        }
                        gap_fraction = MIN_GAP;
                    } else if (gap_fraction > MAX_GAP) {
                        gap_fraction = MAX_GAP;
                    }
                    
                    // 比尔-朗伯定律: LAI = -ln(P_gap) / (k * dz)
                    double lai = -std::log(gap_fraction) / (k_coefficient * z_layer_height);
                    
                    // 限制单层LAI最大值，避免异常值
                    const double MAX_LAYER_LAI = 8.0;  // 单层LAI理论上限
                    if (lai > MAX_LAYER_LAI) {
                        if (debug) {
                            py::print("  警告: 层", layer, "LAI异常高:", lai, 
                                     "gap:", gap_fraction, "occupied:", occupied_in_layer);
                        }
                        lai = MAX_LAYER_LAI;
                    }
                    
                    lai_per_layer[layer] += lai;
                    valid_columns_per_layer[layer]++;
                }
            }
        }
    }
    
    // -------------------- 步骤5: 对每层取平均 --------------------
    for (int layer = 0; layer < n_layers; ++layer) {
        if (valid_columns_per_layer[layer] > 0) {
            lai_per_layer[layer] /= valid_columns_per_layer[layer];
            
            if (debug && lai_per_layer[layer] > 1.0) {
                py::print("层", layer, 
                         "高度:", layer_bottoms[layer], "-", layer_tops[layer],
                         "LAI:", lai_per_layer[layer],
                         "有效柱数:", valid_columns_per_layer[layer],
                         "总柱数:", total_columns_per_layer[layer],
                         "占据率:", static_cast<double>(valid_columns_per_layer[layer])/total_columns_per_layer[layer]);
            }
        } else {
            lai_per_layer[layer] = 0.0;
        }
    }
    
    // 计算总LAI用于调试
    if (debug) {
        double total = 0.0;
        for (double val : lai_per_layer) {
            total += val;
        }
        py::print("总LAI:", total);
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
    double voxel_size = 0.5,
    double k_coefficient = 0.5,
    double min_z = 0.0,
    double max_z = 100.0,
    bool debug = false) {
    
    auto profile = compute_lai_profile(
        x_array, y_array, z_array, treeid_array,
        voxel_size, 1.0, k_coefficient, min_z, max_z, debug
    );
    
    double total_lai = 0.0;
    for (double val : profile) {
        if (std::isfinite(val) && val > 0) {
            total_lai += val;
        }
    }
    return total_lai;
}

// -------------------- Python模块绑定 --------------------
PYBIND11_MODULE(lai_calculator, m) {
    m.doc() = "LAI calculation from normalized LiDAR point clouds using voxel-based gap fraction method";
    
    m.def("compute_lai_profile", &compute_lai_profile,
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("treeid"),
          py::arg("voxel_size") = 0.5,
          py::arg("z_layer_height") = 1.0,
          py::arg("k_coefficient") = 0.5,
          py::arg("min_z") = 0.0,
          py::arg("max_z") = 100.0,
          py::arg("debug") = false,
          "计算垂直LAI剖面，返回每个高度层的平均LAI值\n"
          "优化：每个体素无论有多少个点，只计为1次占据");
    
    m.def("compute_total_lai", &compute_total_lai,
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("treeid"),
          py::arg("voxel_size") = 0.5,
          py::arg("k_coefficient") = 0.5,
          py::arg("min_z") = 0.0,
          py::arg("max_z") = 100.0,
          py::arg("debug") = false,
          "计算总LAI（各层之和）");
}
#include <vector>
#include <map>
#include <iostream>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// 辅助函数：将点映射到网格
std::pair<int, int> get_voxel_index(float x, float y, float voxel_size) {
    int x_index = std::floor(x / voxel_size);
    int y_index = std::floor(y / voxel_size);
    return {x_index, y_index};
}

// 辅助函数：根据网格计算最低点
float find_ground_point(const std::vector<float>& z_values) {
    float ground_point = z_values[0];
    for (const auto& z : z_values) {
        if (z < ground_point) {
            ground_point = z;
        }
    }
    return ground_point;
}

std::vector<float> remove_ground_points(
    py::array_t<float>& x_np,
    py::array_t<float>& y_np,
    py::array_t<float>& z_np,
    py::array_t<int>& treeid_np,
    float voxel_size) {

    // 获取数组的大小
    auto x_buf = x_np.request();
    auto y_buf = y_np.request();
    auto z_buf = z_np.request();
    auto treeid_buf = treeid_np.request();

    size_t size = x_buf.size;

    // 将 NumPy 数组转换为 C++ 中的 std::vector<float>，并按网格划分
    std::map<std::pair<int, int>, std::vector<float>> voxel_grid_z_values;

    // 将每个点映射到对应的网格
    for (size_t i = 0; i < size; ++i) {
        float x = static_cast<float*>(x_buf.ptr)[i];
        float y = static_cast<float*>(y_buf.ptr)[i];
        float z = static_cast<float*>(z_buf.ptr)[i];
        auto index = get_voxel_index(x, y, voxel_size);
        voxel_grid_z_values[index].push_back(z);
    }

    // 计算每个网格的地面点（最低点）
    std::map<std::pair<int, int>, float> voxel_ground_points;
    for (auto& voxel : voxel_grid_z_values) {
        float ground_point = find_ground_point(voxel.second);
        voxel_ground_points[voxel.first] = ground_point;
    }

    // 去除异常地面点（可以自定义逻辑）
    std::map<std::pair<int, int>, float> valid_voxel_ground_points;
    for (auto& voxel : voxel_ground_points) {
        if (voxel.second < 10.0f) {  // 设定一个阈值
            valid_voxel_ground_points[voxel.first] = voxel.second;
        }
    }

    // 插值处理（为空的网格插值）
    for (auto& voxel : voxel_grid_z_values) {
        if (valid_voxel_ground_points.find(voxel.first) == valid_voxel_ground_points.end()) {
            float avg_z = 0.0f;
            int count = 0;
            for (auto& neighbor : valid_voxel_ground_points) {
                avg_z += neighbor.second;
                count++;
            }
            if (count > 0) {
                avg_z /= count;
                valid_voxel_ground_points[voxel.first] = avg_z;
            }
        }
    }

    // 去除地面：计算每个点的高度减去其网格的地面点高度
    std::vector<float> result_z_values;
    for (size_t i = 0; i < size; ++i) {
        float x = static_cast<float*>(x_buf.ptr)[i];
        float y = static_cast<float*>(y_buf.ptr)[i];
        float z = static_cast<float*>(z_buf.ptr)[i];
        auto index = get_voxel_index(x, y, voxel_size);
        float ground_point = valid_voxel_ground_points[index];
        result_z_values.push_back(z - ground_point);  // 高度去除
    }

    return result_z_values;
}

PYBIND11_MODULE(cloud_processing, m) {
    m.def("remove_ground_points", &remove_ground_points, 
          "Remove ground points from point cloud",
          py::arg("x_np"), py::arg("y_np"), py::arg("z_np"), py::arg("treeid_np"), py::arg("voxel_size"));
}

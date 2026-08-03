#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // Required for handling NumPy arrays
#include <pybind11/stl.h>    // Required for automatic STL conversion to Python types
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

struct Point {
    float x, y, z;
};

// C++ function that computes the CHM from a NumPy array
std::vector<std::vector<float>> computeCHM(py::array_t<float> clipxyz, float xmin, float ymin, float xmax, float ymax, float cell_size) {
    // Ensure that the input NumPy array is of shape (N, 3)
    py::buffer_info buf_info = clipxyz.request();
    if (buf_info.ndim != 2 || buf_info.shape[1] != 3) {
        throw std::runtime_error("Input NumPy array must have shape (N, 3), where N is the number of points.");
    }

    // Convert NumPy array to a vector of Points
    std::vector<Point> points;
    float* ptr = static_cast<float*>(buf_info.ptr);
    size_t num_points = buf_info.shape[0];
    
    // Fill the points vector with the (x, y, z) values from the NumPy array
    for (size_t i = 0; i < num_points; ++i) {
        points.push_back({ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2]});
    }

    // Calculate the number of cells in each dimension
    int num_cells_x = std::ceil((xmax - xmin) / cell_size);
    int num_cells_y = std::ceil((ymax - ymin) / cell_size);

    // Initialize a 2D array to hold the largest size for each grid cell
    std::vector<std::vector<float>> chm(num_cells_y, std::vector<float>(num_cells_x, 0.0f));

    // Divide points into cells and calculate the largest z value (height) for each cell
    for (const auto& point : points) {
        if (point.x >= xmin && point.x <= xmax && point.y >= ymin && point.y <= ymax) {
            int cell_x = std::floor((point.x - xmin) / cell_size);
            int cell_y = std::floor((point.y - ymin) / cell_size);

            // Update the maximum z value in this cell
            chm[cell_y][cell_x] = std::max(chm[cell_y][cell_x], point.z);
        }
    }

    return chm;
}

// Python bindings for the computeCHM function
PYBIND11_MODULE(point_cloud_processing, m) {
    m.def("computeCHM", &computeCHM, "Compute Canopy Height Model (CHM) from point cloud data");
}

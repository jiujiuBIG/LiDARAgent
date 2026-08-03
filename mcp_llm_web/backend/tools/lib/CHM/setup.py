import os
import sys
import subprocess
from setuptools import setup, Extension
import pybind11
import numpy

# Check if pybind11 is installed
try:
    import pybind11
except ImportError:
    print("pybind11 is not installed. Please install it with 'pip install pybind11'")
    sys.exit(1)

# Set up the C++ extension module
module_name = "point_cloud_processing"

# Path to the C++ source file (change this path if needed)
cpp_file = "point_cloud_processing.cpp"

# Define the extension module
ext_modules = [
    Extension(
        module_name,
        sources=[cpp_file],
        include_dirs=[pybind11.get_include(), pybind11.get_include(True), numpy.get_include()],  # Include pybind11 and numpy headers
        language="c++",  # Set the language to C++
        extra_compile_args=["-std=c++14"]  # You can add other flags if necessary
    )
]

# Call the setup function to build and install the module
setup(
    name=module_name,
    version="0.1.0",
    author="Your Name",  # Change this to your name
    description="Point cloud processing with CHM generation",
    ext_modules=ext_modules,
    zip_safe=False,
)

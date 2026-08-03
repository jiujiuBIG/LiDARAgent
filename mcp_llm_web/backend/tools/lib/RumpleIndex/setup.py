from setuptools import setup, Extension
import pybind11

rumple_module = Extension(
    'rumple_index',
    sources=['rumple_index.cpp'],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=['-std=c++11'],
    language='c++'
)

setup(
    name='rumple_index',
    version='0.1',
    description='Compute Rumple Index from lidar point cloud',
    ext_modules=[rumple_module],
    zip_safe=False,
)
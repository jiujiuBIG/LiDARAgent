from setuptools import setup, Extension
import pybind11

# 定义扩展模块
top_rugosity_module = Extension(
    'top_rugosity',                     # 模块名，Python中导入时使用
    sources=['top_rugosity.cpp'],        # 源文件列表
    include_dirs=[pybind11.get_include()],
    extra_compile_args=['-std=c++11'],   # 使用C++11标准
    language='c++'
)

setup(
    name='top_rugosity',
    version='0.1',
    description='Compute Top Rugosity from lidar point cloud',
    ext_modules=[top_rugosity_module],
    zip_safe=False,
)
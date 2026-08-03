from setuptools import setup, Extension
import pybind11

setup(
    name="cloud_processing",
    ext_modules=[
        Extension(
            "cloud_processing",  # Python模块名称
            ["cloud_processing.cpp"],  # C++文件列表
            include_dirs=[pybind11.get_include()],  # 引入pybind11的头文件
            language="c++",
            extra_compile_args=["-std=c++14"],  # 指定使用 C++14 编译标准
        ),
    ],
)

from setuptools import setup, Extension
import pybind11

setup(
    name="fill_index_module",  # 模块名称
    version="0.1",  # 版本号
    ext_modules=[  # 扩展模块配置
        Extension(
            "fill_index_module",  # 生成的 Python 模块名称
            ["fill_index_module.cpp"],  # C++ 源文件路径
            include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],  # pybind11 头文件路径
            language="c++",  # 使用 C++ 编译
            extra_compile_args=["-O3", "-std=c++17"],  # 编译选项
        ),
    ],
)

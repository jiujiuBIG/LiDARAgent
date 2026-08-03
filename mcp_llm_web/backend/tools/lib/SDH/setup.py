from setuptools import setup, Extension
import pybind11

# C++ 编译扩展模块
ext_modules = [
    Extension(
        "my_lidar_tools",  # 这是生成的 Python 模块名称
        ["my_lidar_tools.cpp"],  # C++ 源文件路径
        include_dirs=[
            pybind11.get_include(),  # 包含 pybind11 头文件
            pybind11.get_include(user=True),  # 如果你安装了 pybind11 到用户目录，也要加上这个路径
        ],
        language="c++",  # 指定 C++ 编译
        extra_compile_args=["-O3", "-Wall"],  # 编译时的额外参数
        extra_link_args=[],  # 链接时的额外参数，如果需要
    ),
]

setup(
    name="my_lidar_tools",  # 模块名称
    version="0.1",  # 版本号
    ext_modules=ext_modules,  # 传入扩展模块列表
    zip_safe=False,  # 是否允许将扩展模块压缩
)

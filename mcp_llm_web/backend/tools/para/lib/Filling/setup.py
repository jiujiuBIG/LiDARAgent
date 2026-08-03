from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "filling_grid",             # import 时的模块名
        ["filling_grid.cpp"],       # 源文件
        cxx_std=14,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="filling_grid",
    version="0.1.0",
    description="Compute Filling image from voxelized point cloud (C++/pybind11)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

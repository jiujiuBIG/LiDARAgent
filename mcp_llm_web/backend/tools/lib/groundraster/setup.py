# setup.py
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "ground_raster",              # 模块名，import 时用这个
        ["ground_raster.cpp"],        # 源文件
        cxx_std=14,                   # C++ 标准（14 或 17 都行）
        extra_compile_args=["-O3"],   # 优化
    ),
]

setup(
    name="ground_raster",
    version="0.1.0",
    description="Ground raster based Z correction for point clouds (C++/pybind11)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

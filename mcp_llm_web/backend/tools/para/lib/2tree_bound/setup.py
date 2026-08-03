from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "tree_boundary",              # 模块名，import tree_boundary
        ["tree_boundary.cpp"],        # 源文件
        cxx_std=14,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="tree_boundary",
    version="0.1.0",
    description="Convex-hull based boundary tree filtering (C++/pybind11)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

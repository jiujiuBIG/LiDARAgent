from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "tree_height",              # import 时的模块名
        ["tree_height.cpp"],        # 源码文件
        cxx_std=14,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="tree_height",
    version="0.1.0",
    description="Per-tree height computation with outlier removal (C++/pybind11)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

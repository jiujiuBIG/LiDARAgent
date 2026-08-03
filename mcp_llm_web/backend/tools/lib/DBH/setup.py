from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "tree_dbh",              # Python 里的模块名: import tree_dbh
        ["tree_dbh.cpp"],        # 源文件
        cxx_std=14,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="tree_dbh",
    version="0.1.0",
    description="DBH computation per tree from xyz + treeid (C++/pybind11)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

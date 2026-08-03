from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "lad_fhd_canopy",
        ["lad_fhd_canopy.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"]
    ),
]

setup(
    name="lad_fhd_canopy",
    version="1.0",
    ext_modules=ext_modules,
)

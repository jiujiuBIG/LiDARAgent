from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'fraction_canopy_gaps',
        ['canopy_gaps.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++14']
    ),
]

setup(
    name='fraction_canopy_gaps',
    version='0.1',
    author='Your Name',
    ext_modules=ext_modules,
)

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'lai_calculator',
        ['lai.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++14']
    ),
]

setup(
    name='lai_calculator',
    version='0.1',
    author='Your Name',
    ext_modules=ext_modules,
)

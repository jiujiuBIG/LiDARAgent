from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'rumple_index_module',
        ['rumple_index_module.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++14']
    )
]

setup(
    name='rumple_index_module',
    version='0.1',
    author='Your Name',
    description='Compute Rumple Index (canopy surface area / projected area) from point cloud',
    ext_modules=ext_modules,
)

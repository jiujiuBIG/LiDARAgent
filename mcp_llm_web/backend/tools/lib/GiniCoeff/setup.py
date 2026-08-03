# setup.py
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'gini_voxel_module',
        ['gini_voxel_module.cpp'],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
        language='c++',
        extra_compile_args=['-O3', '-std=c++14']
    )
]

setup(
    name='gini_voxel_module',
    version='0.1',
    author='Your Name',
    description='Gini coefficient of canopy heights computed after voxelization and XY max filtering',
    ext_modules=ext_modules,
)

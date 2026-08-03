from setuptools import setup, Extension
import pybind11
import os

tin_module = Extension(
    'rumple_index_tin',
    sources=['rumple_index_tin.cpp'],
    include_dirs=[
        pybind11.get_include(),
        os.path.join(os.environ['CONDA_PREFIX'], 'include')
    ],
    library_dirs=[
        os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    ],
    libraries=['gmp', 'mpfr'],   # 只需链接 gmp 和 mpfr
    extra_compile_args=['-std=c++14', '-frounding-math'],
    language='c++'
)

setup(
    name='rumple_index_tin',
    version='0.1',
    description='Rumple Index using TIN (CGAL)',
    ext_modules=[tin_module],
    zip_safe=False,
)
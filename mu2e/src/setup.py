from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    #ext_modules = cythonize("helloworld.pyx")
    ext_modules = cythonize("RowTransformations.pyx"),
    include_dirs = [numpy.get_include()]
    #ext_modules = cythonize("fib.pyx")
    )

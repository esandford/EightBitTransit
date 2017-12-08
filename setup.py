# to run: python setup.py build_ext --inplace
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('cGridFunctions.pyx'))
setup(ext_modules=cythonize('cTransitingImage.pyx'))
setup(ext_modules=cythonize('misc.pyx'))
setup(ext_modules=cythonize('inversion.pyx'),include_dirs=[np.get_include()])

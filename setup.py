# to run: python setup.py build_ext --inplace
#from setuptools import setup,Extension
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#setup(ext_modules=cythonize('EightBitTransit/cGridFunctions.pyx'))
#setup(ext_modules=cythonize('EightBitTransit/cTransitingImage.pyx'))
#setup(ext_modules=cythonize('EightBitTransit/misc.pyx'))
#setup(ext_modules=cythonize('EightBitTransit/deprecated.pyx'),include_dirs=[np.get_include()])
#setup(ext_modules=cythonize('EightBitTransit/inversion.pyx'),include_dirs=[np.get_include()])

extensions = [
			Extension('EightBitTransit.cGridFunctions',['EightBitTransit/cGridFunctions.pyx']),
			Extension('EightBitTransit.cTransitingImage',['EightBitTransit/cTransitingImage.pyx']),
			Extension('EightBitTransit.misc',['EightBitTransit/misc.pyx']),
			Extension('EightBitTransit.inversion',['EightBitTransit/inversion.pyx'],include_dirs=[np.get_include()])
]


setup(name='EightBitTransit',
      version='1.0',
      description='Shadow imaging of transiting objects',
      author='Emily Sandford',
      author_email='esandford@astro.columbia.edu',
      url='https://github.com/esandford/EightBitTransit',
      license='MIT',
      packages=['EightBitTransit'],
      include_dirs=[np.get_include()],
      #install_requires=['numpy','matplotlib','warnings','scipy','copy','math','itertools','collections'],
      ext_modules=cythonize(extensions))

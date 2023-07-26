# to run: python setup.py build_ext --inplace
from setuptools import setup, Extension, find_packages
import numpy as np
from Cython.Build import cythonize


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ["matplotlib==3.5.3", "numpy==1.21.6", "scipy==1.7.3",
                "imageio==2.31.1", "numba==0.56.4", "jupyterlab==3.4.8"]

extensions = [
    Extension('EightBitTransit.cGridFunctions',
              ['EightBitTransit/cGridFunctions.pyx']),
    Extension('EightBitTransit.cTransitingImage',
              ['EightBitTransit/cTransitingImage.pyx']),
    Extension('EightBitTransit.misc',
              ['EightBitTransit/misc.pyx']),
    Extension('EightBitTransit.inversion',
              ['EightBitTransit/inversion.pyx'],
              include_dirs=[np.get_include()])
]

setup(name='EightBitTransit',
      version='1.1',
      description='Shadow imaging of transiting objects',
      python_requires='>=3.6',
      author='Daniel Giles (Orig. Emily Sandford)',
      author_email='daniel.k.giles@gmail.com',
      url='https://github.com/d-giles/EightBitTransit',
      license='MIT',
      packages=find_packages(include=['EightBitTransit', 'EightBitTransit.*']),
      include_dirs=[np.get_include()],
      install_requires=requirements,
      ext_modules=cythonize(extensions),
      long_description=readme + '\n\n' + history,
      include_package_data=True,
      test_suite='tests',
      tests_require=test_requirements,
      version='1.0.0',
      zip_safe=False,
      )

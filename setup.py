# to run: python setup.py build_ext --inplace
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ["matplotlib==3.5", "numpy==1.21", "scipy==1.7",
                "imageio==2.31", "numba==0.56", "jupyterlab==3.4"]

test_requirements = []

setup(name='EightBitTransit',
      description='Shadow imaging of transiting objects',
      python_requires='>=3.6',
      author='Daniel Giles (Orig. Emily Sandford)',
      author_email='daniel.k.giles@gmail.com',
      url='https://github.com/d-giles/EightBitTransit',
      license='MIT',
      packages=find_packages(include=['EightBitTransit', 'EightBitTransit.*']),
      install_requires=requirements,
      long_description=readme + '\n\n' + history,
      include_package_data=True,
      test_suite='tests',
      tests_require=test_requirements,
      version='1.0.0',
      zip_safe=False,
      )

# README #

`EightBitTransit` is an MIT-licensed Cython code that:
1. Can calculate the light curve of any pixelated image transiting a star;
2. Can invert a light curve to recover the "shadow image" that produced it.

# Installation #

To install EightBitTransit, download this directory, navigate to it, and run:

`python setup.py build-ext --inplace`

`python setup.py install`

# Dependencies #
* Numpy
* Scipy
* Matplotlib
* Copy
* Math
* Itertools
* Collections

# Examples #

See `./examples/examples.ipynb` for examples of both the forward and inverse shadow imaging problem, including for dip 5 of Boyajian's Star. This code reproduces figure 12 of Sandford et al. 2018.

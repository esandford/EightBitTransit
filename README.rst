===========
phasefolder
===========

`EightBitTransit` is an MIT-licensed Cython code that:
1. Can calculate the light curve of any pixelated image transiting a star;
2. Can invert a light curve to recover the "shadow image" that produced it.

* Free software: MIT license

Installation
--------

To install EightBitTransit, download this directory, navigate to it, and run:

`python setup.py build_ext --inplace`

`python setup.py install`

Dependencies
--------
* Numpy
* Scipy
* Matplotlib
* Copy
* Math
* Itertools
* Collections

Examples
--------

See `./examples/examples.ipynb` for examples of both the forward and inverse shadow imaging problem, including for dip 5 of Boyajian's Star. This code reproduces figure 12 of Sandford & Kipping 2018 (https://arxiv.org/abs/1812.01618).

Note on memory
--------

Repeated TransitingImage() calls can cause python to run out of memory in certain cases (thanks to textadactyl for pointing this out!)--when you are done with a TransitingImage object, deallocate it explicitly, i.e.:

`ti = TransitingImage(...)`

*some operations on ti, etc.*

`ti = None`

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
#! /usr/bin/env python
"""Module for implementing the functional forms used for fitting the magnetic field.

This module contains an assortment of factory functions used to produce mathematical expressions
used for the purpose of fitting the magnetic field of the Mu2E experiment.  The expressions
themselves can be expressed much more simply than they are in the following classes and functions;
the reason for their obfuscation is to optimize their performance.  Typically, a given function will
be fit to ~20,000 data points, and have ~1000 free parameters.  Without special care taken to
improve performance, the act of fitting such functions would be untenable.

Notes:
    * All function outputs are created in order to conform with the `lmfit` curve-fitting package.
    * Factory functions are used to precalculate constants based on the user-defined free
        parameters. This prevents unnecessary calculations when the derived objective function is
        undergoing minimization.
    * The `numexpr` and `numba` packages are used in order to compile and parallelize routines that
        would cause a bottleneck during fitting.
    * This module will be updated soon with a class structure to improve readability, reduce code
        redundancy, and implement the new GPU acceleration features that have been tested in other
        branches.

*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""

from __future__ import division
from scipy import special
import numpy as np
import numexpr as ne
from numba import guvectorize
from math import cos, sin
from itertools import izip


def pairwise(iterable):
    """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
    a = iter(iterable)
    return izip(a, a)


class FunctionProducer(object):
    def __init__(self, z, r, phi, L=None, R=None, ns=1, ms=1, version='modbessel'):
        self._versions = ['modbessel', 'bessel', 'cartesian']

        if version not in self._versions:
            raise AttributeError('`version` must be one of: '+','.join(self._versions))




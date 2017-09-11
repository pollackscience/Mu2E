#! /usr/bin/env python
"""Module for implementing the functional forms used for fitting the magnetic field.

This module contains an assortment of factory class used to produce mathematical expressions
used for the purpose of fitting the magnetic field of the Mu2E experiment.  The expressions
themselves can be expressed much more simply than they are in the following class methods;
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
# import numexpr as ne
from numba import guvectorize
# from math import cos, sin
from itertools import izip


def pairwise(iterable):
    """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
    a = iter(iterable)
    return izip(a, a)


class FunctionProducer(object):
    def __init__(self, coord_1, coord_2, coord_3, ns=1, ms=1, version='modbessel', **kwargs):

        self._versions = ['modbessel', 'bessel', 'cartesian']
        self.ns = ns
        self.ms = ms
        self.version = version

        if version not in self._versions:
            raise AttributeError('`version` must be one of: '+','.join(self._versions))

        if version == 'modbessel':

            self.r = coord_1
            self.phi = coord_2
            self.z = coord_3
            self.L = kwargs['L']
            self.kms = []
            for n in range(self.ns):
                self.kms.append([])
                for m in range(self.ms):
                    self.kms[-1].append((m+1)*np.pi/self.L)
            self.kms = np.asarray(self.kms)
            self.iv = np.empty((self.ns, self.ms) + self.r.shape)
            self.ivp = np.empty((self.ns, self.ms) + self.r.shape)
            for n in range(self.ns):
                for m in range(self.ms):
                    self.iv[n][m] = special.iv(n, self.kms[n][m]*np.abs(self.r))
                    self.ivp[n][m] = special.ivp(n, self.kms[n][m]*np.abs(self.r))

        elif version == 'bessel':

            self.r = coord_1
            self.phi = coord_2
            self.z = coord_3
            self.R = kwargs['R']
            b_zeros = []
            for n in range(self.ns):
                b_zeros.append(special.jn_zeros(n, self.ms))
            self.kms = np.asarray([b/self.R for b in b_zeros])
            self.jv = np.empty((self.ns, self.ms) + self.r.shape)
            self.jvp = np.empty((self.ns, self.ms) + self.r.shape)
            for n in range(self.ns):
                for m in range(self.ms):
                    self.jv[n][m] = special.jv(n, self.kms[n][m]*np.abs(self.r))
                    self.jvp[n][m] = special.jvp(n, self.ms[n][m]*np.abs(self.r))

        elif version == 'cartesian':

            self.x = coord_1
            self.y = coord_2
            self.z = coord_3
            self.a = kwargs['a']
            self.b = kwargs['b']
            self.c = kwargs['c']

            self.alpha_2d = np.zeros((self.ns, self.ms))
            self.beta_2d = np.zeros((self.ns, self.ms))
            self.gamma_2d = np.zeros((self.ns, self.ms))

    def get_fit_function(self):

        if self.version == 'modbessel':
            return self.fit_func_modbessel

        elif self.version == 'bessel':
            return self.fit_func_bessel

        elif self.version == 'cartesian':
            return self.fit_func_cartesian

    def fit_func_modbessel(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        R = R
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in AB_params.iteritems() if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, d in enumerate(Ds):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                D = np.array([AB_params[d]], dtype=np.float64)
                _ivp = self.ivp[n][i]
                _iv = self.iv[n][i]
                _kms = np.array([self.kms[n][i]])
                _n = np.array([n])
                self.calc_b_fields_modbessel(z, phi, r, _n, A, B, D, _ivp, _iv, _kms, model_r,
                                             model_z, model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:])"],
                 '(m), (m), (m), (), (), (), (), (m), (m), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_modbessel(z, phi, r, n, A, B, D, ivp, iv, kms, model_r, model_z,
                                model_phi):
        for i in range(z.shape[0]):
            model_r[i] += np.cos(n[0]*phi[i]-D[0])*ivp[i]*kms[0] * \
                (A[0]*np.cos(kms[0]*z[i]) + B[0]*np.sin(kms[0]*z[i]))
            model_z[i] += np.cos(n[0]*phi[i]-D[0])*iv[i]*kms[0] * \
                (-A[0]*np.sin(kms[0]*z[i]) + B[0]*np.cos(kms[0]*z[i]))
            model_phi[i] += n[0]*(-np.sin(n[0]*phi[i]-D[0])) * \
                (1/np.abs(r[i]))*iv[i]*(A[0]*np.cos(kms[0]*z[i]) + B[0]*np.sin(kms[0]*z[i]))

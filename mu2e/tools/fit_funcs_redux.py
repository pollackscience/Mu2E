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
from __future__ import absolute_import
from scipy import special
import numpy as np
import numexpr as ne
from numba import vectorize, guvectorize, float64, int64, njit, prange
from numba.types import UniTuple
from math import cos, sin
import mpmath
import six
from six.moves import range, zip


def pairwise(iterable):
    """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a)


def tripwise(iterable):
    """s -> (s0,s1, s2), (s3, s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a, a)


def quadwise(iterable):
    """s -> (s0,s1,s2,s3), (s4,s5,s6,s7), ..."""
    a = iter(iterable)
    return zip(a, a, a, a)


def pentwise(iterable):
    """s -> (s0,s1,s2,s3,s4), (s5,s6,s7,s8,s9), ..."""
    a = iter(iterable)
    return zip(a, a, a, a, a)


def hexwise(iterable):
    """s -> (s0,s1,s2,s3,s4,s5), (s6,s7,s8,s9,s10,s11), ..."""
    a = iter(iterable)
    return zip(a, a, a, a, a, a)


def brzphi_3d_producer(z, r, phi, R, ns, ms):
    b_zeros = []
    for n in range(ns):
        b_zeros.append(special.jn_zeros(n, ms))
    kms = np.asarray([b/R for b in b_zeros])
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms[n][m]*np.abs(r))

    def brzphi_3d_fast(z, r, phi, R, ns, ms, delta1, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        def numexpr_model_r_calc(z, phi, n, D1, A, B, ivp, kms):
            return ne.evaluate('(cos(n*phi+D1))*ivp*kms*(A*cos(kms*z) + B*sin(-kms*z))')

        def numexpr_model_z_calc(z, phi, n, D1, A, B, iv, kms):
            return ne.evaluate('-(cos(n*phi+D1))*iv*kms*(A*sin(kms*z) + B*cos(-kms*z))')

        def numexpr_model_phi_calc(z, r, phi, n, D1, A, B, iv, kms):
            return ne.evaluate('n*(-sin(n*phi+D1))*(1/abs(r))*iv*(A*cos(kms*z) + B*sin(-kms*z))')

        model_r = 0.0
        model_z = 0.0
        model_phi = 0.0
        R = R
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                             x.split('_')[0])))

        for n in range(ns):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                model_r += numexpr_model_r_calc(z, phi, n, delta1, AB_params[ab[0]],
                                                AB_params[ab[1]], ivp[n][i], kms[n][i])
                model_z += numexpr_model_z_calc(z, phi, n, delta1, AB_params[ab[0]],
                                                AB_params[ab[1]], iv[n][i], kms[n][i])
                model_phi += numexpr_model_phi_calc(z, r, phi, n, delta1, AB_params[ab[0]],
                                                    AB_params[ab[1]], iv[n][i], kms[n][i])

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def b_external_3d_producer(a, b, c, x, y, z, cns, cms):
    a = a
    b = b
    c = c

    alpha_2d = np.zeros((cns, cms))
    beta_2d = np.zeros((cns, cms))
    gamma_2d = np.zeros((cns, cms))

    for cn in range(1, cns+1):
        for cm in range(1, cms+1):
            alpha_2d[cn-1][cm-1] = (cn*np.pi-np.pi/2)/a
            beta_2d[cn-1][cm-1] = (cm*np.pi-np.pi/2)/b
            gamma_2d[cn-1][cm-1] = np.sqrt(alpha_2d[cn-1][cm-1]**2+beta_2d[cn-1][cm-1]**2)

    def b_external_3d_fast(x, y, z, cns, cms, epsilon1, epsilon2, **AB_params):
        """ 3D model for Bz Bx and By vs Z and X. Can take any number of Cnm terms."""

        def numexpr_model_x_ext_calc(x, y, z, C, alpha, beta, gamma, c, epsilon1, epsilon2):
            return ne.evaluate(
                'C*alpha*sin(alpha*x+epsilon1)*cos(beta*y+epsilon2)*sinh(gamma*(z-c))')

        def numexpr_model_y_ext_calc(x, y, z, C, alpha, beta, gamma, c, epsilon1, epsilon2):
            return ne.evaluate(
                'C*beta*cos(alpha*x+epsilon1)*sin(beta*y+epsilon2)*sinh(gamma*(z-c))')

        def numexpr_model_z_ext_calc(x, y, z, C, alpha, beta, gamma, c, epsilon1, epsilon2):
            return ne.evaluate(
                '(-C)*gamma*cos(epsilon1 + alpha*x)*cos(epsilon2 + beta*y)*cosh(gamma*(-c + z))')

        model_x = 0.0
        model_y = 0.0
        model_z = 0.0
        Cs = sorted({k: v for (k, v) in six.iteritems(AB_params) if 'C' in k})

        for cn in range(1, cns+1):
            for cm in range(1, cms+1):
                alpha = alpha_2d[cn-1][cm-1]
                beta = beta_2d[cn-1][cm-1]
                gamma = gamma_2d[cn-1][cm-1]

                # using C's
                model_x += numexpr_model_x_ext_calc(x, y, z, AB_params[Cs[cm-1+(cn-1)*cms]],
                                                    alpha, beta, gamma, c, epsilon1, epsilon2)
                model_y += numexpr_model_y_ext_calc(x, y, z, AB_params[Cs[cm-1+(cn-1)*cms]],
                                                    alpha, beta, gamma, c, epsilon1, epsilon2)
                model_z += numexpr_model_z_ext_calc(x, y, z, AB_params[Cs[cm-1+(cn-1)*cms]],
                                                    alpha, beta, gamma, c, epsilon1, epsilon2)

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return b_external_3d_fast


def b_full_3d_producer(a, b, c, R, z, r, phi, ns, ms, cns, cms):
    b_zeros = []
    a = a
    b = b
    c = c
    for n in range(ns):
        b_zeros.append(special.jn_zeros(n, ms))
    kms = np.asarray([b0/R for b0 in b_zeros])
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms[n][m]*np.abs(r))

    alpha_2d = np.zeros((cns, cms))
    beta_2d = np.zeros((cns, cms))
    gamma_2d = np.zeros((cns, cms))

    for cn in range(1, cns+1):
        for cm in range(1, cms+1):
            alpha_2d[cn-1][cm-1] = (cn*np.pi-np.pi/2)/a
            beta_2d[cn-1][cm-1] = (cm*np.pi-np.pi/2)/b
            gamma_2d[cn-1][cm-1] = np.sqrt(alpha_2d[cn-1][cm-1]**2+beta_2d[cn-1][cm-1]**2)

    def b_full_3d_fast(z, r, phi, R, ns, ms, delta1, cns, cms, epsilon1, epsilon2, **AB_params):
        """ 3D model for Bz Bx and By vs Z and X. Can take any number of Cnm terms."""
        def numexpr_model_r_calc(z, phi, n, D1, A, B, ivp, kms):
            return ne.evaluate('(cos(n*phi+D1))*ivp*kms*(A*cos(kms*z) + B*sin(-kms*z))')

        def numexpr_model_z_calc(z, phi, n, D1, A, B, iv, kms):
            return ne.evaluate('-(cos(n*phi+D1))*iv*kms*(A*sin(kms*z) + B*cos(-kms*z))')

        def numexpr_model_phi_calc(z, r, phi, n, D1, A, B, iv, kms):
            return ne.evaluate('n*(-sin(n*phi+D1))*(1/abs(r))*iv*(A*cos(kms*z) + B*sin(-kms*z))')

        def numexpr_model_r_ext_calc(z, r, phi, C, alpha, beta, gamma, c, epsilon1, epsilon2):
            return ne.evaluate('C*(alpha*cos(phi)*cos(epsilon2 + beta*abs(r)*sin(phi))*'
                               'sin(epsilon1 + alpha*abs(r)*cos(phi)) + '
                               'beta*cos(epsilon1 + alpha*abs(r)*cos(phi))*sin(phi)*'
                               'sin(epsilon2 + beta*abs(r)*sin(phi)))*sinh(gamma*(-c + z))')

        def numexpr_model_phi_ext_calc(z, r, phi, C, alpha, beta, gamma, c, epsilon1, epsilon2):
            return ne.evaluate('C*((-alpha)*cos(epsilon2 + beta*abs(r)*sin(phi))*'
                               'sin(phi)*sin(epsilon1 + alpha*abs(r)*cos(phi)) + '
                               'beta*cos(phi)*cos(epsilon1 + alpha*abs(r)*cos(phi))*'
                               'sin(epsilon2 + beta*abs(r)*sin(phi)))*sinh(gamma*(-c + z))')

        def numexpr_model_z_ext_calc(z, r, phi, C, alpha, beta, gamma, c, epsilon1, epsilon2):
            return ne.evaluate('(-C)*gamma*cos(epsilon1 + alpha*abs(r)*cos(phi))*'
                               'cos(epsilon2 + beta*abs(r)*sin(phi))*cosh(gamma*(-c + z))')

        model_r = 0.0
        model_z = 0.0
        model_phi = 0.0
        R = R
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                             x.split('_')[0])))
        Cs = sorted({k: v for (k, v) in six.iteritems(AB_params) if 'C' in k})

        for n in range(ns):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                model_r += numexpr_model_r_calc(z, phi, n, delta1, AB_params[ab[0]],
                                                AB_params[ab[1]], ivp[n][i], kms[n][i])
                model_z += numexpr_model_z_calc(z, phi, n, delta1, AB_params[ab[0]],
                                                AB_params[ab[1]], iv[n][i], kms[n][i])
                model_phi += numexpr_model_phi_calc(z, r, phi, n, delta1, AB_params[ab[0]],
                                                    AB_params[ab[1]], iv[n][i], kms[n][i])

        for cn in range(1, cns+1):
            for cm in range(1, cms+1):
                alpha = alpha_2d[cn-1][cm-1]
                beta = beta_2d[cn-1][cm-1]
                gamma = gamma_2d[cn-1][cm-1]

                model_r += numexpr_model_r_ext_calc(z, r, phi, AB_params[Cs[cm-1+(cn-1)*cms]],
                                                    alpha, beta, gamma, c, epsilon1, epsilon2)
                model_z += numexpr_model_z_ext_calc(z, r, phi, AB_params[Cs[cm-1+(cn-1)*cms]],
                                                    alpha, beta, gamma, c, epsilon1, epsilon2)
                model_phi += numexpr_model_phi_ext_calc(z, r, phi, AB_params[Cs[cm-1+(cn-1)*cms]],
                                                        alpha, beta, gamma, c, epsilon1, epsilon2)

        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return b_full_3d_fast


def brzphi_3d_producer_v2(z, r, phi, R, ns, ms):
    b_zeros = []
    for n in range(ns):
        b_zeros.append(special.jn_zeros(n, ms))
    kms = np.asarray([b/R for b in b_zeros])
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms[n][m]*np.abs(r))

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        def numexpr_model_r_calc(z, phi, n, A, B, C, D, ivp, kms):
            return ne.evaluate('(C*cos(n*phi)+D*sin(n*phi))*ivp*kms*(A*cos(kms*z) + B*sin(-kms*z))')

        def numexpr_model_z_calc(z, phi, n, A, B, C, D, iv, kms):
            return ne.evaluate('-(C*cos(n*phi)+D*sin(n*phi))*iv*kms*(A*sin(kms*z) + B*cos(-kms*z))')

        def numexpr_model_phi_calc(z, r, phi, n, A, B, C, D, iv, kms):
            return ne.evaluate('n*(-C*sin(n*phi)+D*cos(n*phi))*(1/abs(r))*iv*'
                               '(A*cos(kms*z) + B*sin(-kms*z))')

        model_r = 0.0
        model_z = 0.0
        model_phi = 0.0
        R = R
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                             x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('C' in k or 'D' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, cd in enumerate(pairwise(CDs)):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                model_r += numexpr_model_r_calc(z, phi, n, AB_params[ab[0]], AB_params[ab[1]],
                                                AB_params[cd[0]], AB_params[cd[1]], ivp[n][i],
                                                kms[n][i])
                model_z += numexpr_model_z_calc(z, phi, n, AB_params[ab[0]], AB_params[ab[1]],
                                                AB_params[cd[0]], AB_params[cd[1]], iv[n][i],
                                                kms[n][i])
                model_phi += numexpr_model_phi_calc(z, r, phi, n, AB_params[ab[0]],
                                                    AB_params[ab[1]], AB_params[cd[0]],
                                                    AB_params[cd[1]], iv[n][i], kms[n][i])

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_phase(z, r, phi, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    kms = []
    for n in range(ns):
        kms.append([])
        for m in range(ms):
            kms[-1].append((m+1)*np.pi/L)
    kms = np.asarray(kms)
    iv = np.empty((ns, ms, r.shape[0]))
    ivp = np.empty((ns, ms, r.shape[0]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms[n][m]*np.abs(r))

    @njit(parallel=True)
    def calc_b_fields(z, phi, r, n, A, B, D, ivp, iv, kms, model_r, model_z, model_phi):
        for i in prange(z.shape[0]):
            model_r[i] += (D*np.sin(n*phi[i]) + (1-D)*np.cos(n*phi[i])) * \
                ivp[i]*kms*(A*np.cos(kms*z[i]) + B*np.sin(kms*z[i]))
            model_z[i] += (D*np.sin(n*phi[i]) + (1-D)*np.cos(n*phi[i])) * \
                iv[i]*kms*(-A*np.sin(kms*z[i]) + B*np.cos(kms*z[i]))
            model_phi[i] += n*(D*np.cos(n*phi[i]) - (1-D)*np.sin(n*phi[i])) * \
                (1/r[i])*iv[i]*(A*np.cos(kms*z[i]) + B*np.sin(kms*z[i]))

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        R = R
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, d in enumerate(Ds):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = AB_params[ab[0]]
                B = AB_params[ab[1]]
                D = AB_params[d]
                _ivp = ivp[n][m]
                _iv = iv[n][m]
                _kms = kms[n][m]
                calc_b_fields(z, phi, r, n, A, B, D, _ivp, _iv, _kms, model_r, model_z, model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_phase_ext(z, r, phi, L, ns, ms, cns, cms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''
    kms = []
    for n in range(ns):
        kms.append([])
        for m in range(ms):
            kms[-1].append((m+1)*np.pi/L)
    kms = np.asarray(kms)
    iv = np.empty((ns, ms, r.shape[0]))
    ivp = np.empty((ns, ms, r.shape[0]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms[n][m]*np.abs(r))

    @njit(parallel=True)
    def calc_b_fields(z, phi, r, n, A, B, D, ivp, iv, kms, model_r, model_z, model_phi):
        for i in prange(z.shape[0]):
            model_r[i] += (D*np.sin(n*phi[i]) + (1-D)*np.cos(n*phi[i])) * \
                ivp[i]*kms*(A*np.cos(kms*z[i]) + B*np.sin(kms*z[i]))
            model_z[i] += (D*np.sin(n*phi[i]) + (1-D)*np.cos(n*phi[i])) * \
                iv[i]*kms*(-A*np.sin(kms*z[i]) + B*np.cos(kms*z[i]))
            model_phi[i] += n*(D*np.cos(n*phi[i]) - (1-D)*np.sin(n*phi[i])) * \
                (1/r[i])*iv[i]*(A*np.cos(kms*z[i]) + B*np.sin(kms*z[i]))

    @njit(parallel=True)
    def calc_b_fields_cart(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                           model_r, model_phi, model_z):
        for i in prange(z.shape[0]):
            model_x = k1 + k4*x[i] + k7*y[i] + k8*z[i] + k10*y[i]*z[i]

            model_y = k2 + k5*y[i] + k7*x[i] + k9*z[i] + k10*x[i]*z[i]

            model_z[i] += k3 + k6*z[i] + k8*x[i] + k9*y[i] + k10*x[i]*y[i]

            model_r[i] += model_x*np.cos(phi[i]) + model_y*np.sin(phi[i])
            model_phi[i] += -model_x*np.sin(phi[i]) + model_y*np.cos(phi[i])

    def brzphi_3d_fast(z, r, phi, x, y, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        R = R
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, d in enumerate(Ds):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = AB_params[ab[0]]
                B = AB_params[ab[1]]
                D = AB_params[d]
                _ivp = ivp[n][m]
                _iv = iv[n][m]
                _kms = kms[n][m]
                calc_b_fields(z, phi, r, n, A, B, D, _ivp, _iv, _kms, model_r, model_z, model_phi)

        k1 = AB_params['k1']
        k2 = AB_params['k2']
        k3 = AB_params['k3']
        k4 = AB_params['k4']
        k5 = AB_params['k5']
        k6 = AB_params['k6']
        k7 = AB_params['k7']
        k8 = AB_params['k8']
        k9 = AB_params['k9']
        k10 = AB_params['k10']
        calc_b_fields_cart(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                           model_r, model_phi, model_z)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_v8(z, r, phi, L, ns, ms, cns, cms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''
    kms = []
    for n in range(ns):
        kms.append([])
        for m in range(ms):
            kms[-1].append((m+1)*np.pi/L)
    kms = np.asarray(kms)
    iv = np.empty((ns, ms, r.shape[0]))
    ivp = np.empty((ns, ms, r.shape[0]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms[n][m]*np.abs(r))

    @njit(parallel=True)
    def calc_b_fields(z, phi, r, n, A, B, D, ivp, iv, kms, model_r, model_z, model_phi):
        for i in prange(z.shape[0]):
            model_r[i] += (D*np.sin(n*phi[i]) + (1-D)*np.cos(n*phi[i])) * \
                ivp[i]*kms*(A*np.cos(kms*z[i]) + B*np.sin(kms*z[i]))
            model_z[i] += (D*np.sin(n*phi[i]) + (1-D)*np.cos(n*phi[i])) * \
                iv[i]*kms*(-A*np.sin(kms*z[i]) + B*np.cos(kms*z[i]))
            model_phi[i] += n*(D*np.cos(n*phi[i]) - (1-D)*np.sin(n*phi[i])) * \
                (1/r[i])*iv[i]*(A*np.cos(kms*z[i]) + B*np.sin(kms*z[i]))

    @njit(parallel=True)
    def calc_b_fields_cart(x, y, z, phi, vx1, vy1, vz1, x1, y1, z1, vx2, vy2, vz2, x2, y2, z2,
                           model_r, model_phi, model_z):
        for i in prange(z.shape[0]):
            v1 = np.array([vx1, vy1, vz1])
            r1 = np.array([x[i]-x1, y[i]-y1, z[i]-z1])
            rcube1 = np.linalg.norm(r1)**3
            # res1 = np.cross(v1, r1)/rcube1
            res1 = np.array([v1[1]*r1[2]-v1[2]*r1[1], v1[2]*r1[0]-v1[0]*r1[2],
                             v1[0]*r1[1]-v1[1]*r1[0]])/rcube1
            model_x1, model_y1, model_z1 = res1

            v2 = np.array([vx2, vy2, vz2])
            r2 = np.array([x[i]-x2, y[i]-y2, z[i]-z2])
            rcube2 = np.linalg.norm(r2)**3
            # res2 = np.cross(v2, r2)/rcube2
            res2 = np.array([v2[1]*r2[2]-v2[2]*r2[1], v2[2]*r2[0]-v2[0]*r2[2],
                             v2[0]*r2[1]-v2[1]*r2[0]])/rcube2
            model_x2, model_y2, model_z2 = res2

            model_z[i] += model_z1 + model_z2

            model_r[i] += model_x1*np.cos(phi[i]) + model_y1*np.sin(phi[i])
            model_r[i] += model_x2*np.cos(phi[i]) + model_y2*np.sin(phi[i])
            model_phi[i] += -model_x1*np.sin(phi[i]) + model_y1*np.cos(phi[i])
            model_phi[i] += -model_x2*np.sin(phi[i]) + model_y2*np.cos(phi[i])

    @njit(parallel=True)
    def calc_b_fields_cart2(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                            model_r, model_phi, model_z):
        for i in prange(z.shape[0]):
            model_x = k1 + k4*x[i] + k7*y[i] + k8*z[i] + k10*y[i]*z[i]

            model_y = k2 + k5*y[i] + k7*x[i] + k9*z[i] + k10*x[i]*z[i]

            model_z[i] += k3 + k6*z[i] + k8*x[i] + k9*y[i] + k10*x[i]*y[i]

            model_r[i] += model_x*np.cos(phi[i]) + model_y*np.sin(phi[i])
            model_phi[i] += -model_x*np.sin(phi[i]) + model_y*np.cos(phi[i])

    def brzphi_3d_fast(z, r, phi, x, y, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        R = R
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, d in enumerate(Ds):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = AB_params[ab[0]]
                B = AB_params[ab[1]]
                D = AB_params[d]
                _ivp = ivp[n][m]
                _iv = iv[n][m]
                _kms = kms[n][m]
                calc_b_fields(z, phi, r, n, A, B, D, _ivp, _iv, _kms, model_r, model_z, model_phi)

        vx1 = AB_params['vx1']
        vy1 = AB_params['vy1']
        vz1 = AB_params['vz1']
        x1 = AB_params['x1']
        y1 = AB_params['y1']
        z1 = AB_params['z1']
        vx2 = AB_params['vx2']
        vy2 = AB_params['vy2']
        vz2 = AB_params['vz2']
        x2 = AB_params['x2']
        y2 = AB_params['y2']
        z2 = AB_params['z2']
        calc_b_fields_cart(x, y, z, phi, vx1, vy1, vz1, x1, y1, z1, vx2, vy2, vz2, x2, y2, z2,
                           model_r, model_phi, model_z)

        k1 = AB_params['k1']
        k2 = AB_params['k2']
        k3 = AB_params['k3']
        k4 = AB_params['k4']
        k5 = AB_params['k5']
        k6 = AB_params['k6']
        k7 = AB_params['k7']
        k8 = AB_params['k8']
        k9 = AB_params['k9']
        k10 = AB_params['k10']
        calc_b_fields_cart2(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                            model_r, model_phi, model_z)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_phase_hybrid(z, r, phi, L, ns, ms, cns, cms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''
    R = 20000

    b_zeros = []
    for cn in range(cns):
        b_zeros.append(special.jn_zeros(cn, cms))
    kms_j = np.asarray([b/R for b in b_zeros])
    jv = np.empty((cns, cms, r.shape[0]))
    jvp = np.empty((cns, cms, r.shape[0]))
    for cn in range(cns):
        for cm in range(cms):
            jv[cn][cm] = special.jv(cn, kms_j[cn][cm]*r)
            jvp[cn][cm] = special.jvp(cn, kms_j[cn][cm]*r)

    kms_i = []
    for n in range(ns):
        kms_i.append([])
        for m in range(ms):
            kms_i[-1].append((m+1)*np.pi/L)
    kms_i = np.asarray(kms_i)
    iv = np.empty((ns, ms, r.shape[0]))
    ivp = np.empty((ns, ms, r.shape[0]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms_i[n][m]*r)
            ivp[n][m] = special.ivp(n, kms_i[n][m]*r)

    @njit(parallel=True)
    def calc_b_fields_mb(z, phi, r, n, A, B, D, ivp, iv, kms_i, model_r, model_z, model_phi):
        for i in prange(z.shape[0]):
            model_r[i] += (D*np.sin(n*phi[i]) + (1-D)*np.cos(n*phi[i])) * \
                ivp[i]*kms_i*(A*np.cos(kms_i*z[i]) + B*np.sin(kms_i*z[i]))
            model_z[i] += (D*np.sin(n*phi[i]) + (1-D)*np.cos(n*phi[i])) * \
                iv[i]*kms_i*(-A*np.sin(kms_i*z[i]) + B*np.cos(kms_i*z[i]))
            model_phi[i] += n*(D*np.cos(n*phi[i]) - (1-D)*np.sin(n*phi[i])) * \
                (1/r[i])*iv[i]*(A*np.cos(kms_i*z[i]) + B*np.sin(kms_i*z[i]))

    @njit(parallel=True)
    def calc_b_fields_b(z, phi, r, n, E, F, G, jvp, jv, kms_j, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (G*np.sin(n*phi[i]) + (1-G)*np.cos(n*phi[i])) * \
                jvp[i]*kms_j * (E*np.sinh(kms_j*z[i]) + F*np.cosh(kms_j*z[i]))
            model_z[i] += (G*np.sin(n*phi[i]) + (1-G)*np.cos(n*phi[i])) * \
                jv[i]*kms_j*(E*np.cosh(kms_j*z[i]) + F*np.sinh(kms_j*z[i]))
            model_phi[i] += n*(G*np.cos(n*phi[i]) - (1-G)*np.sin(n*phi[i])) * \
                (1/r[i])*jv[i]*(E*np.sinh(kms_j*z[i]) + F*np.cosh(kms_j*z[i]))

    def brzphi_3d_fast(z, r, phi, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        EFs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('E' in k or 'F' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Gs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('G' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, d in enumerate(Ds):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = AB_params[ab[0]]
                B = AB_params[ab[1]]
                D = AB_params[d]
                _ivp = ivp[n][m]
                _iv = iv[n][m]
                _kms = kms_i[n][m]
                calc_b_fields_mb(z, phi, r, n, A, B, D, _ivp, _iv, _kms, model_r, model_z,
                                 model_phi)

        for cn, g in enumerate(Gs):
            for cm, ef in enumerate(pairwise(EFs[cn*cms*2:(cn+1)*cms*2])):

                E = AB_params[ef[0]]
                F = AB_params[ef[1]]
                G = AB_params[g]
                _jvp = jvp[cn][cm]
                _jv = jv[cn][cm]
                _kms = kms_j[cn][cm]
                calc_b_fields_b(z, phi, r, cn, E, F, G, _jvp, _jv, _kms, model_r, model_z,
                                model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_phase_hybrid_disp(z, r, phi, rp, phip, L, ns, ms, cns, cms):
    '''
    Fit function for the GA05 map.  Requires a set of displaced coordinates that correspond to a
    return wire located outside of the solenoid.

    The displaced field is modeled by the Regular Bessel Function solution to Laplace's EQ.
    '''
    R = 30000

    b_zeros = []
    for cn in range(cns):
        b_zeros.append(special.jn_zeros(cn, cms))
    kms_j = np.asarray([b/R for b in b_zeros])
    jv = np.empty((cns, cms, rp.shape[0], rp.shape[1]))
    jvp = np.empty((cns, cms, rp.shape[0], rp.shape[1]))
    for cn in range(cns):
        for cm in range(cms):
            jv[cn][cm] = special.jv(cn, kms_j[cn][cm]*np.abs(rp))
            jvp[cn][cm] = special.jvp(cn, kms_j[cn][cm]*np.abs(rp))

    kms_i = []
    for n in range(ns):
        kms_i.append([])
        for m in range(ms):
            kms_i[-1].append((m+1)*np.pi/L)
    kms_i = np.asarray(kms_i)
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms_i[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms_i[n][m]*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:])"],
                 '(m), (m), (m), (), (), (), (), (m), (m), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_mb(z, phi, r, n, A, B, D, ivp, iv, kms_i, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += np.cos(n[0]*phi[i]-D[0])*ivp[i]*kms_i[0] * \
                (A[0]*np.cos(kms_i[0]*z[i]) + B[0]*np.sin(kms_i[0]*z[i]))
            model_z[i] += np.cos(n[0]*phi[i]-D[0])*iv[i]*kms_i[0] * \
                (-A[0]*np.sin(kms_i[0]*z[i]) + B[0]*np.cos(kms_i[0]*z[i]))
            model_phi[i] += n[0]*(-np.sin(n[0]*phi[i]-D[0])) * \
                (1/np.abs(r[i]))*iv[i]*(A[0]*np.cos(kms_i[0]*z[i]) + B[0]*np.sin(kms_i[0]*z[i]))

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (m), (m), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_b(z, phi, r, n, E, F, G, jvp, jv, kms_j, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += np.cos(n[0]*phi[i]-G[0])*jvp[i]*kms_j[0] * \
                (E[0]*np.sinh(kms_j[0]*z[i]) + F[0]*np.cosh(kms_j[0]*z[i]))
            model_z[i] += np.cos(n[0]*phi[i]-G[0])*jv[i]*kms_j[0] * \
                (E[0]*np.cosh(kms_j[0]*z[i]) + F[0]*np.sinh(kms_j[0]*z[i]))
            model_phi[i] += n[0]*(-np.sin(n[0]*phi[i]-G[0])) * \
                (1/np.abs(r[i]))*jv[i]*(E[0]*np.sinh(kms_j[0]*z[i]) + F[0]*np.cosh(kms_j[0]*z[i]))

    def brzphi_3d_fast(z, r, phi, rp, phip, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        EFs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('E' in k or 'F' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Gs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('G' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, d in enumerate(Ds):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                D = np.array([AB_params[d]], dtype=np.float64)
                _ivp = ivp[n][i]
                _iv = iv[n][i]
                _kms = np.array([kms_i[n][i]])
                _n = np.array([n])
                calc_b_fields_mb(z, phi, r, _n, A, B, D, _ivp, _iv, _kms, model_r, model_z,
                                 model_phi)

        for cn, g in enumerate(Gs):
            for i, ef in enumerate(pairwise(EFs[cn*cms*2:(cn+1)*cms*2])):

                E = np.array([AB_params[ef[0]]], dtype=np.float64)
                F = np.array([AB_params[ef[1]]], dtype=np.float64)
                G = np.array([AB_params[g]], dtype=np.float64)
                _jvp = jvp[cn][i]
                _jv = jv[cn][i]
                _kms = np.array([kms_j[cn][i]])
                _n = np.array([cn])
                calc_b_fields_b(z, phip, rp, _n, E, F, G, _jvp, _jv, _kms, model_r, model_z,
                                model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_phase_hybrid_disp2(z, r, phi, rp, phip, L, ns, ms, cns, cms):
    '''
    Fit function for the GA05 map.  Requires a set of displaced coordinates that correspond to a
    return wire located outside of the solenoid.

    The displaced field is modeled by the Modified Bessel Function solution to Laplace's EQ.
    '''
    R = 7000

    kms_j = []
    for n in range(cns):
        kms_j.append([])
        for m in range(cms):
            kms_j[-1].append((m+1)*np.pi/R)
    kms_j = np.asarray(kms_j)
    iv_p = np.empty((cns, cms, rp.shape[0], rp.shape[1]))
    ivp_p = np.empty((cns, cms, rp.shape[0], rp.shape[1]))
    for n in range(cns):
        for m in range(cms):
            iv_p[n][m] = special.iv(n, kms_j[n][m]*np.abs(rp))
            ivp_p[n][m] = special.ivp(n, kms_j[n][m]*np.abs(rp))

    kms_i = []
    for n in range(ns):
        kms_i.append([])
        for m in range(ms):
            kms_i[-1].append((m+1)*np.pi/L)
    kms_i = np.asarray(kms_i)
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms_i[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms_i[n][m]*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:])"],
                 '(m), (m), (m), (), (), (), (), (m), (m), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_mb(z, phi, r, n, A, B, D, ivp, iv, kms_i, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += np.cos(n[0]*phi[i]-D[0])*ivp[i]*kms_i[0] * \
                (A[0]*np.cos(kms_i[0]*z[i]) + B[0]*np.sin(kms_i[0]*z[i]))
            model_z[i] += np.cos(n[0]*phi[i]-D[0])*iv[i]*kms_i[0] * \
                (-A[0]*np.sin(kms_i[0]*z[i]) + B[0]*np.cos(kms_i[0]*z[i]))
            model_phi[i] += n[0]*(-np.sin(n[0]*phi[i]-D[0])) * \
                (1/np.abs(r[i]))*iv[i]*(A[0]*np.cos(kms_i[0]*z[i]) + B[0]*np.sin(kms_i[0]*z[i]))

    def brzphi_3d_fast(z, r, phi, rp, phip, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        EFs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('E' in k or 'F' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Gs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('G' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, d in enumerate(Ds):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                D = np.array([AB_params[d]], dtype=np.float64)
                _ivp = ivp[n][i]
                _iv = iv[n][i]
                _kms = np.array([kms_i[n][i]])
                _n = np.array([n])
                calc_b_fields_mb(z, phi, r, _n, A, B, D, _ivp, _iv, _kms, model_r, model_z,
                                 model_phi)

        for cn, g in enumerate(Gs):
            for i, ef in enumerate(pairwise(EFs[cn*cms*2:(cn+1)*cms*2])):

                E = np.array([AB_params[ef[0]]], dtype=np.float64)
                F = np.array([AB_params[ef[1]]], dtype=np.float64)
                G = np.array([AB_params[g]], dtype=np.float64)
                _ivp = ivp_p[cn][i]
                _iv = iv_p[cn][i]
                _kms = np.array([kms_j[cn][i]])
                _n = np.array([cn])
                calc_b_fields_mb(z, phip, rp, _n, E, F, G, _ivp, _iv, _kms, model_r, model_z,
                                 model_phi)

        # model_phi[np.isinf(model_phi)] = 0

        # Modify Br and Bphi to have contributions from two external fields in the X and Y plane
        model_r += AB_params['X']*np.cos(phi)+AB_params['Y']*np.sin(phi)
        model_phi += -1.0*AB_params['X']*np.sin(phi)+AB_params['Y']*np.cos(phi)
        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_phase_hybrid_disp3(z, r, phi, rp, phip, L, ns, ms, cns, cms):
    '''
    Fit function for the GA05 map.  Requires a set of displaced coordinates that correspond to a
    return wire located outside of the solenoid.

    The displaced field is modeled by the Modified Bessel Function solution to Laplace's EQ.
    '''
    R = 7000

    kms_j = []
    for n in range(cns):
        kms_j.append([])
        for m in range(cms):
            kms_j[-1].append((m+1)*np.pi/R)
    kms_j = np.asarray(kms_j)
    iv_p = np.empty((cns, cms, rp.shape[0], rp.shape[1]))
    ivp_p = np.empty((cns, cms, rp.shape[0], rp.shape[1]))
    for n in range(cns):
        for m in range(cms):
            iv_p[n][m] = special.iv(n, kms_j[n][m]*np.abs(rp))
            ivp_p[n][m] = special.ivp(n, kms_j[n][m]*np.abs(rp))

    kms_i = []
    for n in range(ns):
        kms_i.append([])
        for m in range(ms):
            kms_i[-1].append((m+1)*np.pi/L)
    kms_i = np.asarray(kms_i)
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms_i[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms_i[n][m]*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:])"],
                 '(m), (m), (m), (), (), (), (), (m), (m), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_mb(z, phi, r, n, A, B, D, ivp, iv, kms_i, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += np.cos(n[0]*phi[i]-D[0])*ivp[i]*kms_i[0] * \
                (A[0]*np.cos(kms_i[0]*z[i]) + B[0]*np.sin(kms_i[0]*z[i]))
            model_z[i] += np.cos(n[0]*phi[i]-D[0])*iv[i]*kms_i[0] * \
                (-A[0]*np.sin(kms_i[0]*z[i]) + B[0]*np.cos(kms_i[0]*z[i]))
            model_phi[i] += n[0]*(-np.sin(n[0]*phi[i]-D[0])) * \
                (1/np.abs(r[i]))*iv[i]*(A[0]*np.cos(kms_i[0]*z[i]) + B[0]*np.sin(kms_i[0]*z[i]))

    def brzphi_3d_fast(z, r, phi, rp, phip, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        EFs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('E' in k or 'F' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Gs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('G' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, d in enumerate(Ds):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                D = np.array([AB_params[d]], dtype=np.float64)
                _ivp = ivp[n][i]
                _iv = iv[n][i]
                _kms = np.array([kms_i[n][i]])
                _n = np.array([n])
                calc_b_fields_mb(z, phi, r, _n, A, B, D, _ivp, _iv, _kms, model_r, model_z,
                                 model_phi)

        for cn, g in enumerate(Gs):
            for i, ef in enumerate(pairwise(EFs[cn*cms*2:(cn+1)*cms*2])):

                E = np.array([AB_params[ef[0]]], dtype=np.float64)
                F = np.array([AB_params[ef[1]]], dtype=np.float64)
                G = np.array([AB_params[g]], dtype=np.float64)
                _ivp = ivp_p[cn][i]
                _iv = iv_p[cn][i]
                _kms = np.array([kms_j[cn][i]])
                _n = np.array([cn])
                calc_b_fields_mb(z, phip, rp, _n, E, F, G, _ivp, _iv, _kms, model_r, model_z,
                                 model_phi)

        # Modify Br and Bphi to have contributions from two external fields in the X and Y plane
        model_r += AB_params['X']*np.cos(phi)+AB_params['Y']*np.sin(phi)
        model_phi += -1.0*AB_params['X']*np.sin(phi)+AB_params['Y']*np.cos(phi)
        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_bessel(z, r, phi, R, ns, ms):
    b_zeros = []
    for n in range(ns):
        b_zeros.append(special.jn_zeros(n, ms))
    kms = np.asarray([b/R for b in b_zeros])
    jv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    jvp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            jv[n][m] = special.jv(n, kms[n][m]*np.abs(r))
            jvp[n][m] = special.jvp(n, kms[n][m]*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (m), (m), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, n, A, B, C, D, jvp, jv, kms, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (C[0]*np.cos(n[0]*phi[i])+D[0]*np.sin(n[0]*phi[i]))*jvp[i]*kms[0] * \
                (A[0]*np.sinh(kms[0]*z[i]) + B[0]*np.cosh(kms[0]*z[i]))
            model_z[i] += (C[0]*np.cos(n[0]*phi[i])+D[0]*np.sin(n[0]*phi[i]))*jv[i]*kms[0] * \
                (A[0]*np.cosh(kms[0]*z[i]) + B[0]*np.sinh(kms[0]*z[i]))
            model_phi[i] += n[0]*(-C[0]*np.sin(n[0]*phi[i])+D[0]*np.cos(n[0]*phi[i])) * \
                (1/np.abs(r[i]))*jv[i]*(A[0]*np.sinh(kms[0]*z[i]) + B[0]*np.cosh(kms[0]*z[i]))

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        R = R
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ', '.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                              x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('C' in k or 'D' in k)},
                     key=lambda x: ', '.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, cd in enumerate(pairwise(CDs)):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[cd[0]]], dtype=np.float64)
                D = np.array([AB_params[cd[1]]], dtype=np.float64)
                _jvp = jvp[n][i]
                _jv = jv[n][i]
                _kms = np.array([kms[n][i]])
                _n = np.array([n])
                calc_b_fields(z, phi, r, _n, A, B, C, D, _jvp, _jv, _kms, model_r, model_z,
                              model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_bessel_hybrid(z, r, phi, R, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''
    R = 8000
    L = 15000

    b_zeros = []
    for n in range(ns):
        b_zeros.append(special.jn_zeros(n, ms))
    kms_j = np.asarray([b/R for b in b_zeros])
    jv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    jvp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            jv[n][m] = special.jv(n, kms_j[n][m]*np.abs(r))
            jvp[n][m] = special.jvp(n, kms_j[n][m]*np.abs(r))

    kms_i = []
    for n in range(ns):
        kms_i.append([])
        for m in range(ms):
            kms_i[-1].append((m+1)*np.pi/L)
    kms_i = np.asarray(kms_i)
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms_i[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms_i[n][m]*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (), (m), (m), (m), (m), (), ()'
                 '->(m), (m), (m)', nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, n, A, B, C, D, E, F, ivp, iv, jvp, jv,
                      kms_i, kms_j, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (C[0]*np.cos(n[0]*phi[i])+D[0]*np.sin(n[0]*phi[i])) * \
                (ivp[i]*kms_i[0]*(A[0]*np.cos(kms_i[0]*z[i]) + B[0]*np.sin(kms_i[0]*z[i])) +
                 jvp[i]*kms_j[0]*(E[0]*np.cosh(kms_j[0]*z[i]) + F[0]*np.sinh(kms_j[0]*z[i])))

            model_z[i] += (C[0]*np.cos(n[0]*phi[i])+D[0]*np.sin(n[0]*phi[i])) * \
                (iv[i]*kms_i[0]*(-A[0]*np.sin(kms_i[0]*z[i]) + B[0]*np.cos(kms_i[0]*z[i])) +
                 jv[i]*kms_j[0]*(E[0]*np.sinh(kms_j[0]*z[i]) + F[0]*np.cosh(kms_j[0]*z[i])))

            model_phi[i] += (-C[0]*np.sin(n[0]*phi[i])+D[0]*np.cos(n[0]*phi[i])) * \
                n[0]/np.abs(r[i])*(iv[i]*(A[0]*np.cos(kms_i[0]*z[i]) + B[0]*np.sin(kms_i[0]*z[i])) +
                                   jv[i]*(E[0]*np.cosh(kms_j[0]*z[i]) +
                                          F[0]*np.sinh(kms_j[0]*z[i])))

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        R = R
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ', '.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                              x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('C' in k or 'D' in k)},
                     key=lambda x: ', '.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, cd in enumerate(pairwise(CDs)):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):
                ef = (ab[0].replace('A', 'E'), ab[1].replace('B', 'F'))

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[cd[0]]], dtype=np.float64)
                D = np.array([AB_params[cd[1]]], dtype=np.float64)
                E = np.array([AB_params[ef[0]]], dtype=np.float64)
                F = np.array([AB_params[ef[1]]], dtype=np.float64)
                _ivp = ivp[n][i]
                _iv = iv[n][i]
                _jvp = jvp[n][i]
                _jv = jv[n][i]
                _kms_i = np.array([kms_i[n][i]])
                _kms_j = np.array([kms_j[n][i]])
                _n = np.array([n])
                calc_b_fields(z, phi, r, _n, A, B, C, D, E, F, _ivp, _iv, _jvp, _jv,
                              _kms_i, _kms_j, model_r, model_z, model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_profile(z, r, phi, R, ns, ms):
    b_zeros = []
    for n in range(ns):
        b_zeros.append(special.jn_zeros(n, ms))
    kms = np.asarray([b/R for b in b_zeros])
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms[n][m]*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], int64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (), (), (), (), (), (m), ()->(m)',
                 nopython=True, target='parallel')
    def numba_parallel_model_r_calc(z, phi, n, A, B, C, D, ivp, kms, model_r):
        for i in range(z.shape[0]):
            model_r[i] += (C[0]*np.cos(n[0]*phi[i])+D[0]*np.sin(n[0]*phi[i]))*ivp[i]*kms[0] * \
                (A[0]*np.cos(kms[0]*z[i]) + B[0]*np.sin(-kms[0]*z[i]))

    @guvectorize(["void(float64[:], float64[:], int64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (), (), (), (), (), (m), ()->(m)', nopython=True, target='cpu')
    def numba_single_model_r_calc(z, phi, n, A, B, C, D, ivp, kms, model_r):
        for i in range(z.shape[0]):
            model_r[i] += (C[0]*np.cos(n[0]*phi[i])+D[0]*np.sin(n[0]*phi[i]))*ivp[i]*kms[0] * \
                (A[0]*np.cos(kms[0]*z[i]) + B[0]*np.sin(-kms[0]*z[i]))

    def numexpr_model_r_calc(z, phi, n, A, B, C, D, ivp, kms):
        return ne.evaluate('(C*cos(n*phi)+D*sin(n*phi))*ivp*kms*(A*cos(kms*z) + B*sin(-kms*z))')

    def numpy_model_r_calc(z, phi, n, A, B, C, D, ivp, kms):
        return (C*np.cos(n*phi)+D*np.sin(n*phi))*ivp*kms*(A*np.cos(kms*z) + B*np.sin(-kms*z))

    def pythonloops_model_r_calc(z, phi, n, A, B, C, D, ivp, kms, model_r):
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                model_r[i][j] += (C*cos(n*phi[i][j])+D*sin(n*phi[i][j]))*ivp[i][j]*kms * \
                    (A*cos(kms*z[i][j]) + B*sin(-kms*z[i][j]))

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        R = R
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ', '.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                              x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('C' in k or 'D' in k)},
                     key=lambda x: ', '.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, cd in enumerate(pairwise(CDs)):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[cd[0]]], dtype=np.float64)
                D = np.array([AB_params[cd[1]]], dtype=np.float64)
                _ivp = ivp[n][i]
                # _iv = iv[n][i]
                # _kms = np.array([kms[n][i]])
                # _n = np.array([n])

                # model_r += numpy_model_r_calc(z, phi, n, A, B, C, D, _ivp, _kms)
                # numba_single_model_r_calc(z, phi, _n, A, B, C, D, _ivp, _kms, model_r)
                # model_r += numexpr_model_r_calc(z, phi, n, A, B, C, D, _ivp, _kms)
                # numba_parallel_model_r_calc(z, phi, _n, A, B, C, D, _ivp, _kms, model_r)
                pythonloops_model_r_calc(z, phi, n, A, B, C, D, _ivp, kms[n][i], model_r)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_hel_v0(z, r, phi, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    P = L/(2*np.pi)

    iv1 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    iv2 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    ivp1 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    ivp2 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            # if n <= m:
                iv1[n][m] = special.iv(m-n, (n/P)*np.abs(r))
                iv2[n][m] = special.iv(m+n, (n/P)*np.abs(r))
                ivp1[n][m] = 0.5*(special.iv(-1+m-n, (n/P)*np.abs(r)) +
                                  special.iv(1+m-n, (n/P)*np.abs(r)))
                ivp2[n][m] = 0.5*(special.iv(-1+m+n, (n/P)*np.abs(r)) +
                                  special.iv(1+m+n, (n/P)*np.abs(r)))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], int64[:], int64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (), (m), (m), (m), (m)->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, P, m, n, A, B, C, D,
                      iv1, iv2, ivp1, ivp2, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (n[0]/P[0])*(ivp1[i]*(A[0]*np.cos(n[0]*z[i]/P[0]+(m[0]-n[0])*phi[i]) +
                                                B[0]*np.sin(n[0]*z[i]/P[0]+(m[0]-n[0])*phi[i])) +
                                       ivp2[i]*(C[0]*np.cos(n[0]*z[i]/P[0]-(m[0]+n[0])*phi[i]) -
                                                D[0]*np.sin(n[0]*z[i]/P[0]-(m[0]+n[0])*phi[i])))

            model_z[i] += (n[0]/P[0])*(iv1[i]*(-A[0]*np.sin(n[0]*z[i]/P[0]+(m[0]-n[0])*phi[i]) +
                                               B[0]*np.cos(n[0]*z[i]/P[0]+(m[0]-n[0])*phi[i])) -
                                       iv2[i]*(C[0]*np.sin(n[0]*z[i]/P[0]-(m[0]+n[0])*phi[i]) +
                                               D[0]*np.cos(n[0]*z[i]/P[0]-(m[0]+n[0])*phi[i])))

            model_phi[i] += (1.0/np.abs(r[i])) * \
                ((-m[0]+n[0])*iv1[i]*(A[0]*np.sin(n[0]*z[i]/P[0]+(m[0]-n[0])*phi[i]) -
                                      B[0]*np.cos(n[0]*z[i]/P[0]+(m[0]-n[0])*phi[i])) +
                 (m[0]+n[0])*iv2[i]*(C[0]*np.sin(n[0]*z[i]/P[0]-(m[0]+n[0])*phi[i]) +
                                     D[0]*np.cos(n[0]*z[i]/P[0]-(m[0]+n[0])*phi[i])))
            # print(model_r[i], n[0], m[0], P[0], ivp1[i], ivp2[i], A[0], B[0], C[0], D[0], z[i],
            #       phi[i])

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        _P = np.asarray([R/(2*np.pi)])
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            # print('n', n)
            # print(np.any(np.isnan(model_r)))
            # print(np.any(np.isnan(model_z)))
            # print(np.any(np.isnan(model_phi)))
            for m, ab in enumerate(quadwise(ABs[n*ms*4:(n+1)*(ms)*4])):
                if n <= m:
                    # print('n', n, 'm', m, ab)

                    A = np.array([AB_params[ab[0]]], dtype=np.float64)
                    B = np.array([AB_params[ab[1]]], dtype=np.float64)
                    C = np.array([AB_params[ab[2]]], dtype=np.float64)
                    D = np.array([AB_params[ab[3]]], dtype=np.float64)
                    _iv1 = iv1[n][m]
                    _iv2 = iv2[n][m]
                    _ivp1 = ivp1[n][m]
                    _ivp2 = ivp2[n][m]
                    _n = np.array([n])
                    _m = np.array([m])
                    calc_b_fields(z, phi, r, _P, _m, _n, A, B, C, D, _iv1, _iv2, _ivp1, _ivp2,
                                  model_r, model_z, model_phi)

        # model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast

def brzphi_3d_producer_hel_v15(z, r, phi, L, ns, ms, n_scale):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression.
    WARNING: N AND M ARE REINTERPRETED IN THIS CASE
    '''

    P = L/(2*np.pi)
    # P = L
    scale = n_scale

    iv1 = np.zeros((ns, ms, len(r)))
    ivp1 = np.zeros((ns, ms, len(r)))
    for n in range(ns):
        for m in range(ms):
                # iv1[n][m] = special.iv(m, ((n*scale)/P)*r)
                # ivp1[n][m] = 0.5*(special.iv(-1+m, ((n*scale)/P)*r) +
                #                   special.iv(1+m, ((n*scale)/P)*r))
                iv1[n][m] = special.iv(m, (n/P)*r)
                ivp1[n][m] = 0.5*(special.iv(-1+m, (n/P)*r) +
                                  special.iv(1+m, (n/P)*r))

    @njit(parallel=True)
    def calc_b_fields(z, phi, r, P, m, n, scale, A, B,
                      iv1, ivp1, model_r, model_z, model_phi):
        for i in prange(z.shape[0]):
            model_r[i] += (n/P) * \
                (ivp1[i]*(A*np.cos((n*scale)*z[i]/P+m*phi[i]) +
                          B*np.sin((n*scale)*z[i]/P+m*phi[i])))

            model_z[i] += (n/P) * \
                (iv1[i]*(-A*np.sin((n*scale)*z[i]/P+m*phi[i]) +
                         B*np.cos((n*scale)*z[i]/P+m*phi[i])))

            model_phi[i] += (1.0/r[i]) * \
                (-m*iv1[i]*(A*np.sin((n*scale)*z[i]/P+m*phi[i]) -
                            B*np.cos((n*scale)*z[i]/P+m*phi[i])))

    @njit(parallel=True)
    def calc_b_fields_cart(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                           model_r, model_phi, model_z):
        for i in prange(z.shape[0]):
            model_x = k1 + k4*x[i] + k7*y[i] + k8*z[i] + k10*y[i]*z[i]

            model_y = k2 + k5*y[i] + k7*x[i] + k9*z[i] + k10*x[i]*z[i]

            model_z[i] += k3 + k6*z[i] + k8*x[i] + k9*y[i] + k10*x[i]*y[i]

            model_r[i] += model_x*np.cos(phi[i]) + model_y*np.sin(phi[i])
            model_phi[i] += -model_x*np.sin(phi[i]) + model_y*np.cos(phi[i])

    def brzphi_3d_fast(z, r, phi, x, y, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*(ms)*2])):
                A = AB_params[ab[0]]
                B = AB_params[ab[1]]
                # P = R
                _iv1 = iv1[n][m]
                _ivp1 = ivp1[n][m]
                # _iv1 = special.iv(m, ((n*scale)/P)*r)
                # _ivp1 = 0.5*(special.iv(-1+m, ((n*scale)/P)*r) +
                #              special.iv(1+m, ((n*scale)/P)*r))
                calc_b_fields(z, phi, r, P, m, n, scale, A, B, _iv1, _ivp1,
                              model_r, model_z, model_phi)

        k1 = AB_params['k1']
        k2 = AB_params['k2']
        k3 = AB_params['k3']
        k4 = AB_params['k4']
        k5 = AB_params['k5']
        k6 = AB_params['k6']
        k7 = AB_params['k7']
        k8 = AB_params['k8']
        k9 = AB_params['k9']
        k10 = AB_params['k10']
        calc_b_fields_cart(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                           model_r, model_phi, model_z)

        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_hel_v16(z, r, phi, L, ns, ms, n_scale):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression.
    WARNING: N AND M ARE REINTERPRETED IN THIS CASE
    '''

    P = L/(2*np.pi)
    scale = n_scale

    iv1 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    ivp1 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            # if n <= m:
                iv1[n][m] = special.iv(m, ((n*scale)/P)*np.abs(r))
                ivp1[n][m] = 0.5*(special.iv(-1+m, ((n*scale)/P)*np.abs(r)) +
                                  special.iv(1+m, ((n*scale)/P)*np.abs(r)))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], int64[:], float64[:],"
                  "float64[:], float64[:], float64[:], "
                  "float64[:], float64[:], float64[:],"
                  "float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (m), (m)->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, P, m, n, scale, A, B,
                      iv1, ivp1, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (n[0]*scale[0]/P[0]) * \
                (ivp1[i]*(A[0]*np.cos((n[0]*scale[0])*z[i]/P[0]+m[0]*phi[i]) +
                          B[0]*np.sin((n[0]*scale[0])*z[i]/P[0]+m[0]*phi[i])))

            model_z[i] += (n[0]*scale[0]/P[0]) * \
                (iv1[i]*(-A[0]*np.sin((n[0]*scale[0])*z[i]/P[0]+m[0]*phi[i]) +
                         B[0]*np.cos((n[0]*scale[0])*z[i]/P[0]+m[0]*phi[i])))

            model_phi[i] += (1.0/np.abs(r[i])) * \
                (-m[0]*iv1[i]*(A[0]*np.sin((n[0]*scale[0])*z[i]/P[0]+m[0]*phi[i]) -
                               B[0]*np.cos((n[0]*scale[0])*z[i]/P[0]+m[0]*phi[i])))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:])"],
                 '(m), (m), (m), (m), (), (), (), (), (), (), (), ()->(m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, phi, xp1, yp1, zp1, xp2, yp2, zp2, k1, k2,
                           model_r, model_phi):
        for i in range(z.shape[0]):
            xd1 = (x[i]-xp1[0])
            xd2 = (x[i]-xp2[0])
            yd1 = (y[i]-yp1[0])
            yd2 = (y[i]-yp2[0])
            zd1 = (z[i]-zp1[0])
            zd2 = (z[i]-zp2[0])
            rs1 = (x[i]-xp1[0])**2+(y[i]-yp1[0])**2+(z[i]-zp1[0])**2
            rs2 = (x[i]-xp2[0])**2+(y[i]-yp2[0])**2+(z[i]-zp2[0])**2
            rc1 = (x[i]-xp1[0])**2+(y[i]-yp1[0])**2
            rc2 = (x[i]-xp2[0])**2+(y[i]-yp2[0])**2

            model_x = k1[0]*yd1*(
                -(rc1-zd1**2)*np.sqrt(rc1/rs1**2)-zd1/np.sqrt(rs1/rc1))/rc1**(3/2) + \
                k2[0]*yd2*(
                    -(rc2-zd2**2)*np.sqrt(rc2/rs2**2)-zd2/np.sqrt(rs2/rc2))/rc2**(3/2)

            model_y = -k1[0]*xd1*(
                np.sqrt(rc1/rs1**2)*(rs1-2*rc1)-zd1/np.sqrt(rs1/rc1))/rc1**(3/2) - \
                k2[0]*xd2*(
                    np.sqrt(rc2/rs2**2)*(rs2-2*rc2)-zd2/np.sqrt(rs2/rc2))/rc2**(3/2)

            model_r[i] += model_x*np.cos(phi[i]) + model_y*np.sin(phi[i])
            model_phi[i] += -model_x*np.sin(phi[i]) + model_y*np.cos(phi[i])

    def brzphi_3d_fast(z, r, phi, x, y, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        _P = np.asarray([R/(2*np.pi)])
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*(ms)*2])):
                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                _iv1 = iv1[n][m]
                _ivp1 = ivp1[n][m]
                _n = np.array([n])
                _m = np.array([m])
                _scale = np.array([scale])
                calc_b_fields(z, phi, r, _P, _m, _n, _scale, A, B, _iv1, _ivp1,
                              model_r, model_z, model_phi)

        _xp1 = np.array([AB_params['xp1']], dtype=np.float64)
        _xp2 = np.array([AB_params['xp2']], dtype=np.float64)
        _yp1 = np.array([AB_params['yp1']], dtype=np.float64)
        _yp2 = np.array([AB_params['yp2']], dtype=np.float64)
        _zp1 = np.array([AB_params['zp1']], dtype=np.float64)
        _zp2 = np.array([AB_params['zp2']], dtype=np.float64)
        _k1 = np.array([AB_params['k1']], dtype=np.float64)
        _k2 = np.array([AB_params['k2']], dtype=np.float64)
        calc_b_fields_cart(x, y, z, phi, _xp1, _yp1, _zp1, _xp2, _yp2, _zp2, _k1, _k2,
                           model_r, model_phi)

        # model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_hel_v17(z, r, phi, L, ns, ms, n_scale):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression.
    WARNING: N AND M ARE REINTERPRETED IN THIS CASE
    '''

    P = L/(2*np.pi)
    scale = n_scale

    iv1 = np.zeros((ns, ms, len(r)))
    ivp1 = np.zeros((ns, ms, len(r)))
    iv2 = np.zeros((ns, ms, len(r)))
    ivp2 = np.zeros((ns, ms, len(r)))
    for n in range(ns):
        for m in range(ms):
                iv1[n][m] = special.iv(m, ((n*scale)/P)*r)
                ivp1[n][m] = 0.5*(special.iv(-1+m, ((n*scale)/P)*r) +
                                  special.iv(1+m, ((n*scale)/P)*r))
                iv2[n][m] = special.iv(m, (-(n*scale)/P)*r)
                ivp2[n][m] = 0.5*(special.iv(-1+m, (-(n*scale)/P)*r) +
                                  special.iv(1+m, (-(n*scale)/P)*r))

    @njit(parallel=True)
    def calc_b_fields(z, phi, r, P, m, n, scale, A, B, D,
                      iv1, iv2, ivp1, ivp2, model_r, model_z, model_phi):
        for i in prange(z.shape[0]):
            model_r[i] += (n*scale/P) * (D*ivp1[i] - (1-D)*ivp2[i]) * \
                (A*np.cos((n*scale)*z[i]/P+m*phi[i]) +
                 B*np.sin((n*scale)*z[i]/P+m*phi[i]))

            model_z[i] += (n*scale/P) * (D*iv1[i] + (1-D)*iv2[i]) * \
                (-A*np.sin((n*scale)*z[i]/P+m*phi[i]) +
                 B*np.cos((n*scale)*z[i]/P+m*phi[i]))

            model_phi[i] += (-m/r[i]) * (D*iv1[i] + (1-D)*iv2[i]) * \
                (A*np.sin((n*scale)*z[i]/P+m*phi[i]) -
                 B*np.cos((n*scale)*z[i]/P+m*phi[i]))

    @njit(parallel=True)
    def calc_b_fields_cart(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                           model_r, model_phi, model_z):
        for i in prange(z.shape[0]):
            model_x = k1 + k4*x[i] + k7*y[i] + k8*z[i] + k10*y[i]*z[i]

            model_y = k2 + k5*y[i] + k7*x[i] + k9*z[i] + k10*x[i]*z[i]

            model_z[i] += k3 + k6*z[i] + k8*x[i] + k9*y[i] + k10*x[i]*y[i]

            model_r[i] += model_x*np.cos(phi[i]) + model_y*np.sin(phi[i])
            model_phi[i] += -model_x*np.sin(phi[i]) + model_y*np.cos(phi[i])

    def brzphi_3d_fast(z, r, phi, x, y, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*(ms)*2])):
                A = AB_params[ab[0]]
                B = AB_params[ab[1]]
                D = AB_params[f'D_{n}']
                _iv1 = iv1[n][m]
                _ivp1 = ivp1[n][m]
                _iv2 = iv2[n][m]
                _ivp2 = ivp2[n][m]
                calc_b_fields(z, phi, r, P, m, n, scale, A, B, D, _iv1, _iv2, _ivp1, _ivp2,
                              model_r, model_z, model_phi)

        k1 = AB_params['k1']
        k2 = AB_params['k2']
        k3 = AB_params['k3']
        k4 = AB_params['k4']
        k5 = AB_params['k5']
        k6 = AB_params['k6']
        k7 = AB_params['k7']
        k8 = AB_params['k8']
        k9 = AB_params['k9']
        k10 = AB_params['k10']
        calc_b_fields_cart(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                           model_r, model_phi, model_z)

        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_hel_v18(z, r, phi, L, ns, ms, cns, cms, n_scale):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression.
    WARNING: N AND M ARE REINTERPRETED IN THIS CASE
    '''

    P = L/(2*np.pi)
    P_long = n_scale/(2*np.pi)

    iv1 = np.zeros((ns, ms, len(r)))
    ivp1 = np.zeros((ns, ms, len(r)))
    for n in range(ns):
        for m in range(ms):
                iv1[n][m] = special.iv(m, (n/P)*r)
                ivp1[n][m] = 0.5*(special.iv(-1+m, (n/P)*r) +
                                  special.iv(1+m, (n/P)*r))

    iv2 = np.zeros((cns, cms, len(r)))
    ivp2 = np.zeros((cns, cms, len(r)))
    for cn in range(cns):
        for cm in range(cms):
                iv2[cn][cm] = special.iv(cm, (cn/P_long)*r)
                ivp2[cn][cm] = 0.5*(special.iv(-1+cm, (cn/P_long)*r) +
                                    special.iv(1+cm, (cn/P_long)*r))

    @njit(parallel=True)
    def calc_b_fields(z, phi, r, P, m, n, A, B,
                      iv, ivp, model_r, model_z, model_phi):
        for i in prange(z.shape[0]):
            model_r[i] += (n/P) * \
                (ivp[i]*(A*np.cos(n*z[i]/P+m*phi[i]) +
                         B*np.sin(n*z[i]/P+m*phi[i])))

            model_z[i] += (n/P) * \
                (iv[i]*(-A*np.sin(n*z[i]/P+m*phi[i]) +
                        B*np.cos(n*z[i]/P+m*phi[i])))

            model_phi[i] += (1.0/r[i]) * \
                (-m*iv[i]*(A*np.sin(n*z[i]/P+m*phi[i]) -
                           B*np.cos(n*z[i]/P+m*phi[i])))

    @njit(parallel=True)
    def calc_b_fields_cart(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                           model_r, model_phi, model_z):
        for i in prange(z.shape[0]):
            model_x = k1 + k4*x[i] + k7*y[i] + k8*z[i] + k10*y[i]*z[i]

            model_y = k2 + k5*y[i] + k7*x[i] + k9*z[i] + k10*x[i]*z[i]

            model_z[i] += k3 + k6*z[i] + k8*x[i] + k9*y[i] + k10*x[i]*y[i]

            model_r[i] += model_x*np.cos(phi[i]) + model_y*np.sin(phi[i])
            model_phi[i] += -model_x*np.sin(phi[i]) + model_y*np.cos(phi[i])

    def brzphi_3d_fast(z, r, phi, x, y, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)

        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        CDs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('C' in k or 'D' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*(ms)*2])):
                A = AB_params[ab[0]]
                B = AB_params[ab[1]]
                _iv1 = iv1[n][m]
                _ivp1 = ivp1[n][m]
                calc_b_fields(z, phi, r, P, m, n, A, B, _iv1, _ivp1,
                              model_r, model_z, model_phi)

        for cn in range(cns):
            for cm, cd in enumerate(pairwise(CDs[cn*cms*2:(cn+1)*(cms)*2])):
                C = AB_params[cd[0]]
                D = AB_params[cd[1]]
                _iv2 = iv2[cn][cm]
                _ivp2 = ivp2[cn][cm]
                calc_b_fields(z, phi, r, P_long, cm, cn, C, D, _iv2, _ivp2,
                              model_r, model_z, model_phi)

        k1 = AB_params['k1']
        k2 = AB_params['k2']
        k3 = AB_params['k3']
        k4 = AB_params['k4']
        k5 = AB_params['k5']
        k6 = AB_params['k6']
        k7 = AB_params['k7']
        k8 = AB_params['k8']
        k9 = AB_params['k9']
        k10 = AB_params['k10']
        calc_b_fields_cart(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                           model_r, model_phi, model_z)

        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_hel_v19(z, r, phi, L, ns, ms, cns, cms, n_scale):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression.
    WARNING: N AND M ARE REINTERPRETED IN THIS CASE
    '''

    P = L/(2*np.pi)
    P_long = n_scale/(2*np.pi)

    iv1 = np.zeros((ns, ms, len(r)))
    ivp1 = np.zeros((ns, ms, len(r)))
    for n in range(ns):
        for m in range(ms):
                iv1[n][m] = special.iv(m, (n/P)*r)
                ivp1[n][m] = 0.5*(special.iv(-1+m, (n/P)*r) +
                                  special.iv(1+m, (n/P)*r))

    iv2 = np.zeros((cns, cms, len(r)))
    ivp2 = np.zeros((cns, cms, len(r)))
    for cn in range(cns):
        for cm in range(cms):
                iv2[cn][cm] = special.iv(cm, (cn/P_long)*r)
                ivp2[cn][cm] = 0.5*(special.iv(-1+cm, (cn/P_long)*r) +
                                    special.iv(1+cm, (cn/P_long)*r))

    @njit(parallel=True)
    def calc_b_fields(z, phi, r, P, m, n, A, B,
                      iv, ivp, model_r, model_z, model_phi):
        for i in prange(z.shape[0]):
            model_r[i] += (n/P) * \
                (ivp[i]*(A*np.cos(n*z[i]/P+m*phi[i]) +
                         B*np.sin(n*z[i]/P+m*phi[i])))

            model_z[i] += (n/P) * \
                (iv[i]*(-A*np.sin(n*z[i]/P+m*phi[i]) +
                        B*np.cos(n*z[i]/P+m*phi[i])))

            model_phi[i] += (1.0/r[i]) * \
                (-m*iv[i]*(A*np.sin(n*z[i]/P+m*phi[i]) -
                           B*np.cos(n*z[i]/P+m*phi[i])))

    @njit(parallel=True)
    def calc_b_fields_cart(x, y, z, phi, vx1, vy1, vz1, x1, y1, z1, vx2, vy2, vz2, x2, y2, z2,
                           model_r, model_phi, model_z):
        for i in prange(z.shape[0]):
            v1 = np.array([vx1, vy1, vz1])
            r1 = np.array([x[i]-x1, y[i]-y1, z[i]-z1])
            rcube1 = np.linalg.norm(r1)**2
            # res1 = np.cross(v1, r1)/rcube1
            res1 = np.array([v1[1]*r1[2]-v1[2]*r1[1], v1[2]*r1[0]-v1[0]*r1[2],
                             v1[0]*r1[1]-v1[1]*r1[0]])/rcube1
            model_x1, model_y1, model_z1 = res1

            v2 = np.array([vx2, vy2, vz2])
            r2 = np.array([x[i]-x2, y[i]-y2, z[i]-z2])
            rcube2 = np.linalg.norm(r2)**2
            # res2 = np.cross(v2, r2)/rcube2
            res2 = np.array([v2[1]*r2[2]-v2[2]*r2[1], v2[2]*r2[0]-v2[0]*r2[2],
                             v2[0]*r2[1]-v2[1]*r2[0]])/rcube2
            model_x2, model_y2, model_z2 = res2

            model_z[i] += model_z1 + model_z2

            model_r[i] += model_x1*np.cos(phi[i]) + model_y1*np.sin(phi[i])
            model_r[i] += model_x2*np.cos(phi[i]) + model_y2*np.sin(phi[i])
            model_phi[i] += -model_x1*np.sin(phi[i]) + model_y1*np.cos(phi[i])
            model_phi[i] += -model_x2*np.sin(phi[i]) + model_y2*np.cos(phi[i])

    @njit(parallel=True)
    def calc_b_fields_cart2(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                            model_r, model_phi, model_z):
        for i in prange(z.shape[0]):
            model_x = k1 + k4*x[i] + k7*y[i] + k8*z[i] + k10*y[i]*z[i]

            model_y = k2 + k5*y[i] + k7*x[i] + k9*z[i] + k10*x[i]*z[i]

            model_z[i] += k3 + k6*z[i] + k8*x[i] + k9*y[i] + k10*x[i]*y[i]

            model_r[i] += model_x*np.cos(phi[i]) + model_y*np.sin(phi[i])
            model_phi[i] += -model_x*np.sin(phi[i]) + model_y*np.cos(phi[i])

    def brzphi_3d_fast(z, r, phi, x, y, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)

        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        CDs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('C' in k or 'D' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*(ms)*2])):
                A = AB_params[ab[0]]
                B = AB_params[ab[1]]
                _iv1 = iv1[n][m]
                _ivp1 = ivp1[n][m]
                calc_b_fields(z, phi, r, P, m, n, A, B, _iv1, _ivp1,
                              model_r, model_z, model_phi)

        for cn in range(cns):
            for cm, cd in enumerate(pairwise(CDs[cn*cms*2:(cn+1)*(cms)*2])):
                C = AB_params[cd[0]]
                D = AB_params[cd[1]]
                _iv2 = iv2[cn][cm]
                _ivp2 = ivp2[cn][cm]
                calc_b_fields(z, phi, r, P_long, cm, cn, C, D, _iv2, _ivp2,
                              model_r, model_z, model_phi)

        vx1 = AB_params['vx1']
        vy1 = AB_params['vy1']
        vz1 = AB_params['vz1']
        x1 = AB_params['x1']
        y1 = AB_params['y1']
        z1 = AB_params['z1']
        vx2 = AB_params['vx2']
        vy2 = AB_params['vy2']
        vz2 = AB_params['vz2']
        x2 = AB_params['x2']
        y2 = AB_params['y2']
        z2 = AB_params['z2']
        calc_b_fields_cart(x, y, z, phi, vx1, vy1, vz1, x1, y1, z1, vx2, vy2, vz2, x2, y2, z2,
                           model_r, model_phi, model_z)

        k1 = AB_params['k1']
        k2 = AB_params['k2']
        k3 = AB_params['k3']
        k4 = AB_params['k4']
        k5 = AB_params['k5']
        k6 = AB_params['k6']
        k7 = AB_params['k7']
        k8 = AB_params['k8']
        k9 = AB_params['k9']
        k10 = AB_params['k10']
        calc_b_fields_cart2(x, y, z, phi, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                            model_r, model_phi, model_z)

        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast

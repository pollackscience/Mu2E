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
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
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
        Cs = sorted({k: v for (k, v) in AB_params.iteritems() if 'C' in k})

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
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                             x.split('_')[0])))
        Cs = sorted({k: v for (k, v) in AB_params.iteritems() if 'C' in k})

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
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                             x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in AB_params.iteritems() if ('C' in k or 'D' in k)},
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


def brzphi_3d_producer_modbessel(z, r, phi, L, ns, ms):
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
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms[n][m]*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (m), (m), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, n, A, B, C, D, ivp, iv, kms, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (C[0]*np.cos(n[0]*phi[i])+D[0]*np.sin(n[0]*phi[i]))*ivp[i]*kms[0] * \
                (A[0]*np.cos(kms[0]*z[i]) + B[0]*np.sin(kms[0]*z[i]))

            model_z[i] += (C[0]*np.cos(n[0]*phi[i])+D[0]*np.sin(n[0]*phi[i]))*iv[i]*kms[0] * \
                (-A[0]*np.sin(kms[0]*z[i]) + B[0]*np.cos(kms[0]*z[i]))

            model_phi[i] += n[0]*(-C[0]*np.sin(n[0]*phi[i])+D[0]*np.cos(n[0]*phi[i])) * \
                (1/np.abs(r[i]))*iv[i]*(A[0]*np.cos(kms[0]*z[i]) + B[0]*np.sin(kms[0]*z[i]))

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        R = R
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                             x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in AB_params.iteritems() if ('C' in k or 'D' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        for n, cd in enumerate(pairwise(CDs)):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[cd[0]]], dtype=np.float64)
                D = np.array([AB_params[cd[1]]], dtype=np.float64)
                _ivp = ivp[n][i]
                _iv = iv[n][i]
                _kms = np.array([kms[n][i]])
                _n = np.array([n])
                calc_b_fields(z, phi, r, _n, A, B, C, D, _ivp, _iv, _kms, model_r, model_z,
                              model_phi)

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
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms[n][m]*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:])"],
                 '(m), (m), (m), (), (), (), (), (m), (m), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, n, A, B, D, ivp, iv, kms, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += np.cos(n[0]*phi[i]-D[0])*ivp[i]*kms[0] * \
                (A[0]*np.cos(kms[0]*z[i]) + B[0]*np.sin(kms[0]*z[i]))
            model_z[i] += np.cos(n[0]*phi[i]-D[0])*iv[i]*kms[0] * \
                (-A[0]*np.sin(kms[0]*z[i]) + B[0]*np.cos(kms[0]*z[i]))
            model_phi[i] += n[0]*(-np.sin(n[0]*phi[i]-D[0])) * \
                (1/np.abs(r[i]))*iv[i]*(A[0]*np.cos(kms[0]*z[i]) + B[0]*np.sin(kms[0]*z[i]))

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
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
                _ivp = ivp[n][i]
                _iv = iv[n][i]
                _kms = np.array([kms[n][i]])
                _n = np.array([n])
                calc_b_fields(z, phi, r, _n, A, B, D, _ivp, _iv, _kms, model_r, model_z, model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_phase_ext(z, r, phi, L, ns, ms, cns, cms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''
    a = 10000
    b = 10000
    c = 10000

    alpha_2d = np.zeros((cns, cms))
    beta_2d = np.zeros((cns, cms))
    gamma_2d = np.zeros((cns, cms))

    for cn in range(1, cns+1):
        for cm in range(1, cms+1):
            alpha_2d[cn-1][cm-1] = (cn*np.pi-np.pi/2)/a
            beta_2d[cn-1][cm-1] = (cm*np.pi-np.pi/2)/b
            gamma_2d[cn-1][cm-1] = np.sqrt(alpha_2d[cn-1][cm-1]**2+beta_2d[cn-1][cm-1]**2)

    kms = []
    for n in range(ns):
        kms.append([])
        for m in range(ms):
            kms[-1].append((m+1)*np.pi/L)
    kms = np.asarray(kms)
    iv = np.empty((ns, ms, r.shape[0], r.shape[1]))
    ivp = np.empty((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n, kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n, kms[n][m]*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:])"],
                 '(m), (m), (m), (), (), (), (), (m), (m), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cyl(z, phi, r, n, A, B, D, ivp, iv, kms, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += np.cos(n[0]*phi[i]-D[0])*ivp[i]*kms[0] * \
                (A[0]*np.cos(kms[0]*z[i]) + B[0]*np.sin(kms[0]*z[i]))
            model_z[i] += np.cos(n[0]*phi[i]-D[0])*iv[i]*kms[0] * \
                (-A[0]*np.sin(kms[0]*z[i]) + B[0]*np.cos(kms[0]*z[i]))
            model_phi[i] += n[0]*(-np.sin(n[0]*phi[i]-D[0])) * \
                (1/np.abs(r[i]))*iv[i]*(A[0]*np.cos(kms[0]*z[i]) + B[0]*np.sin(kms[0]*z[i]))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:])"],
                 '(m), (m), (m), (m), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(z, x, y, phi, c, C, alpha, beta, gamma, e1, e2, model_r, model_z,
                           model_phi):
        for i in range(z.shape[0]):
            model_x = C[0]*alpha[0]*np.sin(alpha[0]*x[i]+e1[0])*np.cos(beta[0]*y[i]+e2[0]) * \
                np.sinh(gamma[0]*(z[i]-c[0]))
            model_y = C[0]*beta[0]*np.cos(alpha[0]*x[i]+e1[0])*np.sin(beta[0]*y[i]+e2[0]) * \
                np.sinh(gamma[0]*(z[i]-c[0]))

            model_z[i] += (-C[0])*gamma[0]*np.cos(e1[0] + alpha[0]*x[i]) * \
                np.cos(e2[0] + beta[0]*y[i])*np.cosh(gamma[0]*(-c[0] + z[i]))

            model_r[i] += model_x*np.cos(phi[i]) + model_y*np.sin(phi[i])
            model_phi[i] += -model_x*np.sin(phi[i]) + model_y*np.cos(phi[i])

    def brzphi_3d_fast(z, r, phi, x, y, R, ns, ms, e1, e2, **AB_params):
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
        Cs = sorted({k: v for (k, v) in AB_params.iteritems() if 'C' in k})

        for n, d in enumerate(Ds):
            for i, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                D = np.array([AB_params[d]], dtype=np.float64)
                _ivp = ivp[n][i]
                _iv = iv[n][i]
                _kms = np.array([kms[n][i]])
                _n = np.array([n])
                calc_b_fields_cyl(z, phi, r, _n, A, B, D, _ivp, _iv, _kms, model_r, model_z,
                                  model_phi)

        for cn in range(1, cns+1):
            for cm in range(1, cms+1):

                alpha = np.array(alpha_2d[cn-1][cm-1], dtype=np.float64)
                beta = np.array(beta_2d[cn-1][cm-1], dtype=np.float64)
                gamma = np.array(gamma_2d[cn-1][cm-1], dtype=np.float64)
                C = np.array([AB_params[Cs[cm-1+(cn-1)*cms]]], dtype=np.float64)
                _e1 = np.array([e1], dtype=np.float64)
                _e2 = np.array([e2], dtype=np.float64)
                _c = np.array([c], dtype=np.float64)
                calc_b_fields_cart(z, x, y, phi, _c, C, alpha, beta, gamma, _e1, _e2, model_r,
                                   model_z, model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_phase_hybrid(z, r, phi, L, ns, ms, cns, cms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''
    R = 15000

    b_zeros = []
    for cn in range(cns):
        b_zeros.append(special.jn_zeros(cn, cms))
    kms_j = np.asarray([b/R for b in b_zeros])
    jv = np.empty((cns, cms, r.shape[0], r.shape[1]))
    jvp = np.empty((cns, cms, r.shape[0], r.shape[1]))
    for cn in range(cns):
        for cm in range(cms):
            jv[cn][cm] = special.jv(cn, kms_j[cn][cm]*np.abs(r))
            jvp[cn][cm] = special.jvp(cn, kms_j[cn][cm]*np.abs(r))

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

    def brzphi_3d_fast(z, r, phi, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in AB_params.iteritems() if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        EFs = sorted({k: v for (k, v) in AB_params.iteritems() if ('E' in k or 'F' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Gs = sorted({k: v for (k, v) in AB_params.iteritems() if ('G' in k)}, key=lambda x:
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
                _kms = np.array([kms_j[n][i]])
                _n = np.array([cn])
                calc_b_fields_b(z, phi, r, _n, E, F, G, _jvp, _jv, _kms, model_r, model_z,
                                model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_phase_hybrid_disp2(z, r, phi, rp, phip, L, ns, ms, cns, cms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''
    R = 5000

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
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in AB_params.iteritems() if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        EFs = sorted({k: v for (k, v) in AB_params.iteritems() if ('E' in k or 'F' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Gs = sorted({k: v for (k, v) in AB_params.iteritems() if ('G' in k)}, key=lambda x:
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
                _kms = np.array([kms_j[n][i]])
                _n = np.array([cn])
                calc_b_fields_mb(z, phip, rp, _n, E, F, G, _ivp, _iv, _kms, model_r, model_z,
                                 model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_modbessel_phase_hybrid_disp(z, r, phi, rp, phip, L, ns, ms, cns, cms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
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
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in AB_params.iteritems() if ('D' in k)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[0])))

        EFs = sorted({k: v for (k, v) in AB_params.iteritems() if ('E' in k or 'F' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Gs = sorted({k: v for (k, v) in AB_params.iteritems() if ('G' in k)}, key=lambda x:
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
                _kms = np.array([kms_j[n][i]])
                _n = np.array([cn])
                calc_b_fields_b(z, phip, rp, _n, E, F, G, _jvp, _jv, _kms, model_r, model_z,
                                model_phi)

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
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
                     key=lambda x: ', '.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                              x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in AB_params.iteritems() if ('C' in k or 'D' in k)},
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
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
                     key=lambda x: ', '.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                              x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in AB_params.iteritems() if ('C' in k or 'D' in k)},
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
        ABs = sorted({k: v for (k, v) in AB_params.iteritems() if ('A' in k or 'B' in k)},
                     key=lambda x: ', '.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                              x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in AB_params.iteritems() if ('C' in k or 'D' in k)},
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

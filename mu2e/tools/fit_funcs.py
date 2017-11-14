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
from numba import guvectorize
from math import cos, sin
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
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                             x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('C' in k or 'D' in k)},
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
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))
        Ds = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('D' in k)}, key=lambda x:
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
    alpha = np.zeros((cns, cms))
    beta = np.zeros((cns, cms))
    gamma = np.zeros((cns, cms))

    for cn in range(1, cns+1):
        for cm in range(1, cms+1):
            alpha[cn-1][cm-1] = cn/4000
            beta[cn-1][cm-1] = cm/4000
            gamma[cn-1][cm-1] = np.sqrt(beta[cn-1][cm-1]**2+alpha[cn-1][cm-1]**2)

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

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (m), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, phi, E, F, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11,
                           alpha, beta, gamma, model_r, model_phi, model_z):
        for i in range(z.shape[0]):
            model_x = -alpha[0]*np.exp(-alpha[0]*x[i]) * \
                np.exp(-beta[0]*y[i]) * \
                (E[0]*np.sin(gamma[0]*z[i]) + F[0]*np.cos(gamma[0]*z[i]))
                # k11[0]*(x[i]+1175)/((x[i]+1175)**2 + (y[i]-200)**2)
                # k1[0] + k4[0]*x[i] + k7[0]*y[i] + k8[0]*z[i] + k10[0]*y[i]*z[i]

            model_y = np.exp(-alpha[0]*x[i]) * \
                -beta[0]*np.exp(-beta[0]*y[i]) * \
                (E[0]*np.sin(gamma[0]*z[i]) + F[0]*np.cos(gamma[0]*z[i]))
                #k11[0]*(y[i]-200)/((x[i]+1175)**2 + (y[i]-200)**2)
                # k2[0] + k5[0]*y[i] + k7[0]*x[i] + k9[0]*z[i] + k10[0]*x[i]*z[i]

            model_z[i] += np.exp(-alpha[0]*x[i]) * \
                np.exp(-beta[0]*y[i]) * \
                gamma[0]*(E[0]*np.cos(gamma[0]*z[i]) - F[0]*np.sin(gamma[0]*z[i]))
                # k3[0] + k6[0]*z[i] + k8[0]*x[i] + k9[0]*y[i] + k10[0]*x[i]*y[i]

            # model_x = alpha[0]*(E[0]*np.exp(alpha[0]*x[i]) - F[0]*np.exp(-alpha[0]*x[i])) * \
            #     np.exp(beta[0]*y[i]) * \
            #     np.sin(gamma[0]*z[i]) + \
            #     k1[0]*(x[i]+1175)/((x[i]+1175)**2 + (y[i]-200)**2)

            # model_y = (E[0]*np.exp(alpha[0]*x[i]) + F[0]*np.exp(-alpha[0]*x[i])) * \
            #     beta[0]*np.exp(beta[0]*y[i]) * \
            #     np.sin(gamma[0]*z[i]) + \
            #     k1[0]*(y[i]-200)/((x[i]+1175)**2 + (y[i]-200)**2)

            # model_z[i] += (E[0]*np.exp(alpha[0]*x[i]) + F[0]*np.exp(-alpha[0]*x[i])) * \
            #     np.exp(beta[0]*y[i]) * \
            #     -gamma[0]*np.cos(gamma[0]*z[i])

            # model_x = alpha[0]*np.cos(alpha[0]*x[i]) * \
            #     np.sin(beta[0]*y[i]) * \
            #     (E[0]*np.exp(gamma[0]*z[i]) + F[0]*np.exp(-gamma[0]*z[i])) + \
            #     k1[0]*(x[i]+1175)/((x[i]+1175)**2 + (y[i]-200)**2)

            # model_y = np.sin(alpha[0]*x[i]) * \
            #     beta[0]*np.cos(beta[0]*y[i]) * \
            #     (E[0]*np.exp(gamma[0]*z[i]) + F[0]*np.exp(-gamma[0]*z[i])) + \
            #     k1[0]*(y[i]-200)/((x[i]+1175)**2 + (y[i]-200)**2)

            # model_z[i] += np.sin(alpha[0]*x[i]) * \
            #     np.sin(beta[0]*y[i]) * \
            #     gamma[0]*(E[0]*np.exp(gamma[0]*z[i]) - F[0]*np.exp(-gamma[0]*z[i]))

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
        EFs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('E' in k or 'F' in k)})

        for n, d in enumerate(Ds):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                D = np.array([AB_params[d]], dtype=np.float64)
                _ivp = ivp[n][m]
                _iv = iv[n][m]
                _kms = np.array([kms[n][m]])
                _n = np.array([n])
                calc_b_fields_cyl(z, phi, r, _n, A, B, D, _ivp, _iv, _kms, model_r, model_z,
                                  model_phi)

        for cn in range(cns):
            for cm, ef in enumerate(pairwise(EFs[cn*cms*2:(cn+1)*cms*2])):

                _alpha = np.array(alpha[cn][cm], dtype=np.float64)
                _beta = np.array(beta[cn][cm], dtype=np.float64)
                _gamma = np.array(gamma[cn][cm], dtype=np.float64)
                E = np.array([AB_params[ef[0]]], dtype=np.float64)
                F = np.array([AB_params[ef[1]]], dtype=np.float64)
                k1 = np.array(AB_params['k1'], dtype=np.float64)
                k2 = np.array(AB_params['k2'], dtype=np.float64)
                k3 = np.array(AB_params['k3'], dtype=np.float64)
                k4 = np.array(AB_params['k4'], dtype=np.float64)
                k5 = np.array(AB_params['k5'], dtype=np.float64)
                k6 = np.array(AB_params['k6'], dtype=np.float64)
                k7 = np.array(AB_params['k7'], dtype=np.float64)
                k8 = np.array(AB_params['k8'], dtype=np.float64)
                k9 = np.array(AB_params['k9'], dtype=np.float64)
                k10 = np.array(AB_params['k10'], dtype=np.float64)
                k11 = np.array(AB_params['k11'], dtype=np.float64)
                calc_b_fields_cart(x, y, z, phi, E, F, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11,
                                   _alpha, _beta, _gamma, model_r, model_phi, model_z)

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
                _kms = np.array([kms_j[n][i]])
                _n = np.array([cn])
                calc_b_fields_b(z, phi, r, _n, E, F, G, _jvp, _jv, _kms, model_r, model_z,
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


#######################################
#######################################
#        CARTESIAN FUNCTIONS          #
#######################################
#######################################


def bxyz_3d_producer_cart(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            beta[n-1][m-1] = n/L
            alpha[n-1][m-1] = m/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += -alpha[0]*np.exp(-alpha[0]*x[i]) * \
                np.exp(-beta[0]*y[i]) * \
                (A[0]*np.cos(gamma[0]*z[i]) + B[0]*np.sin(gamma[0]*z[i]))

            model_y[i] += np.exp(-alpha[0]*x[i]) * \
                -beta[0]*np.exp(-beta[0]*y[i]) * \
                (A[0]*np.cos(gamma[0]*z[i]) + B[0]*np.sin(gamma[0]*z[i]))

            model_z[i] += np.exp(-alpha[0]*x[i]) * \
                np.exp(-beta[0]*y[i]) * \
                gamma[0]*(-A[0]*np.sin(gamma[0]*z[i]) + B[0]*np.cos(gamma[0]*z[i]))

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                # _alpha = np.array(alpha[n][m], dtype=np.float64)
                # _beta = np.array(beta[n][m], dtype=np.float64)
                # _gamma = np.array(gamma[n][m], dtype=np.float64)
                # k1 = np.array(AB_params['k1'], dtype=np.float64)
                # k2 = np.array(AB_params['k2'], dtype=np.float64)
                _alpha = np.array(m/AB_params['k1'], dtype=np.float64)
                _beta = np.array(n/AB_params['k2'], dtype=np.float64)
                _gamma = np.array(np.sqrt(_alpha**2+_beta**2), dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, _alpha, _beta, _gamma, model_x, model_y,
                                   model_z)

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v2(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            # gamma[n-1][m-1] = (m*np.pi+ns+2)/L
            # beta[n-1][m-1] = n/L
            # alpha[n-1][m-1] = np.sqrt(gamma[n-1][m-1]**2-beta[n-1][m-1]**2)

            alpha[n-1][m-1] = m/L
            beta[n-1][m-1] = n/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*(A[0]*np.exp(alpha[0]*x[i]) - B[0]*np.exp(-alpha[0]*x[i])) * \
                np.exp(beta[0]*y[i])*np.cos(gamma[0]*z[i])

            model_y[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                beta[0]*np.exp(beta[0]*y[i])*np.cos(gamma[0]*z[i])

            model_z[i] += -(A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                np.exp(beta[0]*y[i])*gamma[0]*np.sin(gamma[0]*z[i])

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, _alpha, _beta, _gamma, model_x, model_y, model_z)
        # print 'model_x'
        # print model_x
        # print 'model_y'
        # print model_y
        # print 'model_z'
        # print model_z

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v3(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            # gamma[n-1][m-1] = (m*np.pi)/L
            # beta[n-1][m-1] = n/10000
            # alpha[n-1][m-1] = np.sqrt(gamma[n-1][m-1]**2+beta[n-1][m-1]**2)

            beta[n-1][m-1] = n/L
            alpha[n-1][m-1] = m/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, C, D, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*(A[0]*np.exp(alpha[0]*x[i]) - B[0]*np.exp(-alpha[0]*x[i])) * \
                (C[0]*np.exp(beta[0]*y[i]) +
                 D[0]*np.exp(-beta[0]*y[i]))*np.sin(gamma[0]*z[i])

            model_y[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                beta[0]*(C[0]*np.exp(beta[0]*y[i]) -
                         D[0]*np.exp(-beta[0]*y[i]))*np.sin(gamma[0]*z[i])

            model_z[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                (C[0]*np.exp(beta[0]*y[i]) +
                 D[0]*np.exp(-beta[0]*y[i]))*gamma[0]*np.cos(gamma[0]*z[i])

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(quadwise(ABs[n*ms*4:(n+1)*ms*4])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[ab[2]]], dtype=np.float64)
                D = np.array([AB_params[ab[3]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, C, D, _alpha, _beta, _gamma, model_x, model_y,
                                   model_z)
        # print 'model_x'
        # print model_x
        # print 'model_y'
        # print model_y
        # print 'model_z'
        # print model_z

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v4(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            gamma[n-1][m-1] = (m*np.pi)/L
            beta[n-1][m-1] = n/L
            alpha[n-1][m-1] = np.sqrt(gamma[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, C, D, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*(A[0]*np.cosh(alpha[0]*x[i]) + B[0]*np.sinh(alpha[0]*x[i])) * \
                (C[0]*np.sinh(beta[0]*y[i]) +
                 D[0]*np.cosh(beta[0]*y[i]))*np.sin(gamma[0]*z[i])

            model_y[i] += (A[0]*np.sinh(alpha[0]*x[i]) + B[0]*np.cosh(alpha[0]*x[i])) * \
                beta[0]*(C[0]*np.cosh(beta[0]*y[i]) +
                         D[0]*np.sinh(beta[0]*y[i]))*np.sin(gamma[0]*z[i])

            model_z[i] += (A[0]*np.sinh(alpha[0]*x[i]) + B[0]*np.cosh(alpha[0]*x[i])) * \
                (C[0]*np.sinh(beta[0]*y[i]) +
                 D[0]*np.cosh(beta[0]*y[i]))*gamma[0]*np.cos(gamma[0]*z[i])

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(quadwise(ABs[n*ms*4:(n+1)*ms*4])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[ab[2]]], dtype=np.float64)
                D = np.array([AB_params[ab[3]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, C, D, _alpha, _beta, _gamma, model_x, model_y,
                                   model_z)
        # print 'model_x'
        # print model_x
        # print 'model_y'
        # print model_y
        # print 'model_z'
        # print model_z

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v5(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            beta[n-1][m-1] = n/L
            alpha[n-1][m-1] = m/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, C, D, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*(A[0]*np.cosh(alpha[0]*x[i]) + B[0]*np.sinh(alpha[0]*x[i])) * \
                np.exp(beta[0]*y[i])*(C[0]*np.sin(gamma[0]*y[i])+D[0]*np.cos(gamma[0]*z[i]))

            model_y[i] += (A[0]*np.sinh(alpha[0]*x[i]) + B[0]*np.cosh(alpha[0]*x[i])) * \
                beta[0]*np.exp(beta[0]*y[i])*(C[0]*np.sin(gamma[0]*y[i])+D[0]*np.cos(gamma[0]*z[i]))

            model_z[i] += (A[0]*np.sinh(alpha[0]*x[i]) + B[0]*np.cosh(alpha[0]*x[i])) * \
                np.exp(beta[0]*y[i])*gamma[0]*(C[0]*np.cos(gamma[0]*y[i]) -
                                               D[0]*np.sin(gamma[0]*z[i]))

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5),
                                             x.split('_')[2].zfill(5),
                                             x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(quadwise(ABs[n*ms*4:(n+1)*ms*4])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[ab[2]]], dtype=np.float64)
                D = np.array([AB_params[ab[3]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, C, D, _alpha, _beta, _gamma,
                                   model_x, model_y, model_z)

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v6(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            alpha[n-1][m-1] = n/L
            beta[n-1][m-1] = m/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, C, D, E, F, alpha, beta, gamma, model_x, model_y,
                           model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*(A[0]*np.exp(alpha[0]*x[i]) - B[0]*np.exp(-alpha[0]*x[i])) * \
                (C[0]*np.exp(beta[0]*y[i]) + D[0]*np.exp(-beta[0]*y[i])) * \
                (E[0]*np.sin(gamma[0]*z[i]) + F[0]*np.cos(gamma[0]*z[i]))

            model_y[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                beta[0]*(C[0]*np.exp(beta[0]*y[i]) - D[0]*np.exp(-beta[0]*y[i])) * \
                (E[0]*np.sin(gamma[0]*z[i]) + F[0]*np.cos(gamma[0]*z[i]))

            model_z[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                (C[0]*np.exp(beta[0]*y[i]) + D[0]*np.exp(-beta[0]*y[i])) * \
                gamma[0]*(E[0]*np.cos(gamma[0]*z[i]) - F[0]*np.sin(gamma[0]*z[i]))

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(hexwise(ABs[n*ms*6:(n+1)*ms*6])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[ab[2]]], dtype=np.float64)
                D = np.array([AB_params[ab[3]]], dtype=np.float64)
                E = np.array([AB_params[ab[4]]], dtype=np.float64)
                F = np.array([AB_params[ab[5]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, C, D, E, F, _alpha, _beta, _gamma, model_x,
                                   model_y, model_z)

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v7(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            # gamma[n-1][m-1] = (m*np.pi+ns+2)/L
            # beta[n-1][m-1] = n/L
            # alpha[n-1][m-1] = np.sqrt(gamma[n-1][m-1]**2-beta[n-1][m-1]**2)

            beta[n-1][m-1] = n/L
            gamma[n-1][m-1] = m/L
            alpha[n-1][m-1] = np.sqrt(gamma[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, C, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += -alpha[0]*A[0]*np.exp(-alpha[0]*x[i]) * \
                np.sin(beta[0]*y[i] + B[0]) * \
                np.sin(gamma[0]*z[i] + C[0])

            model_y[i] += A[0]*np.exp(-alpha[0]*x[i]) * \
                beta[0]*np.cos(beta[0]*y[i] + B[0]) * \
                np.sin(gamma[0]*z[i] + C[0])

            model_z[i] += A[0]*np.exp(-alpha[0]*x[i]) * \
                np.sin(beta[0]*y[i] + B[0]) * \
                gamma[0]*np.cos(gamma[0]*z[i] + C[0])

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)}, key=lambda x:
                     ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                               x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(tripwise(ABs[n*ms*3:(n+1)*ms*3])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[ab[2]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, C, _alpha, _beta, _gamma, model_x, model_y,
                                   model_z)
        # print 'model_x'
        # print model_x
        # print 'model_y'
        # print model_y
        # print 'model_z'
        # print model_z

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v8(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            # gamma[n-1][m-1] = (m*np.pi+ns+1)/L
            # beta[n-1][m-1] = n/L
            # alpha[n-1][m-1] = np.sqrt(gamma[n-1][m-1]**2-beta[n-1][m-1]**2)

            beta[n-1][m-1] = n/L
            alpha[n-1][m-1] = m/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*np.exp(alpha[0]*x[i])*np.exp(beta[0]*y[i]) * \
                (A[0]*np.sin(gamma[0]*z[i]))

            model_y[i] += np.exp(alpha[0]*x[i])*beta[0]*np.exp(beta[0]*y[i]) * \
                (A[0]*np.sin(gamma[0]*z[i]))

            model_z[i] += np.exp(alpha[0]*x[i])*np.exp(beta[0]*y[i]) * \
                gamma[0]*(A[0]*np.cos(gamma[0]*z[i]))

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        As = sorted({k: v for (k, v) in six.iteritems(AB_params)}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                              x.split('_')[0])))

        for n in range(ns):
            for m, a in enumerate(As[n*ms:(n+1)*ms]):

                A = np.array([AB_params[a]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, _alpha, _beta, _gamma, model_x,
                                   model_y, model_z)
        # print 'model_x'
        # print model_x
        # print 'model_y'
        # print model_y
        # print 'model_z'
        # print model_z

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v9(x, y, z, L, ns, ms, cns, cms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha1 = np.zeros((ns, ms))
    beta1 = np.zeros((ns, ms))
    gamma1 = np.zeros((ns, ms))

    alpha2 = np.zeros((cns, cms))
    beta2 = np.zeros((cns, cms))
    gamma2 = np.zeros((cns, cms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            alpha1[n-1][m-1] = m/L
            beta1[n-1][m-1] = n/L
            gamma1[n-1][m-1] = np.sqrt(alpha1[n-1][m-1]**2+beta1[n-1][m-1]**2)

    for cn in range(1, cns+1):
        for cm in range(1, cms+1):
            alpha2[cn-1][cm-1] = cm/5000
            beta2[cn-1][cm-1] = cn/5000
            gamma2[cn-1][cm-1] = np.sqrt(alpha2[cn-1][cm-1]**2+beta2[cn-1][cm-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart1(x, y, z, A, B, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*np.exp(alpha[0]*x[i])*np.exp(beta[0]*y[i]) * \
                (A[0]*np.sin(gamma[0]*z[i]) + B[0]*np.cos(gamma[0]*z[i]))

            model_y[i] += np.exp(alpha[0]*x[i])*beta[0]*np.exp(beta[0]*y[i]) * \
                (A[0]*np.sin(gamma[0]*z[i]) + B[0]*np.cos(gamma[0]*z[i]))

            model_z[i] += np.exp(alpha[0]*x[i])*np.exp(beta[0]*y[i]) * \
                gamma[0]*(A[0]*np.cos(gamma[0]*z[i]) - B[0]*np.sin(gamma[0]*z[i]))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart2(x, y, z, C, D, E, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += C[0]*alpha[0]*np.cos(alpha[0]*x[i] + D[0]) * \
                np.sin(beta[0]*y[i] + E[0]) * \
                np.exp(-gamma[0]*z[i])

            model_y[i] += C[0]*np.sin(alpha[0]*x[i] + D[0]) * \
                beta[0]*np.cos(beta[0]*y[i] + E[0]) * \
                np.exp(-gamma[0]*z[i])

            model_z[i] += C[0]*np.sin(alpha[0]*x[i] + D[0]) * \
                np.sin(beta[0]*y[i] + E[0]) * \
                -gamma[0]*np.exp(-gamma[0]*z[i])

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                             x.split('_')[0])))
        CDs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('C' in k or 'D' in k or 'E' in
                                                                   k)}, key=lambda x:
                     ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                               x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                _alpha1 = np.array(alpha1[n][m], dtype=np.float64)
                _beta1 = np.array(beta1[n][m], dtype=np.float64)
                _gamma1 = np.array(gamma1[n][m], dtype=np.float64)

                calc_b_fields_cart1(x, y, z, A, B, _alpha1, _beta1, _gamma1, model_x,
                                    model_y, model_z)

        for cn in range(cns):
            for cm, cd in enumerate(tripwise(CDs[cn*cms*3:(cn+1)*cms*3])):

                C = np.array([AB_params[cd[0]]], dtype=np.float64)
                D = np.array([AB_params[cd[1]]], dtype=np.float64)
                E = np.array([AB_params[cd[2]]], dtype=np.float64)
                _alpha2 = np.array(alpha2[cn][cm], dtype=np.float64)
                _beta2 = np.array(beta2[cn][cm], dtype=np.float64)
                _gamma2 = np.array(gamma2[cn][cm], dtype=np.float64)

                calc_b_fields_cart2(x, y, z, C, D, E, _alpha2, _beta2, _gamma2, model_x,
                                    model_y, model_z)

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v10(x, y, z, L, ns, ms, cns, cms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha1 = np.zeros((ns, ms))
    beta1 = np.zeros((ns, ms))
    gamma1 = np.zeros((ns, ms))

    alpha2 = np.zeros((cns, cms))
    beta2 = np.zeros((cns, cms))
    gamma2 = np.zeros((cns, cms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            alpha1[n-1][m-1] = m/L
            beta1[n-1][m-1] = n/L
            gamma1[n-1][m-1] = np.sqrt(alpha1[n-1][m-1]**2+beta1[n-1][m-1]**2)

    for cn in range(1, cns+1):
        for cm in range(1, cms+1):
            alpha2[cn-1][cm-1] = cm/L
            beta2[cn-1][cm-1] = cn/L
            gamma2[cn-1][cm-1] = np.sqrt(alpha2[cn-1][cm-1]**2+beta2[cn-1][cm-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart1(x, y, z, A, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*np.exp(alpha[0]*x[i])*np.exp(beta[0]*y[i]) * \
                (A[0]*np.sin(gamma[0]*z[i]))

            model_y[i] += np.exp(alpha[0]*x[i])*beta[0]*np.exp(beta[0]*y[i]) * \
                (A[0]*np.sin(gamma[0]*z[i]))

            model_z[i] += np.exp(alpha[0]*x[i])*np.exp(beta[0]*y[i]) * \
                gamma[0]*(A[0]*np.cos(gamma[0]*z[i]))

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        As = sorted({k: v for (k, v) in six.iteritems(AB_params) if 'A' in k}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                              x.split('_')[0])))
        Bs = sorted({k: v for (k, v) in six.iteritems(AB_params) if 'B' in k}, key=lambda x:
                    ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                              x.split('_')[0])))

        for n in range(ns):
            for m, a in enumerate(As[n*ms:(n+1)*ms]):

                A = np.array([AB_params[a]], dtype=np.float64)
                _alpha1 = np.array(alpha1[n][m], dtype=np.float64)
                _beta1 = np.array(beta1[n][m], dtype=np.float64)
                _gamma1 = np.array(gamma1[n][m], dtype=np.float64)

                calc_b_fields_cart1(x, y, z, A, _alpha1, _beta1, _gamma1, model_x,
                                    model_y, model_z)

        for cn in range(cns):
            for cm, b in enumerate(Bs[cn*cms:(cn+1)*cms]):

                B = np.array([AB_params[b]], dtype=np.float64)
                _alpha2 = np.array(alpha2[cn][cm], dtype=np.float64)
                _beta2 = np.array(beta2[cn][cm], dtype=np.float64)
                _gamma2 = np.array(gamma2[cn][cm], dtype=np.float64)

                calc_b_fields_cart1(x, y, z, B, _alpha2, _beta2, _gamma2, model_x,
                                    model_y, model_z)

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v11(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):

            alpha[n-1][m-1] = m/L
            beta[n-1][m-1] = n/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], "
                  "float64[:], float64[:], float64[:], float64[:], "
                  "float64[:], float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, C, D, alpha, beta, gamma, model_x, model_y,
                           model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*(A[0]*np.exp(alpha[0]*x[i]) - B[0]*np.exp(-alpha[0]*x[i])) * \
                (C[0] + np.exp(beta[0]*y[i]))  * \
                np.sin(gamma[0]*z[i]+D[0])

            model_y[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                beta[0]*(C[0] + np.exp(beta[0]*y[i]))  * \
                np.sin(gamma[0]*z[i]+D[0])

            model_z[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                (C[0] + np.exp(beta[0]*y[i]))  * \
                gamma[0]*np.cos(gamma[0]*z[i]+D[0])

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(quadwise(ABs[n*ms*4:(n+1)*ms*4])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[ab[2]]], dtype=np.float64)
                D = np.array([AB_params[ab[3]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, C, D, _alpha, _beta, _gamma, model_x,
                                   model_y, model_z)
        # print 'model_x'
        # print model_x
        # print 'model_y'
        # print model_y
        # print 'model_z'
        # print model_z

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v12(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            gamma[n-1][m-1] = n/L
            alpha[n-1][m-1] = m/L
            beta[n-1][m-1] = np.sqrt(gamma[n-1][m-1]**2+alpha[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, C, D, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += (A[0]*np.sinh(beta[0]*y[i]) + B[0]*np.cosh(beta[0]*y[i])) * \
                alpha[0]*np.cos(alpha[0]*x[i] + C[0])*np.sin(gamma[0]*z[i] + D[0])

            model_y[i] += beta[0]*(A[0]*np.cosh(beta[0]*y[i]) + B[0]*np.sinh(beta[0]*y[i])) * \
                np.sin(alpha[0]*x[i] + C[0])*np.sin(gamma[0]*z[i] + D[0])

            model_z[i] += (A[0]*np.sinh(beta[0]*y[i]) + B[0]*np.cosh(beta[0]*y[i])) * \
                np.sin(alpha[0]*x[i] + C[0])*gamma[0]*np.cos(gamma[0]*z[i] + D[0])

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5),
                                             x.split('_')[2].zfill(5),
                                             x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(quadwise(ABs[n*ms*4:(n+1)*ms*4])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[ab[2]]], dtype=np.float64)
                D = np.array([AB_params[ab[3]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, C, D, _alpha, _beta, _gamma,
                                   model_x, model_y, model_z)

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v13(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            beta[n-1][m-1] = n/L
            alpha[n-1][m-1] = m/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (), (), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                           alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += -alpha[0]*np.exp(-alpha[0]*x[i]) * \
                np.exp(-beta[0]*y[i]) * \
                (A[0]*np.cos(gamma[0]*z[i]) + B[0]*np.sin(gamma[0]*z[i])) + \
                k1[0] + k4[0]*x[i] + k7[0]*y[i] + k8[0]*z[i] + k10[0]*y[i]*z[i]

            model_y[i] += np.exp(-alpha[0]*x[i]) * \
                -beta[0]*np.exp(-beta[0]*y[i]) * \
                (A[0]*np.cos(gamma[0]*z[i]) + B[0]*np.sin(gamma[0]*z[i])) + \
                k2[0] + k5[0]*y[i] + k7[0]*x[i] + k9[0]*z[i] + k10[0]*x[i]*z[i]

            model_z[i] += np.exp(-alpha[0]*x[i]) * \
                np.exp(-beta[0]*y[i]) * \
                gamma[0]*(-A[0]*np.sin(gamma[0]*z[i]) + B[0]*np.cos(gamma[0]*z[i])) + \
                k3[0] + k6[0]*z[i] + k8[0]*x[i] + k9[0]*y[i] + k10[0]*x[i]*y[i]

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)
                k1 = np.array(AB_params['k1'], dtype=np.float64)
                k2 = np.array(AB_params['k2'], dtype=np.float64)
                k3 = np.array(AB_params['k3'], dtype=np.float64)
                k4 = np.array(AB_params['k4'], dtype=np.float64)
                k5 = np.array(AB_params['k5'], dtype=np.float64)
                k6 = np.array(AB_params['k6'], dtype=np.float64)
                k7 = np.array(AB_params['k7'], dtype=np.float64)
                k8 = np.array(AB_params['k8'], dtype=np.float64)
                k9 = np.array(AB_params['k9'], dtype=np.float64)
                k10 = np.array(AB_params['k10'], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
                                   _alpha, _beta, _gamma, model_x, model_y, model_z)

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v14(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            beta[n-1][m-1] = n/L
            alpha[n-1][m-1] = m/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    z_k1 = np.piecewise(z, [(z < 8996) | (z > 13821), (z >= 8996) & (z <= 13821)], [1, 1])
    z_k2 = np.piecewise(z, [(z < 3971) | (z > 8146), (z >= 3971) & (z <= 8146)], [1, 1])
    z_k3 = np.piecewise(z, [(z < 3971) | (z > 7921), (z >= 3971) & (z <= 7921)], [1, 1])

    x_k4 = np.piecewise(x, [(x < -500) | (x > 75), (x >= -500) & (x <= 75)], [0, 1])
    y_k5 = np.piecewise(y, [(y < -750) | (y > 225), (y >= -750) & (y <= 225)], [0, 1])



    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m),(m), (m), (m), (m), (m), (), (), (), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, z_k1, z_k2, z_k3, x_k4, y_k5, A, B, k1, k2, k3, k4, k5, alpha,
                           beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += -alpha[0]*np.exp(-alpha[0]*x[i]) * \
                np.exp(-beta[0]*y[i]) * \
                (A[0]*np.cos(gamma[0]*z[i]) + B[0]*np.sin(gamma[0]*z[i])) + \
                k1[0]*(x[i]+400)/((x[i]+400)**2 + (y[i]-1125)**2) * z_k1[i] + \
                k2[0]*(x[i]+525)/((x[i]+525)**2 + (y[i]-1100)**2) * z_k2[i] + \
                k3[0]*(x[i]+1175)/((x[i]+1175)**2 + (y[i]-225)**2) * z_k3[i] + \
                k5[0]*(x[i]+1025)/((x[i]+1025)**2 + (z[i]-5971)**2) * y_k5[i]

            model_y[i] += np.exp(-alpha[0]*x[i]) * \
                -beta[0]*np.exp(-beta[0]*y[i]) * \
                (A[0]*np.cos(gamma[0]*z[i]) + B[0]*np.sin(gamma[0]*z[i])) + \
                k1[0]*(y[i]-1125)/((x[i]+400)**2 + (y[i]-1125)**2) * z_k1[i] + \
                k2[0]*(y[i]-1100)/((x[i]+525)**2 + (y[i]-1100)**2) * z_k2[i] +\
                k3[0]*(y[i]-225)/((x[i]+1175)**2 + (y[i]-225)**2) * z_k3[i] + \
                k4[0]*(y[i]-1100)/((z[i]-5971)**2 + (y[i]-1100)**2) * x_k4[i]

            model_z[i] += np.exp(-alpha[0]*x[i]) * \
                np.exp(-beta[0]*y[i]) * \
                gamma[0]*(-A[0]*np.sin(gamma[0]*z[i]) + B[0]*np.cos(gamma[0]*z[i])) + \
                k4[0]*(z[i]-5971)/((z[i]-5971)**2 + (y[i]-1100)**2) * x_k4[i] + \
                k5[0]*(z[i]-5971)/((z[i]-5971)**2 + (x[i]+1025)**2) * y_k5[i]

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)
                k1 = np.array(AB_params['k1'], dtype=np.float64)
                k2 = np.array(AB_params['k2'], dtype=np.float64)
                k3 = np.array(AB_params['k3'], dtype=np.float64)
                k4 = np.array(AB_params['k4'], dtype=np.float64)
                k5 = np.array(AB_params['k5'], dtype=np.float64)

                calc_b_fields_cart(x, y, z, z_k1, z_k2, z_k3, x_k4, y_k5, A, B, k1, k2, k3, k4, k5,
                                   _alpha, _beta, _gamma, model_x, model_y, model_z)

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v15(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            beta[n-1][m-1] = n/L
            alpha[n-1][m-1] = m/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, C, D, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*A[0]*(np.cosh(alpha[0]*x[i] + B[0])) * \
                np.sinh(beta[0]*y[i] + C[0]) + \
                np.sin(gamma[0]*z[i] + D[0])

            model_y[i] += A[0]*(np.sinh(alpha[0]*x[i] + B[0])) * \
                beta[0]*np.cosh(beta[0]*y[i] + C[0]) + \
                np.sin(gamma[0]*z[i] + D[0])

            model_z[i] += A[0]*(np.sinh(alpha[0]*x[i] + B[0])) * \
                np.sinh(beta[0]*y[i] + C[0]) + \
                gamma[0]*np.cos(gamma[0]*z[i] + D[0])

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(quadwise(ABs[n*ms*4:(n+1)*ms*4])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[ab[2]]], dtype=np.float64)
                D = np.array([AB_params[ab[3]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, C, D, _alpha, _beta, _gamma, model_x, model_y,
                                   model_z)
        # print 'model_x'
        # print model_x
        # print 'model_y'
        # print model_y
        # print 'model_z'
        # print model_z

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v16(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):

            beta[n-1][m-1] = m/L
            gamma[n-1][m-1] = n/L
            alpha[n-1][m-1] = np.sqrt(beta[n-1][m-1]**2+gamma[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, C, D, E, F, alpha, beta, gamma, model_x, model_y,
                           model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*(A[0]*np.exp(alpha[0]*x[i]) - B[0]*np.exp(-alpha[0]*x[i])) * \
                (C[0]*np.sin(beta[0]*y[i]) + D[0]*np.cos(beta[0]*y[i])) * \
                (E[0]*np.sin(gamma[0]*z[i]) + F[0]*np.cos(gamma[0]*z[i]))

            model_y[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                beta[0]*(C[0]*np.cos(beta[0]*y[i]) - D[0]*np.sin(beta[0]*y[i])) * \
                (E[0]*np.sin(gamma[0]*z[i]) + F[0]*np.cos(gamma[0]*z[i]))

            model_z[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                (C[0]*np.sin(beta[0]*y[i]) + D[0]*np.cos(beta[0]*y[i])) * \
                gamma[0]*(E[0]*np.cos(gamma[0]*z[i]) - F[0]*np.sin(gamma[0]*z[i]))

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(hexwise(ABs[n*ms*6:(n+1)*ms*6])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                C = np.array([AB_params[ab[2]]], dtype=np.float64)
                D = np.array([AB_params[ab[3]]], dtype=np.float64)
                E = np.array([AB_params[ab[4]]], dtype=np.float64)
                F = np.array([AB_params[ab[5]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, C, D, E, F, _alpha, _beta, _gamma, model_x,
                                   model_y, model_z)

        return np.concatenate([model_x, model_y, model_z]).ravel()
    return brzphi_3d_fast


def bxyz_3d_producer_cart_v17(x, y, z, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((ns, ms))
    beta = np.zeros((ns, ms))
    gamma = np.zeros((ns, ms))

    for n in range(1, ns+1):
        for m in range(1, ms+1):
            beta[n-1][m-1] = n/L
            alpha[n-1][m-1] = m/L
            gamma[n-1][m-1] = np.sqrt(alpha[n-1][m-1]**2+beta[n-1][m-1]**2)

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (), () ->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, A, B, k1, k2, k3, alpha, beta, gamma, model_x, model_y, model_z):
        for i in range(z.shape[0]):
            model_x[i] += alpha[0]*(A[0]*np.exp(alpha[0]*x[i]) - B[0]*np.exp(-alpha[0]*x[i])) * \
                np.exp(-beta[0]*y[i]) * \
                np.sin(gamma[0]*z[i]) #+ k1[0]*y[i] + k2[0]*z[i]

            model_y[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                -beta[0]*np.exp(-beta[0]*y[i]) * \
                np.sin(gamma[0]*z[i]) #+ k1[0]*y[i] + k2[0]*z[i]

            model_z[i] += (A[0]*np.exp(alpha[0]*x[i]) + B[0]*np.exp(-alpha[0]*x[i])) * \
                np.exp(-beta[0]*y[i]) * \
                gamma[0]*np.cos(gamma[0]*z[i]) + \
                k1[0]*x[i] + k2[0]*y[i] + k3[0]*x[i]**2

    def brzphi_3d_fast(x, y, z, L, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_x = np.zeros(z.shape, dtype=np.float64)
        model_y = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(ns):
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                A = np.array([AB_params[ab[0]]], dtype=np.float64)
                B = np.array([AB_params[ab[1]]], dtype=np.float64)
                _alpha = np.array(alpha[n][m], dtype=np.float64)
                _beta = np.array(beta[n][m], dtype=np.float64)
                _gamma = np.array(gamma[n][m], dtype=np.float64)
                k1 = np.array(AB_params['k1'], dtype=np.float64)
                k2 = np.array(AB_params['k2'], dtype=np.float64)
                k3 = np.array(AB_params['k3'], dtype=np.float64)

                calc_b_fields_cart(x, y, z, A, B, k1, k2, k3, _alpha, _beta, _gamma, model_x,
                                   model_y, model_z)

        return np.concatenate([model_x, model_y, model_z]).ravel()
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
                iv1[n][m] = special.iv(m-n, ((n)/P)*np.abs(r))
                iv2[n][m] = special.iv(m+n, ((n)/P)*np.abs(r))
                ivp1[n][m] = special.ivp(m-n, ((n)/P)*np.abs(r))
                ivp2[n][m] = special.ivp(m+n, ((n)/P)*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], int64[:], int64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (), (m), (m), (m), (m)->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, P, m, n, A, B, C, D,
                      iv1, iv2, ivp1, ivp2, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (n[0]/P[0])*(ivp1[i]*(A[0]*np.cos(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i]) +
                                                B[0]*np.sin(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i])) +
                                       ivp2[i]*(C[0]*np.cos(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i]) -
                                                D[0]*np.sin(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i])))

            model_z[i] += (n[0]/P[0])*(iv1[i]*(-A[0]*np.sin(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i]) +
                                               B[0]*np.cos(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i])) -
                                       iv2[i]*(C[0]*np.sin(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i]) +
                                               D[0]*np.cos(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i])))

            model_phi[i] += (1.0/np.abs(r[i])) * \
                ((-m[0]+n[0])*iv1[i]*(A[0]*np.sin(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i]) -
                                      B[0]*np.cos(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i])) +
                 (m[0]+n[0])*iv2[i]*(C[0]*np.sin(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i]) +
                                     D[0]*np.cos(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i])))
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

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_hel_v1(z, r, phi, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    P = (2*np.pi)/L

    iv1 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    iv2 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    ivp1 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    ivp2 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            # if n <= m:
            iv1[n][m] = special.iv(n, (m-n)*P*np.abs(r))
            iv2[n][m] = special.iv(n, (m+n)*P*np.abs(r))
            ivp1[n][m] = special.ivp(n, (m-n)*P*np.abs(r))
            ivp2[n][m] = special.ivp(n, (m+n)*P*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], int64[:], int64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (), (m), (m), (m), (m)->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, P, m, n, A, B, C, D,
                      iv1, iv2, ivp1, ivp2, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += P[0]*(ivp1[i]*(m[0]-n[0]) *
                                (A[0]*np.cos((m[0]-n[0])*z[i]*P[0]+n[0]*phi[i]) +
                                 B[0]*np.sin((m[0]-n[0])*z[i]*P[0]+n[0]*phi[i])) +
                                ivp2[i]*(m[0]+n[0]) *
                                (C[0]*np.cos((m[0]+n[0])*z[i]*P[0]-n[0]*phi[i]) -
                                 D[0]*np.sin((m[0]+n[0])*z[i]*P[0]-n[0]*phi[i])))

            model_z[i] += P[0]*(iv1[i]*(n[0]-m[0]) *
                                (A[0]*np.sin((m[0]-n[0])*z[i]*P[0]+n[0]*phi[i]) -
                                 B[0]*np.cos((m[0]-n[0])*z[i]*P[0]+n[0]*phi[i])) +
                                iv2[i]*(m[0]+n[0]) *
                                (C[0]*np.sin((m[0]+n[0])*z[i]*P[0]-n[0]*phi[i]) +
                                 D[0]*np.cos((m[0]+n[0])*z[i]*P[0]-n[0]*phi[i])))

            model_phi[i] += (n[0]/np.abs(r[i])) * \
                (iv1[i]*(-A[0]*np.sin((m[0]-n[0])*z[i]*P[0]+n[0]*phi[i]) +
                         B[0]*np.cos((m[0]-n[0])*z[i]*P[0]+n[0]*phi[i])) +
                 iv2[i]*(C[0]*np.sin((m[0]+n[0])*z[i]*P[0]-n[0]*phi[i]) +
                         D[0]*np.cos((m[0]+n[0])*z[i]*P[0]-n[0]*phi[i])))
            # print(model_r[i], n[0], m[0], P[0], ivp1[i], ivp2[i], A[0], B[0], C[0], D[0], z[i],
            #       phi[i])

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        _P = np.asarray([(2*np.pi)/R])
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        for n in range(1, ns+1):
            # print('n', n)
            # print(np.any(np.isnan(model_r)))
            # print(np.any(np.isnan(model_z)))
            # print(np.any(np.isnan(model_phi)))
            for m, ab in enumerate(quadwise(ABs[n*ms*4:(n+1)*ms*4])):
                # print('m', m)

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

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_hel_v2(z, r, phi, L, ns, ms):
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
                iv1[n][m] = special.iv(m-n, ((n)/P)*np.abs(r))
                iv2[n][m] = special.iv(m+n, ((n)/P)*np.abs(r))
                ivp1[n][m] = special.ivp(m-n, ((n)/P)*np.abs(r))
                ivp2[n][m] = special.ivp(m+n, ((n)/P)*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], int64[:], int64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (), (), (m), (m), (m), (m)->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, P, m, n, A, B, C, D, E,
                      iv1, iv2, ivp1, ivp2, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (n[0]/P[0])*(ivp1[i]*(A[0]*np.cos(n[0]*z[i]/P[0]+(m[0]-n[0])*(phi[i]+E[0])) +
                                                B[0]*np.sin(n[0]*z[i]/P[0]+(m[0]-n[0])*(phi[i]+E[0]))) +
                                       ivp2[i]*(C[0]*np.cos(n[0]*z[i]/P[0]-(m[0]+n[0])*(phi[i]+E[0])) -
                                                D[0]*np.sin(n[0]*z[i]/P[0]-(m[0]+n[0])*(phi[i]+E[0]))))

            model_z[i] += (n[0]/P[0])*(iv1[i]*(-A[0]*np.sin(n[0]*z[i]/P[0]+(m[0]-n[0])*(phi[i]+E[0])) +
                                               B[0]*np.cos(n[0]*z[i]/P[0]+(m[0]-n[0])*(phi[i]+E[0]))) -
                                       iv2[i]*(C[0]*np.sin(n[0]*z[i]/P[0]-(m[0]+n[0])*(phi[i]+E[0])) +
                                               D[0]*np.cos(n[0]*z[i]/P[0]-(m[0]+n[0])*(phi[i]+E[0]))))

            model_phi[i] += (1/np.abs(r[i])) * \
                ((-m[0]+n[0])*iv1[i]*(A[0]*np.sin(n[0]*z[i]/P[0]+(m[0]-n[0])*(phi[i] + E[0])) -
                                      B[0]*np.cos(n[0]*z[i]/P[0]+(m[0]-n[0])*(phi[i] + E[0]))) +
                 (m[0]+n[0])*iv2[i]*(C[0]*np.sin(n[0]*z[i]/P[0]-(m[0]+n[0])*(phi[i] + E[0])) +
                                     D[0]*np.cos(n[0]*z[i]/P[0]-(m[0]+n[0])*(phi[i] + E[0]))))
            # print(model_r[i], n[0], m[0], P[0], ivp1[i], ivp2[i], A[0], B[0], C[0], D[0], z[i],
            #       phi[i])

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        _P = np.asarray([R/(2*np.pi)])
        ABs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k or 'C' in
                                                                      k or 'D' in k)},
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
                    E = np.array([AB_params['E']], dtype=np.float64)
                    _iv1 = iv1[n][m]
                    _iv2 = iv2[n][m]
                    _ivp1 = ivp1[n][m]
                    _ivp2 = ivp2[n][m]
                    _n = np.array([n])
                    _m = np.array([m])
                    calc_b_fields(z, phi, r, _P, _m, _n, A, B, C, D, E, _iv1, _iv2, _ivp1, _ivp2,
                                  model_r, model_z, model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_hel_v3(z, r, phi, L, ns, ms):
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
                iv1[n][m] = special.iv(m-n, ((n)/P)*np.abs(r))
                iv2[n][m] = special.iv(m+n, ((n)/P)*np.abs(r))
                ivp1[n][m] = special.ivp(m-n, ((n)/P)*np.abs(r))
                ivp2[n][m] = special.ivp(m+n, ((n)/P)*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], int64[:], int64[:],"
                  "float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (m), (m), (m), (m)->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, P, m, n, A, B,
                      iv1, iv2, ivp1, ivp2, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (n[0]/P[0])*(ivp1[i]*(A[0]*np.cos(n[0]*z[i]/P[0]+(m[0]-n[0])*phi[i])) +
                                       ivp2[i]*(B[0]*np.cos(n[0]*z[i]/P[0]-(m[0]+n[0])*phi[i])))

            model_z[i] += (n[0]/P[0])*(iv1[i]*(-A[0]*np.sin(n[0]*z[i]/P[0]+(m[0]-n[0])*phi[i])) -
                                       iv2[i]*(B[0]*np.sin(n[0]*z[i]/P[0]-(m[0]+n[0])*phi[i])))

            model_phi[i] += (1.0/np.abs(r[i])) * \
                ((-m[0]+n[0])*iv1[i]*(A[0]*np.sin(n[0]*z[i]/P[0]+(m[0]-n[0])*phi[i])) +
                 (m[0]+n[0])*iv2[i]*(B[0]*np.sin(n[0]*z[i]/P[0]-(m[0]+n[0])*phi[i])))
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
            for m, ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*(ms)*2])):
                if n <= m:
                    # print('n', n, 'm', m, ab)

                    A = np.array([AB_params[ab[0]]], dtype=np.float64)
                    B = np.array([AB_params[ab[1]]], dtype=np.float64)
                    _iv1 = iv1[n][m]
                    _iv2 = iv2[n][m]
                    _ivp1 = ivp1[n][m]
                    _ivp2 = ivp2[n][m]
                    _n = np.array([n])
                    _m = np.array([m])
                    calc_b_fields(z, phi, r, _P, _m, _n, A, B, _iv1, _iv2, _ivp1, _ivp2,
                                  model_r, model_z, model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_hel_v4(z, r, phi, L, ns, ms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    iv1 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    iv2 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    ivp1 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    ivp2 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            # if n <= m:
                iv1[n][m] = special.iv(m-n, ((n)/1000)*np.abs(r))
                iv2[n][m] = special.iv(m+n, ((n)/1000)*np.abs(r))
                ivp1[n][m] = special.ivp(m-n, ((n)/1000)*np.abs(r))
                ivp2[n][m] = special.ivp(m+n, ((n)/1000)*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], int64[:], int64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (m), (m), (m), (m)->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, m, n, A, B, C, D,
                      iv1, iv2, ivp1, ivp2, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (n[0]/1000)*(ivp1[i]*(A[0]*np.cos(n[0]*(z[i]-8000)/2000.0+(m[0]-n[0])*phi[i]) +
                                                B[0]*np.sin(n[0]*(z[i]-8000)/2000.0+(m[0]-n[0])*phi[i])) +
                                       ivp2[i]*(C[0]*np.cos(n[0]*(z[i]-8000)/2000.0-(m[0]+n[0])*phi[i]) -
                                                D[0]*np.sin(n[0]*(z[i]-8000)/2000.0-(m[0]+n[0])*phi[i])))

            model_z[i] += (n[0]/2000.0)*(iv1[i]*(-A[0]*np.sin(n[0]*(z[i]-8000)/2000.0+(m[0]-n[0])*phi[i]) +
                                               B[0]*np.cos(n[0]*(z[i]-8000)/2000.0+(m[0]-n[0])*phi[i])) -
                                       iv2[i]*(C[0]*np.sin(n[0]*(z[i]-8000)/2000.0-(m[0]+n[0])*phi[i]) +
                                               D[0]*np.cos(n[0]*(z[i]-8000)/2000.0-(m[0]+n[0])*phi[i])))

            model_phi[i] += (1.0/np.abs(r[i])) * \
                ((-m[0]+n[0])*iv1[i]*(A[0]*np.sin(n[0]*(z[i]-8000)/2000.0+(m[0]-n[0])*phi[i]) -
                                      B[0]*np.cos(n[0]*(z[i]-8000)/2000.0+(m[0]-n[0])*phi[i])) +
                 (m[0]+n[0])*iv2[i]*(C[0]*np.sin(n[0]*(z[i]-8000)/2000.0-(m[0]+n[0])*phi[i]) +
                                     D[0]*np.cos(n[0]*(z[i]-8000)/2000.0-(m[0]+n[0])*phi[i])))
            # print(model_r[i], n[0], m[0], P[0], ivp1[i], ivp2[i], A[0], B[0], C[0], D[0], z[i],
            #       phi[i])

    def brzphi_3d_fast(z, r, phi, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
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
                    calc_b_fields(z, phi, r, _m, _n, A, B, C, D, _iv1, _iv2, _ivp1, _ivp2,
                                  model_r, model_z, model_phi)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast


def brzphi_3d_producer_hel_v5(z, r, phi, L, ns, ms, cns, cms):
    '''
    Factory function that readies a potential fit function for a 3D magnetic field.
    This function creates a modified bessel function expression/
    '''

    alpha = np.zeros((cns, cms))
    beta = np.zeros((cns, cms))
    gamma = np.zeros((cns, cms))

    for cn in range(1, cns+1):
        for cm in range(1, cms+1):
            alpha[cn-1][cm-1] = cn/4000
            beta[cn-1][cm-1] = cm/4000
            gamma[cn-1][cm-1] = np.sqrt(beta[cn-1][cm-1]**2+alpha[cn-1][cm-1]**2)

    P = L/(2*np.pi)

    iv1 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    iv2 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    ivp1 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    ivp2 = np.zeros((ns, ms, r.shape[0], r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            # if n <= m:
                iv1[n][m] = special.iv(m-n, ((n)/P)*np.abs(r))
                iv2[n][m] = special.iv(m+n, ((n)/P)*np.abs(r))
                ivp1[n][m] = special.ivp(m-n, ((n)/P)*np.abs(r))
                ivp2[n][m] = special.ivp(m+n, ((n)/P)*np.abs(r))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:], int64[:], int64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (), (), (), (), (), (), (), (m), (m), (m), (m)->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields(z, phi, r, P, m, n, A, B, C, D,
                      iv1, iv2, ivp1, ivp2, model_r, model_z, model_phi):
        for i in range(z.shape[0]):
            model_r[i] += (n[0]/P[0])*(ivp1[i]*(A[0]*np.cos(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i]) +
                                                B[0]*np.sin(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i])) +
                                       ivp2[i]*(C[0]*np.cos(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i]) -
                                                D[0]*np.sin(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i])))

            model_z[i] += (n[0]/P[0])*(iv1[i]*(-A[0]*np.sin(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i]) +
                                               B[0]*np.cos(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i])) -
                                       iv2[i]*(C[0]*np.sin(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i]) +
                                               D[0]*np.cos(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i])))

            model_phi[i] += (1.0/np.abs(r[i])) * \
                ((-m[0]+n[0])*iv1[i]*(A[0]*np.sin(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i]) -
                                      B[0]*np.cos(n[0]*(z[i])/P[0]+(m[0]-n[0])*phi[i])) +
                 (m[0]+n[0])*iv2[i]*(C[0]*np.sin(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i]) +
                                     D[0]*np.cos(n[0]*(z[i])/P[0]-(m[0]+n[0])*phi[i])))

    @guvectorize(["void(float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:], float64[:],"
                  "float64[:], float64[:], float64[:], float64[:])"],
                 '(m), (m), (m), (m), (m), (m), (m), (m), (m), (m), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()->(m), (m), (m)',
                 nopython=True, target='parallel')
    def calc_b_fields_cart(x, y, z, xp1, yp1, zp1, xp2, yp2, zp2, phi, E, F, k1, k2, k3, k4, k5, k6,
                           k7, k8, k9, k10, k11, alpha, beta, gamma, model_r, model_phi, model_z):
        for i in range(z.shape[0]):
            model_x = -alpha[0]*np.exp(-alpha[0]*x[i]) * \
                np.exp(-beta[0]*y[i]) * \
                (E[0]*np.sin(gamma[0]*z[i]) + F[0]*np.cos(gamma[0]*z[i])) + \
                k1[0]*xp1[i]*zp1[i]/(
                    (xp1[i]**2+yp1[i]**2+zp1[i]**2) *
                    (zp1[i]+np.sqrt(xp1[i]**2+yp1[i]**2+zp1[i]**2))) + \
                k2[0]*xp2[i]*zp2[i]/(
                    (xp2[i]**2+yp2[i]**2+zp2[i]**2) *
                    (zp2[i]+np.sqrt(xp2[i]**2+yp2[i]**2+zp2[i]**2)))

            model_y = np.exp(-alpha[0]*x[i]) * \
                -beta[0]*np.exp(-beta[0]*y[i]) * \
                (E[0]*np.sin(gamma[0]*z[i]) + F[0]*np.cos(gamma[0]*z[i])) + \
                k1[0]*yp1[i]*zp1[i]/(
                    (xp1[i]**2+yp1[i]**2+zp1[i]**2) *
                    (zp1[i]+np.sqrt(xp1[i]**2+yp1[i]**2+zp1[i]**2))) + \
                k2[0]*yp2[i]*zp2[i]/(
                    (xp2[i]**2+yp2[i]**2+zp2[i]**2) *
                    (zp2[i]+np.sqrt(xp2[i]**2+yp2[i]**2+zp2[i]**2)))

            model_z[i] += np.exp(-alpha[0]*x[i]) * \
                np.exp(-beta[0]*y[i]) * \
                gamma[0]*(E[0]*np.cos(gamma[0]*z[i]) - F[0]*np.sin(gamma[0]*z[i])) + \
                k1[0]*(-zp1[i]+np.sqrt(xp1[i]**2+yp1[i]**2+zp1[i]**2)) / \
                (xp1[i]**2+yp1[i]**2+zp1[i]**2) + \
                k2[0]*(-zp2[i]+np.sqrt(xp2[i]**2+yp2[i]**2+zp2[i]**2)) / \
                (xp2[i]**2+yp2[i]**2+zp2[i]**2)

            model_r[i] += model_x*np.cos(phi[i]) + model_y*np.sin(phi[i])
            model_phi[i] += -model_x*np.sin(phi[i]) + model_y*np.cos(phi[i])

    def brzphi_3d_fast(z, r, phi, x, y, R, ns, ms, **AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        model_r = np.zeros(z.shape, dtype=np.float64)
        model_z = np.zeros(z.shape, dtype=np.float64)
        model_phi = np.zeros(z.shape, dtype=np.float64)
        _P = np.asarray([R/(2*np.pi)])
        ABCDs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('A' in k or 'B' in k or 'C'
                                                                        in k or 'D' in k)},
                     key=lambda x: ','.join((x.split('_')[1].zfill(5), x.split('_')[2].zfill(5),
                                            x.split('_')[0])))

        EFs = sorted({k: v for (k, v) in six.iteritems(AB_params) if ('E' in k or 'F' in k)})
        xp1 = x+1100
        yp1 = y+400
        zp1 = z-7896
        xp2 = x+525
        yp2 = y-1100
        zp2 = z-9746

        for n in range(ns):
            for m, ab in enumerate(quadwise(ABCDs[n*ms*4:(n+1)*(ms)*4])):
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

        for cn in range(cns):
            for cm, ef in enumerate(pairwise(EFs[cn*cms*2:(cn+1)*cms*2])):

                _alpha = np.array(alpha[cn][cm], dtype=np.float64)
                _beta = np.array(beta[cn][cm], dtype=np.float64)
                _gamma = np.array(gamma[cn][cm], dtype=np.float64)
                E = np.array([AB_params[ef[0]]], dtype=np.float64)
                F = np.array([AB_params[ef[1]]], dtype=np.float64)
                k1 = np.array(AB_params['k1'], dtype=np.float64)
                k2 = np.array(AB_params['k2'], dtype=np.float64)
                k3 = np.array(AB_params['k3'], dtype=np.float64)
                k4 = np.array(AB_params['k4'], dtype=np.float64)
                k5 = np.array(AB_params['k5'], dtype=np.float64)
                k6 = np.array(AB_params['k6'], dtype=np.float64)
                k7 = np.array(AB_params['k7'], dtype=np.float64)
                k8 = np.array(AB_params['k8'], dtype=np.float64)
                k9 = np.array(AB_params['k9'], dtype=np.float64)
                k10 = np.array(AB_params['k10'], dtype=np.float64)
                k11 = np.array(AB_params['k11'], dtype=np.float64)
                calc_b_fields_cart(x, y, z, xp1, yp1, zp1, xp2, yp2, zp2, phi, E, F, k1, k2, k3, k4,
                                   k5, k6, k7, k8, k9, k10, k11, _alpha, _beta, _gamma, model_r,
                                   model_phi, model_z)

        model_phi[np.isinf(model_phi)] = 0
        return np.concatenate([model_r, model_z, model_phi]).ravel()
    return brzphi_3d_fast

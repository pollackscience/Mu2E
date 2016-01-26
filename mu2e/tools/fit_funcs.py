#! /usr/bin/env python

from __future__ import division
from scipy import special
from lmfit import minimize, Parameters, Parameter, report_fit, Model
import numpy as np
import numexpr as ne
from numba import double, int32, jit

from itertools import izip

def pairwise(iterable):
    """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
    a = iter(iterable)
    return izip(a, a)

def brzphi_3d_producer(z,r,phi,R,ns,ms):
    b_zeros = []
    for n in range(ns):
        b_zeros.append(special.jn_zeros(n,ms))
    kms = np.asarray([b/R for b in b_zeros])
    iv = np.empty((ns,ms,r.shape[0],r.shape[1]))
    ivp = np.empty((ns,ms,r.shape[0],r.shape[1]))
    for n in range(ns):
        for m in range(ms):
            iv[n][m] = special.iv(n,kms[n][m]*np.abs(r))
            ivp[n][m] = special.ivp(n,kms[n][m]*np.abs(r))

    def brzphi_3d_fast(z,r,phi,R,ns,ms,**AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        def numexpr_model_r_calc(z,phi,n,D,A,B,ivp,kms):
            return ne.evaluate('cos(n*phi+D)*ivp*kms*(A*cos(kms*z) + B*sin(-kms*z))')
        def numexpr_model_z_calc(z,phi,n,D,A,B,iv,kms):
            return ne.evaluate('-cos(n*phi+D)*iv*kms*(A*sin(kms*z) + B*cos(-kms*z))')
        def numexpr_model_phi_calc(z,r,phi,n,D,A,B,iv,kms):
            return ne.evaluate('-n*sin(n*phi+D)*(1/abs(r))*iv*(A*cos(kms*z) + B*sin(-kms*z))')

        model_r = 0.0
        model_z = 0.0
        model_phi = 0.0
        R = R
        ABs = sorted({k:v for (k,v) in AB_params.iteritems() if ('A' in k or 'B' in k)},key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[2].zfill(5),x.split('_')[0])))
        Deltas = sorted({k:v for (k,v) in AB_params.iteritems() if 'delta' in k})

        for n in range(ns):
            for i,ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                model_r += numexpr_model_r_calc(z,phi,n,AB_params[Deltas[n]],AB_params[ab[0]],AB_params[ab[1]],ivp[n][i],kms[n][i])
                model_z += numexpr_model_z_calc(z,phi,n,AB_params[Deltas[n]],AB_params[ab[0]],AB_params[ab[1]],iv[n][i],kms[n][i])
                model_phi += numexpr_model_phi_calc(z,r,phi,n,AB_params[Deltas[n]],AB_params[ab[0]],AB_params[ab[1]],iv[n][i],kms[n][i])

        model_phi[np.isinf(model_phi)]=0
        return np.concatenate([model_r,model_z,model_phi]).ravel()
    return brzphi_3d_fast

def b_external_3d_producer(a,b,c,x,y,z,cns,cms):
    a = a
    b = b
    c = c

    alpha_2d = np.zeros((cns,cms))
    beta_2d = np.zeros((cns,cms))
    gamma_2d = np.zeros((cns,cms))

    for cn in range(1, cns+1):
        for cm in range(1, cms+1):
            alpha_2d[cn-1][cm-1] = (cn*np.pi-np.pi/2)/a
            beta_2d[cn-1][cm-1] = (cm*np.pi-np.pi/2)/b
            gamma_2d[cn-1][cm-1] = np.sqrt(alpha_2d[cn-1][cm-1]**2+beta_2d[cn-1][cm-1]**2)


    def b_external_3d_fast(x,y,z,cns,cms,epsilon1,epsilon2,**AB_params):
        """ 3D model for Bz Bx and By vs Z and X. Can take any number of Cnm terms."""

        def numexpr_model_x_ext_calc(x,y,z,C,alpha,beta,gamma,c,epsilon1,epsilon2):
            return ne.evaluate('C*alpha*sin(alpha*x+epsilon1)*cos(beta*y+epsilon2)*sinh(gamma*(z-c))')
        def numexpr_model_y_ext_calc(x,y,z,C,alpha,beta,gamma,c,epsilon1,epsilon2):
            return ne.evaluate('C*beta*cos(alpha*x+epsilon1)*sin(beta*y+epsilon2)*sinh(gamma*(z-c))')
        def numexpr_model_z_ext_calc(x,y,z,C,alpha,beta,gamma,c,epsilon1,epsilon2):
            return ne.evaluate('(-C)*gamma*cos(epsilon1 + alpha*x)*cos(epsilon2 + beta*y)*cosh(gamma*(-c + z))')

        model_x = 0.0
        model_y = 0.0
        model_z = 0.0
        Cs = sorted({k:v for (k,v) in AB_params.iteritems() if 'C' in k})

        for cn in range(1, cns+1):
            for cm in range(1, cms+1):
            #for cm,cd in enumerate(pairwise(CDs[cn*cms*2:(cn+1)*cms*2])):
                alpha = alpha_2d[cn-1][cm-1]
                beta = beta_2d[cn-1][cm-1]
                gamma = gamma_2d[cn-1][cm-1]

                #using C's
                model_x += numexpr_model_x_ext_calc(x,y,z,AB_params[Cs[cm-1+(cn-1)*cms]],alpha,beta,gamma,c,epsilon1,epsilon2)
                model_y += numexpr_model_y_ext_calc(x,y,z,AB_params[Cs[cm-1+(cn-1)*cms]],alpha,beta,gamma,c,epsilon1,epsilon2)
                model_z += numexpr_model_z_ext_calc(x,y,z,AB_params[Cs[cm-1+(cn-1)*cms]],alpha,beta,gamma,c,epsilon1,epsilon2)


        return np.concatenate([model_x,model_y,model_z]).ravel()
    return b_external_3d_fast


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
        #def model_r_calc(z,phi,n,D,A,B,ivp,kms):
        #        return np.cos(n*phi+D)*ivp*kms*(A*np.cos(kms*z) + B*np.sin(-kms*z))
        #jit_model_r_calc = jit(double[:,:](double[:,:],double[:,:],int32,double,double,double,double[:,:],double))(model_r_calc)

        def numexpr_model_r_calc(z,phi,n,D,A,B,ivp,kms):
                return ne.evaluate('cos(n*phi+D)*ivp*kms*(A*cos(kms*z) + B*sin(-kms*z))')
        def numexpr_model_z_calc(z,phi,n,D,A,B,iv,kms):
                return ne.evaluate('-cos(n*phi+D)*iv*kms*(A*sin(kms*z) + B*cos(-kms*z))')
        def numexpr_model_phi_calc(z,r,phi,n,D,A,B,iv,kms):
                return ne.evaluate('-n*sin(n*phi+D)*(1/abs(r))*iv*(A*cos(kms*z) + B*sin(-kms*z))')

        def numexpr_model_r_ext_calc(z,r,phi,C,alpha,beta,gamma):
            return ne.evaluate('C*sinh(gamma*z)*(beta*sin(phi)*sin(alpha*r*cos(phi))*cos(beta*r*sin(phi))+alpha*cos(phi)*cos(alpha*r*cos(phi))*sin(beta*r*sin(phi)))')
        def numexpr_model_phi_ext_calc(z,r,phi,C,alpha,beta,gamma):
            return ne.evaluate('C*sinh(gamma*z)*(beta*cos(phi)*sin(alpha*r*cos(phi))*cos(beta*r*sin(phi))-alpha*sin(phi)*cos(alpha*r*cos(phi))*sin(beta*r*sin(phi)))')
        def numexpr_model_z_ext_calc(z,r,phi,C,alpha,beta,gamma):
            return ne.evaluate('gamma*C*cosh(gamma*z)*sin(beta*r*sin(phi))*sin(alpha*r*cos(phi))')

        model_r = 0.0
        model_z = 0.0
        model_phi = 0.0
        R = R
        ABs = sorted({k:v for (k,v) in AB_params.iteritems() if ('A' in k or 'B' in k)},key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[2].zfill(5),x.split('_')[0])))
        Ds = sorted({k:v for (k,v) in AB_params.iteritems() if 'delta' in k})
        Cs = sorted({k:v for (k,v) in AB_params.iteritems() if 'C' in k})
        a = 20000
        b = 20000

        for n in range(ns):
            for i,ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                model_r += numexpr_model_r_calc(z,phi,n,AB_params[Ds[n]],AB_params[ab[0]],AB_params[ab[1]],ivp[n][i],kms[n][i])
                model_z += numexpr_model_z_calc(z,phi,n,AB_params[Ds[n]],AB_params[ab[0]],AB_params[ab[1]],iv[n][i],kms[n][i])
                model_phi += numexpr_model_phi_calc(z,r,phi,n,AB_params[Ds[n]],AB_params[ab[0]],AB_params[ab[1]],iv[n][i],kms[n][i])

        for cn in range(5):
            for cm in range(5):
                alpha = cn*np.pi/a
                beta = cm*np.pi/b
                gamma = np.pi*np.sqrt(cn**2/a**2+cm**2/b**2)

                model_r += numexpr_model_r_ext_calc(z,r,phi,AB_params[Cs[cm+cn*5]],alpha,beta,gamma)
                model_z += numexpr_model_z_ext_calc(z,r,phi,AB_params[Cs[cm+cn*5]],alpha,beta,gamma)
                model_phi += numexpr_model_phi_ext_calc(z,r,phi,AB_params[Cs[cm+cn*5]],alpha,beta,gamma)

        model_phi[np.isinf(model_phi)]=0
        return np.concatenate([model_r,model_z,model_phi]).ravel()
    return brzphi_3d_fast

def brzphi_ext_producer(z,r,phi,ns,ms):
    a = 20000
    b = 20000

    def brzphi_ext_fast(z,r,phi,ns,ms,**A_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""
        def numexpr_model_r_calc(z,r,phi,A,n,m,alpha,beta,gamma):
            return ne.evaluate('A*sin(gamma*z)*(beta*sin(phi)*sinh(alpha*r*cos(phi))*cosh(beta*r*sin(phi))+alpha*cos(phi)*cosh(alpha*r*cos(phi))*sinh(beta*r*sin(phi)))')
        def numexpr_model_phi_calc(z,r,phi,A,n,m,alpha,beta,gamma):
            return ne.evaluate('A*sin(gamma*z)*(beta*cos(phi)*sinh(alpha*r*cos(phi))*cosh(beta*r*sin(phi))-alpha*sin(phi)*cosh(alpha*r*cos(phi))*sinh(beta*r*sin(phi)))')
        def numexpr_model_z_calc(z,r,phi,A,n,m,alpha,beta,gamma):
            return ne.evaluate('A*gamma*cos(gamma*z)*sinh(alpha*r*cos(phi))*sinh(beta*r*sin(phi))')

        model_r = 0.0
        model_z = 0.0
        model_phi = 0.0
        As = sorted({k:v for (k,v) in A_params.iteritems()})
        for n in range(ns):
            for m,A_nm in enumerate(As[n*ms:(n+1)*ms]):
                alpha = n*np.pi/a
                beta = m*np.pi/b
                gamma = np.pi*np.sqrt(n**2/a**2+m**2/b**2)

                model_r += numexpr_model_r_calc(z,r,phi,A_params[A_nm],n,m,alpha,beta,gamma)
                model_z += numexpr_model_z_calc(z,r,phi,A_params[A_nm],n,m,alpha,beta,gamma)
                model_phi += numexpr_model_phi_calc(z,r,phi,A_params[A_nm],n,m,alpha,beta,gamma)

        model_phi[np.isinf(model_phi)]=0
        return np.concatenate([model_r,model_z,model_phi]).ravel()
    return brzphi_ext_fast

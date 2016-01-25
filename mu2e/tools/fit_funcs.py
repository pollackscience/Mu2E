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

    #def brzphi_3d_fast(z,r,phi,R,ns,ms,cns,cms,x0,x1,y0,y1,c,**AB_params):
    def brzphi_3d_fast(z,r,phi,R,ns,ms,cns,cms,a,b,c,epsilon1,epsilon2,**AB_params):
        """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

        def numexpr_model_r_calc(z,phi,n,D,A,B,ivp,kms):
            return ne.evaluate('cos(n*phi+D)*ivp*kms*(A*cos(kms*z) + B*sin(-kms*z))')
        def numexpr_model_z_calc(z,phi,n,D,A,B,iv,kms):
            return ne.evaluate('-cos(n*phi+D)*iv*kms*(A*sin(kms*z) + B*cos(-kms*z))')
        def numexpr_model_phi_calc(z,r,phi,n,D,A,B,iv,kms):
            return ne.evaluate('-n*sin(n*phi+D)*(1/abs(r))*iv*(A*cos(kms*z) + B*sin(-kms*z))')

        #def numexpr_model_r_ext_calc(z,r,phi,C,alpha,beta,gamma,c):
        #    return ne.evaluate('C*sinh(-gamma*(z-c))*(-alpha*cos(phi)*sin(alpha*abs(r)*cos(phi))*cos(beta*abs(r)*sin(phi))-beta*sin(phi)*cos(alpha*abs(r)*cos(phi))*sin(beta*abs(r)*sin(phi)))')
        #def numexpr_model_phi_ext_calc(z,r,phi,C,alpha,beta,gamma,c):
        #    return ne.evaluate('C*sinh(-gamma*(z-c))*(alpha*sin(phi)*sin(alpha*abs(r)*cos(phi))*cos(beta*abs(r)*sin(phi))-beta*cos(phi)*cos(alpha*abs(r)*cos(phi))*sin(beta*abs(r)*sin(phi)))')
        #def numexpr_model_z_ext_calc(z,r,phi,C,alpha,beta,gamma,c):
        #    return ne.evaluate('-gamma*C*cosh(-gamma*(z-c))*cos(beta*abs(r)*sin(phi))*cos(alpha*abs(r)*cos(phi))')

        #def numexpr_model_r_ext_calc(z,r,phi,C,D,alpha1,beta1,gamma1, alpha2,beta2,gamma2,c):
        #    return ne.evaluate('(beta*sin(phi)*((-D)*cos(beta*r*sin(phi))*sin(alpha*r*cos(phi)) + C*cos(alpha*r*cos(phi))*sin(beta*r*sin(phi))) + alpha*cos(phi)*(C*cos(beta*r*sin(phi))*sin(alpha*r*cos(phi)) - D*cos(alpha*r*cos(phi))*sin(beta*r*sin(phi))))*sinh(gamma*(-c + z))')
        #def numexpr_model_phi_ext_calc(z,r,phi,C,D,alpha1,beta1,gamma1,alpha2,beta2,gamma2,c):
        #    return ne.evaluate('((-alpha1)*C*cos(beta1*r*sin(phi))*sin(phi)*sin(alpha1*r*cos(phi)) +beta1*C*cos(phi)*cos(alpha1*r*cos(phi))*sin(beta1*r*sin(phi)))*sinh(gamma1*(-c + z)) + D*((-beta2)*cos(phi)*cos(beta2*r*sin(phi))*sin(alpha2*r*cos(phi)) + alpha2*cos(alpha2*r*cos(phi))*sin(phi)*sin(beta2*r*sin(phi)))*sinh(gamma2*(-c + z))')
        #def numexpr_model_z_ext_calc(z,r,phi,C,D,alpha1,beta1,gamma1,alpha2,beta2,gamma2,c):
        #    return ne.evaluate('(-C)*gamma1*cos(alpha1*r*cos(phi))*cos(beta1*r*sin(phi))*cosh(gamma1*(-c + z)) -D*gamma2*cosh(gamma2*(-c + z))*sin(alpha2*r*cos(phi))*sin(beta2*r*sin(phi))')

        def numexpr_model_r_ext_calc(z,r,phi,C,alpha,beta,gamma,c,epsilon1,epsilon2):
            return ne.evaluate('C*(alpha*cos(phi)*cos(epsilon2 + beta*abs(r)*sin(phi))*sin(epsilon1 + alpha*abs(r)*cos(phi)) + beta*cos(epsilon1 + alpha*abs(r)*cos(phi))*sin(phi)*sin(epsilon2 + beta*abs(r)*sin(phi)))*sinh(gamma*(-c + z))')
        def numexpr_model_phi_ext_calc(z,r,phi,C,alpha,beta,gamma,c,epsilon1,epsilon2):
            return ne.evaluate('C*((-alpha)*cos(epsilon2 + beta*abs(r)*sin(phi))*sin(phi)*sin(epsilon1 + alpha*abs(r)*cos(phi)) + beta*cos(phi)*cos(epsilon1 + alpha*abs(r)*cos(phi))*sin(epsilon2 + beta*abs(r)*sin(phi)))*sinh(gamma*(-c + z))')
        def numexpr_model_z_ext_calc(z,r,phi,C,alpha,beta,gamma,c,epsilon1,epsilon2):
            return ne.evaluate('(-C)*gamma*cos(epsilon1 + alpha*abs(r)*cos(phi))*cos(epsilon2 + beta*abs(r)*sin(phi))*cosh(gamma*(-c + z))')

        model_r = 0.0
        model_z = 0.0
        model_phi = 0.0
        R = R
        ABs = sorted({k:v for (k,v) in AB_params.iteritems() if ('A' in k or 'B' in k)},key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[2].zfill(5),x.split('_')[0])))
        Deltas = sorted({k:v for (k,v) in AB_params.iteritems() if 'delta' in k})
        Cs = sorted({k:v for (k,v) in AB_params.iteritems() if 'C' in k})
        #CDs = sorted({k:v for (k,v) in AB_params.iteritems() if ('C' in k or 'D' in k)},key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[2].zfill(5),x.split('_')[0])))
        #a = 50000
        #b = 50000

        for n in range(ns):
            for i,ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

                model_r += numexpr_model_r_calc(z,phi,n,AB_params[Deltas[n]],AB_params[ab[0]],AB_params[ab[1]],ivp[n][i],kms[n][i])
                model_z += numexpr_model_z_calc(z,phi,n,AB_params[Deltas[n]],AB_params[ab[0]],AB_params[ab[1]],iv[n][i],kms[n][i])
                model_phi += numexpr_model_phi_calc(z,r,phi,n,AB_params[Deltas[n]],AB_params[ab[0]],AB_params[ab[1]],iv[n][i],kms[n][i])

        for cn in range(1, cns+1):
            for cm in range(1, cms+1):
            #for cm,cd in enumerate(pairwise(CDs[cn*cms*2:(cn+1)*cms*2])):
                #alpha = (cn*np.pi-np.pi/2)/a
                #beta = (cm*np.pi-np.pi/2)/b
                #gamma = np.pi*np.sqrt((cn-1/2)**2/a**2+(cm-1/2)**2/b**2)

                alpha = (cn*np.pi-np.pi/2)/a
                beta = (cm*np.pi-np.pi/2)/b
                gamma = np.sqrt(alpha**2+beta**2)

                #using C's
                model_r += numexpr_model_r_ext_calc(z,r,phi,AB_params[Cs[cm-1+(cn-1)*cms]],alpha,beta,gamma,c,epsilon1,epsilon2)
                model_z += numexpr_model_z_ext_calc(z,r,phi,AB_params[Cs[cm-1+(cn-1)*cms]],alpha,beta,gamma,c,epsilon1,epsilon2)
                model_phi += numexpr_model_phi_ext_calc(z,r,phi,AB_params[Cs[cm-1+(cn-1)*cms]],alpha,beta,gamma,c,epsilon1,epsilon2)

                #using Phi boundary soln
                #model_r += numexpr_model_r_ext_calc(z,r,phi,x0,x1,y0,y1,alpha,beta,gamma,a,b,c)
                #model_z += numexpr_model_z_ext_calc(z,r,phi,x0,x1,y0,y1,alpha,beta,gamma,a,b,c)
                #model_phi += numexpr_model_phi_ext_calc(z,r,phi,x0,x1,y0,y1,alpha,beta,gamma,a,b,c)

        model_phi[np.isinf(model_phi)]=0
        return np.concatenate([model_r,model_z,model_phi]).ravel()
    return brzphi_3d_fast

def b_external_3d_producer(x,y,z,cns,cms):

    def b_external_3d_fast(x,y,z,cns,cms,a,b,c,epsilon1,epsilon2,**AB_params):
        """ 3D model for Bz Bx and By vs Z and X. Can take any number of Cnm terms."""

        def numexpr_model_x_ext_calc(x,y,z,C,alpha,beta,gamma,c,epsilon1,epsilon2):
            return ne.evaluate('C*(alpha*cos(phi)*cos(epsilon2 + beta*abs(r)*sin(phi))*sin(epsilon1 + alpha*abs(r)*cos(phi)) + beta*cos(epsilon1 + alpha*abs(r)*cos(phi))*sin(phi)*sin(epsilon2 + beta*abs(r)*sin(phi)))*sinh(gamma*(-c + z))')
        def numexpr_model_y_ext_calc(x,y,z,C,alpha,beta,gamma,c,epsilon1,epsilon2):
            return ne.evaluate('C*((-alpha)*cos(epsilon2 + beta*abs(r)*sin(phi))*sin(phi)*sin(epsilon1 + alpha*abs(r)*cos(phi)) + beta*cos(phi)*cos(epsilon1 + alpha*abs(r)*cos(phi))*sin(epsilon2 + beta*abs(r)*sin(phi)))*sinh(gamma*(-c + z))')
        def numexpr_model_z_ext_calc(x,y,z,C,alpha,beta,gamma,c,epsilon1,epsilon2):
            return ne.evaluate('(-C)*gamma*cos(epsilon1 + alpha*abs(r)*cos(phi))*cos(epsilon2 + beta*abs(r)*sin(phi))*cosh(gamma*(-c + z))')

        model_x = 0.0
        model_y = 0.0
        model_z = 0.0
        Cs = sorted({k:v for (k,v) in AB_params.iteritems() if 'C' in k})

        for cn in range(1, cns+1):
            for cm in range(1, cms+1):
            #for cm,cd in enumerate(pairwise(CDs[cn*cms*2:(cn+1)*cms*2])):
                #alpha = (cn*np.pi-np.pi/2)/a
                #beta = (cm*np.pi-np.pi/2)/b
                #gamma = np.pi*np.sqrt((cn-1/2)**2/a**2+(cm-1/2)**2/b**2)

                alpha = (cn*np.pi-np.pi/2)/a
                beta = (cm*np.pi-np.pi/2)/b
                gamma = np.sqrt(alpha**2+beta**2)

                #using C's
                model_x += numexpr_model_x_ext_calc(x,y,z,AB_params[Cs[cm-1+(cn-1)*cms]],alpha,beta,gamma,c,epsilon1,epsilon2)
                model_y += numexpr_model_y_ext_calc(x,y,z,AB_params[Cs[cm-1+(cn-1)*cms]],alpha,beta,gamma,c,epsilon1,epsilon2)
                model_z += numexpr_model_z_ext_calc(x,y,z,AB_params[Cs[cm-1+(cn-1)*cms]],alpha,beta,gamma,c,epsilon1,epsilon2)


        return np.concatenate([model_x,model_y,model_z]).ravel()
    return b_external_3d_fast


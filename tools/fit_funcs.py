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


def bz_r0_1d(z, R, **AB_params):
  """ 1D model for Bz vs Z, R=0. Can take any number of AnBn terms."""
  model = 0.0
  R = R
  ABs = sorted(AB_params.keys(),key=lambda x:x[::-1])
  b_zeros = special.jn_zeros(0,len(ABs)/2)
  kms = b_zeros/R
  for i,ab in enumerate(pairwise(ABs)):
    model += kms[i]*(AB_params[ab[0]]*np.exp(kms[i]*z) - AB_params[ab[1]]*np.exp(-kms[i]*z))
  return model

def bz_2d(z,r, R, **AB_params):
#def bz_2d(z,r, R, C,**AB_params):
  """ 2D model for Bz vs Z and R. Can take any number of AnBn terms."""
  model = 0.0
  #model = C*z
  R = R
  ABs = sorted(AB_params.keys(),key=lambda x:x[::-1])
  b_zeros = special.jn_zeros(0,len(ABs)/2)
  kms = b_zeros/R
  for i,ab in enumerate(pairwise(ABs)):
    model += special.jn(0,kms[i]*abs(r))*kms[i]*(AB_params[ab[0]]*np.exp(kms[i]*z) - AB_params[ab[1]]*np.exp(-kms[i]*z))
  return model.ravel()

def bz_2d_mod(z,r, R, **AB_params):
  """ 2D model for Bz vs Z and R. Can take any number of AnBn terms."""
  model = 0.0
  R = R
  ABs = sorted(AB_params.keys(),key=lambda x:x[::-1])
  b_zeros = special.jn_zeros(0,len(ABs)/2)
  kms = b_zeros/R
  for i,ab in enumerate(pairwise(ABs)):
    model += special.iv(0,kms[i]*abs(r))*kms[i]*(AB_params[ab[0]]*np.exp(kms[i]*z) - AB_params[ab[1]]*np.exp(-kms[i]*z))
  return model.ravel()

#def br_2d(z,r, R, **AB_params):
def br_2d(z,r, R, C, **AB_params):
  """ 2D model for Bz vs Z and R. Can take any number of AnBn terms."""
  model = C*z
  #model = 0.0
  R = R
  ABs = sorted(AB_params.keys(),key=lambda x:x[::-1])
  b_zeros = special.jn_zeros(0,len(ABs)/2)
  kms = b_zeros/R
  for i,ab in enumerate(pairwise(ABs)):
    model += -special.jn(1,kms[i]*abs(r))*kms[i]*(AB_params[ab[0]]*np.exp(kms[i]*z) + AB_params[ab[1]]*np.exp(-kms[i]*z))
  return model.ravel()

def brz_2d(z,r, R, C, **AB_params):
#def brz_2d(z,r, R, **AB_params):
  """ 2D model for Bz vs Z and R. Can take any number of AnBn terms."""
  model_r = C*z
  model_z = 0.0
  R = R
  ABs = sorted(AB_params.keys(),key=lambda x:x[::-1])
  b_zeros = special.jn_zeros(0,len(ABs)/2)
  kms = b_zeros/R
  for i,ab in enumerate(pairwise(ABs)):
    model_r += -special.jv(1,kms[i]*abs(r))*kms[i]*(AB_params[ab[0]]*np.exp(kms[i]*(z-8000)) + AB_params[ab[1]]*np.exp(-kms[i]*(z-8000)))
    model_z += special.jv(0,kms[i]*abs(r))*kms[i]*(AB_params[ab[0]]*np.exp(kms[i]*(z-8000)) - AB_params[ab[1]]*np.exp(-kms[i]*(z-8000)))
  return np.concatenate([model_r,model_z]).ravel()

def brz_2d_trig(z,r,R,offset,**AB_params):
#def brz_2d(z,r, R, **AB_params):
  """ 2D model for Bz and Br vs Z and R. Can take any number of AnBn terms."""
  #model_r = C*(1.0/z)
  model_r = 0.0
  model_z = 0.0
  R = R
  ABs = sorted(AB_params.keys(),key=lambda x:x[::-1])
  b_zeros = special.jn_zeros(0,len(ABs)/2)
  kms = b_zeros/R
  for i,ab in enumerate(pairwise(ABs)):
    model_r += special.iv(1,kms[i]*abs(r))*kms[i]*(AB_params[ab[0]]*np.cos(kms[i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[i]*(z-offset)))
    model_z += -special.iv(0,kms[i]*abs(r))*kms[i]*(AB_params[ab[0]]*np.sin(kms[i]*(z-offset)) + AB_params[ab[1]]*np.cos(-kms[i]*(z-offset)))
  return np.concatenate([model_r,model_z]).ravel()

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

  def brzphi_3d_fast(z,r,phi,R,C,ns,ms,**AB_params):
    """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""
    #def model_r_calc(z,phi,n,D,A,B,ivp,kms):
    #    return np.cos(n*phi+D)*ivp*kms*(A*np.cos(kms*z) + B*np.sin(-kms*z))
    #jit_model_r_calc = jit(double[:,:](double[:,:],double[:,:],int32,double,double,double,double[:,:],double))(model_r_calc)
    def numexpr_model_r_calc(z,phi,n,D,A,B,ivp,kms):
        return ne.evaluate('cos(n*phi+D)*ivp*kms*(A*cos(kms*z) + B*sin(-kms*z))')

    model_r = -np.cos(phi)*(C)
    #model_r = 0.0
    model_z = 0.0
    model_phi = np.sin(phi)*(C)
    #model_phi = 0.0
    R = R
    ABs = sorted({k:v for (k,v) in AB_params.iteritems() if 'delta' not in k},key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[2].zfill(5),x.split('_')[0])))
    Ds = sorted({k:v for (k,v) in AB_params.iteritems() if 'delta' in k})
    for n in range(ns):
      for i,ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

        #model_r += model_r_calc(z,phi,n,AB_params[Ds[n]],AB_params[ab[0]],AB_params[ab[1]],kms[n][i],ivp[n][i])

        #model_r += np.cos(n*phi)*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        #model_z += -np.cos(n*phi)*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.cos(-kms[n][i]*(z-offset)))
        #model_phi += -n*np.sin(n*phi)*(1/abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))

######version cos(n*phi+delta)
        #model_r += jit_model_r_calc(z,phi,n,AB_params[Ds[n]],AB_params[ab[0]],AB_params[ab[1]],ivp[n][i],kms[n][i])
        model_r += numexpr_model_r_calc(z,phi,n,AB_params[Ds[n]],AB_params[ab[0]],AB_params[ab[1]],ivp[n][i],kms[n][i])
        #model_r += np.cos(n*phi+AB_params[Ds[n]])*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*z) + AB_params[ab[1]]*np.sin(-kms[n][i]*z))
        model_z += -np.cos(n*phi+AB_params[Ds[n]])*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*z) + AB_params[ab[1]]*np.cos(-kms[n][i]*z))
        model_phi += -n*np.sin(n*phi+AB_params[Ds[n]])*(1/np.abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*z) + AB_params[ab[1]]*np.sin(-kms[n][i]*z))

######version cos(n*phi)cos(delta)-sin(n*phi)sin(delta)
        #model_r += (np.cos(n*phi)*np.cos(AB_params[Ds[n]])-np.sin(n*phi)*np.sin(AB_params[Ds[n]]))*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*z) + AB_params[ab[1]]*np.sin(-kms[n][i]*z))
        #model_z += -(np.cos(n*phi)*np.cos(AB_params[Ds[n]])-np.sin(n*phi)*np.sin(AB_params[Ds[n]]))*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*z) + AB_params[ab[1]]*np.cos(-kms[n][i]*z))
        #model_phi += -n*(np.sin(n*phi)*np.cos(AB_params[Ds[n]])+np.cos(n*phi)*np.sin(AB_params[Ds[n]]))*(1/np.abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*z) + AB_params[ab[1]]*np.sin(-kms[n][i]*z))

######version cos(n*phi)cos(delta)-sin(n*phi)sin(delta), small angle for delta
        #model_r += (np.cos(n*phi)*(1-AB_params[Ds[n]]**2)-np.sin(n*phi)*(AB_params[Ds[n]]))*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*z) + AB_params[ab[1]]*np.sin(-kms[n][i]*z))
        #model_z += -(np.cos(n*phi)*(1-AB_params[Ds[n]]**2)-np.sin(n*phi)*(AB_params[Ds[n]]))*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*z) + AB_params[ab[1]]*np.cos(-kms[n][i]*z))
        #model_phi += -n*(np.sin(n*phi)*(1-AB_params[Ds[n]]**2)+np.cos(n*phi)*(AB_params[Ds[n]]))*(1/abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*z) + AB_params[ab[1]]*np.sin(-kms[n][i]*z))


######version cos(n*phi+delta)+sin(n*phi+delta)
        #model_r += (np.cos(n*phi+delta)+np.sin(n*phi+delta))*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        #model_z += -(np.cos(n*phi+delta)+np.sin(n*phi+delta))*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.cos(-kms[n][i]*(z-offset)))
        #model_phi += n*(-np.sin(n*phi+delta)+np.cos(n*phi+delta))*(1/abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))

    model_phi[np.isinf(model_phi)]=0
    print model_r.shape
    return np.concatenate([model_r,model_z,model_phi]).ravel()
  return brzphi_3d_fast


def brzphi_3d_producer_n2(z,r,phi,R,ns,ms):
  b_zeros = []
  for n in range(ns):
    b_zeros.append(special.jn_zeros(n+2,ms))
  kms = np.asarray([b/R for b in b_zeros])
  iv = np.empty((ns,ms,r.shape[0],r.shape[1]))
  ivp = np.empty((ns,ms,r.shape[0],r.shape[1]))
  for n in range(ns):
    for m in range(ms):
      iv[n][m] = special.iv(n+2,kms[n][m]*abs(r))
      ivp[n][m] = special.ivp(n+2,kms[n][m]*abs(r))

  #def brzphi_3d_fast(z,r,phi,R,ns,ms,delta,offset,**AB_params):
  def brzphi_3d_fast(z,r,phi,R,ns,ms,**AB_params):
    """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""

    model_r = 0.0
    model_z = 0.0
    model_phi = 0.0
    R = R
    ABs = sorted({k:v for (k,v) in AB_params.iteritems() if 'delta' not in k},key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[2].zfill(5),x.split('_')[0])))
    Ds = sorted({k:v for (k,v) in AB_params.iteritems() if 'delta' in k})
    for n in range(ns):
      for i,ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):

######version cos(n*phi)cos(delta)-sin(n*phi)sin(delta)
        model_r += (np.cos((n+2)*phi)*np.cos(AB_params[Ds[n]])-np.sin((n+2)*phi)*np.sin(AB_params[Ds[n]]))*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*z) + AB_params[ab[1]]*np.sin(-kms[n][i]*z))
        model_z += -(np.cos((n+2)*phi)*np.cos(AB_params[Ds[n]])-np.sin((n+2)*phi)*np.sin(AB_params[Ds[n]]))*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*z) + AB_params[ab[1]]*np.cos(-kms[n][i]*z))
        model_phi += -(n+2)*(np.sin((n+2)*phi)*np.cos(AB_params[Ds[n]])+np.cos((n+2)*phi)*np.sin(AB_params[Ds[n]]))*(1/abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*z) + AB_params[ab[1]]*np.sin(-kms[n][i]*z))

    model_phi[np.isinf(model_phi)]=0
    return np.concatenate([model_r,model_z,model_phi]).ravel()
  return brzphi_3d_fast

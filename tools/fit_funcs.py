#! /usr/bin/env python

from scipy import special
from lmfit import minimize, Parameters, Parameter, report_fit, Model
import numpy as np

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

def brzphi_3d(z,r,phi,R,ns,ms,offset,**AB_params):
#def brz_2d(z,r, R, **AB_params):
  """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""
  model_r = 0.0
  model_z = 0.0
  model_phi = 0.0
  R = R
  ABs = sorted(AB_params,key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[2].zfill(5),x.split('_')[0])))
  b_zeros = []
  for n in range(ns):
    b_zeros.append(special.jn_zeros(n,ms))
  kms = [b/R for b in b_zeros]
  for n in range(ns):
    #print n
    for i,ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):
      model_r += np.cos(n*phi)*special.ivp(n,kms[n][i]*abs(r))*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
      model_z += -np.cos(n*phi)*special.iv(n,kms[n][i]*abs(r))*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.cos(-kms[n][i]*(z-offset)))
      model_phi += -n*np.sin(n*phi)*(1/abs(r))*special.iv(n,kms[n][i]*abs(r))*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
      #model_r += np.cos(n*phi)*special.jvp(n,kms[n][i]*abs(r))*kms[n][i]*(AB_params[ab[0]]*np.exp(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.exp(-kms[n][i]*(z-offset)))
      #model_z += np.cos(n*phi)*special.jv(n,kms[n][i]*abs(r))*kms[n][i]*(AB_params[ab[0]]*np.exp(kms[n][i]*(z-offset)) - AB_params[ab[1]]*np.exp(-kms[n][i]*(z-offset)))
      #model_phi += -n*np.sin(n*phi)*special.jv(n,kms[n][i]*abs(r))*(AB_params[ab[0]]*np.exp(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.exp(-kms[n][i]*(z-offset)))
      #model_phi += phi*0.0
  model_phi[np.isinf(model_phi)]=0
  return np.concatenate([model_r,model_z,model_phi]).ravel()

def brzphi_3d_producer(z,r,phi,R,ns,ms):
  b_zeros = []
  for n in range(ns):
    b_zeros.append(special.jn_zeros(n,ms))
  kms = [b/R for b in b_zeros]
  iv = np.empty((ns,ms,r.shape[0],r.shape[1]))
  ivp = np.empty((ns,ms,r.shape[0],r.shape[1]))
  for n in range(ns):
    for m in range(ms):
      iv[n][m] = special.iv(n,kms[n][m]*abs(r))
      ivp[n][m] = special.ivp(n,kms[n][m]*abs(r))

  def brzphi_3d_fast(z,r,phi,R,ns,ms,delta,offset,**AB_params):
  #def brzphi_3d_fast(z,r,phi,R,ns,ms,offset,**AB_params):
    """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""
    model_r = 0.0
    model_z = 0.0
    model_phi = 0.0
    R = R
    ABs = sorted(AB_params,key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[2].zfill(5),x.split('_')[0])))
    for n in range(ns):
      for i,ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):
        #model_r += np.cos(n*phi)*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        #model_z += -np.cos(n*phi)*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.cos(-kms[n][i]*(z-offset)))
        #model_phi += -n*np.sin(n*phi)*(1/abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        model_r += np.cos(n*phi+delta)*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        model_z += -np.cos(n*phi+delta)*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.cos(-kms[n][i]*(z-offset)))
        model_phi += -n*np.sin(n*phi+delta)*(1/abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        #model_r += (np.cos(n*phi+delta)+np.sin(n*phi+delta))*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        #model_z += -(np.cos(n*phi+delta)+np.sin(n*phi+delta))*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.cos(-kms[n][i]*(z-offset)))
        #model_phi += n*(-np.sin(n*phi+delta)+np.cos(n*phi+delta))*(1/abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
    model_phi[np.isinf(model_phi)]=0
    return np.concatenate([model_r,model_z,model_phi]).ravel()
  return brzphi_3d_fast



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

def brz_2d_trig(z,r,R,offset,C,D, **AB_params):
#def brz_2d(z,r, R, **AB_params):
  """ 2D model for Bz vs Z and R. Can take any number of AnBn terms."""
  #model_r = C*(1.0/z**2)
  model_r = 0.0
  model_z = 0.0
  R = R
  ABs = sorted(AB_params.keys(),key=lambda x:x[::-1])
  b_zeros = special.jn_zeros(0,len(ABs)/2)
  kms = b_zeros/R
  for i,ab in enumerate(pairwise(ABs)):
    model_r += -special.iv(1,kms[i]*abs(r))*kms[i]*(AB_params[ab[0]]*np.cos(kms[i]*(z-offset)) + AB_params[ab[1]]*np.sin(kms[i]*(z-offset)))
    model_z += special.iv(0,kms[i]*abs(r))*kms[i]*(AB_params[ab[0]]*np.sin(kms[i]*(z-offset)) - AB_params[ab[1]]*np.cos(-kms[i]*(z-offset)))
  return np.concatenate([model_r,model_z]).ravel()

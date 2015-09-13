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
  """ 2D model for Bz vs Z and R. Can take any number of AnBn terms."""
  model = 0.0
  R = R
  ABs = sorted(AB_params.keys(),key=lambda x:x[::-1])
  b_zeros = special.jn_zeros(0,len(ABs)/2)
  kms = b_zeros/R
  for i,ab in enumerate(pairwise(ABs)):
    model += special.jn(0,kms[i]*abs(r))*kms[i]*(AB_params[ab[0]]*np.exp(kms[i]*z) - AB_params[ab[1]]*np.exp(-kms[i]*z))
  return model.ravel()

#def br_2d(z,r, R,C, **AB_params):
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

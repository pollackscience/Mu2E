#! /usr/bin/env python

from scipy import special
from lmfit import minimize, Parameters, Parameter, report_fit, Model
import numpy as np


def big_poly_2d((x,y), O,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p):
  leg = O+a*x+b*y+ c*x*y + d*x**2 + e*y**2+ f*(x*y**2) + g*(y*x**2) + h*x**3 + i*y**3 +j*x**2*y**2 +k*x**3*y +l*y**3*x
  + m*x**4 + n*y**4 + o*x**5 + p*y**5
  return leg.ravel()


def bz_r0_1d(z, *params):
  R=1000.0
  b_zeros = special.jn_zeros(0,len(params)/2)
  kms = b_zeros/R
  bz = 0
  for i in range(len(params)/2):
    bz+=kms[i]*(params[2*i]*np.exp(kms[i]*z)-params[2*i+1]*np.exp(-kms[i]*z))

  return bz

def bz_r0_1d_lm(z, R, A0,B0,A1,B1,A2,B2,A3,B3,A4,B4,A5,B5,A6,B6):
  """ model decaying sine wave, subtract data"""
  R = R
  b_zeros = special.jn_zeros(0,(bz_r0_1d_lm.func_code.co_argcount-2)/2)
  kms = b_zeros/R
  model = (kms[0]*(A0*np.exp(kms[0]*z) - B0*np.exp(-kms[0]*z))+
      kms[1]*(A1*np.exp(kms[1]*z) - B1*np.exp(-kms[1]*z))+
      kms[2]*(A2*np.exp(kms[2]*z) - B2*np.exp(-kms[2]*z))+
      kms[3]*(A3*np.exp(kms[3]*z) - B3*np.exp(-kms[3]*z))+
      kms[4]*(A4*np.exp(kms[4]*z) - B4*np.exp(-kms[4]*z))+
      kms[5]*(A5*np.exp(kms[5]*z) - B5*np.exp(-kms[5]*z))+
      kms[6]*(A6*np.exp(kms[6]*z) - B6*np.exp(-kms[6]*z))
      )
  return model


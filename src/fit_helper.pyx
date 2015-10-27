#! /usr/bin/env python

from __future__ import division
cimport cython
cimport numpy as np
import numpy as np
from scipy import special
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport fabs


cpdef model_r_calc(np.ndarray[np.float64_t, ndim=2] res, np.ndarray[np.float64_t, ndim=2] z, np.ndarray[np.float64_t, ndim=2] phi,
  int n, int i, np.float64_t delta, np.float64_t offset, np.float64_t a, np.float64_t b,
  np.ndarray[np.float64_t, ndim=2] kms, np.ndarray[np.float64_t, ndim=4] ivp):

  #cdef unsigned int r1,r2
  #for r1 in range(phi.shape[0]):
   # for r2 in range(phi.shape[1]):
    #  pass
      #res[r1][r2] += (cos(n*phi[r1][r2]+delta)*ivp[n][i][r1][r2]*kms[n][i]*(a*cos(kms[n][i]*(z[r1][r2]-offset)) + b*sin(-kms[n][i]*(z[r1][r2]-offset))))
  pass

def main():
  r1 = np.asarray(range(-49,0), dtype=np.float64)
  r2 = np.asarray(range(1,50), dtype=np.float64)
  z = np.asarray(range(8000,9000), dtype=np.float64)


  zz,rr = np.meshgrid(z,np.concatenate([r1,r2]))
  pp = np.full_like(rr,-2)
  pp[:,pp.shape[1]/2:]*=-1

  ns = 2
  ms = 5

  delta =0.5
  offset = 0.0
  a=1.1
  b=1.1
  R=9000.0


  model_r = np.zeros((rr.shape[0],rr.shape[1]))


  b_zeros = []
  for n in range(ns):
    b_zeros.append(special.jn_zeros(n,ms))
  kms = np.asarray([b/R for b in b_zeros])
  iv = np.empty((ns,ms,rr.shape[0],rr.shape[1]))
  ivp = np.empty((ns,ms,rr.shape[0],rr.shape[1]))
  for n in range(ns):
    for m in range(ms):
      iv[n][m] = special.iv(n,kms[n][m]*np.abs(rr))
      ivp[n][m] = special.ivp(n,kms[n][m]*np.abs(rr))

  n=1
  i=1
  print model_r

  model_r_calc(model_r,zz,pp,n,i,delta,offset,a,b,kms,ivp)
  print model_r

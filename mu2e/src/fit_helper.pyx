#! /usr/bin/env python

from __future__ import division
cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport fabs


        #model_r += (np.cos(n*phi)*(1-AB_params[Ds[n]]**2)-np.sin(n*phi)*(AB_params[Ds[n]]))*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*z) + AB_params[ab[1]]*np.sin(-kms[n][i]*z))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[double, ndim=2] model_r_calc(np.ndarray[double, ndim=2] z, np.ndarray[double, ndim=2] phi,
  double n, double d, double a, double b,
  double kms, np.ndarray[double, ndim=2] ivp):

  cdef unsigned int s1 = z.shape[0]
  cdef unsigned int s2 = z.shape[1]
  cdef np.ndarray[double, ndim=2] res = np.empty((s1,s2))
  cdef unsigned int r1,r2
  for r1 in range(s1):
    for r2 in range(s2):
      res[r1][r2] = (cos(n*phi[r1][r2]+d)*ivp[r1][r2]*kms*(a*cos(kms*(z[r1][r2])) + b*sin(-kms*(z[r1][r2]))))

  return res

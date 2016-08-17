#! /usr/bin/env python

from __future__ import division
cimport cython
from scipy import special
from lmfit import minimize, Parameters, Parameter, report_fit, Model
cimport numpy as np
import numpy as np
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport fabs

from itertools import izip

def pairwise(iterable):
  """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
  a = iter(iterable)
  return izip(a, a)

def brzphi_3d_producer_c(np.ndarray[np.float64_t, ndim=2] z, np.ndarray[np.float64_t, ndim=2] r, np.ndarray[np.float64_t, ndim=2] phi, int R, int ns, int ms):
  cdef unsigned int n
  cdef unsigned int m
  cdef np.float64_t sp
  cdef unsigned int i
  cdef unsigned int r1
  cdef unsigned int r2
  #cdef np.float64_t[:,:] b_zeros = np.empty((ns,ms))
  cdef np.float64_t[:,:] kms = np.empty((ns,ms))
  for n in range(ns):
    for i,sp in enumerate(special.jn_zeros(n,ms)):
      kms[n][i] = sp/R
  cdef np.float64_t[:,:,:,:] iv = np.empty((ns,ms,r.shape[0],r.shape[1]))
  cdef np.float64_t[:,:,:,:] ivp = np.empty((ns,ms,r.shape[0],r.shape[1]))
  for n in range(ns):
    for m in range(ms):
      for r1 in range(r.shape[0]):
        for r2 in range(r.shape[1]):
          iv[n][m][r1][r2] = special.iv(n,kms[n][m]*fabs(r[r1][r2]))
          ivp[n][m][r1][r2] = special.ivp(n,kms[n][m]*fabs(r[r1][r2]))

  def brzphi_3d_fast_c(np.float64_t[:,:] z, np.float64_t[:,:] r, np.float64_t[:,:] phi, int R, int ns, int ms,np.float64_t delta, np.float64_t offset, **AB_params):
  #def brzphi_3d_fast(z,r,phi,R,ns,ms,offset,**AB_params):
    """ 3D model for Bz Br and Bphi vs Z and R. Can take any number of AnBn terms."""
    def model_r_calc(int n, int i, str a, str b):
      cdef np.float64_t[:,:] res = np.empty((phi.shape[0],phi.shape[1]))
      for r1 in xrange(phi.shape[0]):
        for r2 in xrange(phi.shape[1]):
          res[r1][r2] = (cos(n*phi[r1][r2]+delta)*ivp[n][i][r1][r2]*kms[n][i]*(AB_params[a]*cos(kms[n][i]*(z[r1][r2]-offset)) + AB_params[b]*sin(-kms[n][i]*(z[r1][r2]-offset))))
      return res
#
#    mr = np.vectorize(model_r,otypes=[np.ndarray])
#    n_args = []
#    i_args = []
#    a_args = []
#    b_args = []

    cdef np.float64_t[:,:] model_r = np.zeros((r.shape[0],r.shape[1]))
    model_z = 0.0
    model_phi = 0.0
    R = R
    ABs = sorted(AB_params,key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[2].zfill(5),x.split('_')[0])))
    for n in range(ns):
      for i,ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):
#        n_args.append(n)
#        i_args.append(i)
#        a_args.append(ab[0])
#        b_args.append(ab[1])

        #model_r += np.cos(n*phi)*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        #model_z += -np.cos(n*phi)*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.cos(-kms[n][i]*(z-offset)))
        #model_phi += -n*np.sin(n*phi)*(1/abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        #model_r += np.cos(n*phi+delta)*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        model_r += np.array(model_r_calc(n, i, ab[0], ab[1]))

        #model_z += -np.cos(n*phi+delta)*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.cos(-kms[n][i]*(z-offset)))
        #model_phi += -n*np.sin(n*phi+delta)*(1/abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        #model_r += (np.cos(n*phi+delta)+np.sin(n*phi+delta))*ivp[n][i]*kms[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
        #model_z += -(np.cos(n*phi+delta)+np.sin(n*phi+delta))*iv[n][i]*kms[n][i]*(AB_params[ab[0]]*np.sin(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.cos(-kms[n][i]*(z-offset)))
        #model_phi += n*(-np.sin(n*phi+delta)+np.cos(n*phi+delta))*(1/abs(r))*iv[n][i]*(AB_params[ab[0]]*np.cos(kms[n][i]*(z-offset)) + AB_params[ab[1]]*np.sin(-kms[n][i]*(z-offset)))
    #model_phi[np.isinf(model_phi)]=0
    #m = mr(n_args,i_args,a_args,b_args)
    #model_r = np.sum(mr(n_args,i_args,a_args,b_args),dtype=np.ndarray)
    #model_r = np.add.reduce(m)
    #model_r = np.sum(mr(n_args,i_args,a_args,b_args),dtype=np.ndarray)
    #return np.concatenate([model_r,model_z,model_phi]).ravel()
    model_rr = np.array(model_r)
    return model_rr.ravel()
  return brzphi_3d_fast_c

def main():

  r1 = np.asarray(range(-49,0), dtype=np.float64)
  r2 = np.asarray(range(1,50), dtype=np.float64)
  z = np.asarray(range(8000,9000), dtype=np.float64)


  zz,rr = np.meshgrid(z,np.concatenate([r1,r2]))
  pp = np.full_like(rr,-2)
  pp[:,pp.shape[1]/2:]*=-1

  ns = 2
  ms = 5

  params = {}
  for n in range(ns):
    for m in range(ms):
      params['A_{0}_{1}'.format(n,m)]=1
      params['B_{0}_{1}'.format(n,m)]=1

  f = brzphi_3d_producer_c(zz,rr,pp,9000,ns,ms)
  fout = f(zz,rr,pp,9000,ns,ms,0.5,0,**params)
  print fout

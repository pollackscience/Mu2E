#! /usr/bin/env python

from __future__ import division
import os
import mu2e
import numpy as np
import pandas as pd
import cPickle as pkl
from mu2e.tools.fit_funcs import *
from scipy import special
from mu2e.datafileprod import DataFileMaker
from numba import double, int32, jit, vectorize, float64, guvectorize
from itertools import izip


def pairwise(iterable):
    """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
    a = iter(iterable)
    return izip(a, a)




# load up the model, set all the parameters from the pickle, then start using that function.  Possibly binarize that entire object?

#test point (R,phi,Z)
point = (50,0,8000)


def get_mag_field_function(param_name):
    '''pre-calculate what can be done, cache, return function to calc mag field'''
    pickle_path = os.path.abspath(os.path.dirname(mu2e.__file__))+'/../fit_params/'
    params = pkl.load(open(pickle_path+param_name+'_results.p',"rb"))
#params.pretty_print()
    param_dict = params.valuesdict()
    Reff = param_dict['R']
    ns = param_dict['ns']
    ms = param_dict['ms']

    del param_dict['R']
    del param_dict['ns']
    del param_dict['ms']

    As = np.zeros((ns,ms))
    Bs = np.zeros((ns,ms))
    Cs = np.zeros(ns)
    Ds = np.zeros(ns)

    ABs = sorted({k:v for (k,v) in param_dict.iteritems() if ('A' in k or 'B' in k)},key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[2].zfill(5),x.split('_')[0])))
    CDs = sorted({k:v for (k,v) in param_dict.iteritems() if ('C' in k or 'D' in k)},key=lambda x:','.join((x.split('_')[1].zfill(5),x.split('_')[0])))

    for n,cd in enumerate(pairwise(CDs)):
        Cs[n] = param_dict[cd[0]]
        Ds[n] = param_dict[cd[1]]
        for m,ab in enumerate(pairwise(ABs[n*ms*2:(n+1)*ms*2])):
            As[n,m] = param_dict[ab[0]]
            Bs[n,m] = param_dict[ab[1]]

    b_zeros = []
    for n in range(ns):
        b_zeros.append(special.jn_zeros(n,ms))
    kms = np.asarray([b/Reff for b in b_zeros])

    @jit
    def mag_field_function(a,b,z,cart=False):
        '''give r,phi,z, (or x,y,z) and return br,bphi,bz (or bx,by,bz)'''
        if cart:
            #if a==0: a+=1e-8
            #if b==0: b+=1e-8
            r = np.sqrt(a**2+b**2)
            phi = np.arctan2(b,a)
        else:
            r = a
            phi = b

        iv = np.empty((ns,ms))
        ivp = np.empty((ns,ms))
        for n in range(ns):
            iv[n,:] = special.iv(n,kms[n,:]*np.abs(r))
            ivp[n,:] = special.ivp(n,kms[n,:]*np.abs(r))

        br = 0.0
        bphi = 0.0
        bz = 0.0
        for n in range(ns):
            for m in range(ms):
                br += (Cs[n]*np.cos(n*phi)+Ds[n]*np.sin(n*phi))*ivp[n][m]*kms[n][m]*(As[n][m]*np.cos(kms[n][m]*z) + Bs[n][m]*np.sin(-kms[n][m]*z))
                if abs(r)>1e-10:
                    bphi += n*(-Cs[n]*np.sin(n*phi)+Ds[n]*np.cos(n*phi))*(1/abs(r))*iv[n][m]*(As[n][m]*np.cos(kms[n][m]*z) + Bs[n][m]*np.sin(-kms[n][m]*z))
                bz += -(Cs[n]*np.cos(n*phi)+Ds[n]*np.sin(n*phi))*iv[n][m]*kms[n][m]*(As[n][m]*np.sin(kms[n][m]*z) + Bs[n][m]*np.cos(-kms[n][m]*z))

        if cart:
            bx = br*np.cos(phi)-bphi*np.sin(phi)
            by = br*np.sin(phi)+bphi*np.cos(phi)
            return (bx,by,bz)
        else:
            return (br,bphi,bz)
    return mag_field_function

def quick_print(df, a,b,z, cart=False):
    if cart:
        df_tmp = df[(np.isclose(df.X,a)) & (np.isclose(df.Y,b)) & (df.Z==z)]
        print df_tmp.Bx.values[0], df_tmp.By.values[0], df_tmp.Bz.values[0]
    else:
        df_tmp = df[(np.isclose(df.R,a)) & (np.isclose(df.Phi,b)) & (df.Z==z)]
        print df_tmp.Br.values[0], df_tmp.Bphi.values[0], df_tmp.Bz.values[0]

if __name__=='__main__':
    df = DataFileMaker('../datafiles/Mau10/Standard_Maps/Mu2e_DSMap', use_pickle=True).data_frame
    mag_field_function = get_mag_field_function()




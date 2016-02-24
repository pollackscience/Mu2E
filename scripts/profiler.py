#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from hallprobesim import *
from mu2e.tools.fit_funcs import *
from pyprof2calltree import convert, visualize


#use for profiling and optimizing the fitting procedure.
#do:
#  %run profiler
#  %prun -D out.stats ff.fit_3d_v4(5,10) or whatever fit thing
#  visualize('out.stats')
#
#  also, for the fit function, do
# %lprun -f  f f(ZZ,RR,PP,r,ns,ms,**params)
# or something similar to test the individual function lines

if __name__ == "__main__":


    cfg_params_profile = cfg_params(ns = 70, ms = 4, cns = 0, cms=0, Reff = 7000, a=None,b=None,c=None)
    cfg_pickle_profile = cfg_pickle(use_pickle = False, save_pickle = False, load_name = 'profile', save_name = 'profile', recreate = False)
    ZZ,RR,PP,Bz,Br,Bphi = field_map_analysis('profile', cfg_data_DS_Mau10, cfg_geom_cyl_800mm, cfg_params_profile, cfg_pickle_profile, cfg_plot_mpl, profile=True)
    R = cfg_params_profile.Reff
    ns = cfg_params_profile.ns
    ms = cfg_params_profile.ms

    params = {}

    for n in range(ns):
        params['C_{0}'.format(n)] = 1
        params['D_{0}'.format(n)] = 0.001
        for m in range(ms):
            params['A_{0}_{1}'.format(n,m)]=0
            params['B_{0}_{1}'.format(n,m)]=0


    f = brzphi_3d_producer_profile(ZZ.T,RR.T,PP.T,R,ns,ms)
    #fout = f(ZZ,RR,PP,R,ns,ms,**params)

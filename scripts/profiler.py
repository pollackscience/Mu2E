#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from mu2e.datafileprod import DataFileMaker
from mu2e.hallprober import HallProbeGenerator
from mu2e.fieldfitter import FieldFitter
from mu2e.plotter import Plotter
from tools.fit_funcs import brzphi_3d_producer


#use for profiling and optimizing the fitting procedure.
#do:
#  %run profiler
#  %prun ff.fit_3d() or whatever fit thing
#
#  also, for the fit function, do
# %lprun -f  f f(zz,rr,2,9000,ns,ms,0, **params)
# or something similar to test the individual function lines


def profiler_setup(magnet = 'DS',A='Y',B='Z',nparams=10,fullsim=False,suffix='halltoy',
    r_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all', phi_steps = (0,np.pi/2), conditions = ('X==0','Z>4000','Z<14000')):

  data_maker = DataFileMaker('../FieldMapData_1760_v5/Mu2e_'+magnet+'map',use_pickle = True)
  input_data = data_maker.data_frame
  for condition in conditions:
    input_data = input_data.query(condition)
  hpg = HallProbeGenerator(input_data, z_steps = z_steps, r_steps = r_steps, phi_steps = phi_steps)
  toy = hpg.get_toy()
  toy.By = abs(toy.By)
  toy.Bx = abs(toy.Bx)

  if A=='X':Br='Bx'
  elif A=='Y':Br='By'
  elif A=='R':Br='Br'

  ff = FieldFitter(toy,phi_steps,r_steps)
  return ff

if __name__ == "__main__":
  pi4r = [35.35533906,   70.71067812,  141.42135624, 176.7766953 ,  212.13203436, 282.84271247,
  353.55339059,  424.26406871, 494.97474683,  530.33008589,601.04076401,  671.75144213]
  phi_steps = (0, np.pi/4, np.pi/2, 3*np.pi/4)
  r_steps = (range(25,625,50), pi4r, range(25,625,50), pi4r)

  ff = profiler_setup(magnet = 'DS',A='R',B='Z',nparams=60,fullsim=False,suffix='halltoy3d_test',
           r_steps = r_steps, phi_steps = phi_steps, z_steps = 'all', conditions = ('Z>5000','Z<14000','R!=0'))

  ns = 5
  ms = 50

  ZZ,RR,PP,Bz,Br,Bphi = ff.fit_3d_v4(ns,ms,line_profile=True)

  params = {}
  for n in range(ns):
    params['delta_{0}'.format(n)]=0.5
    for m in range(ms):
      params['A_{0}_{1}'.format(n,m)]=1.0
      params['B_{0}_{1}'.format(n,m)]=1.0

  #f_c = brzphi_3d_producer_c(zz,rr,pp,9000,ns,ms)
  #fout_c = f_c(zz,rr,pp,9000,ns,ms,0.5,0,**params)
  f = brzphi_3d_producer(ZZ,RR,PP,9000,ns,ms)
  fout = f(ZZ,RR,PP,9000,0,ns,ms,**params)
  #src.helloworld.say_hello()

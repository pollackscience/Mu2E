#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from mu2e.datafileprod import DataFileMaker
from mu2e.hallprober import HallProbeGenerator
from mu2e.fieldfitter import FieldFitter
from mu2e.plotter import Plotter

#use for profiling and optimizing the fitting procedure.
#do:
#  %run profiler
#  %prun ff.fit_3d() or whatever fit thing
#
#  also, for the fit function, do
# %lprun -f  brzphi_3d brzphi_3d(zz,rr,2,9000,ns,ms,0, **params)
# or something similar to test the individual function lines


def profiler_setup(do_3d = False, magnet = 'DS',A='Y',B='Z',nparams=10,fullsim=False,suffix='halltoy',
    r_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all', conditions = ('X==0','Z>4000','Z<14000')):
  plt.close('all')
  data_maker = DataFileMaker('../FieldMapData_1760_v5/Mu2e_'+magnet+'map',use_pickle = True)
  input_data = data_maker.data_frame
  for condition in conditions:
    input_data = input_data.query(condition)
  hpg = HallProbeGenerator(input_data, z_steps = z_steps, x_steps = r_steps, y_steps = r_steps)
  toy = hpg.get_toy()
  toy.By = abs(toy.By)
  toy.Bx = abs(toy.Bx)

  if A=='X':Br='Bx'
  elif A=='Y':Br='By'
  elif A=='R':Br='Br'

  ff = FieldFitter(toy)

  return ff



if __name__ == "__main__":

  #data_maker,hpg,plot_maker = hallprobesim(magnet = 'DS',A='X',B='Z',nparams=60,fullsim=False,suffix='halltoy',
  #         r_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all', conditions = ('Y==0','Z>4000','Z<14000'))
  #data_maker,hpg,plot_maker = hallprobesim(do_3d=True, magnet = 'DS',A='R',B='Z',nparams=60,fullsim=False,suffix='halltoy3d_test',
           #r_steps = range(0,600,50), z_steps = 'all', conditions = ('X==0','Z>5000','Z<14000','R!=0'))

  ff = profiler_setup(do_3d=True, magnet = 'DS',A='R',B='Z',nparams=60,fullsim=False,suffix='profiler',
           r_steps = range(0,600,50), z_steps = 'all', conditions = ('X==0','Z>5000','Z<14000','R!=0'))



  from tools.fit_funcs import brzphi_3d_producer
  from tools.fit_funcs import brzphi_3d_producer_c

  zz,rr = np.meshgrid(np.asarray(range(8000,9000),dtype=np.float64),np.asarray(range(-49,0)+range(1,50),dtype=np.float64))
  pp = np.full_like(rr,-2)
  pp[:,pp.shape[1]/2:]*=-1

  ns = 2
  ms = 5

  params = {}
  for n in range(ns):
    for m in range(ms):
      params['A_{0}_{1}'.format(n,m)]=1
      params['B_{0}_{1}'.format(n,m)]=1

  f_c = brzphi_3d_producer_c(zz,rr,pp,9000,ns,ms)
  fout_c = f_c(zz,rr,pp,9000,ns,ms,0.5,0,**params)
  f = brzphi_3d_producer(zz,rr,pp,9000,ns,ms)
  fout = f(zz,rr,pp,9000,ns,ms,0.5,0,**params)
  #src.helloworld.say_hello()

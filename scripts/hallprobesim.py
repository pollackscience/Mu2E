#! /usr/bin/env python

import matplotlib.pyplot as plt
from mu2e.datafileprod import DataFileMaker
from mu2e.hallprober import HallProbeGenerator
from mu2e.fieldfitter import FieldFitter
from mu2e.plotter import Plotter


def hallprobesim(do_3d = False, magnet = 'DS',A='Y',B='Z',nparams=10,fullsim=False,suffix='halltoy',
    r_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all', conditions = ('X==0','Z>4000','Z<14000')):
  plt.close('all')
  data_maker = DataFileMaker('../FieldMapData_1760_v5/Mu2e_'+magnet+'map',use_pickle = True)
  input_data = data_maker.data_frame
  for condition in conditions:
    input_data = input_data.query(condition)
  hpg = HallProbeGenerator(input_data, z_steps = z_steps, r_steps = r_steps)
  toy = hpg.get_toy()
  toy.By = abs(toy.By)
  toy.Bx = abs(toy.Bx)

  if A=='X':Br='Bx'
  elif A=='Y':Br='By'
  elif A=='R':Br='Br'

  ff = FieldFitter(toy)
  if do_3d:
    ff.fit_3d_v2(ns=4,ms=20)
  else:
    ff.fit_2d_sim(A,B,nparams = nparams)

  if fullsim:
    df = data_maker.data_frame
    df.By = abs(df.By)
    df.Bx = abs(df.Bx)
    plot_maker = Plotter.from_hall_study({magnet+'_Mau':df},fit_result = ff.result)
    plot_maker.extra_suffix = suffix
    plot_maker.plot_A_v_B_and_C_fit('Bz',A,B,sim=True,do_3d=do_3d,do_eval=True,*conditions)
    plot_maker.plot_A_v_B_and_C_fit(Br,A,B,sim=True,do_3d=do_3d,do_eval=True,*conditions)
  else:
    plot_maker = Plotter.from_hall_study({magnet+'_Mau':ff.input_data},fit_result = ff.result)
    plot_maker.extra_suffix = suffix
    plot_maker.plot_A_v_B_and_C_fit_cyl('Bz',A,B,False,*conditions)
    plot_maker.plot_A_v_B_and_C_fit_cyl(Br,A,B,False,*conditions)
    if do_3d:
      plot_maker.plot_A_v_B_and_C_fit_cyl('Bphi',A,B,False,*conditions)

  return data_maker, hpg, plot_maker


if __name__ == "__main__":

  #data_maker,hpg,plot_maker = hallprobesim(magnet = 'DS',A='X',B='Z',nparams=60,fullsim=False,suffix='halltoy',
  #         r_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all', conditions = ('Y==0','Z>4000','Z<14000'))

  data_maker,hpg,plot_maker = hallprobesim(do_3d=True, magnet = 'DS',A='R',B='Z',nparams=60,fullsim=False,suffix='halltoy3d_test',
           r_steps = range(0,600,50), z_steps = 'all', conditions = ('X==0','Z>5000','Z<14000','R!=0'))


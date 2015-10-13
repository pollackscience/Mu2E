#! /usr/bin/env python

import matplotlib.pyplot as plt
from mu2e.datafileprod import DataFileMaker
from mu2e.hallprober import HallProbeGenerator
from mu2e.fieldfitter import FieldFitter
from mu2e.plotter import Plotter


def hallprobesim(magnet = 'DS',A='Y',B='Z',nparams=10,fullsim=False,suffix='halltoy',
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

  ff = FieldFitter(toy)
  ff.fit_2d_sim(A,B,nparams = nparams)

  if fullsim:
    df = data_maker.data_frame
    df.By = abs(df.By)
    df.Bx = abs(df.Bx)
    plot_maker = Plotter.from_hall_study({magnet+'_Mau':df},fit_result = ff.result)
    plot_maker.extra_suffix = suffix
    plot_maker.plot_A_v_B_and_C_fit('Bz',A,B,sim=True,do_eval=True,*conditions)
    plot_maker.plot_A_v_B_and_C_fit(Br,A,B,sim=True,do_eval=True,*conditions)
  else:
    plot_maker = Plotter.from_hall_study({magnet+'_Mau':ff.input_data},fit_result = ff.result)
    plot_maker.extra_suffix = suffix
    plot_maker.plot_A_v_B_and_C_fit('Bz',A,B,True,False,*conditions)
    plot_maker.plot_A_v_B_and_C_fit(Br,A,B,True,False,*conditions)

  return data_maker, hpg, plot_maker


if __name__ == "__main__":

  hallprobesim(magnet = 'DS',A='X',B='Z',nparams=60,fullsim=False,suffix='halltoy',
           r_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all', conditions = ('Y==0','Z>4000','Z<14000'))

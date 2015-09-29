#! /usr/bin/env python

import matplotlib.pyplot as plt
from mu2e.datafileprod import DataFileMaker
from mu2e.hallprober import HallProbeGenerator
from mu2e.fieldfitter import FieldFitter
from mu2e.plotter import Plotter

if __name__ == "__main__":

  plt.close('all')
  data_maker1=DataFileMaker('../FieldMapData_1760_v5/Mu2e_DSmap',use_pickle = True)
  #hpg = HallProbeGenerator(data_maker1.data_frame.query('X==0 & Z>8200 & Z <12000'),z_steps = 100,y_steps = 10)
  #hpg2 = HallProbeGenerator(data_maker1.data_frame.query('X==0 & Z>4000 & Z <14000'),z_steps = 100,y_steps = 10)
  #hpg2 = HallProbeGenerator(data_maker1.data_frame.query('X==0 & Z>4000 & Z <14000'),z_steps = 'all',y_steps = 10)
  #ff = FieldFitter(hpg.get_toy())
  #ff2 = FieldFitter(hpg2.get_toy())
  #ff.fit_2d_sim('Y','Z')
  #ff.pickle_results()
  #plot_maker = Plotter.from_hall_study({'DS_Mau':ff.input_data},fit_result = ff.result)
  #fig1,name1 = plot_maker.plot_A_v_B_and_C_fit('Bz','Y','Z',True,'X==0','Z>8200','Z<12000')
  #fig2,name2 = plot_maker.plot_A_v_B_and_C_fit('Br','Y','Z',True,'X==0','Z>8200','Z<13000')
  #plot_maker.make_gif(fig1,name1)
  #plot_maker.make_gif(fig2,name2)
  #ff2.fit_2d_sim('Y','Z')
  #plot_maker2 = Plotter.from_hall_study({'DS_Mau':ff2.input_data},fit_result = ff2.result)
  #plot_maker2.plot_A_v_B_and_C_fit('Bz','Y','Z',True,'X==0','Z>4000','Z<14000')
  #plot_maker2.plot_A_v_B_and_C_fit('Br','Y','Z',True,'X==0','Z>4000','Z<14000')


  hpg3 = HallProbeGenerator(data_maker1.data_frame.query('X==0 & Z>4000 & Z <14000'),z_steps = 'all', y_steps = [-825,-650,-475,-325,0,325,475,650,825])
  ff3 = FieldFitter(hpg3.get_toy())
  ff3.fit_2d_sim('Y','Z')
  plot_maker3 = Plotter.from_hall_study({'DS_Mau':ff3.input_data},fit_result = ff3.result)
  plot_maker3.extra_suffix = 'halltoy'
  plot_maker3.plot_A_v_B_and_C_fit('Bz','Y','Z',True,'X==0')
  plot_maker3.plot_A_v_B_and_C_fit('Br','Y','Z',True,'X==0')





"""
ff3.result
ff3.result.eval
df = data_maker1.data_frame
df = df.eval('Z>4000')
df = df.eval('Z<14000')
df
df = data_maker1.data_frame
df
df.eval('Z<14000')
df = df.query('Z<14000')
df = df.query('Z>4000')
df = df.query('X==0')
df = df.query('R<700')
df
piv_bz = df.pivot('Z','Y','Bz')
piv_bz
piv_br = df.pivot('Z','Y','Br')
X=piv_br.columns.values
Y=piv_br.index.values
Bz=piv_bz.values
Br=piv_br.values
X,Y = np.meshgrid(X, Y)
full_fit_res = ff3.result.eval(r=X,z=Y)
full_fit_res
len(full_fit_res)
fig=plt.figure()
ax = fig.add_subplot(111)
bz_fit_diff = (Bz-full_fit_res[len(full_fit_res)/2:].reshape(bz.shape))*10000
bz_fit_diff = (Bz-full_fit_res[len(full_fit_res)/2:].reshape(Bz.shape))*10000
heat = ax.pcolor(X,Y,bz_fit_diff,vmin=-1,vmax=1)
cb = plt.colorbar(heat, aspect=7)
cb.set_label('Data-Fit (G)')
xa.set_xlabel('Y (mm)')
ax.set_xlabel('Y (mm)')
ax.set_ylabel('Z (mm)')
ax.set_title('Residual for Bz (full simulation)')
"""

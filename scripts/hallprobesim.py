#! /usr/bin/env python

import matplotlib.pyplot as plt
from mu2e.datafileprod import DataFileMaker
from mu2e.hallprober import HallProbeGenerator
from mu2e.fieldfitter import FieldFitter
from mu2e.plotter import Plotter

if __name__ == "__main__":

  plt.close('all')
  data_maker1=DataFileMaker('../FieldMapData_1760_v5/Mu2e_DSmap',use_pickle = True)
  hpg = HallProbeGenerator(data_maker1.data_frame.query('X==0 & Z>4000'),z_steps = 100)
  hall_toy = hpg.get_toy()
  ff = FieldFitter(hall_toy)
  #ff.fit_2d('Bz','Y','Z')
  ff.fit_2d_sim('Y','Z')
  plot_maker = Plotter.from_hall_study({'DS_Mau':ff.input_data},fit_result = ff.result)
  #plot_maker.plot_A_v_B_and_C_fit('Bz','Y','Z',False,'X==0')
  fig = plot_maker.plot_A_v_B_and_C_fit('Bz','Y','Z',True,'X==0')
  plot_maker.plot_A_v_B_and_C_fit('Br','Y','Z',True,'X==0')
  #plot_maker.make_gif(fig)






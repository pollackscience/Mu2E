#! /usr/bin/env python

import matplotlib.pyplot as plt
from mu2e.datafileprod import DataFileMaker
from mu2e.hallprober import HallProbeGenerator
from mu2e.fieldfitter import FieldFitter

if __name__ == "__main__":

  plt.close('all')
  data_maker1=DataFileMaker('../FieldMapData_1760_v5/Mu2e_DSmap',use_pickle = True)
  hpg = HallProbeGenerator(data_maker1.data_frame.query('Y==0 & Z < 8000'),z_steps = 100)
  hall_toy = hpg.get_toy()
  ff = FieldFitter(hall_toy)




#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from mu2e.datafileprod import DataFileMaker
from mu2e.hallprober import HallProbeGenerator
from mu2e.fieldfitter import FieldFitter
from mu2e.plotter import Plotter
import pandas as pd
pd.options.mode.chained_assignment = None

#for creating plot.ly plots

def gen_fit_result(magnet = 'DS',A='Y',B='Z',nparams=10,fullsim=False,suffix='halltoy',
        r_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all', phi_steps = (0,np.pi/2), conditions = ('X==0','Z>4000','Z<14000')):
    plt.close('all')
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
    ff.fit_3d_v4(ns=10,ms=50,recreate=True)
    return ff


if __name__ == "__main__":
    pi4r = [35.35533906,   70.71067812,  141.42135624, 176.7766953 ,  212.13203436, 282.84271247,
          353.55339059,  424.26406871, 494.97474683,  530.33008589,601.04076401,  671.75144213]
    phi_steps = (0, np.pi/4, np.pi/2, 3*np.pi/4)
    r_steps = (range(25,625,50), pi4r, range(25,625,50), pi4r)
    magnet = 'DS'
    A = 'R'
    B = 'Z'
    suffix = 'plotly_test'
    conditions =  ('Z>5000','Z<14000','R!=0')

    ff = gen_fit_result(magnet = 'DS',A=A,B=B,nparams=60,fullsim=False,suffix=suffix,
          r_steps = r_steps, phi_steps = phi_steps, z_steps = 'all', conditions = conditions)

    plot_maker = Plotter.from_hall_study({magnet+'_Mau':ff.input_data},fit_result = ff.result)
    plot_maker.extra_suffix = suffix
    plot_maker.plot_A_v_B_and_C_fit_cyl_plotly('Bz',A,B,phi_steps,False,*conditions)
    #print 'finished'


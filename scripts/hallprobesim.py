#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from mu2e.datafileprod import DataFileMaker
from mu2e.hallprober import HallProbeGenerator
from mu2e.fieldfitter import FieldFitter
from mu2e.plotter import Plotter


def hallprobesim(magnet = 'DS',A='Y',B='Z',fullsim=False,suffix='halltoy',
        r_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all', phi_steps = (0,np.pi/2),
        ns = 10, ms = 40, cns = 10, cms = 10, use_pickle = False, pickle_name='default',
        conditions = ('X==0','Z>4000','Z<14000')):
    plt.close('all')
    #data_maker = DataFileMaker('../FieldMapData_1760_v5/Mu2e_'+magnet+'map',use_pickle = True)
    #data_maker=DataFileMaker('../FieldMapsGA04/Mu2e_DS_GA0',use_pickle = True)
    data_maker= DataFileMaker('../Mau10/TS_and_PS_OFF/Mu2e_DSMap',use_pickle = True)
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
    ff.fit_3d_v4(ns=ns,ms=ms,cns=cns,cms=cms,use_pickle=use_pickle,pickle_name = pickle_name)

    if fullsim:
        df = data_maker.data_frame
        df.By = abs(df.By)
        df.Bx = abs(df.Bx)
        plot_maker = Plotter.from_hall_study({magnet+'_Mau':df},fit_result = ff.result)
        plot_maker.extra_suffix = suffix
        plot_maker.plot_A_v_B_and_C_fit('Bz',A,B,sim=True,do_3d=True,do_eval=True,*conditions)
        plot_maker.plot_A_v_B_and_C_fit(Br,A,B,sim=True,do_3d=True,do_eval=True,*conditions)
    else:
        plot_maker = Plotter.from_hall_study({magnet+'_Mau':ff.input_data},fit_result = ff.result)
        plot_maker.extra_suffix = suffix
        plot_maker.plot_A_v_B_and_C_fit_cyl_v2('Bz',A,B,phi_steps,False,*conditions)
        plot_maker.plot_A_v_B_and_C_fit_cyl_v2(Br,A,B,phi_steps,False,*conditions)
        plot_maker.plot_A_v_B_and_C_fit_cyl_v2('Bphi',A,B,phi_steps,False,*conditions)

    return data_maker, hpg, plot_maker, ff


if __name__ == "__main__":

#default four-phi settings
    '''
    pi4r = [35.35533906,   70.71067812,  141.42135624, 176.7766953 ,  212.13203436, 282.84271247,
          353.55339059,  424.26406871, 494.97474683,  530.33008589,601.04076401,  671.75144213]
    phi_steps = (0, np.pi/4, np.pi/2, 3*np.pi/4)
    r_steps = (range(25,625,50), pi4r, range(25,625,50), pi4r)
    data_maker,hpg,plot_maker,ff = hallprobesim(magnet = 'DS',A='R',B='Z',fullsim=False,suffix='halltoy3d_test',
          r_steps = r_steps, phi_steps = phi_steps, z_steps = 'all',
          ns = 10, ms = 40, cns = 10, cms = 10,
          use_pickle = False, pickle_name='four_phi',
          conditions = ('Z>5000','Z<13000','R!=0'))
    '''

#four-phi settings, less R, less Z
    '''
    pi4r = [35.35533906,    70.71067812,   106.06601718,   141.42135624,
            212.13203436,   247.48737342,   282.84271247]
    pi2r = range(25,326,50)

    phi_steps = (0,  np.pi/4, np.pi/2, 3*np.pi/4)
    r_steps = (pi2r, pi4r, pi2r, pi4r)
    data_maker,hpg,plot_maker,ff = hallprobesim(magnet = 'DS',A='R',B='Z',fullsim=False,suffix='halltoy3d_test',
          r_steps = r_steps, phi_steps = phi_steps, z_steps = range(5021,13021,50),
          ns = 10, ms = 40, cns = 10, cms = 10,
          use_pickle = True, pickle_name='four_phi_reduced1',
          conditions = ('Z>5000','Z<13000','R!=0'))
    '''

#eight-phi settings, less R, less Z
    pi8r = [55.90169944,   111.80339887,   167.70509831,   223.60679775,
            279.50849719,   335.41019662,   391.31189606]
    pi4r = [35.35533906,    106.06601718,   141.42135624,
            212.13203436,   282.84271247, 318.19805153, 388.90872965]
    pi2r = range(25,326,50)

    phi_steps = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444,  3*np.pi/4, 2.677945)
    r_steps = (pi2r, pi8r, pi4r, pi8r, pi2r, pi8r, pi4r, pi8r)
    data_maker,hpg,plot_maker,ff = hallprobesim(magnet = 'DS',A='R',B='Z',fullsim=False,suffix='halltoy_DS_only',
          r_steps = r_steps, phi_steps = phi_steps, z_steps = range(5021,13021,50),
          ns = 10, ms = 50, cns =0, cms = 0,
          use_pickle = True, pickle_name='eight_phi',
          conditions = ('Z>5000','Z<13000','R!=0'))

#four-phi settings, half rotation
    '''
    pi8r = [55.90169944,   111.80339887,   167.70509831,   223.60679775,
            279.50849719,   335.41019662,   391.31189606]
    pi4r = [35.35533906,    106.06601718,   141.42135624,
            212.13203436,   282.84271247, 318.19805153, 388.90872965]
    pi2r = range(25,326,50)

    #phi_steps = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444)
    phi_steps = (2.034444,  3*np.pi/4, 2.677945)
    r_steps = (pi8r, pi4r, pi8r)
    data_maker,hpg,plot_maker,ff = hallprobesim(magnet = 'DS',A='R',B='Z',fullsim=False,suffix='halltoy_DS_only',
          r_steps = r_steps, phi_steps = phi_steps, z_steps = range(5021,13021,50),
          ns = 10, ms = 50, cns =0, cms = 0,
          use_pickle = True, pickle_name='three_phi',
          conditions = ('Z>5000','Z<13000','R!=0'))
    '''

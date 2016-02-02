#! /usr/bin/env python

#import matplotlib.pyplot as plt
#import numpy as np
#from mu2e.datafileprod import DataFileMaker
from mu2e.hallprober import *
#from mu2e.fieldfitter import FieldFitter
#from mu2e.plotter import Plotter

from collections import namedtuple

cfg_data = namedtuple('cfg_data', 'datatype magnet path conditions')
cfg_geom = namedtuple('cfg_geom', 'geom z_steps r_steps phi_steps xy_steps')
cfg_params = namedtuple('cfg_params', 'ns ms cns cms Reff a b c')
cfg_pickle = namedtuple('cfg_pickle', 'use_pickle save_pickle load_name save_name recreate')
cfg_plot = namedtuple('cfg_plot', 'plot_type')

cfg_data_Mau10 = cfg_data('Mau10', 'DS', '../Mau10/Standard_Maps/Mu2e_DSMap',('Z>5000','Z<13000','R!=0'))


field_map_analysis('halltoy_600mm', cfg_data_Mau10,

def solenoid_field_cyl(magnet = 'DS',fullsim=False,suffix='halltoy',
        r_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all', phi_steps = (0,np.pi/2),
        ns = 10, ms = 40, use_pickle = False, pickle_name='default',
        conditions = ('X==0','Z>4000','Z<14000'), recreate=False ):
    plt.close('all')
    #data_maker= DataFileMaker('../Mau10/TS_and_PS_OFF/Mu2e_DSMap',use_pickle = True)
    #data_maker= DataFileMaker('../Mau10/DS_OFF/Mu2e_DSMap',use_pickle = True)
    #data_maker = DataFileMaker('../Mau10/Standard_Maps/Mu2e_DSMap',use_pickle = True)
    input_data = DataFileMaker('../FieldMapsGA04/Mu2e_DS_GA0',use_pickle = True)
    input_data = data_maker.data_frame
    for condition in conditions:
        input_data = input_data.query(condition)
    hpg = HallProbeGenerator(input_data, z_steps = z_steps, r_steps = r_steps, phi_steps = phi_steps)
    toy = hpg.get_toy()

    ff = FieldFitter(toy,phi_steps,r_steps)
    ff.fit_solenoid(ns=ns,ms=ms,use_pickle=use_pickle,pickle_name = pickle_name,recreate=recreate)

    plot_maker = Plotter.from_hall_study({magnet+'_GA04':ff.input_data},fit_result = ff.result)

    make_fit_plots(plot_maker, plot_type, suffix, geom,
        phi_steps = phi_steps, conditions = conditions)


    return data_maker, hpg, plot_maker, ff

def external_field_cart(magnet = 'DS',fullsim=False,suffix='halltoy',
        xy_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all',
        cns = 10, cms = 10, use_pickle = False, pickle_name='default',
        conditions = ('Z>4000','Z<14000'), recreate=False ):

    plt.close('all')
    #data_maker = DataFileMaker('../FieldMapData_1760_v5/Mu2e_'+magnet+'map',use_pickle = True)
    #data_maker=DataFileMaker('../FieldMapsGA04/Mu2e_DS_GA0',use_pickle = True)
    #data_maker= DataFileMaker('../Mau10/TS_and_PS_OFF/Mu2e_DSMap',use_pickle = True)
    data_maker= DataFileMaker('../Mau10/DS_OFF/Mu2e_DSMap',use_pickle = True)
    #data_maker = DataFileMaker('../Mau10/Standard_Maps/Mu2e_DSMap',use_pickle = True)
    input_data = data_maker.data_frame
    for condition in conditions:
        input_data = input_data.query(condition)
    hpg = HallProbeGenerator(input_data, z_steps = z_steps, x_steps = xy_steps, y_steps = xy_steps)
    toy = hpg.get_toy()

    ff = FieldFitter(toy,xy_steps=xy_steps)
    ff.fit_external(cns=cns,cms=cms,use_pickle=use_pickle,pickle_name = pickle_name,recreate=recreate)

    if recreate:
        plot_maker = Plotter.from_hall_study({magnet+'_Mau10':ff.input_data},fit_result = ff.result)
        plot_maker.extra_suffix = suffix
        plot_maker.plot_A_v_B_and_C_fit_ext_plotly('Bz','X','Z',xy_steps,False,*conditions)
        plot_maker.plot_A_v_B_and_C_fit_ext_plotly('Bx','X','Z',xy_steps,False,*conditions)
        plot_maker.plot_A_v_B_and_C_fit_ext_plotly('By','X','Z',xy_steps,False,*conditions)
    elif fullsim:
        pass
#fix this
        #df = data_maker.data_frame
        #df.By = abs(df.By)
        #df.Bx = abs(df.Bx)
        #plot_maker = Plotter.from_hall_study({magnet+'_Mau10':df},fit_result = ff.result)
        #plot_maker.extra_suffix = suffix
        #plot_maker.plot_A_v_B_and_C_fit('Bz',A,B,sim=True,do_3d=True,do_eval=True,*conditions)
        #plot_maker.plot_A_v_B_and_C_fit(Br,A,B,sim=True,do_3d=True,do_eval=True,*conditions)
    else:
        plot_maker = Plotter.from_hall_study({magnet+'_Mau10':ff.input_data},fit_result = ff.result)
        plot_maker.extra_suffix = suffix
        plot_maker.plot_A_v_B_and_C_fit_ext('Bz','X','Z',xy_steps,False,*conditions)
        plot_maker.plot_A_v_B_and_C_fit_ext('Bx','X','Z',xy_steps,False,*conditions)
        plot_maker.plot_A_v_B_and_C_fit_ext('By','X','Z',xy_steps,False,*conditions)

    return data_maker, hpg, plot_maker, ff

def full_field_cyl(magnet = 'DS',fullsim=False,suffix='halltoy',
        r_steps = (-825,-650,-475,-325,0,325,475,650,825), z_steps = 'all', phi_steps = (0,np.pi/2),
        ns = 7, ms = 40, cns= 7, cms = 7, use_pickle = True, pickle_name='default',
        conditions = ('R!=0','Z>5000','Z<13000'), recreate=False ):
    plt.close('all')
    #data_maker= DataFileMaker('../Mau10/TS_and_PS_OFF/Mu2e_DSMap',use_pickle = True)
    #data_maker= DataFileMaker('../Mau10/DS_OFF/Mu2e_DSMap',use_pickle = True)
    data_maker = DataFileMaker('../Mau10/Standard_Maps/Mu2e_DSMap',use_pickle = True)
    input_data = data_maker.data_frame
    for condition in conditions:
        input_data = input_data.query(condition)
    hpg = HallProbeGenerator(input_data, z_steps = z_steps, r_steps = r_steps, phi_steps = phi_steps)
    toy = hpg.get_toy()

    ff = FieldFitter(toy,phi_steps,r_steps)
    ff.fit_full(ns=ns, ms=ms, cns=cns, cms=cms,
            use_pickle=use_pickle, pickle_name=pickle_name, recreate=recreate)

    if recreate:
        plot_maker = Plotter.from_hall_study({magnet+'_Mau10':ff.input_data},fit_result = ff.result)
        plot_maker.extra_suffix = suffix
        plot_maker.plot_A_v_B_and_C_fit_cyl_plotly('Bz','R','Z',phi_steps,False,*conditions)
        plot_maker.plot_A_v_B_and_C_fit_cyl_plotly('Br','R','Z',phi_steps,False,*conditions)
        plot_maker.plot_A_v_B_and_C_fit_cyl_plotly('Bphi','R','Z',phi_steps,False,*conditions)
    elif fullsim:
        pass
#fix this
        #df = data_maker.data_frame
        #df.By = abs(df.By)
        #df.Bx = abs(df.Bx)
        #plot_maker = Plotter.from_hall_study({magnet+'_Mau10':df},fit_result = ff.result)
        #plot_maker.extra_suffix = suffix
        #plot_maker.plot_A_v_B_and_C_fit('Bz',A,B,sim=True,do_3d=True,do_eval=True,*conditions)
        #plot_maker.plot_A_v_B_and_C_fit(Br,A,B,sim=True,do_3d=True,do_eval=True,*conditions)
    else:
        plot_maker = Plotter.from_hall_study({magnet+'_Mau10':ff.input_data},fit_result = ff.result)
        plot_maker.extra_suffix = suffix
        plot_maker.plot_A_v_B_and_C_fit_cyl_v2('Bz','R','Z',phi_steps,False,*conditions)
        plot_maker.plot_A_v_B_and_C_fit_cyl_v2('Br','R','Z',phi_steps,False,*conditions)
        plot_maker.plot_A_v_B_and_C_fit_cyl_v2('Bphi','R','Z',phi_steps,False,*conditions)

    return data_maker, hpg, plot_maker, ff


if __name__ == "__main__":

#eight-phi settings, DS only, R values similar to hall probe
    pi8r = [55.90169944, 167.70509831, 279.50849719, 447.2135955, 614.91869381]
    pi4r = [35.35533906, 141.42135624, 318.19805153, 494.97474683, 601.04076401]
    pi2r = [25,150,325,475,600]

    #phi_steps = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444,  3*np.pi/4, 2.677945)
    #r_steps = (pi2r, pi8r, pi4r, pi8r, pi2r, pi8r, pi4r, pi8r)
    phi_steps = (0, 0.463648, np.pi/4, 1.107149)
    r_steps = (pi2r, pi8r, pi4r, pi8r)
    #data_maker,hpg,plot_maker,ff = solenoid_field_cyl(magnet = 'DS',fullsim=False,suffix='halltoy_full_Mau10_v2',
    data_maker,hpg,plot_maker,ff = solenoid_field_cyl(magnet = 'DS',fullsim=False,suffix='halltoy_full_GA04',
          r_steps = r_steps, phi_steps = phi_steps, z_steps = range(5021,13021,50),
          ns = 10, ms = 50,
          #use_pickle = False, pickle_name='eight_phi_full_Mau10_v2',
          use_pickle = False, pickle_name='eight_phi_full_GA04',
          conditions = ('Z>5000','Z<13000','R!=0'),recreate=False)
#cartestian settings, external field
    '''
    xy_steps = [-600,-450,-300,-150,0,150,300,450,600]
    data_maker,hpg,plot_maker,ff = external_field_cart(magnet = 'DS',fullsim=False,suffix='halltoy_ext_only',
          xy_steps = xy_steps, z_steps = range(5021,13021,50),
          cns = 7, cms = 7,
          #use_pickle = True, pickle_name='eight_phi_and_ext',
          use_pickle = False, pickle_name='cart_ext_only',
          conditions = ('Z>5000','Z<13000'),recreate=False)
    '''

#eight-phi settings, DS and External combined fit, R values similar to hall probe
    '''
    pi8r = [55.90169944, 167.70509831, 279.50849719, 447.2135955, 614.91869381]
    pi4r = [35.35533906, 141.42135624, 318.19805153, 494.97474683, 601.04076401]
    pi2r = [25,150,325,475,600]

    phi_steps = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444,  3*np.pi/4, 2.677945)
    r_steps = (pi2r, pi8r, pi4r, pi8r, pi2r, pi8r, pi4r, pi8r)
    #phi_steps = (0, 0.463648, np.pi/4, 1.107149)
    #r_steps = (pi2r, pi8r, pi4r, pi8r)
    data_maker,hpg,plot_maker,ff = full_field_cyl(magnet = 'DS',fullsim=False,suffix='halltoy_full',
          r_steps = r_steps, phi_steps = phi_steps, z_steps = range(5021,13021,50),
          ns = 7, ms = 40, cns=7, cms=7,
          use_pickle = True, pickle_name=['eight_phi_DS_only_probeRs','cart_ext_only'],
          conditions = ('Z>5000','Z<13000','R!=0'),recreate=False)
    '''

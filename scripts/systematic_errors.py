#! /usr/bin/env python

from mu2e.hallprober import field_map_analysis
from collections import namedtuple
import numpy as np
from mu2e import mu2e_ext_path
import cPickle as pkl


cfg_data   = namedtuple('cfg_data', 'datatype magnet path conditions')
cfg_geom   = namedtuple('cfg_geom', 'geom z_steps r_steps phi_steps xy_steps bad_calibration '
                        'interpolate')
cfg_params = namedtuple('cfg_params', 'ns ms cns cms Reff func_version')
cfg_pickle = namedtuple('cfg_pickle', 'use_pickle save_pickle load_name save_name recreate')
cfg_plot   = namedtuple('cfg_plot', 'plot_type zlims save_loc sub_dir')

syst_set = 20

path_DS_Mau10          = mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap'
cfg_data_DS_Mau10_long = cfg_data('Mau10', 'DS', path_DS_Mau10, ('Z>4200', 'Z<13900', 'R!=0'))

z_steps_DS_fullsim    = range(4221, 13921, 25)
pi8r_fullsim800       = [55.90169944, 111.80339887, 167.70509831, 223.60679775, 279.50849719,
                         335.41019663, 391.31189606, 447.2135955, 503.11529494, 559.01699437,
                         614.91869381, 670.82039325, 726.72209269, 782.62379213]
pi4r_fullsim800       = [35.35533906, 106.06601718, 176.7766953, 247.48737342, 282.84271247,
                         318.19805153, 388.90872965, 424.26406871, 494.97474683, 565.68542495,
                         601.04076401, 671.75144213, 742.46212025, 813.17279836]
pi2r_fullsim800       = [25, 75, 125, 175, 225, 275, 325, 400, 475, 525, 575, 625, 700, 800]
r_steps_fullsim_trunc = (pi2r_fullsim800, pi8r_fullsim800, pi4r_fullsim800, pi8r_fullsim800,
                         pi2r_fullsim800, pi8r_fullsim800, pi4r_fullsim800, pi8r_fullsim800)
phi_steps_8 = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444, 3*np.pi/4, 2.677945)

cfg_geom_cyl_fullsim_trunc  = cfg_geom('cyl', z_steps_DS_fullsim, r_steps_fullsim_trunc,
                                       phi_steps_8, xy_steps=None,
                                       bad_calibration=[False, False, False],
                                       interpolate=False)

cfg_params_Mau_DS_800mm_long = cfg_params(ns=3, ms=70, cns=0, cms=0, Reff=7000, func_version=5)


cfg_pickle_set_Mau_bad_m     = [cfg_pickle(use_pickle=True, save_pickle=True,
                                           load_name='Mau10_bad_m_{}'.format(i),
                                           save_name='Mau10_bad_m_{}'.format(i), recreate=True) for
                                i in range(syst_set)]

cfg_plot_mpl = cfg_plot('mpl', [-2, 2], 'html', None)


# for i in range(syst_set):
df_set = []
for i in range(1):
    _, ff = field_map_analysis('halltoy_Mau10_800mm_bad_m_full_{}'.format(i),
                               cfg_data_DS_Mau10_long, cfg_geom_cyl_fullsim_trunc,
                               cfg_params_Mau_DS_800mm_long, cfg_pickle_set_Mau_bad_m[i],
                               cfg_plot_mpl)
    df_set.append(ff.input_data)

pkl.dump(df_set, open('df_set.p', 'wb'), protocol=pkl.HIGHEST_PROTOCOL)

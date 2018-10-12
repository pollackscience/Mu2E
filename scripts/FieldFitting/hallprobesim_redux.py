#! /usr/bin/env python

from __future__ import absolute_import
from mu2e.hallprober import field_map_analysis
from collections import namedtuple
import numpy as np
from mu2e import mu2e_ext_path
from mu2e.src.make_csv import make_csv
from six.moves import range

############################
# defining the cfg structs #
############################

cfg_data   = namedtuple('cfg_data', 'datatype magnet path conditions')
cfg_geom   = namedtuple('cfg_geom', 'geom z_steps r_steps phi_steps x_steps y_steps '
                        'bad_calibration interpolate do2pi')
cfg_plot   = namedtuple('cfg_plot', 'plot_type zlims save_loc sub_dir')
cfg_params = namedtuple('cfg_params', 'pitch1 ms_h1 ns_h1 pitch2 ms_h2 ns_h2 '
                        ' length1 ms_c1 ns_c1 length2 ms_c2 ns_c2 version')
cfg_pickle = namedtuple('cfg_pickle', 'use_pickle save_pickle load_name save_name recreate')

#################
# the data cfgs #
#################

path_DS_Mau13       = mu2e_ext_path+'datafiles/Mau13/Mu2e_DSMap_V13'
path_Cole_250mm_long_cyl = mu2e_ext_path +\
    'datafiles/FieldMapsCole/10x_bfield_map_cylin_985152pts_09-20_162454'
path_Cole_250mm_long_cyl_hg = mu2e_ext_path +\
    'datafiles/FieldMapsCole/10x_high_granularity_cylin_3846784pts_r250mm_p10cm_10-06_004607'

cfg_data_DS_Mau13        = cfg_data('Mau13', 'DS', path_DS_Mau13,
                                    ('Z>4.200', 'Z<13.900', 'R!=0'))

cfg_data_Cole_250mm_long_cyl  = cfg_data('Cole', 'DS', path_Cole_250mm_long_cyl,
                                         ('Z>-3.5', 'Z<3.5', 'R!=0'))
cfg_data_Cole_250mm_long_cyl_hg = cfg_data('Cole', 'DS', path_Cole_250mm_long_cyl_hg,
                                           ('Z>-3.5', 'Z<3.5', 'R!=0'))

#################
# the geom cfgs #
#################
# For cartesian DS
pi8r_800mm = [0.05590169944, 0.16770509831, 0.33541019663, 0.55901699437, 0.78262379213]
pi4r_800mm = [0.03535533906, 0.1767766953, 0.35355339059, 0.53033008589, 0.81317279836]
pi2r_800mm = [0.025, 0.175, 0.375, 0.525, 0.800]
r_steps_800mm = (pi2r_800mm, pi8r_800mm, pi4r_800mm, pi8r_800mm,
                 pi2r_800mm, pi8r_800mm, pi4r_800mm, pi8r_800mm)

# For cylindrical Cole 250 maps
piall_250mm = [0.0125, 0.0375, 0.0625, 0.0875, 0.1125, 0.150]
r_steps_250mm_true = (piall_250mm,)*8

piall_250mm_hg = [0.00625, 0.0125, 0.01875, 0.025, 0.03125, 0.0375, 0.04375,
                  0.05, 0.05625, 0.0625, 0.06875, 0.075, 0.08125, 0.0875,
                  0.09375, 0.1, 0.10625, 0.1125, 0.11875, 0.125, 0.13125,
                  0.1375, 0.14375, 0.15]
r_steps_250mm_true_hg = (piall_250mm,)*64

# For all maps
phi_steps_8 = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444, 3*np.pi/4, 2.677945)
phi_steps_true = (0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8)
phi_steps_true_hg = [0., 0.0491, 0.0982, 0.1473, 0.1963, 0.2454, 0.2945, 0.3436,
                     0.3927, 0.4418, 0.4909, 0.54, 0.589, 0.6381, 0.6872, 0.7363,
                     0.7854, 0.8345, 0.8836, 0.9327, 0.9817, 1.0308, 1.0799, 1.129,
                     1.1781, 1.2272, 1.2763, 1.3254, 1.3744, 1.4235, 1.4726, 1.5217,
                     1.5708, 1.6199, 1.669, 1.7181, 1.7671, 1.8162, 1.8653, 1.9144,
                     1.9635, 2.0126, 2.0617, 2.1108, 2.1598, 2.2089, 2.258, 2.3071,
                     2.3562, 2.4053, 2.4544, 2.5035, 2.5525, 2.6016, 2.6507, 2.6998,
                     2.7489, 2.798, 2.8471, 2.8962, 2.9452, 2.9943, 3.0434, 3.0925]


z_steps_DS_long = [i*1e-3 for i in range(4221, 13921, 50)]
z_steps_cole_small = [i*1e-3 for i in range(-1500, 1500, 25)]

# Actual configs
cfg_geom_cyl_800mm_long         = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           bad_calibration=[False, False, False], interpolate=False,
                                           do2pi=False)

cfg_geom_Cole_250mm_long_cyl     = cfg_geom('cyl', z_steps_cole_small, r_steps_250mm_true[0:],
                                            phi_steps_true[0:], x_steps=None, y_steps=None,
                                            bad_calibration=[False, False, False],
                                            interpolate=False, do2pi=True)

cfg_geom_Cole_250mm_long_cyl_hg  = cfg_geom('cyl', z_steps_cole_small, r_steps_250mm_true_hg[0:],
                                            phi_steps_true_hg[0:], x_steps=None, y_steps=None,
                                            bad_calibration=[False, False, False],
                                            interpolate=False, do2pi=True)

#################
# the plot cfgs #
#################
cfg_plot_none = cfg_plot('none', [-2, 2], 'html', None)
cfg_plot_mpl = cfg_plot('mpl', [-2, 2], 'html', None)
cfg_plot_mpl_high_lim = cfg_plot('mpl', [-10, 10], 'html', None)
cfg_plot_plotly_img = cfg_plot('plotly_html_img', [-2, 2], 'html', None)
cfg_plot_plotly_html = cfg_plot('plotly_html', [-2, 2], 'html', None)
cfg_plot_plotly_high_lim = cfg_plot('plotly', [-10, 10], 'html', None)

##############################
# the params and pickle cfgs #
##############################
cfg_params_DS_Mau13                 = cfg_params(pitch1=0.0075, ms_h1=0, ns_h1=0,
                                                 pitch2=0, ms_h2=0, ns_h2=0,
                                                 length1=0, ms_c1=0, ns_c1=0,
                                                 length2=9, ms_c2=20, ns_c2=3,
                                                 version=1000)
cfg_pickle_Mau13                    = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Mau13',
                                                 save_name='Mau13', recreate=False)

cfg_params_Cole_Hel                 = cfg_params(pitch1=0.1, ms_h1=4, ns_h1=6,
                                                 pitch2=0, ms_h2=0, ns_h2=0,
                                                 length1=0, ms_c1=0, ns_c1=0,
                                                 length2=0, ms_c2=0, ns_c2=0,
                                                 version=1000)
cfg_pickle_Cole_Hel                 = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Cole_Hel',
                                                 save_name='Cole_Hel', recreate=False)

cfg_params_Cole_Cyl                 = cfg_params(pitch1=0, ms_h1=0, ns_h1=0,
                                                 pitch2=0, ms_h2=0, ns_h2=0,
                                                 length1=0.05, ms_c1=1, ns_c1=60,
                                                 length2=0, ms_c2=0, ns_c2=0,
                                                 version=1000)
cfg_pickle_Cole_Cyl                 = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Cole_Cyl',
                                                 save_name='Cole_Cyl', recreate=False)

if __name__ == "__main__":

    # hmd, ff = field_map_analysis('fma_cole_hel', cfg_data_Cole_250mm_long_cyl,
    #                              cfg_geom_Cole_250mm_long_cyl, cfg_params_Cole_Hel,
    #                              cfg_pickle_Cole_Hel, cfg_plot_mpl)

    hmd, ff = field_map_analysis('fma_cole_cyl', cfg_data_Cole_250mm_long_cyl,
                                 cfg_geom_Cole_250mm_long_cyl, cfg_params_Cole_Cyl,
                                 cfg_pickle_Cole_Cyl, cfg_plot_mpl)

    # hmd, ff = field_map_analysis('fma_mau13', cfg_data_DS_Mau13,
    #                              cfg_geom_cyl_800mm_long, cfg_params_DS_Mau13,
    #                              cfg_pickle_Mau13, cfg_plot_mpl)

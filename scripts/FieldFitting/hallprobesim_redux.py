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

cfg_data_DS_Mau13        = cfg_data('Mau13', 'DS', path_DS_Mau13,
                                    ('Z>4200', 'Z<13900', 'R!=0'))

cfg_data_Cole_250mm_long_cyl  = cfg_data('Cole', 'DS', path_Cole_250mm_long_cyl,
                                         ('Z>-3.5', 'Z<3.5', 'R!=0'))

#################
# the geom cfgs #
#################
# For cartesian DS
pi8r_800mm = [55.90169944, 167.70509831, 335.41019663, 559.01699437, 782.62379213]
pi4r_800mm = [35.35533906, 176.7766953, 353.55339059, 530.33008589, 813.17279836]
pi2r_800mm = [25, 175, 375, 525, 800]
r_steps_800mm = (pi2r_800mm, pi8r_800mm, pi4r_800mm, pi8r_800mm,
                 pi2r_800mm, pi8r_800mm, pi4r_800mm, pi8r_800mm)

# For cylindrical Cole 250 maps
piall_250mm = [0.0125, 0.0375, 0.0625, 0.0875, 0.1125, 0.150]
r_steps_250mm_true = (piall_250mm,)*8

# For all maps
phi_steps_8 = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444, 3*np.pi/4, 2.677945)
phi_steps_true = (0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8)

z_steps_DS_long = list(range(4221, 13921, 50))
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
cfg_params_DS_Mau13                 = cfg_params(pitch1=0.075, ms_h1=4, ns_h1=4,
                                                 pitch2=0, ms_h2=0, ns_h2=0,
                                                 length1=0, ms_c1=0, ns_c1=0,
                                                 length2=15, ms_c2=5, ns_c2=5,
                                                 version=1000)
cfg_pickle_Mau13                    = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Mau13',
                                                 save_name='Mau13', recreate=False)

cfg_params_Cole_Hel                 = cfg_params(pitch1=0.1, ms_h1=1, ns_h1=2,
                                                 pitch2=0, ms_h2=0, ns_h2=0,
                                                 length1=0, ms_c1=0, ns_c1=0,
                                                 length2=0, ms_c2=0, ns_c2=0,
                                                 version=1000)
cfg_pickle_Cole_Hel                 = cfg_pickle(use_pickle=False, save_pickle=True,
                                                 load_name='Cole_Hel',
                                                 save_name='Cole_Hel', recreate=False)

cfg_params_Cole_Cyl                 = cfg_params(pitch1=0, ms_h1=0, ns_h1=0,
                                                 pitch2=0, ms_h2=0, ns_h2=0,
                                                 length1=0.1, ms_c1=2, ns_c1=2,
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

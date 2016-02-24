#! /usr/bin/env python

from mu2e.hallprober import field_map_analysis
from collections import namedtuple
import numpy as np

############################
# defining the cfg structs #
############################
cfg_data = namedtuple('cfg_data', 'datatype magnet path conditions')
cfg_geom = namedtuple('cfg_geom', 'geom z_steps r_steps phi_steps xy_steps bad_calibration')
cfg_params = namedtuple('cfg_params', 'ns ms cns cms Reff a b c')
cfg_pickle = namedtuple('cfg_pickle', 'use_pickle save_pickle load_name save_name recreate')
cfg_plot = namedtuple('cfg_plot', 'plot_type zlims html_loc')

#################
# the data cfgs #
#################
cfg_data_DS_Mau10 = cfg_data('Mau10', 'DS', '../Mau10/Standard_Maps/Mu2e_DSMap', ('Z>5000','Z<13000','R!=0'))
cfg_data_DS2_Mau10 = cfg_data('Mau10', 'DS', '../Mau10/Standard_Maps/Mu2e_DSMap', ('Z>4000','Z<14000','R!=0'))
cfg_data_PS_Mau10 = cfg_data('Mau10', 'PS', '../Mau10/Standard_Maps/Mu2e_PSMap', ('Z>-7900','Z<-4000','R!=0'))


cfg_data_DS_GA05_no_ext = cfg_data('GA05', 'DS', '../FieldMapsGA_Special/Mu2e_DS_noPSTS_GA0', ('Z>5000','Z<13000','R!=0'))
cfg_data_DS_GA05_no_DS = cfg_data('GA05', 'DS', '../FieldMapsGA_Special/Mu2e_DS_noDS_GA0', ('Z>5000','Z<13000','R!=0'))
#################
# the geom cfgs #
#################
#DS
pi8r_600mm = [55.90169944, 167.70509831, 279.50849719, 447.2135955, 614.91869381]
pi4r_600mm = [35.35533906, 141.42135624, 318.19805153, 494.97474683, 601.04076401]
pi2r_600mm = [25,150,325,475,600]

pi8r_700mm = [55.90169944, 167.70509831, 335.41019663, 559.01699437, 726.72209269]
pi4r_700mm = [35.35533906, 176.7766953, 353.55339059, 530.33008589, 707.10678119]
pi2r_700mm = [25,175,375,525,700]

pi8r_825mm_v1 = [55.90169944, 167.70509831, 335.41019663, 559.01699437, 838.52549156]
pi4r_825mm_v1 = [35.35533906, 176.7766953, 353.55339059, 530.33008589, 813.17279836]
pi2r_825mm_v1 = [25,175,375,525,825]

pi8r_825mm_v2 = [55.90169944, 335.41019663, 503.11529494, 726.72209269, 838.52549156]
pi4r_825mm_v2 = [35.35533906, 318.19805153, 494.97474683, 707.10678119, 813.17279836]
pi2r_825mm_v2 = [25,325,500,725,825]

pi8r_800mm = [55.90169944, 335.41019663, 503.11529494, 726.72209269, 782.62379213]
pi4r_800mm = [35.35533906, 318.19805153, 494.97474683, 707.10678119, 813.17279836]
pi2r_800mm = [25,325,500,725,800]


pi8r_fullsim = [55.90169944, 111.80339887, 167.70509831, 223.60679775, 279.50849719,
        335.41019663, 391.31189606, 447.2135955 ,503.11529494, 559.01699437,
        614.91869381, 670.82039325, 726.72209269, 782.62379213, 838.52549156, 894.427191,1006.23058987]
pi4r_fullsim = [35.35533906, 106.06601718, 176.7766953, 247.48737342, 282.84271247,
        318.19805153, 388.90872965, 424.26406871, 494.97474683, 565.68542495,
        601.04076401, 671.75144213, 742.46212025, 777.81745931, 813.17279836, 883.88347648,989.94949366]
pi2r_fullsim = [25, 75, 125, 175, 225, 275, 325, 400, 475, 525, 575, 625, 675, 725, 800,900,1000]

#PS
pi8r_150mm = [55.90169944, 111.80339887, 167.70509831]
pi4r_150mm = [35.35533906, 106.06601718, 141.42135624]
pi2r_150mm = [25,100,150]

pi8r_150mm = [55.90169944, 111.80339887, 167.70509831]
pi4r_150mm = [35.35533906, 106.06601718, 141.42135624]
pi2r_150mm = [25,100,150]



r_steps_150mm = (pi2r_150mm, pi8r_150mm, pi4r_150mm, pi8r_150mm,
        pi2r_150mm, pi8r_150mm, pi4r_150mm, pi8r_150mm)
r_steps_600mm = (pi2r_600mm, pi8r_600mm, pi4r_600mm, pi8r_600mm,
        pi2r_600mm, pi8r_600mm, pi4r_600mm, pi8r_600mm)
r_steps_700mm = (pi2r_700mm, pi8r_700mm, pi4r_700mm, pi8r_700mm,
        pi2r_700mm, pi8r_700mm, pi4r_700mm, pi8r_700mm)

r_steps_825mm_v1 = (pi2r_825mm_v1, pi8r_825mm_v1, pi4r_825mm_v1, pi8r_825mm_v1,
        pi2r_825mm_v1, pi8r_825mm_v1, pi4r_825mm_v1, pi8r_825mm_v1)
r_steps_825mm_v2 = (pi2r_825mm_v2, pi8r_825mm_v2, pi4r_825mm_v2, pi8r_825mm_v2,
        pi2r_825mm_v2, pi8r_825mm_v2, pi4r_825mm_v2, pi8r_825mm_v2)

r_steps_800mm = (pi2r_800mm, pi8r_800mm, pi4r_800mm, pi8r_800mm,
        pi2r_800mm, pi8r_800mm, pi4r_800mm, pi8r_800mm)

r_steps_fullsim = (pi2r_fullsim, pi8r_fullsim, pi4r_fullsim, pi8r_fullsim,
        pi2r_fullsim, pi8r_fullsim, pi4r_fullsim, pi8r_fullsim)
r_steps_fullsim_trunc = (pi2r_fullsim[:-2], pi8r_fullsim[:-2], pi4r_fullsim[:-2], pi8r_fullsim[:-2],
        pi2r_fullsim[:-2], pi8r_fullsim[:-2], pi4r_fullsim[:-2], pi8r_fullsim[:-2])

phi_steps_8 = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444,  3*np.pi/4, 2.677945)

z_steps_DS = range(5021,13021,50)
z_steps_DS2 = range(4171,13921,50)
z_steps_DS_fullsim = range(5021,13021,25)
z_steps_DS_fullsim2 = range(4171,13921,25)
z_steps_PS = range(-7879,-4004,50)


#DS
cfg_geom_cyl_600mm = cfg_geom('cyl',z_steps_DS, r_steps_600mm, phi_steps_8, xy_steps = None, bad_calibration = [False, False, False])
cfg_geom_cyl_700mm = cfg_geom('cyl',z_steps_DS, r_steps_700mm, phi_steps_8, xy_steps = None, bad_calibration = [False, False, False])
cfg_geom_cyl_825mm_v1 = cfg_geom('cyl',z_steps_DS, r_steps_825mm_v1, phi_steps_8, xy_steps = None, bad_calibration = [False, False, False])
cfg_geom_cyl_825mm_v2 = cfg_geom('cyl',z_steps_DS, r_steps_825mm_v2, phi_steps_8, xy_steps = None, bad_calibration = [False, False, False])
cfg_geom_cyl_800mm = cfg_geom('cyl',z_steps_DS2, r_steps_800mm, phi_steps_8, xy_steps = None, bad_calibration = [False, False, False])
cfg_geom_cyl_fullsim = cfg_geom('cyl',z_steps_DS_fullsim, r_steps_fullsim, phi_steps_8, xy_steps = None, bad_calibration = [False, False, False])
cfg_geom_cyl_fullsim_trunc = cfg_geom('cyl',z_steps_DS_fullsim, r_steps_fullsim_trunc, phi_steps_8, xy_steps = None, bad_calibration = [False, False, False])
cfg_geom_cyl_fullsim2 = cfg_geom('cyl',z_steps_DS_fullsim2, r_steps_fullsim, phi_steps_8, xy_steps = None, bad_calibration = [False, False, False])
cfg_geom_cyl_bad_measure_v1 = cfg_geom('cyl',z_steps_DS, r_steps_825mm_v1, phi_steps_8, xy_steps = None, bad_calibration = [True, False, False])
cfg_geom_cyl_bad_position_v1 = cfg_geom('cyl',z_steps_DS, r_steps_825mm_v1, phi_steps_8, xy_steps = None, bad_calibration = [False, True, False])
cfg_geom_cyl_bad_rotation_v1 = cfg_geom('cyl',z_steps_DS, r_steps_825mm_v1, phi_steps_8, xy_steps = None, bad_calibration = [False, False, True])
cfg_geom_cyl_bad_rotation2_v1 = cfg_geom('cyl',z_steps_DS, r_steps_825mm_v1, phi_steps_8, xy_steps = None, bad_calibration = [False, False, True])
cfg_geom_cyl_bad_measure_v2 = cfg_geom('cyl',z_steps_DS, r_steps_825mm_v2, phi_steps_8, xy_steps = None, bad_calibration = [True, False, False])
cfg_geom_cyl_bad_position_v2 = cfg_geom('cyl',z_steps_DS, r_steps_825mm_v2, phi_steps_8, xy_steps = None, bad_calibration = [False, True, False])
cfg_geom_cyl_bad_rotation_v2 = cfg_geom('cyl',z_steps_DS, r_steps_825mm_v2, phi_steps_8, xy_steps = None, bad_calibration = [False, False, True])
#PS
cfg_geom_cyl_150mm = cfg_geom('cyl',z_steps_PS, r_steps_150mm, phi_steps_8, xy_steps = None, bad_calibration = [False, False, False])


###################
# the params cfgs #
###################
cfg_params_Mau_DS_opt = cfg_params(ns = 3, ms = 40, cns = 0, cms=0, Reff = 7000, a=None,b=None,c=None)
cfg_params_Mau_DS_825mm_v1 = cfg_params(ns = 3, ms = 80, cns = 0, cms=0, Reff = 7000, a=None,b=None,c=None)
cfg_params_Mau_DS_825mm_v2 = cfg_params(ns = 3, ms = 80, cns = 0, cms=0, Reff = 7000, a=None,b=None,c=None)
cfg_params_Mau_DS_800mm = cfg_params(ns = 3, ms = 80, cns = 0, cms=0, Reff = 7000, a=None,b=None,c=None)
cfg_params_Mau_DS_700 = cfg_params(ns = 3, ms = 70, cns = 0, cms=0, Reff = 7000, a=None,b=None,c=None)
cfg_params_Mau_DS_bad = cfg_params(ns = 3, ms = 80, cns = 0, cms=0, Reff = 7000, a=None,b=None,c=None)

cfg_params_GA05_DS_opt = cfg_params(ns = 6, ms = 40, cns = 0, cms=0, Reff = 7000, a=None,b=None,c=None)

cfg_params_Mau_PS_opt = cfg_params(ns = 3, ms = 40, cns = 0, cms=0, Reff = 9000, a=None,b=None,c=None)

###################
# the pickle cfgs #
###################
cfg_pickle_new_Mau = cfg_pickle(use_pickle = False, save_pickle = True, load_name = None, save_name = 'Mau10_opt', recreate = False)
cfg_pickle_Mau_700 = cfg_pickle(use_pickle = True, save_pickle = False, load_name = 'Mau10_700', save_name = 'Mau10_700', recreate = True)

cfg_pickle_Mau_825mm_v1 = cfg_pickle(use_pickle = True, save_pickle = True, load_name = 'Mau10_825mm_v1', save_name = 'Mau10_825mm_v1', recreate = True)
cfg_pickle_Mau_825mm_v2 = cfg_pickle(use_pickle = True, save_pickle = True, load_name = 'Mau10_825mm_v2', save_name = 'Mau10_825mm_v2', recreate = True)

cfg_pickle_Mau_800mm = cfg_pickle(use_pickle = False, save_pickle = True, load_name = 'Mau10_800mm', save_name = 'Mau10_800mm', recreate = False)

cfg_pickle_Mau_bad_m_test_v1 = cfg_pickle(use_pickle = True, save_pickle = True, load_name = 'Mau10_bad_m_test_v1', save_name = 'Mau10_bad_m_test_v1', recreate = True)
cfg_pickle_Mau_bad_p_test_v1 = cfg_pickle(use_pickle = True, save_pickle = True, load_name = 'Mau10_bad_p_test_v1', save_name = 'Mau10_bad_p_test_v1', recreate = True)
cfg_pickle_Mau_bad_r_test_v1 = cfg_pickle(use_pickle = False, save_pickle = True, load_name = 'Mau10_bad_r_test_v1', save_name = 'Mau10_bad_r_test_v1', recreate = True)
cfg_pickle_Mau_bad_r2_test_v1 = cfg_pickle(use_pickle = False, save_pickle = True, load_name = 'Mau10_bad_r2_test_v1', save_name = 'Mau10_bad_r2_test_v1', recreate = True)

cfg_pickle_Mau_bad_m_test_v2 = cfg_pickle(use_pickle = True, save_pickle = True, load_name = 'Mau10_bad_m_test_v2', save_name = 'Mau10_bad_m_test_v2', recreate = True)
cfg_pickle_Mau_bad_p_test_v2 = cfg_pickle(use_pickle = True, save_pickle = True, load_name = 'Mau10_bad_p_test_v2', save_name = 'Mau10_bad_p_test_v2', recreate = True)
cfg_pickle_Mau_bad_r_test_v2 = cfg_pickle(use_pickle = False, save_pickle = True, load_name = 'Mau10_bad_r_test_v2', save_name = 'Mau10_bad_r_test_v2', recreate = True)

cfg_pickle_Mau_700_plotly = cfg_pickle(use_pickle = True, save_pickle = False, load_name = 'Mau10_700', save_name = 'Mau10_700', recreate = True)

cfg_pickle_GA05_no_ext = cfg_pickle(use_pickle = True, save_pickle = True, load_name = 'GA05_no_ext', save_name = 'GA05_no_ext', recreate = False)
cfg_pickle_GA05_no_DS = cfg_pickle(use_pickle = False, save_pickle = True, load_name = 'GA05_no_DS', save_name = 'GA05_no_DS', recreate = False)

cfg_pickle_new_Mau_PS = cfg_pickle(use_pickle = True, save_pickle = False, load_name = 'Mau10_PS', save_name = 'Mau10_PS', recreate = True)
cfg_pickle_new_Mau_PS_plotly = cfg_pickle(use_pickle = True, save_pickle = False, load_name = 'Mau10_PS', save_name = 'Mau10_PS', recreate = True)

cfg_pickle_Mau_fullsim = cfg_pickle(use_pickle = True, save_pickle = False, load_name = 'Mau10_700_test', save_name = None, recreate = True)

#################
# the plot cfgs #
#################
cfg_plot_mpl = cfg_plot('mpl',[-2,2],True)
cfg_plot_mpl_high_lim = cfg_plot('mpl',[-5,5],True)
cfg_plot_plotly = cfg_plot('plotly',[-2,2],True)
cfg_plot_plotly_high_lim = cfg_plot('plotly',[-10,10],True)




if __name__ == "__main__":

    field_map_analysis('halltoy_600mm', cfg_data_DS_Mau10, cfg_geom_cyl_600mm, cfg_params_Mau_DS_opt, cfg_pickle_new_Mau, cfg_plot_mpl)
#do 700
#create 700 in plotly
    #df, ff, plotter = field_map_analysis('halltoy_700mm', cfg_data_DS_Mau10, cfg_geom_cyl_700mm, cfg_params_Mau_DS_700, cfg_pickle_Mau_700_plotly, cfg_plot_plotly)
#do PS stuff
    #field_map_analysis('halltoy_150mm', cfg_data_PS_Mau10, cfg_geom_cyl_150mm, cfg_params_Mau_PS_opt, cfg_pickle_new_Mau_PS, cfg_plot_mpl)
    #field_map_analysis('halltoy_150mm', cfg_data_PS_Mau10, cfg_geom_cyl_150mm, cfg_params_Mau_PS_opt, cfg_pickle_new_Mau_PS_plotly, cfg_plot_plotly)

    #field_map_analysis('halltoy_825mm_v1_test', cfg_data_DS_Mau10, cfg_geom_cyl_825mm_v1, cfg_params_Mau_DS_825mm_v1, cfg_pickle_Mau_825mm_v1, cfg_plot_mpl)
    #field_map_analysis('halltoy_825mm_v2_test', cfg_data_DS_Mau10, cfg_geom_cyl_825mm_v2, cfg_params_Mau_DS_825mm_v2, cfg_pickle_Mau_825mm_v2, cfg_plot_mpl)
    #field_map_analysis('halltoy_825mm_v1_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim, cfg_params_Mau_DS_825mm_v1, cfg_pickle_Mau_825mm_v1, cfg_plot_mpl)
    #field_map_analysis('halltoy_825mm_v2_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim, cfg_params_Mau_DS_825mm_v2, cfg_pickle_Mau_825mm_v2, cfg_plot_mpl)
    #field_map_analysis('halltoy_825mm_v1_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim, cfg_params_Mau_DS_825mm_v1, cfg_pickle_Mau_825mm_v1, cfg_plot_plotly)
    #field_map_analysis('halltoy_825mm_v2_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim, cfg_params_Mau_DS_825mm_v2, cfg_pickle_Mau_825mm_v2, cfg_plot_plotly)

    #field_map_analysis('halltoy_800mm', cfg_data_DS2_Mau10, cfg_geom_cyl_800mm, cfg_params_Mau_DS_800mm, cfg_pickle_Mau_800mm, cfg_plot_mpl)

    #field_map_analysis('halltoy_825mm_v1_bad_measure', cfg_data_DS_Mau10, cfg_geom_cyl_bad_measure_v1, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_m_test_v1, cfg_plot_mpl_high_lim)
    #field_map_analysis('halltoy_825mm_v1_bad_position', cfg_data_DS_Mau10, cfg_geom_cyl_bad_position_v1, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_p_test_v1, cfg_plot_plotly_high_lim)
    #field_map_analysis('halltoy_825mm_v1_bad_rotation', cfg_data_DS_Mau10, cfg_geom_cyl_bad_rotation_v1, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r_test_v1, cfg_plot_mpl_high_lim)
    #field_map_analysis('halltoy_825mm_v1_bad_rotation2', cfg_data_DS_Mau10, cfg_geom_cyl_bad_rotation2_v1, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r2_test_v1, cfg_plot_mpl_high_lim)

    #field_map_analysis('halltoy_825mm_v2_bad_measure', cfg_data_DS_Mau10, cfg_geom_cyl_bad_measure_v2, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_m_test_v2, cfg_plot_plotly_high_lim)
    #field_map_analysis('halltoy_825mm_v2_bad_position', cfg_data_DS_Mau10, cfg_geom_cyl_bad_position_v2, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_p_test_v2, cfg_plot_plotly_high_lim)
    #field_map_analysis('halltoy_825mm_v2_bad_rotation', cfg_data_DS_Mau10, cfg_geom_cyl_bad_rotation_v2, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r_test_v2, cfg_plot_plotly_high_lim)

    #field_map_analysis('halltoy_825mm_v1_bad_measure_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_m_test_v1, cfg_plot_mpl_high_lim)
    #field_map_analysis('halltoy_825mm_v1_bad_position_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_p_test_v1, cfg_plot_mpl_high_lim)
    #field_map_analysis('halltoy_825mm_v1_bad_rotation_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r_test_v1, cfg_plot_mpl_high_lim)
    #field_map_analysis('halltoy_825mm_v1_bad_rotation_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r_test_v1, cfg_plot_plotly_high_lim)

    #field_map_analysis('halltoy_825mm_v1_bad_rotation_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim_trunc, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r2_test_v1, cfg_plot_mpl_high_lim)

    #field_map_analysis('halltoy_825mm_v2_bad_measure_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_m_test_v2, cfg_plot_plotly_high_lim)
    #field_map_analysis('halltoy_825mm_v2_bad_position_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_p_test_v2, cfg_plot_plotly_high_lim)
    #field_map_analysis('halltoy_825mm_v2_bad_rotation_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r_test_v2, cfg_plot_mpl_high_lim)
    #field_map_analysis('halltoy_825mm_v2_bad_rotation_fullsim', cfg_data_DS_Mau10, cfg_geom_cyl_fullsim, cfg_params_Mau_DS_bad, cfg_pickle_Mau_bad_r_test_v2, cfg_plot_plotly_high_lim)

    #field_map_analysis('halltoy_GA05_no_ext', cfg_data_DS_GA05_no_ext, cfg_geom_cyl_700mm, cfg_params_GA05_DS_opt, cfg_pickle_GA05_no_ext, cfg_plot_mpl)
    #field_map_analysis('halltoy_GA05_no_DS', cfg_data_DS_GA05_no_DS, cfg_geom_cyl_700mm, cfg_params_GA05_DS_opt, cfg_pickle_GA05_no_DS, cfg_plot_mpl)

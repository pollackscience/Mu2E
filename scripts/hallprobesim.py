#! /usr/bin/env python

from mu2e.hallprober import field_map_analysis
from collections import namedtuple
import numpy as np

############################
# defining the cfg structs #
############################
cfg_data = namedtuple('cfg_data', 'datatype magnet path conditions')
cfg_geom = namedtuple('cfg_geom', 'geom z_steps r_steps phi_steps xy_steps')
cfg_params = namedtuple('cfg_params', 'ns ms cns cms Reff a b c')
cfg_pickle = namedtuple('cfg_pickle', 'use_pickle save_pickle load_name save_name recreate')
cfg_plot = namedtuple('cfg_plot', 'plot_type html_loc')

#################
# the data cfgs #
#################
cfg_data_DS_Mau10 = cfg_data('Mau10', 'DS', '../Mau10/Standard_Maps/Mu2e_DSMap', ('Z>5000','Z<13000','R!=0'))
cfg_data_PS_Mau10 = cfg_data('Mau10', 'PS', '../Mau10/Standard_Maps/Mu2e_PSMap', ('Z>-7900','Z<-4000','R!=0'))

#################
# the geom cfgs #
#################
pi8r_600mm = [55.90169944, 167.70509831, 279.50849719, 447.2135955, 614.91869381]
pi4r_600mm = [35.35533906, 141.42135624, 318.19805153, 494.97474683, 601.04076401]
pi2r_600mm = [25,150,325,475,600]

pi8r_700mm = [55.90169944, 167.70509831, 335.41019663, 559.01699437, 726.72209269]
pi4r_700mm = [35.35533906, 176.7766953, 353.55339059, 530.33008589, 707.10678119]
pi2r_700mm = [25,175,375,525,700]

pi8r_700mm2 = [55.90169944, 167.70509831, 335.41019663, 559.01699437,614.91869381, 726.72209269]
pi4r_700mm2 = [35.35533906, 176.7766953, 353.55339059, 530.33008589, 601.04076401, 707.10678119]
pi2r_700mm2 = [25,175,375,525,600,700]

pi8r_150mm = [55.90169944, 111.80339887, 167.70509831]
pi4r_150mm = [35.35533906, 106.06601718, 141.42135624]
pi2r_150mm = [25,100,150]

r_steps_150mm = (pi2r_150mm, pi8r_150mm, pi4r_150mm, pi8r_150mm,
        pi2r_150mm, pi8r_150mm, pi4r_150mm, pi8r_150mm)
r_steps_600mm = (pi2r_600mm, pi8r_600mm, pi4r_600mm, pi8r_600mm,
        pi2r_600mm, pi8r_600mm, pi4r_600mm, pi8r_600mm)
r_steps_700mm = (pi2r_700mm, pi8r_700mm, pi4r_700mm, pi8r_700mm,
        pi2r_700mm, pi8r_700mm, pi4r_700mm, pi8r_700mm)
r_steps_700mm2 = (pi2r_700mm2, pi8r_700mm2, pi4r_700mm2, pi8r_700mm2,
        pi2r_700mm2, pi8r_700mm2, pi4r_700mm2, pi8r_700mm2)
phi_steps_8 = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444,  3*np.pi/4, 2.677945)
z_steps_DS = range(5021,13021,50)
z_steps_PS = range(-7879,-4004,50)


cfg_geom_cyl_600mm = cfg_geom('cyl',z_steps_DS, r_steps_600mm, phi_steps_8, xy_steps = None)
cfg_geom_cyl_700mm = cfg_geom('cyl',z_steps_DS, r_steps_700mm, phi_steps_8, xy_steps = None)
cfg_geom_cyl_700mm2 = cfg_geom('cyl',z_steps_DS, r_steps_700mm2, phi_steps_8, xy_steps = None)
cfg_geom_cyl_150mm = cfg_geom('cyl',z_steps_PS, r_steps_150mm, phi_steps_8, xy_steps = None)


###################
# the params cfgs #
###################
cfg_params_Mau_DS_opt = cfg_params(ns = 3, ms = 40, cns = 0, cms=0, Reff = 9000, a=None,b=None,c=None)
cfg_params_Mau_DS_700 = cfg_params(ns = 3, ms = 70, cns = 0, cms=0, Reff = 9000, a=None,b=None,c=None)
cfg_params_Mau_PS_opt = cfg_params(ns = 3, ms = 40, cns = 0, cms=0, Reff = 9000, a=None,b=None,c=None)

###################
# the pickle cfgs #
###################
cfg_pickle_new_Mau = cfg_pickle(use_pickle = False, save_pickle = True, load_name = None, save_name = 'Mau10_opt', recreate = False)
cfg_pickle_Mau_700 = cfg_pickle(use_pickle = True, save_pickle = False, load_name = 'Mau10_700', save_name = 'Mau10_700', recreate = True)
cfg_pickle_Mau_700_plotly = cfg_pickle(use_pickle = True, save_pickle = False, load_name = 'Mau10_700', save_name = 'Mau10_700', recreate = True)
cfg_pickle_new_Mau_PS = cfg_pickle(use_pickle = True, save_pickle = False, load_name = 'Mau10_PS', save_name = 'Mau10_PS', recreate = True)
cfg_pickle_new_Mau_PS_plotly = cfg_pickle(use_pickle = True, save_pickle = False, load_name = 'Mau10_PS', save_name = 'Mau10_PS', recreate = True)

#################
# the plot cfgs #
#################
cfg_plot_mpl = cfg_plot('mpl',True)
cfg_plot_plotly = cfg_plot('plotly',True)




if __name__ == "__main__":

    #field_map_analysis('halltoy_600mm', cfg_data_Mau10, cfg_geom_cyl_600mm, cfg_params_Mau_opt, cfg_pickle_new_Mau, cfg_plot_mpl)
#do 700
    #field_map_analysis('halltoy_700mm', cfg_data_DS_Mau10, cfg_geom_cyl_700mm, cfg_params_Mau_DS_700, cfg_pickle_Mau_700, cfg_plot_mpl)
#create 700 in plotly
    field_map_analysis('halltoy_700mm', cfg_data_DS_Mau10, cfg_geom_cyl_700mm, cfg_params_Mau_DS_700, cfg_pickle_Mau_700_plotly, cfg_plot_plotly)
#do PS stuff
    #field_map_analysis('halltoy_150mm', cfg_data_PS_Mau10, cfg_geom_cyl_150mm, cfg_params_Mau_PS_opt, cfg_pickle_new_Mau_PS, cfg_plot_mpl)
    #field_map_analysis('halltoy_150mm', cfg_data_PS_Mau10, cfg_geom_cyl_150mm, cfg_params_Mau_PS_opt, cfg_pickle_new_Mau_PS_plotly, cfg_plot_plotly)

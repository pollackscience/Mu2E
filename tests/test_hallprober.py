import unittest
from collections import namedtuple
import numpy as np
from matplotlib.testing.decorators import cleanup
import mu2e.hallprober as hp
from mu2e import mu2e_ext_path


class TestHallProber(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Defining cfg namedtuples
        cfg_data = namedtuple('cfg_data', 'datatype magnet path conditions')
        cfg_geom = namedtuple('cfg_geom', 'geom z_steps r_steps phi_steps xy_steps bad_calibration '
                              'interpolate')
        cfg_params = namedtuple('cfg_params', 'ns ms cns cms Reff func_version')
        cfg_pickle = namedtuple('cfg_pickle', 'use_pickle save_pickle load_name save_name recreate')
        cfg_plot = namedtuple('cfg_plot', 'plot_type zlims save_loc sub_dir')

        # Using Mau10 Map as default testing map
        path_DS_Mau10 = mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap'

        # Making cfgs

        # Data cfg
        cls.cfg_data_DS_Mau10 = cfg_data('Mau10', 'DS',
                                         path_DS_Mau10, ('Z>8000', 'Z<11000', 'R!=0'))

        # Geom cfg
        pi4r_800mm = [35.35533906, 176.7766953, 353.55339059, 530.33008589, 813.17279836]
        pi2r_800mm = [25, 175, 375, 525, 800]
        r_steps_800mm = (pi2r_800mm, pi4r_800mm,
                         pi2r_800mm, pi4r_800mm)
        z_steps_DS = range(8021, 11021, 50)
        phi_steps_8 = (0, np.pi/4, np.pi/2, 3*np.pi/4)
        cls.cfg_geom_cyl_test  = cfg_geom('cyl', z_steps_DS, r_steps_800mm, phi_steps_8,
                                          xy_steps=None, bad_calibration=[False, False, False],
                                          interpolate=False)

        # Param cfg
        cls.cfg_params_Mau_DS_small = cfg_params(ns=3, ms=10, cns=0, cms=0, Reff=7000,
                                                 func_version=5)

        # Pickle cfg
        cls.cfg_pickle_Mau_test = cfg_pickle(use_pickle=False, save_pickle=True,
                                             load_name='Mau10_test_in', save_name='Mau10_test_out',
                                             recreate=False)

        # Plotting fg
        cls.cfg_plot_mpl = cfg_plot('mpl_none', [-2, 2], 'html', None)

    @cleanup
    def test_field_map_analysis(self):
        hmd, ff = hp.field_map_analysis('field_map_analysis_test', self.cfg_data_DS_Mau10,
                                        self.cfg_geom_cyl_test, self.cfg_params_Mau_DS_small,
                                        self.cfg_pickle_Mau_test, self.cfg_plot_mpl)

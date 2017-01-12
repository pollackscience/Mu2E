import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
# import matplotlib.pyplot as plt
import mu2e
from mu2e import mu2eplots, mu2e_ext_path, dataframeprod


class TestMu2ePlots(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.path = mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap'
        self.load_df = dataframeprod.DataFrameMaker(
            mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap', use_pickle=True).data_frame

    def test_mu2e_plot(self):
        x = 'Z'
        y = 'Bz'
        query = 'X==600 and Y==0 and 11000<Z<14000'
        savename = os.path.dirname(mu2e.__file__)+'/../tests/tmp_outputs/test_mu2e_plot.html'
        ax = mu2eplots.mu2e_plot(self.load_df, x, y, query, info='Mau10', mode='mpl')
        assert_allclose(ax.lines[0].get_xydata(), self.load_df.query(query)[[x, y]].values)

        with self.assertRaises(ValueError):
            mu2eplots.mu2e_plot(self.load_df, x, y, query, info='Mau10', mode='fake_plotter')

        mu2eplots.mu2e_plot(self.load_df, x, y, query, info='Mau10', mode='plotly_html',
                            savename=savename, auto_open=False)
        assert os.path.exists(savename)

    def test_mu2e_plot3d(self):
        x = 'X'
        y = 'Z'
        z = 'Bz'
        query = '-1000<=X<=1000 and Phi=={} and 3000<Z<15000'.format(np.pi/4.0)
        savedir = os.path.dirname(mu2e.__file__)+'/../tests/tmp_outputs/'
        mu2eplots.mu2e_plot3d(self.load_df, x, y, z, query, mode='mpl', ptype='3d',
                              save_dir=savedir, save_name='plot3d')
        assert os.path.exists(savedir+'plot3d.png')
        mu2eplots.mu2e_plot3d(self.load_df, x, y, z, query, mode='mpl', ptype='heat',
                              save_dir=savedir, save_name='heat')
        assert os.path.exists(savedir+'heat.png')

        with self.assertRaises(ValueError):
            mu2eplots.mu2e_plot3d(self.load_df, x, y, z, query, info='Mau10', mode='fake_plotter')

    @classmethod
    def tearDownClass(self):
        folder = os.path.dirname(mu2e.__file__)+'/../tests/tmp_outputs/'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

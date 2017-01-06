import unittest
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import matplotlib.pyplot as plt
from mu2e import mu2eplots, mu2e_ext_path, dataframeprod


class TestMu2ePlots(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.path = mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap'
        self.load_df = dataframeprod.DataFrameMaker(mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap',
                                          use_pickle=True).data_frame

    def test_mu2e_plot(self):
        x = 'Z'
        y = 'Bz'
        query = 'X==600 and Y==0 and 11000<Z<14000'
        ax = mu2eplots.mu2e_plot(self.load_df, x, y, query, info='Mau10', mode='mpl')
        assert_allclose(ax.lines[0].get_xydata(), self.load_df.query(query)[[x,y]].values)
        with self.assertRaises(ValueError):
            mu2eplots.mu2e_plot(self.load_df, x, y, query, info='Mau10', mode='fake_plotter')



    #def test_mu2e_plot_2(self):
    #    x = 'Z'
    #    y = 'Bz'
    #    query = 'X==600 and Y==0 and 11000<Z<14000'
    #    ax = mu2eplots.mu2e_plot(self.load_df, x, y, query,
    #                        info='Mau10', mode='fake_plotter')

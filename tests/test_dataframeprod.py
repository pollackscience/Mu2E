import unittest
import pandas as pd
import mu2e.dataframeprod as dfp
from mu2e import mu2e_ext_path


class TestDataFrameProd(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.load_df = dfp.DataFrameMaker(mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap',
                                          use_pickle=True).data_frame

    def test_invalid_version(self):
        self.assertRaises(
            KeyError, dfp.DataFrameMaker(mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap',
                                         use_pickle=False, field_map_version='Mau10')
        )

    def test_load_df(self):
        self.assertIsInstance(self.load_df, pd.DataFrame)

    def test_make_df(self):
        dm = dfp.DataFrameMaker(mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap',
                                use_pickle=False, field_map_version='Mau10')
        dm.do_basic_modifications(-3896)
        make_df = dm.data_frame
        self.assertTrue(make_df.equals(self.load_df))

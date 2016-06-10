import unittest
import pandas as pd
import mu2e.datafileprod as dfp

class TestDataFileProd(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.load_df = dfp.DataFileMaker('../datafiles/Mau10/Standard_Maps/Mu2e_DSMap',use_pickle = True).data_frame

    def test_load_df(self):
        self.assertIsInstance(self.load_df, pd.DataFrame)

    def test_make_df(self):
        dm = dfp.DataFileMaker('../datafiles/Mau10/Standard_Maps/Mu2e_DSMap',use_pickle = False,field_map_version='Mau10')
        dm.do_basic_modifications(-3896)
        make_df = dm.data_frame
        self.assertTrue(make_df.equals(self.load_df))





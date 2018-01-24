#! /usr/bin/env python
"""Module for converting Mu2e data into DataFrames.

This module defines classes and functions that takes Mu2E-style input files, typically csv or ROOT
files, and generates :class:`pandas.DataFrame` for use in this Python Mu2E package.  This module
can also save the output DataFrames as compressed cPickle files.  The input and output data are
saved to a subdir of the user-defined `mu2e_ext_path`, and are not committed to github.

Example:
    Using the DataFrameMaker class::

        In[1]: from mu2e.dataframeprod import DataFrameMaker
        ...    from mu2e import mu2e_ext_path

        In[2]: print mu2e_ext_path
        '/User/local/local_data'

        In[3]: df = DataFrameMaker(
        ...        mu2e_ext_path + 'datafiles/FieldMapsGA02/Mu2e_DS_GA0',
        ...        use_pickle=True,
        ...        field_map_version='GA02'
        ...    ).data_frame

        In[4]: print df.head()
        ...  X       Y       Z        Bx        By        Bz
        0 -1200.0 -1200.0  3071.0  0.129280  0.132039  0.044327
        1 -1200.0 -1200.0  3096.0  0.132106  0.134879  0.041158
        2 -1200.0 -1200.0  3121.0  0.134885  0.137670  0.037726
        3 -1200.0 -1200.0  3146.0  0.137600  0.140397  0.034024
        4 -1200.0 -1200.0  3171.0  0.140235  0.143042  0.030045

        ...  R          Phi      Bphi        Br
        0 1697.056275 -2.356194 -0.001951 -0.184780
        1 1697.056275 -2.356194 -0.001960 -0.188787
        2 1697.056275 -2.356194 -0.001969 -0.192725
        3 1697.056275 -2.356194 -0.001977 -0.196573
        4 1697.056275 -2.356194 -0.001985 -0.200307


Todo:
    * Update the particle trapping function with something more flexible.


*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""


from __future__ import absolute_import
from __future__ import print_function
import re
import six.moves.cPickle as pkl
import numpy as np
import pandas as pd
import mu2e.src.RowTransformations as rt
from tqdm import tqdm
from six.moves import range


class DataFrameMaker(object):
    """Convert a FieldMap csv into a pandas DataFrame.

    The DataFrameMaker acts as a wrapper for :func:`pandas.DataFrame.read_csv` when
    `use_pickle` is `False`. Due to multiple differing input data csv formats, the exact csv
    options are hardcoded, depending on the `field_map_version`.

    * It is assumed that the plaintext is formatted as a csv file, with comma or space delimiters.
    * The expected headers are: 'X Y Z Bx By Bz'
    * The DataFrameMaker converts these into `pandas` DFs, where each header is its own row, as
      expected.
    * The input must be in units of mm and T (certain GA maps are hard-coded to convert to mm).
    * Offsets in the X direction are applied if specified.
    * If the map only covers one region of Y, the map is doubled and reflected about Y, such that
      Y->-Y and By->-By.
    * The following columns are constructed and added to the DF by default: 'R Phi Br Bphi'

    The outputs should be saved as compressed pickle files, and should be loaded from those files
    for further use.  Each pickle contains a single DF.

    Args:
        file_name (str): File path and name for csv/txt/pickle file.  Do not include suffix.
        field_map_version (str): Specify field map type and version (Mau9, Mau10,
            GA01/2/3/4/5)
        header_names (:obj:`list` of :obj:`str`, optional): List of headers if default is not valid.
            Default is `['X', 'Y', 'Z', 'Bx', 'By', 'Bz']`.
        use_pickle (bool, optional): Load data from pickle (instead of csv). Default is
            False.

    Attributes:
        file_name (str): File path and name, no suffix.
        field_map_version (str, optional): Mau or GA simulation type. Default to 'Mau10'.
        data_frame (pandas.DataFrame): Output DF.
        input_source (str): Indicator for input, `pickle` or `csv`.



    """
    def __init__(self, file_name, field_map_version='Mau10', header_names=None, use_pickle=False):
        """The DataFrameMaker initialization process.

        """

        self.file_name = re.sub('\.\w*$', '', file_name)
        self.field_map_version = field_map_version
        if use_pickle:
            self.input_source = 'pickle'
        else:
            self.input_source = 'csv'
        if header_names is None:
            header_names = ['X', 'Y', 'Z', 'Bx', 'By', 'Bz']

        # Load from pickle (all are identical in format).  Otherwise, load from csv
        if use_pickle:
            # self.data_frame = pkl.load(open(self.file_name+'.p', "rb"), encoding='latin1')
            self.data_frame = pd.read_pickle(self.file_name+'.p')

        elif 'Mau9' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True)

        elif 'Mau10' in self.field_map_version and 'rand' in self.file_name:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True)

        elif 'Mau10' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'GA01' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.1', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'GA02' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.2', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'GA03' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.3', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'GA04' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'GA05' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=4, dtype=np.float64)

        elif 'Pure_Cyl' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'Pure_Hel' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=1, dtype=np.float64)

        elif 'Only' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'Ideal' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'Glass_Helix_v4' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=4)

        elif 'Glass' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'Mau11' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'Mau12' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=4)

        else:
            raise KeyError("'Mau' or 'GA' not found in field_map_version: "+self.field_map_version)

    def do_basic_modifications(self, offset=None, helix=False, pitch=None):
        """Perform the expected modifications to the input, needed for further analysis.

        Modify the field map to add more columns, offset the X axis so it is re-centered to 0,
        and reflect the map about the Y-axis, if applicable.  In general these modifactions should
        not be applied to already-pickled inputs, as they should have already been sufficiently
        modified.

        Note:
            * Default offset is 0.
            * The PS offset is +3904 (for Mau).
            * The DS offset is -3896 (for Mau).
            * GA field maps are converted from meters to millimeters.

        Args:
            offset (float, optional): If specified, apply this offset to x-axis. Value should be in
                millimeters.
            helix (boolean, optional): If True, calculate helical coordinates in addition to other
                modifications.
            pitch (float, optional): If helix is True, pitch must be specified in order to create
                helical coordinates. Expect units of mm.  7.53mm for DS8-9-10. 5.27 for DS1-7,11

        Returns:
            Nothing. Operations are performed in place, and modify the internal `data_frame`.

        """

        print('num of rows start', len(self.data_frame.index))
        print(self.data_frame.head())
        print(self.data_frame.tail())

        # Convert to mm for some hard-coded versions
        if (('GA' in self.field_map_version and '5' not in self.field_map_version) or
                ('rand' in self.file_name) or
                ('Pure' in self.field_map_version) or
                ('MIN' in self.file_name) or
                ('MAX' in self.file_name) or
                ('Only' in self.field_map_version) or
                ('Glass' in self.field_map_version and ('v3' not in self.field_map_version and
                                                        'v4' not in self.field_map_version)) or
                ('Mau11' in self.field_map_version and '5096' not in self.file_name) or
                ('Ideal' in self.field_map_version)):

            self.data_frame.eval('X = X*1000', inplace=True)
            self.data_frame.eval('Y = Y*1000', inplace=True)
            self.data_frame.eval('Z = Z*1000', inplace=True)

        # Offset x-axis
        if offset:
            self.data_frame.eval('X = X-{0}'.format(offset), inplace=True)

        # Generate radial position column
        self.data_frame.loc[:, 'R'] = rt.apply_make_r(self.data_frame['X'].values,
                                                      self.data_frame['Y'].values)

        # Generate negative Y-axis values for some hard-coded versions.
        if (any([vers in self.field_map_version for vers in ['Mau9', 'Mau10', 'GA01']]) and
           ('rand' not in self.file_name)):

            data_frame_lower = self.data_frame.query('Y >0').copy()
            data_frame_lower.eval('Y = Y*-1', inplace=True)
            data_frame_lower.eval('By = By*-1', inplace=True)
            self.data_frame = pd.concat([self.data_frame, data_frame_lower])

        # Generate phi position column
        self.data_frame.loc[:, 'Phi'] = rt.apply_make_theta(self.data_frame['X'].values,
                                                            self.data_frame['Y'].values)
        # Generate Bphi field column
        self.data_frame.loc[:, 'Bphi'] = rt.apply_make_bphi(self.data_frame['Phi'].values,
                                                            self.data_frame['Bx'].values,
                                                            self.data_frame['By'].values)
        # Generate Br field column
        self.data_frame.loc[:, 'Br'] = rt.apply_make_br(self.data_frame['Phi'].values,
                                                        self.data_frame['Bx'].values,
                                                        self.data_frame['By'].values)
        if helix:
            if not isinstance(pitch, float):
                raise TypeError("If `helix` is True, pitch must be a float")

            self.data_frame.loc[:, 'Zeta'] = rt.apply_make_zeta(self.data_frame['Z'].values,
                                                                self.data_frame['Phi'].values,
                                                                pitch)

            self.data_frame.loc[:, 'Bphi_wald'] = rt.apply_make_bphi_wald(
                self.data_frame['Phi'].values, self.data_frame['R'].values,
                self.data_frame['Bx'].values, self.data_frame['By'].values, pitch)

            self.data_frame.loc[:, 'Bzeta'] = rt.apply_make_bzeta(
                self.data_frame['Phi'].values, self.data_frame['R'].values,
                self.data_frame['Bx'].values, self.data_frame['By'].values,
                self.data_frame['Bz'].values,
                pitch)

        # Clean up, sort, round off.
        self.data_frame.sort_values(['X', 'Y', 'Z'], inplace=True)
        self.data_frame.reset_index(inplace=True, drop=True)
        self.data_frame = self.data_frame.round(9)
        print('num of rows end', len(self.data_frame.index))

    def make_dump(self, suffix=''):
        """Create a pickle file containing the data_frame, in the same dir as the input.

        Args:
            suffix (str, optional): Attach a suffix to the file name, before the '.p' suffix.
        """
        pkl.dump(self.data_frame, open(self.file_name+suffix+'.p', "wb"), pkl.HIGHEST_PROTOCOL)

    def make_r(self, row):
        return np.sqrt(row['X']**2+row['Y']**2)

    def make_br(self, row):
        return np.sqrt(row['Bx']**2+row['By']**2)

    def make_theta(self, row):
        return np.arctan2(row['Y'], row['X'])

    def make_bottom_half(self, row):
        return (-row['Y'])


def g4root_to_df(input_name, make_pickle=False, do_basic_modifications=False,
                 trees=['vd', 'tvd', 'part'], cluster='', tree_prefix='readvd'):
    from root_pandas import read_root
    '''
    Quick converter for virtual detector ROOT files from the Mu2E Art Framework.

    Args:
         input_name (str): file path name without suffix.
         make_pickle (bool, optional): If `True`, no dfs are returned, and a pickle is created
             containing a tuple of the relevant dfs.
             *Note: Testing out HDF5 storage instead of pickle.*
         do_basic_modifications(bool, optional): If `True`, recenter x-axis, add column 'runevt' in
             order to identify individual events. Add total momentum for nttvd.
         trees(list, optional): List of up to three potential trees to reproduce. List can have
             'vd', 'tvd', and 'part' as its members.

    Return:
        A tuple of dataframes, or none.
    '''
    do_vd = do_tvd = do_part = False

    for tree in trees:
        if tree not in ['vd', 'tvd', 'part']:
            raise KeyError(str(tree)+' is not a valid tree')
        if tree == 'vd':
            do_vd = True
        elif tree == 'tvd':
            do_tvd = True
        elif tree == 'part':
            do_part = True

    input_root = input_name + '.root'

    df_ntpart = df_nttvd = df_ntvd = None

    if tree_prefix is not '':
        tree_prefix = tree_prefix+'/'
    if do_part:
        df_ntpart = read_root(input_root, tree_prefix+'ntpart', ignore='*vd')
    if do_tvd:
        df_nttvd = read_root(input_root, tree_prefix+'nttvd')
    if do_vd:
        df_ntvd = read_root(input_root, tree_prefix+'ntvd')

    if do_basic_modifications:
        if do_part:
            df_ntpart.eval('x = x+3904', inplace=True)
            df_ntpart.eval('xstop = xstop+3904', inplace=True)
            df_ntpart['runevt'] = (str(cluster)+df_ntpart.subrun.astype(int).astype(str) +
                                   df_ntpart.evt.astype(int).astype(str)).astype(int)
        if do_tvd:
            df_nttvd.eval('x = x+3904', inplace=True)
            df_nttvd['runevt'] = (str(cluster)+df_nttvd.subrun.astype(int).astype(str) +
                                  df_nttvd.evt.astype(int).astype(str)).astype(int)
            df_nttvd.eval('p = sqrt(px**2+py**2+pz**2)', inplace=True)
        if do_vd:
            df_ntvd.eval('x = x+3904', inplace=True)
            df_ntvd['runevt'] = (str(cluster)+df_ntvd.subrun.astype(int).astype(str) +
                                 df_ntvd.evt.astype(int).astype(str)).astype(int)
            df_ntvd.eval('p = sqrt(px**2+py**2+pz**2)', inplace=True)

    if make_pickle:
        print('loading into hdf5')
        # pkl.dump((df_nttvd, df_ntpart), open(input_name + '.p', "wb"), pkl.HIGHEST_PROTOCOL)
        store = pd.HDFStore(input_name+'.h5')
        if do_part:
            store['df_ntpart'] = df_ntpart
        if do_tvd:
            store['df_nttvd'] = df_nttvd
        if do_vd:
            store['df_ntvd'] = df_ntvd
        store.close()
        print('file finished')
    else:
        return (df_ntpart, df_nttvd, df_ntvd)


def g4root_to_df_skim_and_combo(input_name, total_n):
    for i in tqdm(list(range(total_n))):
        df_ntpart, df_nttvd, df_ntvd = g4root_to_df(input_name+str(i), make_pickle=False,
                                                    do_basic_modifications=True, cluster=i)
        good_runevt = df_ntpart.query('pdg==11 and p>75').runevt.unique()
        df_ntpart = df_ntpart[df_ntpart.runevt.isin(good_runevt)]
        df_ntvd = df_ntvd[df_ntvd.runevt.isin(good_runevt)]
        df_nttvd = df_nttvd[df_nttvd.runevt.isin(good_runevt)]
        output_root = input_name+str(i)+'_skim.root'
        df_ntvd.to_root(output_root, tree_key='ntvd', mode='w')
        df_nttvd.to_root(output_root, tree_key='nttvd', mode='a')
        df_ntpart.to_root(output_root, tree_key='ntpart', mode='a')


if __name__ == "__main__":
    from mu2e import mu2e_ext_path
    # for PS
    # data_maker = DataFrameMaker('../datafiles/Mau10/Standard_Maps/Mu2e_PSMap',use_pickle = False,
    #                            field_map_version='Mau10')
    # data_maker.do_basic_modifications(3904)
    # data_maker.make_dump()

    # for DS
    # data_maker = DataFrameMaker('../datafiles/FieldMapData_1760_v5/Mu2e_DSMap',use_pickle = False)
    # data_maker = DataFrameMaker('../datafiles/FieldMapsGA01/Mu2e_DS_GA0',use_pickle = False,
    #                            field_map_version='GA01')
    # data_maker = DataFrameMaker(mu2e_ext_path+'datafiles/FieldMapsGA04/Mu2e_DS_GA04',
    #                             use_pickle=False, field_map_version='GA04')
    # data_maker = DataFrameMaker('../datafiles/FieldMapsGA04/Mu2e_DS_GA0',use_pickle = False,
    #                            field_map_version='GA04')
    # data_maker = DataFrameMaker('../datafiles/FieldMapsGA_Special/Mu2e_DS_noPSTS_GA0',
    #                            use_pickle=False, field_map_version='GA05')
    # data_maker = DataFrameMaker('../datafiles/FieldMapsGA_Special/Mu2e_DS_noDS_GA0',
    #                            use_pickle=False, field_map_version='GA05')
    # data_maker = DataFrameMaker('../datafiles/Mau10/Standard_Maps/Mu2e_DSMap', use_pickle=False,
    #                            field_map_version='Mau10')
    # data_maker = DataFrameMaker(mu2e_ext_path+'datafiles/Mau10/Standard_Maps/Mu2e_DSMap_rand1mil',
    #                             use_pickle=False, field_map_version='Mau10')
    # data_maker.do_basic_modifications()
    # data_maker = DataFrameMaker('../datafiles/Mau10/TS_and_PS_OFF/Mu2e_DSMap',use_pickle=False,
    #                            field_map_version='Mau10')
    # data_maker = DataFrameMaker('../datafiles/Mau10/DS_OFF/Mu2e_DSMap',use_pickle=False,
    #                            field_map_version='Mau10')

    # data_maker = DataFrameMaker(mu2e_ext_path+'datafiles/FieldMapsPure/DS8_Bz_xzplane.table',
    #                             use_pickle=False, field_map_version='Pure_Cyl_2D')
    # data_maker.do_basic_modifications(helix=True, pitch=7.38)

    # data_maker = DataFrameMaker(mu2e_ext_path+'datafiles/FieldMapsPure/DS8_HeliCalcfields',
    #                             use_pickle=False, field_map_version='Pure_Hel_2D')
    # data_maker.do_basic_modifications(helix=True, pitch=7.38)

    # data_maker = DataFrameMaker(mu2e_ext_path+'datafiles/Mau9/MAX',
    #                             use_pickle=False, field_map_version='Mau9')
    # data_maker.do_basic_modifications(-3896)

    # data_maker = DataFrameMaker(
    #     mu2e_ext_path+'datafiles/FieldMapsPure/TS5_DS_longbus',
    #     use_pickle=False, field_map_version='Ideal_w_LongBus_3D')
    # data_maker.do_basic_modifications(-3904)

    # data_maker = DataFrameMaker(
    #     mu2e_ext_path+'datafiles/FieldMapsPure/DS_buswork_only_fullmap',
    #     use_pickle=False, field_map_version='Bus_Only_3D')
    # data_maker.do_basic_modifications(-3904)

    # data_maker = DataFrameMaker(
    #     mu2e_ext_path+'datafiles/FieldMapsPure/DS_longbus_nocoils',
    #     use_pickle=False, field_map_version='Glass_longbus_only')
    # data_maker.do_basic_modifications(-3904)

    # data_maker = DataFrameMaker(
    #     mu2e_ext_path+'datafiles/FieldMapsPure/DS-8_helix_no_leads',
    #     use_pickle=False, field_map_version='Glass_Helix_v3')
    # data_maker.do_basic_modifications(-3904, helix=True, pitch=7.53)

    # data_maker = DataFrameMaker(
    #     mu2e_ext_path+'datafiles/FieldMapsPure/DS-8_with_leads_TOL1e-5',
    #     use_pickle=False, field_map_version='Glass_Helix_v2')
    # data_maker.do_basic_modifications(-3904)

    # data_maker = DataFrameMaker(mu2e_ext_path+'datafiles/FieldMapsGA05/TSdMap', use_pickle=False,
    #                             field_map_version='GA05')
    # data_maker.do_basic_modifications()

    # data_maker = DataFrameMaker(
    #     mu2e_ext_path+'datafiles/FieldMapsPure/DS-8_with_leads_TOL1e-5',
    #     use_pickle=False, field_map_version='Glass_Helix_v2')
    # data_maker.do_basic_modifications(-3904)

    # data_maker = DataFrameMaker(
    #     mu2e_ext_path+'datafiles/Mau11/Mu2e_DSMap_v11',
    #     use_pickle=False, field_map_version='Mau11')
    # data_maker.do_basic_modifications(-3904)

    # data_maker = DataFrameMaker(
    #     mu2e_ext_path+'datafiles/Mau11/Mu2e_DSMap_5096_v11',
    #     use_pickle=False, field_map_version='Mau11')
    # data_maker.do_basic_modifications(-3896)

    # data_maker = DataFrameMaker(
    #     mu2e_ext_path+'datafiles/Mau12/DSMap',
    #     use_pickle=False, field_map_version='Mau12')
    # data_maker.do_basic_modifications(-3896)

    data_maker = DataFrameMaker(
        mu2e_ext_path+'datafiles/FieldMapsPure/test_helix_5L_detail',
        use_pickle=False, field_map_version='Glass_Helix_v4')
    data_maker.do_basic_modifications()

    data_maker.make_dump()
    # data_maker.make_dump('_noOffset')
    print(data_maker.data_frame.head())
    print(data_maker.data_frame.tail())

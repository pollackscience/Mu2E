#! /usr/bin/env python

import pandas as pd
import numpy as np
import cPickle as pkl
import src.RowTransformations as rt
import re

class DataFileMaker:
    """Convert Field Map plain text into pandas Data File"""
    def __init__(self, file_name,header_names = None,use_pickle = False,field_map_version='Mau9'):
        self.file_name = re.sub('\.\w*$','',file_name)
        self.field_map_version = field_map_version
        if header_names == None: header_names = ['X','Y','Z','Bx','By','Bz']
        if use_pickle:
            self.data_frame = pkl.load(open(self.file_name+'.p',"rb"))
        elif 'Mau9' in self.field_map_version:
            self.data_frame = pd.read_csv(self.file_name+'.txt', header=None, names = header_names, delim_whitespace=True)
        elif 'Mau10' in self.field_map_version and 'rand' in self.file_name:
            self.data_frame = pd.read_csv(self.file_name+'.table', header=None, names = header_names, delim_whitespace=True)
        elif 'Mau10' in self.field_map_version:
            self.data_frame = pd.read_csv(self.file_name+'.table', header=None, names = header_names, delim_whitespace=True, skiprows=8)
        elif 'GA01' in self.field_map_version:
            self.data_frame = pd.read_csv(self.file_name+'.1', header=None, names = header_names, delim_whitespace=True, skiprows=8)
        elif 'GA02' in self.field_map_version:
            self.data_frame = pd.read_csv(self.file_name+'.2', header=None, names = header_names, delim_whitespace=True, skiprows=8)
        elif 'GA03' in self.field_map_version:
            self.data_frame = pd.read_csv(self.file_name+'.3', header=None, names = header_names, delim_whitespace=True, skiprows=8)
        elif 'GA04' in self.field_map_version:
            self.data_frame = pd.read_csv(self.file_name+'.4', header=None, names = header_names, delim_whitespace=True, skiprows=8)
        elif 'GA05' in self.field_map_version:
            self.data_frame = pd.read_csv(self.file_name+'.txt', header=None, names = header_names, delim_whitespace=True, skiprows=4,dtype=np.float64)
        else:
            raise KeyError("'Mau' or 'GA' not found in field_map_version: "+self.field_map_version)




    def do_basic_modifications(self,offset=None):
        '''Modify the field map to add more columns, offset the X axis so it is recentered to 0,
        and reflect the map about the Y-axis, if applicable.

        -Default offset is 0
        -The PS offset is +3904 (for Mau)
        -The DS offset is -3896 (for Mau)

        GA field maps are converted from meters to millimeters'''

        print 'num of columns start', len(self.data_frame.index)
        if ('GA' in self.field_map_version and '5' not in self.field_map_version) or ('rand' in self.file_name):
            self.data_frame.eval('X = X*1000', inplace=True)
            self.data_frame.eval('Y = Y*1000', inplace=True)
            self.data_frame.eval('Z = Z*1000', inplace=True)

        if offset: self.data_frame.eval('X = X-{0}'.format(offset), inplace=True)

        self.data_frame.loc[:, 'R'] = rt.apply_make_r(self.data_frame['X'].values, self.data_frame['Y'].values)
        if any([vers in self.field_map_version for vers in ['Mau9','Mau10','GA01']]) and 'rand' not in self.file_name:
            data_frame_lower = self.data_frame.query('Y >0').copy()
            data_frame_lower.eval('Y = Y*-1', inplace=True)
            data_frame_lower.eval('By = By*-1', inplace=True)
            self.data_frame = pd.concat([self.data_frame, data_frame_lower])
        self.data_frame.loc[:, 'Phi'] = rt.apply_make_theta(self.data_frame['X'].values, self.data_frame['Y'].values)
        self.data_frame.loc[:, 'Bphi'] = rt.apply_make_bphi(self.data_frame['Phi'].values, self.data_frame['Bx'].values, self.data_frame['By'].values)
        self.data_frame.loc[:, 'Br'] = rt.apply_make_br(self.data_frame['Phi'].values, self.data_frame['Bx'].values, self.data_frame['By'].values)
        self.data_frame.sort_values(['X','Y','Z'],inplace=True)
        self.data_frame.reset_index(inplace = True, drop=True)
        self.data_frame = self.data_frame.round(9)
        print 'num of columns end', len(self.data_frame.index)

    def make_dump(self,suffix=''):
        pkl.dump( self.data_frame, open( self.file_name+suffix+'.p', "wb" ),pkl.HIGHEST_PROTOCOL )

    def make_r(self,row):
        return np.sqrt(row['X']**2+row['Y']**2)
    def make_br(self,row):
        return np.sqrt(row['Bx']**2+row['By']**2)
    def make_theta(self,row):
        return np.arctan2(row['Y'],row['X'])
    def make_bottom_half(self,row):
        return (-row['Y'])



if __name__ == "__main__":
    #for PS
    #data_maker = DataFileMaker('../datafiles/Mau10/Standard_Maps/Mu2e_PSMap',use_pickle = False, field_map_version='Mau10')
    #data_maker.do_basic_modifications(3904)
    #data_maker.make_dump()

    #for DS
    #data_maker = DataFileMaker('../datafiles/FieldMapData_1760_v5/Mu2e_DSMap',use_pickle = False)
    #data_maker = DataFileMaker('../datafiles/FieldMapsGA01/Mu2e_DS_GA0',use_pickle = False,field_map_version='GA01')
    #data_maker = DataFileMaker('../datafiles/FieldMapsGA02/Mu2e_DS_GA0',use_pickle = False,field_map_version='GA02')
    #data_maker = DataFileMaker('../datafiles/FieldMapsGA04/Mu2e_DS_GA0',use_pickle = False,field_map_version='GA04')
    #data_maker = DataFileMaker('../datafiles/FieldMapsGA_Special/Mu2e_DS_noPSTS_GA0',use_pickle = False,field_map_version='GA05')
    #data_maker = DataFileMaker('../datafiles/FieldMapsGA_Special/Mu2e_DS_noDS_GA0',use_pickle = False,field_map_version='GA05')
    data_maker = DataFileMaker('../datafiles/Mau10/Standard_Maps/Mu2e_DSMap',use_pickle = False,field_map_version='Mau10')
    #data_maker = DataFileMaker('../datafiles/Mau10/Standard_Maps/Mu2e_DSMap_rand1mil',use_pickle = False,field_map_version='Mau10')
    #data_maker = DataFileMaker('../datafiles/Mau10/TS_and_PS_OFF/Mu2e_DSMap',use_pickle = False,field_map_version='Mau10')
    #data_maker = DataFileMaker('../datafiles/Mau10/DS_OFF/Mu2e_DSMap',use_pickle = False,field_map_version='Mau10')
    #data_maker = DataFileMaker('../datafiles/FieldMapsGA05/DSMap',use_pickle = False,field_map_version='GA05')
    data_maker.do_basic_modifications(-3896)
    #data_maker.do_basic_modifications(-3904)
    #data_maker.do_basic_modifications()
    data_maker.make_dump()
    #data_maker.make_dump('_8mmOffset')
    print data_maker.data_frame.head()
    print data_maker.data_frame.tail()




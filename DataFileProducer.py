#! /usr/bin/env python

import pandas as pd
import numpy as np
import cPickle as pkl

class DataFileMaker:
  """Convert Field Map plain text into pandas Data File"""
  def __init__(self, file_name,header_names = None,use_pickle = False):
    self.file_name = file_name
    if header_names == None: header_names = ['X','Y','Z','Bx','By','Bz']
    if use_pickle:
      self.data_file = pkl.load(open(self.file_name+'.p',"rb"))
    else:
      self.data_file = pd.read_csv(self.file_name+'.txt', header=None, names = header_names, delim_whitespace=True)
  def do_basic_modifications(self,offset=None):
    print 'num of columns start', len(self.data_file.index)
    self.data_file['X'] = self.data_file.apply(self.center_x, args = (offset,), axis=1)
    self.data_file['R'] = self.data_file.apply(self.make_r, axis=1)
    self.data_file['Br'] = self.data_file.apply(self.make_br, axis=1)
    data_file_lower = self.data_file.query('Y >0')
    #data_file_lower['Y'] = data_file_lower.apply(self.make_bottom_half, axis=1)
    data_file_lower['Y'] *= -1
    data_file_lower['By'] *= -1
    self.data_file = pd.concat([self.data_file, data_file_lower])
    self.data_file['Theta'] = self.data_file.apply(self.make_theta, axis=1)
    #self.data_file.sort(['Z','X','Y'],inplace=True)
    print 'num of columns end', len(self.data_file.index)
  def make_dump(self):
    pkl.dump( self.data_file, open( self.file_name+'.p', "wb" ),pkl.HIGHEST_PROTOCOL )

  def make_r(self,row):
    return np.sqrt(row['X']**2+row['Y']**2)
  def center_x(self,row,offset = None):
    if offset == None: offset = 3904
    return row['X']-offset
  def make_br(self,row):
    return np.sqrt(row['Bx']**2+row['By']**2)
  def make_theta(self,row):
    return np.arctan2(row['Y'],row['X'])
  def make_bottom_half(self,row):
    return (-row['Y'])



if __name__ == "__main__":
  data_maker = DataFileMaker('FieldMapData_1760_v5/Mu2e_PSMap',use_pickle = False)
  data_maker.do_basic_modifications()
  data_maker.make_dump()
  print data_maker.data_file.head()


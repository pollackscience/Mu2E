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
      self.data_frame = pkl.load(open(self.file_name+'.p',"rb"))
    else:
      self.data_frame = pd.read_csv(self.file_name+'.txt', header=None, names = header_names, delim_whitespace=True)
  def do_basic_modifications(self,offset=None):
    print 'num of columns start', len(self.data_frame.index)
    self.data_frame['X'] = self.data_frame.apply(self.center_x, args = (offset,), axis=1)
    self.data_frame['R'] = self.data_frame.apply(self.make_r, axis=1)
    self.data_frame['Br'] = self.data_frame.apply(self.make_br, axis=1)
    data_frame_lower = self.data_frame.query('Y >0')
    #data_frame_lower['Y'] = data_frame_lower.apply(self.make_bottom_half, axis=1)
    data_frame_lower['Y'] *= -1
    data_frame_lower['By'] *= -1
    self.data_frame = pd.concat([self.data_frame, data_frame_lower])
    self.data_frame['Theta'] = self.data_frame.apply(self.make_theta, axis=1)
    #self.data_frame.sort(['Z','X','Y'],inplace=True)
    print 'num of columns end', len(self.data_frame.index)
  def make_dump(self,suffix=''):
    pkl.dump( self.data_frame, open( self.file_name+suffix+'.p', "wb" ),pkl.HIGHEST_PROTOCOL )

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
  data_maker.do_basic_modifications(3905.5)
  data_maker.make_dump('_-1.5mmOffset')
  print data_maker.data_frame.head()


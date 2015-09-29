#! /usr/bin/env python

import os
import math
import mu2e
import numpy as np
import pandas as pd
import collections
from datafileprod import DataFileMaker
import src.RowTransformations as rt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import gridspec
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.formula.api import wls
from statsmodels.graphics.regressionplots import abline_plot
import matplotlib.ticker as mtick
import re


class HallProbeGenerator:
  """Class for generating toy outputs for mimicing the Argonne hall probe measurements.
  The input should be a dataframe made from the magnetic field simulations."""

  def __init__(self, input_data, z_steps = 15, x_steps = 10, y_steps = 10):
    self.full_field = input_data
    self.sparse_field = self.full_field

    self.apply_selection('Z',z_steps)
    self.apply_selection('X',x_steps)
    self.apply_selection('Y',y_steps)

  def takespread(self, sequence, num):
    length = float(len(sequence))
    spread = []
    for i in range(num):
      spread.append(sequence[int(math.ceil(i * length / num))])
    return spread


  def apply_selection(self, coord, steps):

    if isinstance(steps,int):
      if coord in ['Z','R']:
        coord_vals = np.sort(self.full_field[coord].unique())
        coord_vals = self.takespread(coord_vals, steps)
        #coord_vals = np.sort(self.full_field[coord].abs().unique())[:steps]
      else:
        coord_vals = np.sort(self.full_field[coord].abs().unique())[:steps]
        coord_vals = np.concatenate((coord_vals,-coord_vals[np.where(coord_vals>0)]))

    elif isinstance(steps, collections.Sequence) and type(steps)!=str:
      coord_vals = steps
    elif steps=='all':
        coord_vals = np.sort(self.full_field[coord].unique())
    else:
      raise TypeError(coord+" steps must be scalar or list of values!")

    self.sparse_field = self.sparse_field[self.sparse_field[coord].isin(coord_vals)]
    if len(self.sparse_field[coord].unique()) != len(coord_vals):
      print 'Warning!:',set(coord_vals)-set(self.sparse_field[coord].unique()), 'not valid input_data',coord

    for mag in ['Bz','Br','Bx','By','Bz']:
      self.sparse_field.eval('{0}err = 0.0001*{0}'.format(mag))
      #self.sparse_field[self.sparse_field.Z > 8000][self.sparse_field.Z < 13000].eval('{0}err = 0.0000001*{0}'.format(mag))

  def get_toy(self):
    return self.sparse_field


if __name__=="__main__":
  data_maker1=DataFileMaker('../FieldMapData_1760_v5/Mu2e_DSmap',use_pickle = True)
  hpg = HallProbeGenerator(data_maker1.data_frame)
  hall_toy = hpg.get_toy()

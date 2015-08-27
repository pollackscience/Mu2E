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

  def apply_selection(self, coord, steps):
    if isinstance(steps,int):
      coord_vals = sort(self.full_field[coord].unique())[:steps]
    elif isinstance(steps, collections.Sequence):
      coord_vals = steps
    else:
      raise TypeError(coord+" steps must be scalar or list of values!")

    self.sparse_field = self.sparse_field[self.sparse_field[coord].isin(coord_vals)]
    if len(self.sparse_field[coord].unique()) != len(coord_vals):
      print 'Warning!:',set(coord_vals)-set(self.sparse_field[coord].unique()), 'not valid input_data',coord

  def get_toy(self):
    return self.sparse_field




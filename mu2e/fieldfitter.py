#! /usr/bin/env python

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tools.fit_funcs import *
from lmfit import  Model

class FieldFitter:
  """Input hall probe measurements, perform semi-analytical fit, return fit function and other stuff."""
  def __init__(self, input_data):
    self.input_data = input_data


  def fit_2d(self,A,B,C):

    piv = self.input_data.pivot(B,C,A)
    X=piv.columns.values
    Y=piv.index.values
    self.Z=piv.values
    self.X,self.Y = np.meshgrid(X, Y)
    self.popt,self.pcov = curve_fit(big_poly_2d, (self.X,self.Y), self.Z.ravel(), p0=(0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1))

  def fit_1d(self,A,B):

    data_1d = self.input_data.query('X==0 & Y==0')
    self.X=data_1d['Z'].values
    self.Z=data_1d['Bz'].values
    self.mod = Model(bz_r0_1d_lm)
    self.mod.set_param_hint('R',value=1500,vary=False)
    self.mod.set_param_hint('A0',value=9.68625620,max=100,min=-100)
    self.mod.set_param_hint('B0',value=-58921.8812,max=100,min=-100000000)
    self.mod.set_param_hint('A1',value=-0.00279735,max=100,min=-100)
    self.mod.set_param_hint('B1',value=-34.9145987,max=100,min=-100)
    self.mod.set_param_hint('A2',value=4.0663e-07,max=100,min=-100)
    self.mod.set_param_hint('B2',value=-1.65006612,max=100,min=-100)
    self.mod.set_param_hint('A3',value=4.1283e-11,max=100,min=-100)
    self.mod.set_param_hint('B3',value=-0.03202499,max=100,min=-100)
    self.mod.set_param_hint('A4',value=-1.4211e-14,max=100,min=-100)
    self.mod.set_param_hint('B4',value=-0.02571291,max=100,min=-100)
    self.mod.set_param_hint('A5',value=-1.4211e-14,max=100,min=-100)
    self.mod.set_param_hint('B5',value=-0.02571291,max=100,min=-100)
    self.mod.set_param_hint('A6',value=-1.4211e-14,max=100,min=-100)
    self.mod.set_param_hint('B6',value=-0.02571291,max=100,min=-100)
    self.params = self.mod.make_params()

    self.result = self.mod.fit(self.Z,z=self.X, params = self.params,method='newton')
    report_fit(self.result)

  def plot_fit(self,ds='1d'):

    plt.close('all')
    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.hold(True)

    if ds == '1d':
      ax = fig.gca()
      ax.scatter(self.X, self.Z, color='black')
      ax.set_xlabel('X')
      ax.set_ylabel('Bz')

      ax.plot(self.X, self.result.best_fit, color='green')
      #ax.plot(self.X, self.result.init_fit, color='black',linestyle='--')
    else:

      ax = fig.gca(projection='3d')
      scat = ax.scatter(self.X.ravel(), self.Y.ravel(), self.Z.ravel(), color='black')

      ax.set_xlabel('Z')
      ax.set_ylabel('X')
      ax.set_zlabel('Br')
      #ax.set_title(r'Z=X$\times$(Y+1)$^2$')

      fitted_vals = big_poly_2d((self.X,self.Y),*self.popt).reshape(self.Z.shape)

      surf = ax.plot_wireframe(self.X, self.Y, fitted_vals,color='green')

    plt.show()
    plt.get_current_fig_manager().window.wm_geometry("-2600-600")
    #plt.get_current_fig_manager().window.wm_geometry("-1100-600")
    fig.set_size_inches(10,10,forward=True)


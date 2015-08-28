#! /usr/bin/env python

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

class FieldFitter:
  """Input hall probe measurements, perform semi-analytical fit, return fit function and other stuff."""
  def __init__(self, input_data):
    self.input_data = input_data


  def legendre_2d(self,(x,y), O,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p):
    leg = O+a*x+b*y+ c*x*y + d*x**2 + e*y**2+ f*(x*y**2) + g*(y*x**2) + h*x**3 + i*y**3 +j*x**2*y**2 +k*x**3*y +l*y**3*x
    + m*x**4 + n*y**4 + o*x**5 + p*y**5
    return leg.ravel()

  def fit_2d(self,A,B,C):

    piv = self.input_data.pivot(B,C,A)
    X=piv.columns.values
    Y=piv.index.values
    self.Z=piv.values
    self.X,self.Y = np.meshgrid(X, Y)
    self.popt,self.pcov = curve_fit(self.legendre_2d, (self.X,self.Y), self.Z.ravel(), p0=(0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1))

  def plot_fit(self):

    plt.close('all')
    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.hold(True)
    ax = fig.gca(projection='3d')
    scat = ax.scatter(self.X.ravel(), self.Y.ravel(), self.Z.ravel(), color='black')

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Br')
    #ax.set_title(r'Z=X$\times$(Y+1)$^2$')

    fitted_vals = self.legendre_2d((self.X,self.Y),*self.popt).reshape(self.Z.shape)

    surf = ax.plot_wireframe(self.X, self.Y, fitted_vals,color='green')

    plt.show()

    plt.get_current_fig_manager().window.wm_geometry("-2600-600")
    fig.set_size_inches(10,10,forward=True)


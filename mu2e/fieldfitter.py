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
import cPickle as pkl
from matplotlib import gridspec
import mpld3
from mpld3 import plugins, utils
from matplotlib import animation
import IPython.display as IPdisplay
import glob
from PIL import Image as PIL_Image
from images2gif import writeGif

class FieldFitter:
  """Input hall probe measurements, perform semi-analytical fit, return fit function and other stuff."""
  def __init__(self, input_data):
    self.input_data = input_data
    #self.add_zero_data()

  def add_zero_data(self):
    df_highz = self.input_data[self.input_data.Z==self.input_data.Z.max()]
    df_highz.Z = 1000000
    df_highz.Bz = 0
    df_highz.Bx = 0
    df_highz.By = 0
    df_highz.Br = 0
    df_lowz = df_highz.copy()
    df_lowz.Z = -1000000
    self.zero_data = pd.concat([df_highz, self.input_data, df_lowz], ignore_index=True)
    self.zero_data.sort(['Z','X'],inplace=True)

  def fit_3d(self,ns=5,ms=10,use_pickle = False):

    Reff=9000
    phi_slice = 1.570796
    #input_data_phi = self.input_data[(abs(self.input_data.phi-phi_slice)<1e-6)|(self.input_data.phi==0)]
    input_data_phi = self.input_data[(abs(self.input_data.Phi-phi_slice)<1e-6)]
    piv_bz = input_data_phi.pivot('Z','R','Bz')
    piv_br = input_data_phi.pivot('Z','R','Br')
    piv_bphi = input_data_phi.pivot('Z','R','Bphi')
    R=piv_br.columns.values
    Z=piv_br.index.values
    self.Bz=piv_bz.values
    self.Br=piv_br.values
    self.Bphi=piv_bphi.values
    self.R,self.Z = np.meshgrid(R, Z)
    self.Phi = np.full_like(self.R,input_data_phi.Phi.unique()[0])

    piv_bz_err = input_data_phi.pivot('Z','R','Bzerr')
    piv_br_err = input_data_phi.pivot('Z','R','Brerr')
    piv_bphi_err = input_data_phi.pivot('Z','R','Bphierr')
    self.Bzerr=piv_bz_err.values
    self.Brerr=piv_br_err.values
    self.Bphierr=piv_bphi_err.values

    #self.mod = Model(brzphi_3d, independent_vars=['r','z','phi'])
    brzphi_3d_fast = brzphi_3d_producer(self.Z,self.R,self.Phi,Reff,ns,ms)
    self.mod = Model(brzphi_3d_fast, independent_vars=['r','z','phi'])

    if use_pickle:
      #self.params = pkl.load(open('result.p',"rb"))
      #for param in self.params:
      #  self.params[param].vary = False
      #self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(), weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
      #    r=self.X,z=self.Y, params = self.params,method='leastsq')
      pass
    else:

      #b_zeros = []
      #for n in range(ns):
      #  b_zeros.append(special.jn_zeros(n,ms))
      #kms = [b/R for b in b_zeros]

      self.params = Parameters()
      self.params.add('R',value=Reff,vary=False)
      self.params.add('offset',value=0,vary=False)
      self.params.add('ns',value=ns,vary=False)
      self.params.add('ms',value=ms,vary=False)

      for n in range(ns):
        for m in range(ms):
          self.params.add('A_{0}_{1}'.format(n,m),value=0)
          self.params.add('B_{0}_{1}'.format(n,m),value=0)
      print 'fitting with n={0}, m={1}'.format(ns,ms)
      self.result = self.mod.fit(np.concatenate([self.Br,self.Bz,self.Bphi]).ravel(),
          weights = np.concatenate([self.Brerr,self.Bzerr,self.Bphierr]).ravel(),
          r=self.R, z=self.Z, phi=self.Phi, params = self.params,method='leastsq')

    self.params = self.result.params
    report_fit(self.result)

  def fit_2d_sim(self,B,C,nparams = 20,use_pickle = False):

    if B=='X':Br='Bx'
    elif B=='Y':Br='By'
    piv_bz = self.input_data.pivot(C,B,'Bz')
    piv_br = self.input_data.pivot(C,B,Br)
    X=piv_br.columns.values
    Y=piv_br.index.values
    self.Bz=piv_bz.values
    self.Br=abs(piv_br.values)
    self.X,self.Y = np.meshgrid(X, Y)

    piv_bz_err = self.input_data.pivot(C,B,'Bzerr')
    piv_br_err = self.input_data.pivot(C,B,Br+'err')
    self.Bzerr=piv_bz_err.values
    self.Brerr=piv_br_err.values

    self.mod = Model(brz_2d_trig, independent_vars=['r','z'])

    if use_pickle:
      self.params = pkl.load(open('result.p',"rb"))
      #for param in self.params:
      #  self.params[param].vary = False
      self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(), weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
          r=self.X,z=self.Y, params = self.params,method='leastsq')
    else:
      self.params = Parameters()
      #self.params.add('R',value=1000,vary=False)
      #self.params.add('R',value=22000,vary=False)
      self.params.add('R',value=9000,vary=False)
      #self.params.add('offset',value=-14000,vary=False)
      self.params.add('offset',value=0,vary=False)
      #self.params.add('C',value=0)
      self.params.add('A0',value=0)
      self.params.add('B0',value=0)
      #self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(),r=self.X,z=self.Y, params = self.params,method='leastsq')

      for i in range(nparams):
        print 'refitting with params:',i+1
        self.params.add('A'+str(i+1),value=0)
        self.params.add('B'+str(i+1),value=0)
        #if (i+1)%10==0:
        #  self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(),r=self.X,z=self.Y, params = self.params,method='leastsq')
        #  self.params = self.result.params

      #    fit_kws={'xtol':1e-100,'ftol':1e-100,'maxfev':5000,'epsfcn':1e-40})
      self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(), weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
          r=self.X,z=self.Y, params = self.params,method='leastsq')
      #self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(), weights = np.concatenate([self.Brerr,self.Bzerr]).ravel(),
          #r=self.X,z=self.Y, params = self.params,method='lbfgsb',fit_kws= {'options':{'factr':0.1}})

    self.params = self.result.params
    report_fit(self.result)


  def fit_2d(self,A,B,C,use_pickle = False):
    self.mag_field = A
    self.axis2 = B
    self.axis1 = C

    piv = self.input_data.pivot(C,B,A)
    X=piv.columns.values
    Y=piv.index.values
    self.Z=piv.values
    self.X,self.Y = np.meshgrid(X, Y)

    piv_err = self.input_data.pivot(C,B,A+'err')
    self.Zerr = piv_err.values
    if A == 'Bz':
      #self.mod = Model(bz_2d, independent_vars=['r','z'])
      self.mod = Model(bz_2d, independent_vars=['r','z'])
    elif A == 'Br':
      self.mod = Model(br_2d, independent_vars=['r','z'])
    else:
      raise KeyError('No function available for '+A)

    if use_pickle:
      self.params = pkl.load(open('result.p',"rb"))
      self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params, weights  = self.Zerr.ravel(), method='leastsq')
      #for param in self.params:
      #  self.params[param].vary = False
      self.params = self.result.params
    else:
      self.params = Parameters()
      #self.params.add('R',value=1000,vary=False)
      #self.params.add('R',value=22000,vary=False)
      self.params.add('R',value=11000,vary=False)
      #if A == 'Br':
      self.params.add('C',value=0)
      self.params.add('A0',value=0)
      self.params.add('B0',value=0)
      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params,method='leastsq')

      for i in range(60):
        print 'refitting with params:',i+1
        #self.params = self.result.params
        self.params.add('A'+str(i+1),value=0)
        self.params.add('B'+str(i+1),value=0)
        #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params,method='nelder')

      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params,method='leastsq',
      #    fit_kws={'xtol':1e-100,'ftol':1e-100,'maxfev':5000,'epsfcn':1e-40})
      self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params, weights  = self.Zerr.ravel(), method='leastsq')
      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params,method='powell')
      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.result.params,method='lbfgsb')
      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.result.params,method='lbfgsb')
      self.params = self.result.params
      #self.params['R'].vary=True
      #self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params,method='slsqp')

    report_fit(self.result)

  def fit_1d(self,A,B):

    if A == 'Bz':
      self.mod = Model(bz_2d)
    elif A == 'Br':
      self.mod = Model(br_2d, independent_vars=['r','z'])
    else:
      raise KeyError('No function available for '+A)


    data_1d = self.input_data.query('X==0 & Y==0')
    self.X=data_1d[B].values
    self.Z=data_1d[A].values
    self.mod = Model(bz_r0_1d)

    self.params = Parameters()
    self.params.add('R',value=100000,vary=False)
    self.params.add('A0',value=0)
    self.params.add('B0',value=0)

    self.result = self.mod.fit(self.Z,z=self.X, params = self.params,method='nelder')
    for i in range(9):
      print 'refitting with params:',i+1
      self.params = self.result.params
      self.params.add('A'+str(i+1),value=0)
      self.params.add('B'+str(i+1),value=0)
      self.result = self.mod.fit(self.Z,z=self.X, params = self.params,method='nelder')

    self.params = self.result.params
    report_fit(self.result)
    #report_fit(self.params)

  def pickle_results(self):
    pkl.dump( self.result.params, open( 'result.p', "wb" ),pkl.HIGHEST_PROTOCOL )


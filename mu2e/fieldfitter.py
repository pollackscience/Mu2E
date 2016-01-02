#! /usr/bin/env python

from __future__ import division
import pandas as pd
import numpy as np
from tools.fit_funcs import *
from lmfit import    Model
from lmfit.model import  ModelResult
import cPickle as pkl
from time import time

class FieldFitter:
    """Input hall probe measurements, perform semi-analytical fit, return fit function and other stuff."""
    def __init__(self, input_data, phi_steps = None, r_steps = None, no_save = False):
        self.input_data = input_data
        if phi_steps: self.phi_steps = phi_steps
        else: self.phi_steps = (np.pi/2,)
        if r_steps: self.r_steps = r_steps
        else: self.r_steps = (range(25,625,50),)
        self.no_save = no_save

    def fit_external_field(self,ns=5,ms=10,use_pickle = False, line_profile=False, recreate=False):
        Reff=9000
        Bz = []
        Br =[]
        Bphi = []
        Bzerr = []
        Brerr =[]
        Bphierr = []
        RR =[]
        ZZ = []
        PP = []
        for phi in self.phi_steps:
            if phi==0: nphi = np.pi
            else: nphi=phi-np.pi

            input_data_phi = self.input_data[(np.abs(self.input_data.Phi-phi)<1e-6)|(np.abs(self.input_data.Phi-nphi)<1e-6)]
            input_data_phi.ix[np.abs(input_data_phi.Phi-nphi)<1e-6, 'R']*=-1

            input_data_phi_top = input_data_phi[input_data_phi['R']>0].sort(['Z','R']).reset_index(drop=True)
            input_data_phi_bottom = input_data_phi[input_data_phi['R']<0].sort(['Z','R'],ascending=[True,False]).reset_index(drop=True)
            input_data_phi_ext = input_data_phi_top.copy()
            input_data_phi_ext['Bphi_ext'] = -(input_data_phi_top['Bphi']+input_data_phi_bottom['Bphi'])
            input_data_phi_ext['Br_ext'] = input_data_phi_top['Br']-input_data_phi_bottom['Br']
            input_data_phi_ext['Bz_ext'] = input_data_phi_top['Bz']-input_data_phi_bottom['Bz']

            piv_bz = input_data_phi_ext.pivot('Z','R','Bz_ext')
            piv_br = input_data_phi_ext.pivot('Z','R','Br_ext')
            piv_bphi = input_data_phi_ext.pivot('Z','R','Bphi_ext')
            #print input_data_phi.Phi.unique()
            R = piv_br.columns.values
            Z = piv_br.index.values
            Bz.append(piv_bz.values)
            Br.append(piv_br.values)
            Bphi.append(piv_bphi.values)
            RR_slice,ZZ_slice = np.meshgrid(R, Z)
            RR.append(RR_slice)
            ZZ.append(ZZ_slice)
            PP_slice = np.full_like(RR_slice,input_data_phi_ext.Phi.unique()[0])
            PP_slice[:,PP_slice.shape[1]/2:]=input_data_phi_ext.Phi.unique()[0]
            PP.append(PP_slice)

        ZZ = np.concatenate(ZZ)
        RR = np.concatenate(RR)
        PP = np.concatenate(PP)
        Bz = np.concatenate(Bz)
        Br = np.concatenate(Br)
        Bphi = np.concatenate(Bphi)
        if line_profile:
            return ZZ,RR,PP,Bz,Br,Bphi

        brzphi_ext_fast = brzphi_ext_producer(ZZ,RR,PP,ns,ms)
        self.mod = Model(brzphi_ext_fast, independent_vars=['r','z','phi'])
        self.params = Parameters()
        if 'ns' not in self.params: self.params.add('ns',value=ns,vary=False)
        else: self.params['ns'].value=ns
        if 'ms' not in self.params: self.params.add('ms',value=ms,vary=False)
        else: self.params['ms'].value=ms

        for n in range(ns):
            for m in range(ms):
                if 'A_{0}_{1}'.format(n,m) not in self.params: self.params.add('A_{0}_{1}'.format(n,m),value=1)
                else: self.params['A_{0}_{1}'.format(n,m)].vary=True

        if not recreate: print 'fitting with n={0}, m={1}'.format(ns,ms)
        start_time=time()
        self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
            #weights = np.concatenate([Brerr,Bzerr,Bphierr]).ravel(),
            #r=RR, z=ZZ, phi=PP, params = self.params, method='leastsq',fit_kws={'maxfev':100})
            r=RR, z=ZZ, phi=PP, params = self.params, method='leastsq')

        self.params = self.result.params
        end_time=time()
        if not recreate:
            print("Elapsed time was %g seconds" % (end_time - start_time))
            report_fit(self.result, show_correl=False)
        if not self.no_save and not recreate: self.pickle_results()


    def fit_3d_v4(self,ns=5,ms=10,cns=1,cms=1, use_pickle = False, pickle_name = 'default', line_profile=False, recreate=False):
        Reff=9000
        Bz = []
        Br =[]
        Bphi = []
        Bzerr = []
        Brerr =[]
        Bphierr = []
        RR =[]
        ZZ = []
        PP = []
        for phi in self.phi_steps:
            if phi==0: nphi = np.pi
            else: nphi=phi-np.pi

            input_data_phi = self.input_data[(np.abs(self.input_data.Phi-phi)<1e-6)|(np.abs(self.input_data.Phi-nphi)<1e-6)]
            input_data_phi.ix[np.abs(input_data_phi.Phi-nphi)<1e-6, 'R']*=-1
            if phi>np.pi/2:
                input_data_phi.Phi=input_data_phi.Phi+np.pi
                input_data_phi.ix[input_data_phi.Phi>np.pi, 'Phi']-=(2*np.pi)
            #print input_data_phi.Phi.unique()

            piv_bz = input_data_phi.pivot('Z','R','Bz')
            piv_br = input_data_phi.pivot('Z','R','Br')
            piv_bphi = input_data_phi.pivot('Z','R','Bphi')
            piv_bz_err = input_data_phi.pivot('Z','R','Bzerr')
            piv_br_err = input_data_phi.pivot('Z','R','Brerr')
            piv_bphi_err = input_data_phi.pivot('Z','R','Bphierr')

            R = piv_br.columns.values
            Z = piv_br.index.values
            Bz.append(piv_bz.values)
            Br.append(piv_br.values)
            Bphi.append(piv_bphi.values)
            Bzerr.append(piv_bz_err.values)
            Brerr.append(piv_br_err.values)
            Bphierr.append(piv_bphi_err.values)
            RR_slice,ZZ_slice = np.meshgrid(R, Z)
            RR.append(RR_slice)
            ZZ.append(ZZ_slice)
            PP_slice = np.full_like(RR_slice,input_data_phi.Phi.unique()[0])
            PP_slice[:,PP_slice.shape[1]/2:]=input_data_phi.Phi.unique()[1]
            PP.append(PP_slice)

        ZZ = np.concatenate(ZZ)
        RR = np.concatenate(RR)
        PP = np.concatenate(PP)
        Bz = np.concatenate(Bz)
        Br = np.concatenate(Br)
        Bphi = np.concatenate(Bphi)
        Bzerr = np.concatenate(Bzerr)
        Brerr = np.concatenate(Brerr)
        Bphierr = np.concatenate(Bphierr)
        if line_profile:
            return ZZ,RR,PP,Bz,Br,Bphi

        brzphi_3d_fast = brzphi_3d_producer(ZZ,RR,PP,Reff,ns,ms)
        self.mod = Model(brzphi_3d_fast, independent_vars=['r','z','phi'])

        if use_pickle or recreate:
            self.params = pkl.load(open(pickle_name+'_results.p',"rb"))
        else:
            self.params = Parameters()
        delta_seeds = [0, 0.00059746, 0.00452236, 1.82217664, 1.54383364, 0.92910890, 2.3320e-6, 1.57188824, 3.02599942, 3.04222595]


        if 'R' not in    self.params: self.params.add('R',value=Reff,vary=False)
        if 'ns' not in self.params: self.params.add('ns',value=ns,vary=False)
        else: self.params['ns'].value=ns
        if 'ms' not in self.params: self.params.add('ms',value=ms,vary=False)
        else: self.params['ms'].value=ms
        if 'cns' not in self.params: self.params.add('cns',value=cns,vary=False)
        else: self.params['cns'].value=cns
        if 'cms' not in self.params: self.params.add('cms',value=cms,vary=False)
        else: self.params['cms'].value=cms

        for n in range(ns):
            if 'delta_{0}'.format(n) not in self.params: self.params.add('delta_{0}'.format(n),
                    #value=delta_seeds[n], min=0, max=np.pi, vary=True)
                    value=n*(2*np.pi/ns), min=0, max=2*np.pi, vary=True)
            else: self.params['delta_{0}'.format(n)].vary=True
            for m in range(ms):
                #if 'A_{0}_{1}'.format(n,m) not in self.params: self.params.add('A_{0}_{1}'.format(n,m),value=-100)
                if 'A_{0}_{1}'.format(n,m) not in self.params: self.params.add('A_{0}_{1}'.format(n,m),value=0)
                else: self.params['A_{0}_{1}'.format(n,m)].vary=True
                #if 'B_{0}_{1}'.format(n,m) not in self.params: self.params.add('B_{0}_{1}'.format(n,m),value=100)
                if 'B_{0}_{1}'.format(n,m) not in self.params: self.params.add('B_{0}_{1}'.format(n,m),value=0)
                else: self.params['B_{0}_{1}'.format(n,m)].vary=True
        for cn in range(1,cns+1):
            for cm in range(1,cms+1):
                if 'C_{0}_{1}'.format(cn,cm) not in self.params: self.params.add('C_{0}_{1}'.format(cn,cm),value=0.0001,vary=True)
                else:
                    #self.params['C_{0}_{1}'.format(cn,cm)].value=0
                    self.params['C_{0}_{1}'.format(cn,cm)].vary=True

        if not recreate: print 'fitting with n={0}, m={1}'.format(ns,ms)
        start_time=time()
        if recreate:
            for param in self.params:
                self.params[param].vary=False
            self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
                #weights = np.concatenate([Brerr,Bzerr,Bphierr]).ravel(),
                r=RR, z=ZZ, phi=PP, params = self.params, method='leastsq',fit_kws={'maxfev':1})
        elif use_pickle:
            self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
                #weights = np.concatenate([Brerr,Bzerr,Bphierr]).ravel(),
                r=RR, z=ZZ, phi=PP, params = self.params, method='leastsq',fit_kws={'maxfev':1000})
        else:
            self.result = self.mod.fit(np.concatenate([Br,Bz,Bphi]).ravel(),
                #weights = np.concatenate([Brerr,Bzerr,Bphierr]).ravel(),
                r=RR, z=ZZ, phi=PP, params = self.params, method='leastsq',fit_kws={'maxfev':2000})
                #r=RR, z=ZZ, phi=PP, params = self.params, method='differential_evolution',fit_kws={'maxfun':1})
                #r=RR, z=ZZ, phi=PP, params = self.params, method='leastsq')

        self.params = self.result.params
        end_time=time()
        if not recreate:
            print("Elapsed time was %g seconds" % (end_time - start_time))
            report_fit(self.result, show_correl=False)
        if not self.no_save and not recreate: self.pickle_results(pickle_name)


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
            #    self.params[param].vary = False
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
                #    self.result = self.mod.fit(np.concatenate([self.Br,self.Bz]).ravel(),r=self.X,z=self.Y, params = self.params,method='leastsq')
                #    self.params = self.result.params

            #        fit_kws={'xtol':1e-100,'ftol':1e-100,'maxfev':5000,'epsfcn':1e-40})
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
            self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params, weights    = self.Zerr.ravel(), method='leastsq')
            #for param in self.params:
            #    self.params[param].vary = False
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
            #        fit_kws={'xtol':1e-100,'ftol':1e-100,'maxfev':5000,'epsfcn':1e-40})
            self.result = self.mod.fit(self.Z.ravel(),r=self.X,z=self.Y, params = self.params, weights    = self.Zerr.ravel(), method='leastsq')
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

    def pickle_results(self,pickle_name='default'):
        pkl.dump( self.result.params, open( pickle_name+'_results.p', "wb" ),pkl.HIGHEST_PROTOCOL )


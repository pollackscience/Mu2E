#! /usr/bin/env python
"""Module for fitting magnetic field data with a parametric expression.

This is the main workhorse module for fitting the magnetic field data.  It is assumed that the data
has passed through the :class:`mu2e.dataframeprod.DataFrameMaker`, and thus in the expected format.
In most cases the data has also passed through the :mod:`mu2e.hallprober` module, to imitate the act
of surveying one of the Mu2E solenoids with a series of hall probes.  The
:class:`mu2e.fieldfitter.FieldFitter` preps the data, flattens it into 1D, and uses the :mod:`lmfit`
package extensively for parameter-handling, fitting, optimization, etc.

Example:
    Coming soon...

*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""

from __future__ import division
from time import time
import numpy as np
import cPickle as pkl
import pandas as pd
from lmfit import Model, Parameters, report_fit
from mu2e import mu2e_ext_path
import tools.fit_funcs as ff


class FieldFitter:
    """Input field measurements, perform parametric fit, return relevant quantities.

    The :class:`mu2e.fieldfitter.FieldFitter` takes a 3D set of field measurements and their
    associated position values, and performs a parametric fit.  The parameters and fit model are
    handled by the :mod:`lmfit` package, which in turn wraps the :mod:`scipy.optimize` module, which
    actually performs the parameter optimization.  The default optimizer is the Levenberg–Marquardt
    algorithm.

    The :func:`mu2e.fieldfitter.FieldFitter.fit` requires multiple cfg `namedtuples`, and performs
    the actual fitting (or recreates a fit for a given set of saved parameters).  After fitting, the
    generated class members can be used for further analysis.

    Args:
        input_data (pandas.DataFrame): DF that contains the field component values to be fit.
        cfg_geom (namedtuple): namedtuple with the following members:
            'geom z_steps r_steps phi_steps xy_steps bad_calibration'

    Attributes:
        input_data (pandas.DataFrame): The input DF, with possible modifications.
        phi_steps (List[float]): The axial values of the field data (cylindrial coords)
        r_steps (List[float]): The radial values of the field data (cylindrial coords)
        xy_steps (List[float]): The xy values of the field data (cartesian coords)
        pickle_path (str): Location to read/write the pickled fit parameter values
        params (lmfit.Parameters): Set of Parameters, inherited from `lmfit`
        result (lmfit.ModelResult): Container for resulting fit information, inherited from `lmfit`

    """
    def __init__(self, input_data, cfg_geom):
        self.input_data = input_data
        if cfg_geom.geom == 'cyl':
            self.phi_steps = cfg_geom.phi_steps
            self.r_steps = cfg_geom.r_steps
        elif cfg_geom.geom == 'cart':
            self.xy_steps = cfg_geom.xy_steps
        self.pickle_path = mu2e_ext_path+'fit_params/'

    def fit(self, geom, cfg_params, cfg_pickle, profile=False):
        if profile:
            return self.fit_solenoid(cfg_params, cfg_pickle, profile)
        if geom == 'cyl':
            self.fit_solenoid(cfg_params, cfg_pickle, profile)
        elif geom == 'cart':
            self.fit_external(cfg_params, cfg_pickle)

    def fit_solenoid(self, cfg_params, cfg_pickle, profile):
        Reff         = cfg_params.Reff
        ns           = cfg_params.ns
        ms           = cfg_params.ms
        func_version = cfg_params.func_version
        Bz           = []
        Br           = []
        Bphi         = []
        RR           = []
        ZZ           = []
        PP           = []

        for phi in self.phi_steps:
            # determine phi and negative phi
            if phi == 0:
                nphi = np.pi
            else:
                nphi = phi-np.pi

            # select data with correct pair of phis
            input_data_phi = self.input_data[
                (np.isclose(self.input_data.Phi, phi)) | (np.isclose(self.input_data.Phi, nphi))
            ]
            # make radial values negative for negative phis to prevent degeneracy
            input_data_phi.ix[np.isclose(input_data_phi.Phi, nphi), 'R'] *= -1

            # convert B field components into 2D arrays
            piv_bz = input_data_phi.pivot('Z', 'R', 'Bz')
            piv_br = input_data_phi.pivot('Z', 'R', 'Br')
            piv_bphi = input_data_phi.pivot('Z', 'R', 'Bphi')

            # bookkeeping for field and position values
            R = piv_br.columns.values
            Z = piv_br.index.values
            Bz.append(piv_bz.values)
            Br.append(piv_br.values)
            Bphi.append(piv_bphi.values)
            RR_slice, ZZ_slice = np.meshgrid(R, Z)
            RR.append(RR_slice)
            ZZ.append(ZZ_slice)
            use_phis = np.sort(input_data_phi.Phi.unique())
            # formatting for correct phi ordering
            if phi == 0:
                use_phis = use_phis[::-1]
            PP_slice = np.full_like(RR_slice, use_phis[0])
            PP_slice[:, int(PP_slice.shape[1]/2):] = use_phis[1]
            PP.append(PP_slice)

        # combine all phi slices
        ZZ = np.concatenate(ZZ)
        RR = np.concatenate(RR)
        PP = np.concatenate(PP)
        Bz = np.concatenate(Bz)
        Br = np.concatenate(Br)
        Bphi = np.concatenate(Bphi)
        if profile:
            # terminate here if we are profiling the code for further optimization
            return ZZ, RR, PP, Bz, Br, Bphi

        # Choose the type of fitting function we'll be using.
        if func_version == 1:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel(ZZ, RR, PP, Reff, ns, ms)
        elif func_version == 2:
            brzphi_3d_fast = ff.brzphi_3d_producer_bessel(ZZ, RR, PP, Reff, ns, ms)
        elif func_version == 3:
            brzphi_3d_fast = ff.brzphi_3d_producer_bessel_hybrid(ZZ, RR, PP, Reff, ns, ms)
        elif func_version == 4:
            brzphi_3d_fast = ff.brzphi_3d_producer_numba_v2(ZZ, RR, PP, Reff, ns, ms)
        elif func_version == 5:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel_phase(ZZ, RR, PP, Reff, ns, ms)
        else:
            raise KeyError('func version '+func_version+' does not exist')

        # Generate an lmfit Model
        self.mod = Model(brzphi_3d_fast, independent_vars=['r', 'z', 'phi'])

        # Load pre-defined starting valyes for parameters, or make a new set
        if cfg_pickle.use_pickle or cfg_pickle.recreate:
            self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p', "rb"))
        else:
            self.params = Parameters()

        if 'R' not in self.params:
            self.params.add('R', value=Reff, vary=False)
        if 'ns' not in self.params:
            self.params.add('ns', value=ns, vary=False)
        else:
            self.params['ns'].value = ns
        if 'ms' not in self.params:
            self.params.add('ms', value=ms, vary=False)
        else:
            self.params['ms'].value = ms

        for n in range(ns):
            # If function version 5, `D` parameter is a delta offset for phi
            if func_version == 5:
                if 'D_{0}'.format(n) not in self.params:
                    self.params.add('D_{0}'.format(n), value=0, min=-np.pi*0.5, max=np.pi*0.5)
                else:
                    self.params['D_{0}'.format(n)].vary = True
            # Otherwise `D` parameter is a scaling constant, along with a `C` parameter
            else:
                if 'C_{0}'.format(n) not in self.params:
                    self.params.add('C_{0}'.format(n), value=1)
                else:
                    self.params['C_{0}'.format(n)].vary = True
                if 'D_{0}'.format(n) not in self.params:
                    self.params.add('D_{0}'.format(n), value=0.001)
                else:
                    self.params['D_{0}'.format(n)].vary = True

            for m in range(ms):
                if 'A_{0}_{1}'.format(n, m) not in self.params:
                    self.params.add('A_{0}_{1}'.format(n, m), value=0, vary=True)
                else:
                    self.params['A_{0}_{1}'.format(n, m)].vary = True
                if 'B_{0}_{1}'.format(n, m) not in self.params:
                    self.params.add('B_{0}_{1}'.format(n, m), value=0, vary=True)
                else:
                    self.params['B_{0}_{1}'.format(n, m)].vary = True
                # Additional terms used in func version 3
                if func_version == 3:
                    if 'E_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('E_{0}_{1}'.format(n, m), value=0, vary=True)
                    else:
                        self.params['E_{0}_{1}'.format(n, m)].vary = True
                    if 'F_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('F_{0}_{1}'.format(n, m), value=0, vary=True)
                    else:
                        self.params['F_{0}_{1}'.format(n, m)].vary = True
                    if m > 3:
                        self.params['E_{0}_{1}'.format(n, m)].vary = False
                        self.params['F_{0}_{1}'.format(n, m)].vary = False

        if not cfg_pickle.recreate:
            print 'fitting with n={0}, m={1}'.format(ns, ms)
        else:
            print 'recreating fit with n={0}, m={1}, pickle_file={2}'.format(
                ns, ms, cfg_pickle.load_name)
        start_time = time()
        if cfg_pickle.recreate:
            for param in self.params:
                self.params[param].vary = False
            self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                       r=RR, z=ZZ, phi=PP, params=self.params,
                                       method='leastsq', fit_kws={'maxfev': 1})
        elif cfg_pickle.use_pickle:
            mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
            self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                       weights=np.concatenate([mag, mag, mag]).ravel(),
                                       r=RR, z=ZZ, phi=PP, params=self.params,
                                       method='leastsq', fit_kws={'maxfev': 10000})
        else:
            mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
            self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                       weights=np.concatenate([mag, mag, mag]).ravel(),
                                       r=RR, z=ZZ, phi=PP, cfg_params=self.params,
                                       method='leastsq', fit_kws={'maxfev': 2000})

        self.params = self.result.params
        end_time = time()
        print("Elapsed time was %g seconds" % (end_time - start_time))
        report_fit(self.result, show_correl=False)
        if cfg_pickle.save_pickle and not cfg_pickle.recreate:
            self.pickle_results(self.pickle_path+cfg_pickle.save_name)

    def fit_external(self, cns=1, cms=1, use_pickle=False, pickle_name='default',
                     line_profile=False, recreate=False):

        a     = 3e4
        b     = 3e4
        c     = 3e4

        Bz    = []
        Bx    = []
        By    = []
        Bzerr = []
        Bxerr = []
        Byerr = []
        ZZ    = []
        XX    = []
        YY    = []
        for y in self.xy_steps:

            input_data_y = self.input_data[self.input_data.Y == y]

            piv_bz = input_data_y.pivot('Z', 'X', 'Bz')
            piv_bx = input_data_y.pivot('Z', 'X', 'Bx')
            piv_by = input_data_y.pivot('Z', 'X', 'By')
            piv_bz_err = input_data_y.pivot('Z', 'X', 'Bzerr')
            piv_bx_err = input_data_y.pivot('Z', 'X', 'Bxerr')
            piv_by_err = input_data_y.pivot('Z', 'X', 'Byerr')

            X = piv_bx.columns.values
            Z = piv_bx.index.values
            Bz.append(piv_bz.values)
            Bx.append(piv_bx.values)
            By.append(piv_by.values)
            Bzerr.append(piv_bz_err.values)
            Bxerr.append(piv_bx_err.values)
            Byerr.append(piv_by_err.values)
            XX_slice, ZZ_slice = np.meshgrid(X, Z)
            XX.append(XX_slice)
            ZZ.append(ZZ_slice)
            YY_slice = np.full_like(XX_slice, y)
            YY.append(YY_slice)

        ZZ = np.concatenate(ZZ)
        XX = np.concatenate(XX)
        YY = np.concatenate(YY)
        Bz = np.concatenate(Bz)
        Bx = np.concatenate(Bx)
        By = np.concatenate(By)
        Bzerr = np.concatenate(Bzerr)
        Bxerr = np.concatenate(Bxerr)
        Byerr = np.concatenate(Byerr)
        if line_profile:
            return ZZ, XX, YY, Bz, Bx, By

        b_external_3d_fast = ff.b_external_3d_producer(a, b, c, ZZ, XX, YY, cns, cms)
        self.mod = Model(b_external_3d_fast, independent_vars=['x', 'y', 'z'])

        if use_pickle or recreate:
            self.params = pkl.load(open(pickle_name+'_results.p', "rb"))
        else:
            self.params = Parameters()

        if 'cns' not in self.params:
            self.params.add('cns', value=cns, vary=False)
        else:
            self.params['cns'].value = cns
        if 'cms' not in self.params:
            self.params.add('cms', value=cms, vary=False)
        else:
            self.params['cms'].value = cms

        if 'epsilon1' not in self.params:
            self.params.add('epsilon1', value=0.1, min=0, max=2*np.pi, vary=True)
        else:
            self.params['epsilon1'].vary = True
        if 'epsilon2' not in self.params:
            self.params.add('epsilon2', value=0.1, min=0, max=2*np.pi, vary=True)
        else:
            self.params['epsilon2'].vary = True

        for cn in range(1, cns+1):
            for cm in range(1, cms+1):
                if 'C_{0}_{1}'.format(cn, cm) not in self.params:
                    self.params.add('C_{0}_{1}'.format(cn, cm), value=1, vary=True)
                else:
                    self.params['C_{0}_{1}'.format(cn, cm)].vary = True

        if not recreate:
            print 'fitting external with cn={0}, cm={1}'.format(cns, cms)
        start_time = time()
        if recreate:
            for param in self.params:
                self.params[param].vary = False

            self.result = self.mod.fit(np.concatenate([Bx, By, Bz]).ravel(), x=XX, y=YY, z=ZZ,
                                       params=self.params, method='leastsq', fit_kws={'maxfev': 1})
        elif use_pickle:
            self.result = self.mod.fit(np.concatenate([Bx, By, Bz]).ravel(), x=XX, y=YY, z=ZZ,
                                       params=self.params, method='leastsq',
                                       fit_kws={'maxfev': 1000})
        else:
            self.result = self.mod.fit(np.concatenate([Bx, By, Bz]).ravel(), x=XX, y=YY, z=ZZ,
                                       params=self.params, method='leastsq')

        self.params = self.result.params
        end_time = time()
        if not recreate:
            print("Elapsed time was %g seconds" % (end_time - start_time))
            report_fit(self.result, show_correl=False)
        if not self.no_save and not recreate:
            self.pickle_results(pickle_name)

    def pickle_results(self, pickle_name='default'):
        pkl.dump(self.result.params, open(pickle_name+'_results.p', "wb"), pkl.HIGHEST_PROTOCOL)

    def merge_data_fit_res(self):
        '''Combine the fit results and the input data into one dataframe for easier
        comparison of results.
        Adds three columns to input_data:
            Br_fit, Bphi_fit, Bz_fit
        '''
        best_fit = self.result.best_fit

        df_fit = pd.DataFrame()
        isc = np.isclose
        for i, phi in enumerate(self.phi_steps):
            if phi == 0:
                nphi = np.pi
            else:
                nphi = phi-np.pi
            data_frame_phi = self.input_data[
                (isc(self.input_data.Phi, phi)) | (isc(self.input_data.Phi, nphi))
            ]

            # careful sorting of values to match up with fit output bookkeeping
            df_fit_tmp = data_frame_phi[
                isc(data_frame_phi.Phi, nphi)][['Z', 'R', 'Phi']].sort_values(
                    ['R', 'Phi', 'Z'], ascending=[False, True, True])
            df_fit_tmp = df_fit_tmp.append(
                data_frame_phi[isc(data_frame_phi.Phi, phi)][['Z', 'R', 'Phi']].sort_values(
                    ['R', 'Phi', 'Z'], ascending=[True, True, True]))

            l = int(len(best_fit)/3)
            br = best_fit[:l]
            bz = best_fit[l:int(2*l)]
            bphi = best_fit[int(2*l):]
            p = len(br)
            br = br[(i/len(self.phi_steps))*p:((i+1)/len(self.phi_steps))*p]
            bz = bz[(i/len(self.phi_steps))*p:((i+1)/len(self.phi_steps))*p]
            bphi = bphi[(i/len(self.phi_steps))*p:((i+1)/len(self.phi_steps))*p]

            z_size = len(df_fit_tmp.Z.unique())
            r_size = 2*len(df_fit_tmp.R.unique())
            df_fit_tmp['Br_fit'] = np.transpose(br.reshape((z_size, r_size))).flatten()
            df_fit_tmp['Bz_fit'] = np.transpose(bz.reshape((z_size, r_size))).flatten()
            df_fit_tmp['Bphi_fit'] = np.transpose(bphi.reshape((z_size, r_size))).flatten()

            df_fit = df_fit.append(df_fit_tmp)

        self.input_data = pd.merge(self.input_data, df_fit, on=['Z', 'R', 'Phi'])

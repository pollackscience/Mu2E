#! /usr/bin/env python
"""Module for fitting magnetic field data with a parametric expression.

This is the main workhorse module for fitting the magnetic field data.  It is assumed that the data
has passed through the :class:`mu2e.dataframeprod.DataFrameMaker`, and thus in the expected format.
In most cases the data has also passed through the :mod:`mu2e.hallprober` module, to imitate the act
of surveying one of the Mu2E solenoids with a series of hall probes.  The
:class:`mu2e.fieldfitter.FieldFitter` preps the data, flattens it into 1D, and uses the :mod:`lmfit`
package extensively for parameter-handling, fitting, optimization, etc.

Example:
    Incomplete excerpt, see :func:`mu2e.hallprober.field_map_analysis` and `scripts/hallprobesim`
    for more typical use cases:

    .. code-block:: python

        # assuming config files already defined...

        In [10]: input_data = DataFileMaker(cfg_data.path, use_pickle=True).data_frame
        ...      input_data.query(' and '.join(cfg_data.conditions))

        In [11]: hpg = HallProbeGenerator(
        ...         input_data, z_steps = cfg_geom.z_steps,
        ...         r_steps = cfg_geom.r_steps, phi_steps = cfg_geom.phi_steps,
        ...         x_steps = cfg_geom.x_steps, y_steps = cfg_geom.y_steps)

        In [12]: ff = FieldFitter(hpg.get_toy(), cfg_geom)

        In [13]: ff.fit(cfg_geom.geom, cfg_params, cfg_pickle)
        ...      # This will take some time, especially for many data points and free params

        In [14]: ff.merge_data_fit_res() # merge the results in for easy plotting

        In [15]: make_fit_plots(ff.input_data, cfg_data, cfg_geom, cfg_plot, name)
        ...      # defined in :class:`mu2e.hallprober`

*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
from time import time
import numpy as np
import six.moves.cPickle as pkl
import pandas as pd
from lmfit import Model, Parameters, report_fit
from mu2e import mu2e_ext_path
from mu2e.tools import fit_funcs as ff
# import tools.fit_func_class as ffc


class FieldFitter:
    """Input field measurements, perform parametric fit, return relevant quantities.

    The :class:`mu2e.fieldfitter.FieldFitter` takes a 3D set of field measurements and their
    associated position values, and performs a parametric fit.  The parameters and fit model are
    handled by the :mod:`lmfit` package, which in turn wraps the :mod:`scipy.optimize` module, which
    actually performs the parameter optimization.  The default optimizer is the Levenberg-Marquardt
    algorithm.

    The :func:`mu2e.fieldfitter.FieldFitter.fit` requires multiple cfg `namedtuples`, and performs
    the actual fitting (or recreates a fit for a given set of saved parameters).  After fitting, the
    generated class members can be used for further analysis.

    Args:
        input_data (pandas.DataFrame): DF that contains the field component values to be fit.
        cfg_geom (namedtuple): namedtuple with the following members:
            'geom z_steps r_steps phi_steps x_steps y_steps bad_calibration'

    Attributes:
        input_data (pandas.DataFrame): The input DF, with possible modifications.
        phi_steps (List[float]): The axial values of the field data (cylindrial coords)
        r_steps (List[float]): The radial values of the field data (cylindrial coords)
        x_steps (List[float]): The x values of the field data (cartesian coords)
        y_steps (List[float]): The y values of the field data (cartesian coords)
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
            self.x_steps = cfg_geom.x_steps
            self.y_steps = cfg_geom.y_steps
        self.pickle_path = mu2e_ext_path+'fit_params/'
        self.geom = cfg_geom.geom

    def fit(self, geom, cfg_params, cfg_pickle, profile=False):
        """Helper function that chooses one of the subsequent fitting functions."""

        if profile:
            return self.fit_solenoid(cfg_params, cfg_pickle, profile)
        if geom == 'cyl':
            self.fit_solenoid(cfg_params, cfg_pickle, profile)
        elif geom == 'cart':
            self.fit_external(cfg_params, cfg_pickle)

    def fit_solenoid(self, cfg_params, cfg_pickle, profile=False):
        """Main fitting function for FieldFitter class.

        The typical magnetic field geometry for the Mu2E experiment is determined by one or more
        solenoids, with some contaminating external fields.  The purpose of this function is to fit
        a set of sparse magnetic field data that would, in practice, be generated by a field
        measurement device.

        The following assumptions must hold for the input data:
           * The data is represented in a cylindrical coordiante system.
           * The data forms a series of planes, where all planes intersect at R=0.
           * All planes has the same R and Z values.
           * All positive Phi values have an associated negative phi value, which uniquely defines a
             single plane in R-Z space.

        Args:
           cfg_params (namedtuple): 'ns ms cns cms Reff func_version'
           cfg_pickle (namedtuple): 'use_pickle save_pickle load_name save_name recreate'
           profile (Optional[bool]): True if you want to exit after the model is built, before
               actual fitting is performed for profiling. Default is False.

        Returns:
            Nothing.  Generates class attributes after fitting, and saves parameter values, if
            saving is specified.
        """
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
        XX           = []
        YY           = []
        cns = cfg_params.cns
        cms = cfg_params.cms
        if func_version in [8, 9]:
            self.input_data.eval('Xp = X+1075', inplace=True)
            self.input_data.eval('Yp = Y-440', inplace=True)
            self.input_data.eval('Rp = sqrt(Xp**2+Yp**2)', inplace=True)
            self.input_data.eval('Phip = arctan2(Yp,Xp)', inplace=True)
            RRP = []
            PPP = []

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
            input_data_phi.loc[np.isclose(input_data_phi.Phi, nphi), 'R'] *= -1

            # convert B field components into 2D arrays
            # print input_data_phi
            piv_bz = input_data_phi.pivot('Z', 'R', 'Bz')
            piv_br = input_data_phi.pivot('Z', 'R', 'Br')
            piv_bphi = input_data_phi.pivot('Z', 'R', 'Bphi')
            piv_phi = input_data_phi.pivot('Z', 'R', 'Phi')
            if func_version == 6:
                piv_x = input_data_phi.pivot('Z', 'R', 'X')
                piv_y = input_data_phi.pivot('Z', 'R', 'Y')
            elif func_version in [8, 9]:
                piv_rp = input_data_phi.pivot('Z', 'R', 'Rp')
                piv_phip = input_data_phi.pivot('Z', 'R', 'Phip')

            # bookkeeping for field and position values
            R = piv_br.columns.values
            Z = piv_br.index.values
            Bz.append(piv_bz.values)
            Br.append(piv_br.values)
            Bphi.append(piv_bphi.values)
            RR_slice, ZZ_slice = np.meshgrid(R, Z)
            RR.append(RR_slice)
            ZZ.append(ZZ_slice)
            if func_version == 6:
                XX.append(piv_x.values)
                YY.append(piv_y.values)
            elif func_version in [8, 9]:
                RRP.append(piv_rp.values)
                PPP.append(piv_phip.values)
            # use_phis = np.sort(input_data_phi.Phi.unique())
            # formatting for correct phi ordering
            # if phi == 0:
            #     use_phis = use_phis[::-1]
            # print phi, use_phis
            # PP_slice = np.full_like(RR_slice, use_phis[0])
            # PP_slice[:, int(PP_slice.shape[1]/2):] = use_phis[1]
            PP.append(piv_phi.values)

        # combine all phi slices
        ZZ = np.concatenate(ZZ)
        RR = np.concatenate(RR)
        PP = np.concatenate(PP)
        Bz = np.concatenate(Bz)
        Br = np.concatenate(Br)
        Bphi = np.concatenate(Bphi)
        if func_version == 6:
            XX = np.concatenate(XX)
            YY = np.concatenate(YY)
        if func_version in [8, 9]:
            RRP = np.concatenate(RRP)
            PPP = np.concatenate(PPP)
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
            # factory = ffc.FunctionProducer(RR, PP, ZZ, ns, ms, 'modbessel', L=Reff)
            # brzphi_3d_fast = factory.get_fit_function()
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel_phase(ZZ, RR, PP, Reff, ns, ms)
        elif func_version == 6:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel_phase_ext(ZZ, RR, PP, Reff, ns, ms,
                                                                       cns, cms)
        elif func_version == 7:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel_phase_hybrid(ZZ, RR, PP, Reff, ns, ms,
                                                                          cns, cms)
        elif func_version == 8:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel_phase_hybrid_disp2(ZZ, RR, PP, RRP,
                                                                                PPP, Reff, ns, ms,
                                                                                cns, cms)
        elif func_version == 9:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel_phase_hybrid_disp3(ZZ, RR, PP, RRP,
                                                                                PPP, Reff, ns, ms,
                                                                                cns, cms)
        elif func_version == 100:
            brzphi_3d_fast = ff.brzphi_3d_producer_hel_v0(ZZ, RR, PP, Reff, ns, ms)
        else:
            raise KeyError('func version '+func_version+' does not exist')

        # Generate an lmfit Model
        if func_version in [6]:
            self.mod = Model(brzphi_3d_fast, independent_vars=['r', 'z', 'phi', 'x', 'y'])
        elif func_version in [8, 9, 10]:
            self.mod = Model(brzphi_3d_fast, independent_vars=['r', 'z', 'phi', 'rp', 'phip'])
        else:
            self.mod = Model(brzphi_3d_fast, independent_vars=['r', 'z', 'phi'])

        # Load pre-defined starting valyes for parameters, or make a new set
        if cfg_pickle.use_pickle or cfg_pickle.recreate:
            try:
                self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
                                            "rb"))
            except UnicodeDecodeError:
                self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p',
                                            "rb"), encoding='latin1')
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

        if func_version < 100:
            for n in range(ns):
                # If function version 5, `D` parameter is a delta offset for phi
                if func_version in [5, 6, 7, 8, 9]:
                    d_starts = np.linspace(-np.pi*0.5+0.1, np.pi*0.5-0.1, ns)
                    if 'D_{0}'.format(n) not in self.params:
                        self.params.add('D_{0}'.format(n), value=d_starts[n], min=-np.pi*0.5,
                                        max=np.pi*0.5, vary=True)
                    else:
                        self.params['D_{0}'.format(n)].vary = False
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

        else:
            for n in range(ns):
                for m in range(ms):
                    if 'A_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('A_{0}_{1}'.format(n, m), value=0, vary=True)
                    else:
                        self.params['A_{0}_{1}'.format(n, m)].vary = True
                    if 'B_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('B_{0}_{1}'.format(n, m), value=0, vary=True)
                    else:
                        self.params['B_{0}_{1}'.format(n, m)].vary = True
                    if 'C_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('C_{0}_{1}'.format(n, m), value=0, vary=True)
                    else:
                        self.params['C_{0}_{1}'.format(n, m)].vary = True
                    if 'D_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('D_{0}_{1}'.format(n, m), value=0, vary=True)
                    else:
                        self.params['D_{0}_{1}'.format(n, m)].vary = True

                    if n > m:
                        self.params['A_{0}_{1}'.format(n, m)].vary = False
                        self.params['A_{0}_{1}'.format(n, m)].value = 0
                        self.params['B_{0}_{1}'.format(n, m)].vary = False
                        self.params['B_{0}_{1}'.format(n, m)].value = 0
                        self.params['C_{0}_{1}'.format(n, m)].vary = False
                        self.params['C_{0}_{1}'.format(n, m)].value = 0
                        self.params['D_{0}_{1}'.format(n, m)].vary = False
                        self.params['D_{0}_{1}'.format(n, m)].value = 0

        if func_version == 6:

            if 'cns' not in self.params:
                self.params.add('cns', value=cns, vary=False)
            else:
                self.params['cns'].value = cns
            if 'cms' not in self.params:
                self.params.add('cms', value=cms, vary=False)
            else:
                self.params['cms'].value = cms
            for cn in range(cns):
                for cm in range(cms):
                    if 'E_{0}_{1}'.format(cn, cm) not in self.params:
                        self.params.add('E_{0}_{1}'.format(cn, cm), value=0.01, vary=True)
                    else:
                        self.params['E_{0}_{1}'.format(cn, cm)].vary = True
                    if 'F_{0}_{1}'.format(cn, cm) not in self.params:
                        self.params.add('F_{0}_{1}'.format(cn, cm), value=0.01, vary=True)
                    else:
                        self.params['F_{0}_{1}'.format(cn, cm)].vary = True
            if 'k1' not in self.params:
                self.params.add('k1', value=0, vary=True)
            else:
                self.params['k1'].vary = True
            if 'k2' not in self.params:
                self.params.add('k2', value=0, vary=True)
            else:
                self.params['k2'].vary = True
            if 'k3' not in self.params:
                self.params.add('k3', value=0, vary=True)
            else:
                self.params['k3'].vary = True
            if 'k4' not in self.params:
                self.params.add('k4', value=0, vary=True)
            else:
                self.params['k4'].vary = True
            if 'k5' not in self.params:
                self.params.add('k5', value=0, vary=True)
            else:
                self.params['k5'].vary = True
            if 'k6' not in self.params:
                self.params.add('k6', value=0, vary=True)
            else:
                self.params['k6'].vary = True
            if 'k7' not in self.params:
                self.params.add('k7', value=0, vary=True)
            else:
                self.params['k7'].vary = True
            if 'k8' not in self.params:
                self.params.add('k8', value=0, vary=True)
            else:
                self.params['k8'].vary = True
            if 'k9' not in self.params:
                self.params.add('k9', value=0, vary=True)
            else:
                self.params['k9'].vary = True
            if 'k10' not in self.params:
                self.params.add('k10', value=0, vary=True)
            else:
                self.params['k10'].vary = True
            if 'k11' not in self.params:
                self.params.add('k11', value=0, vary=True)
            else:
                self.params['k10'].vary = True

        if func_version in [7, 8, 9]:
            if 'cns' not in self.params:
                self.params.add('cns', value=cns, vary=False)
            else:
                self.params['cns'].value = cns
            if 'cms' not in self.params:
                self.params.add('cms', value=cms, vary=False)
            else:
                self.params['cms'].value = cms

            for cn in range(cns):
                if 'G_{0}'.format(cn) not in self.params:
                    self.params.add('G_{0}'.format(cn), value=0, min=-np.pi*0.5, max=np.pi*0.5,
                                    vary=True)
                else:
                    self.params['G_{0}'.format(cn)].vary = True

                for cm in range(cms):
                    if 'E_{0}_{1}'.format(cn, cm) not in self.params:
                        self.params.add('E_{0}_{1}'.format(cn, cm), value=0, vary=True)
                    else:
                        self.params['E_{0}_{1}'.format(cn, cm)].vary = True
                    if 'F_{0}_{1}'.format(cn, cm) not in self.params:
                        self.params.add('F_{0}_{1}'.format(cn, cm), value=0, vary=True)
                    else:
                        self.params['F_{0}_{1}'.format(cn, cm)].vary = True

            if func_version in [8, 9]:
                if 'X' not in self.params:
                    self.params.add('X', value=0, vary=True)
                if 'Y' not in self.params:
                    self.params.add('Y', value=0, vary=True)

        if not cfg_pickle.recreate:
            print('fitting with n={0}, m={1}, cn={2}, cm={3}'.format(ns, ms, cns, cms))
        else:
            print('recreating fit with n={0}, m={1}, cn={2}, cm={3}, pickle_file={4}'.format(
                ns, ms, cns, cms, cfg_pickle.load_name))
        start_time = time()
        if func_version not in [6, 8, 9] and func_version < 100:
            if cfg_pickle.recreate:
                for param in self.params:
                    self.params[param].vary = False
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           r=RR, z=ZZ, phi=PP, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 1})
            elif cfg_pickle.use_pickle:
                mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           # weights=np.concatenate([mag, mag, mag]).ravel(),
                                           r=RR, z=ZZ, phi=PP, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 10000})
            else:
                mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           # weights=np.concatenate([mag, mag, mag]).ravel(),
                                           r=RR, z=ZZ, phi=PP, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 10000})
        elif func_version == 6:
            if cfg_pickle.recreate:
                for param in self.params:
                    self.params[param].vary = False
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 1})
            elif cfg_pickle.use_pickle:
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 20000})
            else:
                print('fitting phase ext')
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 10000})

        elif func_version in [8, 9]:
            if cfg_pickle.recreate:
                for param in self.params:
                    self.params[param].vary = False
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           r=RR, z=ZZ, phi=PP, rp=RRP, phip=PPP, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 1})
            elif cfg_pickle.use_pickle:
                mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           weights=np.concatenate([mag, mag, mag]).ravel(),
                                           r=RR, z=ZZ, phi=PP, rp=RRP, phip=PPP, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 12000})
            else:
                mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           weights=np.concatenate([mag, mag, mag]).ravel(),
                                           r=RR, z=ZZ, phi=PP, rp=RRP, phip=PPP, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 2000})
        elif func_version >= 100:
            if cfg_pickle.recreate:
                for param in self.params:
                    self.params[param].vary = False
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           r=RR, z=ZZ, phi=PP, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 1})
            elif cfg_pickle.use_pickle:
                mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           # weights=np.concatenate([mag, mag, mag]).ravel(),
                                           r=RR, z=ZZ, phi=PP, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 10000})
            else:
                mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           # weights=np.concatenate([mag, mag, mag]).ravel(),
                                           r=RR, z=ZZ, phi=PP, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 10000})

        self.params = self.result.params
        end_time = time()
        print(("Elapsed time was %g seconds" % (end_time - start_time)))
        report_fit(self.result, show_correl=False)
        if cfg_pickle.save_pickle:  # and not cfg_pickle.recreate:
            self.pickle_results(self.pickle_path+cfg_pickle.save_name)

    def fit_external(self, cfg_params, cfg_pickle, profile=False):
        """For fitting an external field in cartesian space.

        Note:
            Being revamped!
        """

        Reff         = cfg_params.Reff
        ns           = cfg_params.ns
        ms           = cfg_params.ms
        cns          = cfg_params.cns
        cms          = cfg_params.cms
        func_version = cfg_params.func_version
        Bx           = []
        By           = []
        Bz           = []
        XX           = []
        YY           = []
        ZZ           = []
        for y in self.y_steps:

            input_data_y = self.input_data[self.input_data.Y == y]

            piv_bz = input_data_y.pivot('Z', 'X', 'Bz')
            piv_bx = input_data_y.pivot('Z', 'X', 'Bx')
            piv_by = input_data_y.pivot('Z', 'X', 'By')

            X = piv_bx.columns.values
            Z = piv_bx.index.values
            Bz.append(piv_bz.values)
            Bx.append(piv_bx.values)
            By.append(piv_by.values)
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

        if func_version == 20:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 21:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v2(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 22:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v3(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 23:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v4(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 24:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v5(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 25:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v6(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 26:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v7(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 27:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v8(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 28:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v9(XX, YY, ZZ, Reff, ns, ms, cns, cms)
        elif func_version == 29:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v10(XX, YY, ZZ, Reff, ns, ms, cns, cms)
        elif func_version == 30:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v11(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 31:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v12(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 32:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v13(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 33:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v14(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 34:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v15(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 35:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v16(XX, YY, ZZ, Reff, ns, ms)
        elif func_version == 36:
            bxyz_3d_fast = ff.bxyz_3d_producer_cart_v17(XX, YY, ZZ, Reff, ns, ms)
        self.mod = Model(bxyz_3d_fast, independent_vars=['x', 'y', 'z'])

        # Load pre-defined starting valyes for parameters, or make a new set
        if cfg_pickle.use_pickle or cfg_pickle.recreate:
            self.params = pkl.load(open(self.pickle_path+cfg_pickle.load_name+'_results.p', "rb"))
        else:
            self.params = Parameters()

        if 'L' not in self.params:
            self.params.add('L', value=Reff, vary=False)
        if 'ns' not in self.params:
            self.params.add('ns', value=ns, vary=False)
        else:
            self.params['ns'].value = ns
        if 'ms' not in self.params:
            self.params.add('ms', value=ms, vary=False)
        else:
            self.params['ms'].value = ms

        for n in range(ns):
            for m in range(ms):
                if 'A_{0}_{1}'.format(n, m) not in self.params:
                    self.params.add('A_{0}_{1}'.format(n, m), value=1e-2, vary=True)
                else:
                    self.params['A_{0}_{1}'.format(n, m)].vary = True
                if func_version not in [26, 27, 29]:
                    if 'B_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('B_{0}_{1}'.format(n, m), value=1e-2, vary=True)
                    else:
                        self.params['B_{0}_{1}'.format(n, m)].vary = True
                if func_version in [22, 23, 24, 25, 30, 34, 35]:
                    if 'C_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('C_{0}_{1}'.format(n, m), value=1e-2, vary=True)
                    else:
                        self.params['C_{0}_{1}'.format(n, m)].vary = True
                if func_version in [22, 23, 24, 25, 35]:
                    if 'D_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('D_{0}_{1}'.format(n, m), value=1e-2, vary=True)
                    else:
                        self.params['D_{0}_{1}'.format(n, m)].vary = True
                if func_version in [25, 35]:
                    if 'E_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('E_{0}_{1}'.format(n, m), value=1e-2, vary=True)
                    else:
                        self.params['E_{0}_{1}'.format(n, m)].vary = True
                if func_version in [25, 35]:
                    if 'F_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('F_{0}_{1}'.format(n, m), value=1e-2, vary=True)
                    else:
                        self.params['F_{0}_{1}'.format(n, m)].vary = True
                if func_version in [30, 31, 34]:
                    if 'D_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('D_{0}_{1}'.format(n, m), value=0, min=-np.pi*0.5,
                                        max=np.pi*0.5)
                    else:
                        self.params['D_{0}_{1}'.format(n, m)].vary = True
                if func_version in [26]:
                    if 'B_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('B_{0}_{1}'.format(n, m), value=0, min=-np.pi*0.5,
                                        max=np.pi*0.5)
                    else:
                        self.params['C_{0}_{1}'.format(n, m)].vary = True
                if func_version in [26, 31]:
                    if 'C_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('C_{0}_{1}'.format(n, m), value=0, min=-np.pi*0.5,
                                        max=np.pi*0.5)
                    else:
                        self.params['C_{0}_{1}'.format(n, m)].vary = True

        if func_version in [29]:
            for cn in range(cns):
                for cm in range(cms):
                    if 'B_{0}_{1}'.format(cn, cm) not in self.params:
                        self.params.add('B_{0}_{1}'.format(cn, cm), value=1e-2, vary=True)
                    else:
                        self.params['B_{0}_{1}'.format(cn, cm)].vary = True

        if func_version in [28]:
            for cn in range(cns):
                for cm in range(cms):
                    if 'C_{0}_{1}'.format(cn, cm) not in self.params:
                        self.params.add('C_{0}_{1}'.format(cn, cm), value=1e-2, vary=True)
                    else:
                        self.params['C_{0}_{1}'.format(cn, cm)].vary = True
                    if 'D_{0}_{1}'.format(cn, cm) not in self.params:
                        self.params.add('D_{0}_{1}'.format(cn, cm), value=0, min=-np.pi*0.5,
                                        max=np.pi*0.5)
                    else:
                        self.params['D_{0}_{1}'.format(cn, cm)].vary = True
                    if 'E_{0}_{1}'.format(cn, cm) not in self.params:
                        self.params.add('E_{0}_{1}'.format(cn, cm), value=0, min=-np.pi*0.5,
                                        max=np.pi*0.5)
                    else:
                        self.params['E_{0}_{1}'.format(cn, cm)].vary = True

        if func_version in [20]:
            if 'k1' not in self.params:
                self.params.add('k1', value=5000, vary=True, min=1000, max=10000)
            else:
                self.params['k1'].vary = False
            if 'k2' not in self.params:
                self.params.add('k2', value=5000, vary=True, min=1000, max=10000)
            else:
                self.params['k2'].vary = False

        if func_version in [32, 33, 36]:
            if 'k1' not in self.params:
                self.params.add('k1', value=0, vary=True)
            else:
                self.params['k1'].vary = True
            if 'k2' not in self.params:
                self.params.add('k2', value=0, vary=True)
            else:
                self.params['k2'].vary = True
            if 'k3' not in self.params:
                self.params.add('k3', value=0, vary=True)
            else:
                self.params['k3'].vary = True
        if func_version in [32, 33]:
            if 'k4' not in self.params:
                self.params.add('k4', value=0, vary=True)
            else:
                self.params['k4'].vary = True
            if 'k5' not in self.params:
                self.params.add('k5', value=0, vary=True)
            else:
                self.params['k5'].vary = True
        if func_version in [32]:
            if 'k6' not in self.params:
                self.params.add('k6', value=0, vary=True)
            else:
                self.params['k6'].vary = True
            if 'k7' not in self.params:
                self.params.add('k7', value=0, vary=True)
            else:
                self.params['k7'].vary = True
            if 'k8' not in self.params:
                self.params.add('k8', value=0, vary=True)
            else:
                self.params['k8'].vary = True
            if 'k9' not in self.params:
                self.params.add('k9', value=0, vary=True)
            else:
                self.params['k9'].vary = True
            if 'k10' not in self.params:
                self.params.add('k10', value=0, vary=True)
            else:
                self.params['k10'].vary = True

        if not cfg_pickle.recreate:
            print('fitting with n={0}, m={1}, cn={2}, cm={3}'.format(ns, ms, cns, cms))
        else:
            print('recreating fit with n={0}, m={1}, pickle_file={2}'.format(ns, ms,
                                                                             cfg_pickle.load_name))
        start_time = time()
        if cfg_pickle.recreate:
            for param in self.params:
                self.params[param].vary = False
            self.result = self.mod.fit(np.concatenate([Bx, By, Bz]).ravel(),
                                       x=XX, y=YY, z=ZZ, params=self.params,
                                       method='leastsq', fit_kws={'maxfev': 1})
        elif cfg_pickle.use_pickle:
            # mag = 1/np.sqrt(Bx**2+By**2+Bz**2)
            self.result = self.mod.fit(np.concatenate([Bx, By, Bz]).ravel(),
                                       # weights=np.concatenate([mag, mag, mag]).ravel(),
                                       x=XX, y=YY, z=ZZ, params=self.params,
                                       # method='least_squares', fit_kws={
                                       #     'max_nfev': 200, 'loss': 'soft_l1', 'verbose': 2})
                                       method='leastsq', fit_kws={'maxfev': 10000})
        else:
            # mag = 1/np.sqrt(Bx**2+By**2+Bz**2)
            self.result = self.mod.fit(np.concatenate([Bx, By, Bz]).ravel(),
                                       # weights=np.concatenate([mag, mag, mag]).ravel(),
                                       x=XX, y=YY, z=ZZ, params=self.params,
                                       # method='least_squares', fit_kws={
                                       #     'max_nfev': 1000, 'loss': 'soft_l1', 'verbose': 2})
                                       method='leastsq', fit_kws={'maxfev': 20000})

        self.params = self.result.params
        end_time = time()
        print(("Elapsed time was %g seconds" % (end_time - start_time)))
        report_fit(self.result, show_correl=False)
        if cfg_pickle.save_pickle:  # and not cfg_pickle.recreate:
            self.pickle_results(self.pickle_path+cfg_pickle.save_name)

    def pickle_results(self, pickle_name='default'):
        """Pickle the resulting Parameters after a fit is performed."""

        pkl.dump(self.result.params, open(pickle_name+'_results.p', "wb"), pkl.HIGHEST_PROTOCOL)

    def merge_data_fit_res(self):
        """Combine the fit results and the input data into one dataframe for easier
        comparison of results.

        Adds three columns to input_data: `Br_fit, Bphi_fit, Bz_fit` or `Bx_fit, By_fit, Bz_fit`,
        depending on the geometry.
        """
        best_fit = self.result.best_fit

        df_fit = pd.DataFrame()
        isc = np.isclose

        if self.geom == 'cyl':
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
                br = br[(i*p)//len(self.phi_steps):((i+1)*p)//len(self.phi_steps)]
                bz = bz[(i*p)//len(self.phi_steps):((i+1)*p)//len(self.phi_steps)]
                bphi = bphi[(i*p)//len(self.phi_steps):((i+1)*p)//len(self.phi_steps)]

                z_size = len(df_fit_tmp.Z.unique())
                r_size = 2*len(df_fit_tmp.R.unique())
                df_fit_tmp['Br_fit'] = np.transpose(br.reshape((z_size, r_size))).flatten()
                df_fit_tmp['Bz_fit'] = np.transpose(bz.reshape((z_size, r_size))).flatten()
                df_fit_tmp['Bphi_fit'] = np.transpose(bphi.reshape((z_size, r_size))).flatten()

                df_fit = df_fit.append(df_fit_tmp)

            self.input_data = pd.merge(self.input_data, df_fit, on=['Z', 'R', 'Phi'])

        elif self.geom == 'cart':
            for i, y in enumerate(self.y_steps):
                df_fit_tmp = self.input_data[self.input_data.Y == y][['X', 'Y', 'Z']]

                # careful sorting of values to match up with fit output bookkeeping
                l = int(len(best_fit)/3)
                bx = best_fit[:l]
                by = best_fit[l:int(2*l)]
                bz = best_fit[int(2*l):]
                p = len(bx)
                bx = bx[(i*p)//len(self.y_steps):((i+1)*p)//len(self.y_steps)]
                by = by[(i*p)//len(self.y_steps):((i+1)*p)//len(self.y_steps)]
                bz = bz[(i*p)//len(self.y_steps):((i+1)*p)//len(self.y_steps)]

                z_size = len(df_fit_tmp.Z.unique())
                x_size = len(df_fit_tmp.X.unique())
                df_fit_tmp['Bx_fit'] = np.transpose(bx.reshape((z_size, x_size))).flatten()
                df_fit_tmp['By_fit'] = np.transpose(by.reshape((z_size, x_size))).flatten()
                df_fit_tmp['Bz_fit'] = np.transpose(bz.reshape((z_size, x_size))).flatten()

                df_fit = df_fit.append(df_fit_tmp)

            self.input_data = pd.merge(self.input_data, df_fit, on=['X', 'Y', 'Z'])
        else:
            raise KeyError('Geom is not specified correctly')

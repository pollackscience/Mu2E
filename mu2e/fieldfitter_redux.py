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
from mu2e.tools import fit_funcs_redux as ff
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
        n_scale      = cfg_params.n_scale
        m_scale      = cfg_params.m_scale
        ns           = cfg_params.ns
        ms           = cfg_params.ms
        nms          = cfg_params.nms
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

        ZZ = self.input_data.Z.values
        RR = self.input_data.R.values
        PP = self.input_data.Phi.values
        Bz = self.input_data.Bz.values
        Br = self.input_data.Br.values
        Bphi = self.input_data.Bphi.values
        if func_version in [6, 105, 110, 115, 116, 117]:
            XX = self.input_data.X.values
            YY = self.input_data.Y.values
        if func_version in [8, 9]:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        if profile:
            raise NotImplementedError('Oh no! you got lazy during refactoring')

        # Choose the type of fitting function we'll be using.
        if func_version == 1:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 2:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 3:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 4:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 5:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel_phase(ZZ, RR, PP, Reff, ns, ms)
        elif func_version == 6:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel_phase_ext(ZZ, RR, PP, Reff, ns, ms,
                                                                       cns, cms)
        elif func_version == 7:
            brzphi_3d_fast = ff.brzphi_3d_producer_modbessel_phase_hybrid(ZZ, RR, PP, Reff, ns, ms,
                                                                          cns, cms)
        elif func_version == 8:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 9:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 100:
            brzphi_3d_fast = ff.brzphi_3d_producer_hel_v0(ZZ, RR, PP, Reff, ns, ms)
        elif func_version == 101:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 102:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 103:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 104:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 105:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 106:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 107:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 108:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 109:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 110:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 111:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 112:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 113:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 114:
            brzphi_3d_fast = ff.brzphi_3d_producer_hel_v14(ZZ, RR, PP, Reff, ns, ms, n_scale)
        elif func_version == 115:
            brzphi_3d_fast = ff.brzphi_3d_producer_hel_v15(ZZ, RR, PP, Reff, ns, ms, n_scale)
        elif func_version == 116:
            raise NotImplementedError('Oh no! you got lazy during refactoring')
        elif func_version == 117:
            brzphi_3d_fast = ff.brzphi_3d_producer_hel_v17(ZZ, RR, PP, Reff, ns, ms, n_scale)
        else:
            raise KeyError('func version '+func_version+' does not exist')

        # Generate an lmfit Model
        if func_version in [6, 105, 110, 115, 116, 117]:
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
                        # self.params.add('D_{0}'.format(n), value=d_starts[n], min=-np.pi*0.5,
                        #                 max=np.pi*0.5, vary=True)
                        self.params.add('D_{0}'.format(n), value=0.5, min=0,
                                        max=1, vary=True)
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

        elif func_version < 111:
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

                    if func_version != 103:
                        if 'C_{0}_{1}'.format(n, m) not in self.params:
                            self.params.add('C_{0}_{1}'.format(n, m), value=0, vary=False)
                        else:
                            self.params['C_{0}_{1}'.format(n, m)].vary = False

                        if 'D_{0}_{1}'.format(n, m) not in self.params:
                            self.params.add('D_{0}_{1}'.format(n, m), value=0, vary=False)
                        else:
                            self.params['D_{0}_{1}'.format(n, m)].vary = False

                    if func_version not in [108, 109] and (n*n_scale > m*m_scale) or n*n_scale == 0:
                        self.params['A_{0}_{1}'.format(n, m)].vary = False
                        self.params['A_{0}_{1}'.format(n, m)].value = 0
                        self.params['B_{0}_{1}'.format(n, m)].vary = False
                        self.params['B_{0}_{1}'.format(n, m)].value = 0
                        self.params['C_{0}_{1}'.format(n, m)].vary = False
                        self.params['C_{0}_{1}'.format(n, m)].value = 0
                        self.params['D_{0}_{1}'.format(n, m)].vary = False
                        self.params['D_{0}_{1}'.format(n, m)].value = 0

                    if func_version == 100 and (m-n == 1):
                        self.params['A_{0}_{1}'.format(n, m)].vary = False
                        self.params['A_{0}_{1}'.format(n, m)].value = 0

        elif func_version == 111:
            for n, m in nms:
                if f'A_{n}_{m}' not in self.params:
                    self.params.add(f'A_{n}_{m}', value=1, vary=True)
                else:
                    self.params[f'A_{n}_{m}'].vary = True

                if f'B_{n}_{m}' not in self.params:
                    self.params.add(f'B_{n}_{m}', value=-1, vary=True)
                else:
                    self.params[f'B_{n}_{m}'].vary = True

        elif func_version in [112, 113]:
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
                        self.params.add('C_{0}_{1}'.format(n, m), value=0, vary=True,
                                        min=-0.999, max=0.999)
                    else:
                        self.params['C_{0}_{1}'.format(n, m)].vary = True

                    if func_version == 112:
                        if 'D_{0}_{1}'.format(n, m) not in self.params:
                            self.params.add('D_{0}_{1}'.format(n, m), value=0, vary=True)
                        else:
                            self.params['D_{0}_{1}'.format(n, m)].vary = True

                    if (n*n_scale > m*m_scale) or n*n_scale == 0:
                        self.params['A_{0}_{1}'.format(n, m)].vary = False
                        self.params['A_{0}_{1}'.format(n, m)].value = 0
                        self.params['B_{0}_{1}'.format(n, m)].vary = False
                        self.params['B_{0}_{1}'.format(n, m)].value = 0
                        self.params['C_{0}_{1}'.format(n, m)].vary = False
                        self.params['C_{0}_{1}'.format(n, m)].value = 0
                        if func_version == 112:
                            self.params['D_{0}_{1}'.format(n, m)].vary = False
                            self.params['D_{0}_{1}'.format(n, m)].value = 0

        elif func_version == 114:
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

                    if f'D_{m}' not in self.params:
                        self.params.add(f'D_{m}', value=0, min=-np.pi*0.5,
                                        max=np.pi*0.5, vary=False)
                    else:
                        self.params[f'D_{m}'].vary = False

        elif func_version in [115, 117]:
            for n in range(ns):
                if func_version == 117:
                    if f'D_{n}' not in self.params:
                        self.params.add(f'D_{n}', value=0.5, vary=True, min=0, max=1)
                    else:
                        self.params[f'D_{n}'].vary = True
                for m in range(ms):
                    if 'A_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('A_{0}_{1}'.format(n, m), value=0, vary=True)
                    else:
                        self.params['A_{0}_{1}'.format(n, m)].vary = True

                    if 'B_{0}_{1}'.format(n, m) not in self.params:
                        self.params.add('B_{0}_{1}'.format(n, m), value=0, vary=True)
                    else:
                        self.params['B_{0}_{1}'.format(n, m)].vary = True

                    if n == 0:
                        self.params['A_{0}_{1}'.format(n, m)].vary = False
                        self.params['A_{0}_{1}'.format(n, m)].value = 0
                        self.params['B_{0}_{1}'.format(n, m)].vary = False
                        self.params['B_{0}_{1}'.format(n, m)].value = 0

            if 'k1' not in self.params:
                self.params.add('k1', value=0, vary=False)
            else:
                self.params['k1'].vary = False
            if 'k2' not in self.params:
                self.params.add('k2', value=0, vary=False)
            else:
                self.params['k2'].vary = False
            if 'k3' not in self.params:
                self.params.add('k3', value=0, vary=True)
            else:
                self.params['k3'].vary = True
            if 'k4' not in self.params:
                self.params.add('k4', value=0, vary=False)
            else:
                self.params['k4'].vary = False
            if 'k5' not in self.params:
                self.params.add('k5', value=0, vary=False)
            else:
                self.params['k5'].vary = False
            if 'k6' not in self.params:
                self.params.add('k6', value=0, vary=False)
            else:
                self.params['k6'].vary = False
            if 'k7' not in self.params:
                self.params.add('k7', value=0, vary=False)
            else:
                self.params['k7'].vary = False
            if 'k8' not in self.params:
                self.params.add('k8', value=0, vary=False)
            else:
                self.params['k8'].vary = False
            if 'k9' not in self.params:
                self.params.add('k9', value=0, vary=False)
            else:
                self.params['k9'].vary = False
            if 'k10' not in self.params:
                self.params.add('k10', value=0, vary=False)
            else:
                self.params['k10'].vary = False

        elif func_version == 116:
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

                    if n == 0:
                        self.params['A_{0}_{1}'.format(n, m)].vary = False
                        self.params['A_{0}_{1}'.format(n, m)].value = 0
                        self.params['B_{0}_{1}'.format(n, m)].vary = False
                        self.params['B_{0}_{1}'.format(n, m)].value = 0

            if 'k1' not in self.params:
                self.params.add('k1', value=0, vary=True)
            else:
                self.params['k1'].vary = True
            if 'k2' not in self.params:
                self.params.add('k2', value=0, vary=True)
            else:
                self.params['k2'].vary = True
            if 'xp1' not in self.params:
                self.params.add('xp1', value=1050, vary=False, min=900, max=1200)
            else:
                self.params['xp1'].vary = True
            if 'xp2' not in self.params:
                self.params.add('xp2', value=1050, vary=False, min=900, max=1200)
            else:
                self.params['xp2'].vary = True
            if 'yp1' not in self.params:
                self.params.add('yp1', value=0, vary=False, min=-100, max=100)
            else:
                self.params['yp1'].vary = True
            if 'yp2' not in self.params:
                self.params.add('yp2', value=0, vary=False, min=-100, max=100)
            else:
                self.params['yp2'].vary = True
            if 'zp1' not in self.params:
                self.params.add('zp1', value=4575, vary=False, min=4300, max=4700)
            else:
                self.params['zp1'].vary = True
            if 'zp2' not in self.params:
                self.params.add('zp2', value=-4575, vary=False, min=-4700, max=-4300)
            else:
                self.params['zp2'].vary = True

        if func_version == 102:
            d_starts = np.linspace(-np.pi*0.5+0.1, np.pi*0.5-0.1, ms)
            for m in range(ms):
                # for m in range(ms):
                if f'E_{m}' not in self.params:
                    self.params.add(f'E_{m}', value=d_starts[m], min=-np.pi*0.5,
                                    max=np.pi*0.5, vary=True)
                else:
                    self.params[f'E_{m}'].vary = True
                # if n > m:
                #     self.params['E_{0}_{1}'.format(n, m)].vary = False
                #     self.params['E_{0}_{1}'.format(n, m)].value = 0

        if func_version in [6, 105]:

            if 'k1' not in self.params:
                self.params.add('k1', value=0, vary=False)
            else:
                self.params['k1'].vary = True
            if 'k2' not in self.params:
                self.params.add('k2', value=0, vary=False)
            else:
                self.params['k2'].vary = True
            if 'k3' not in self.params:
                self.params.add('k3', value=0, vary=True)
            else:
                self.params['k3'].vary = True
            if 'k4' not in self.params:
                self.params.add('k4', value=0, vary=False)
            else:
                self.params['k4'].vary = False
            if 'k5' not in self.params:
                self.params.add('k5', value=0, vary=False)
            else:
                self.params['k5'].vary = False
            if 'k6' not in self.params:
                self.params.add('k6', value=0, vary=False)
            else:
                self.params['k6'].vary = False
            if 'k7' not in self.params:
                self.params.add('k7', value=0, vary=False)
            else:
                self.params['k7'].vary = False
            if 'k8' not in self.params:
                self.params.add('k8', value=0, vary=False)
            else:
                self.params['k8'].vary = False
            if 'k9' not in self.params:
                self.params.add('k9', value=0, vary=False)
            else:
                self.params['k9'].vary = False
            if 'k10' not in self.params:
                self.params.add('k10', value=0, vary=False)
            else:
                self.params['k10'].vary = False

        if func_version in [106]:
            if 'k1' not in self.params:
                self.params.add('k1', value=1, vary=True)
            else:
                self.params['k1'].vary = True

        if func_version in [110]:
            if 'k1' not in self.params:
                self.params.add('k1', value=0, vary=True)
            else:
                self.params['k1'].vary = True
            if 'k2' not in self.params:
                self.params.add('k2', value=0, vary=True)
            else:
                self.params['k2'].vary = True
            if 'xp1' not in self.params:
                # self.params.add('xp1', value=-525, vary=True, min=-600, max=-400)
                self.params.add('xp1', value=1050, vary=False, min=900, max=1200)
            else:
                self.params['xp1'].vary = True
            if 'xp2' not in self.params:
                # self.params.add('xp2', value=-1100, vary=True, min=-1200, max=-1000)
                self.params.add('xp2', value=1050, vary=False, min=900, max=1200)
            else:
                self.params['xp2'].vary = True
            if 'yp1' not in self.params:
                # self.params.add('yp1', value=1100, vary=True, min=1000, max=1200)
                self.params.add('yp1', value=0, vary=False, min=-100, max=100)
            else:
                self.params['yp1'].vary = True
            if 'yp2' not in self.params:
                # self.params.add('yp2', value=-400, vary=True, min=-500, max=-300)
                self.params.add('yp2', value=0, vary=False, min=-100, max=100)
            else:
                self.params['yp2'].vary = True
            if 'zp1' not in self.params:
                # self.params.add('zp1', value=9696, vary=True, min=9500, max=9700)
                self.params.add('zp1', value=4575, vary=False, min=4300, max=4700)
            else:
                self.params['zp1'].vary = True
            if 'zp2' not in self.params:
                # self.params.add('zp2', value=7871, vary=True, min=7700, max=8000)
                self.params.add('zp2', value=-4575, vary=False, min=-4700, max=-4300)
            else:
                self.params['zp2'].vary = True

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
                    self.params.add('G_{0}'.format(cn), value=0, min=0, max=1,
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
                # method='least_squares')
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
                # print('fitting phase ext')
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                           # method='leastsq', fit_kws={'maxfev': 10000})
                                           method='least_squares', fit_kws={'verbose': 0,
                                                                            'gtol': 1e-12,
                                                                            'ftol': 1e-12,
                                                                            'xtol': 1e-12})

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
        elif func_version >= 100 and func_version not in [105, 110, 111, 115, 116, 117]:
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
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           # weights=np.concatenate(
                                           #     [np.ones(Br.shape), np.ones(Bz.shape),
                                           #      np.ones(Bphi.shape)*100000]).ravel(),
                                           r=np.abs(RR), z=ZZ, phi=PP, params=self.params,
                                           # method='leastsq', fit_kws={'maxfev': 10000})
                                           method='least_squares', fit_kws={'max_nfev': 10000})

                # 'loss': 'soft_l1'})
                # 'ftol': 1e-11,
                # 'xtol': 1e-11,
                # 'gtol': 1e-11}, verbose=True)

        elif func_version == 111:
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
                                           method='least_squares', fit_kws={'max_nfev': 10000,
                                                                            'loss': 'soft_l1'})
                # 'xtol': 1e-11,
                # 'gtol': 1e-11}, verbose=True)
        elif func_version in [105, 110, 115, 116, 117]:
            if cfg_pickle.recreate:
                for param in self.params:
                    self.params[param].vary = False
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 1})
            elif cfg_pickle.use_pickle:
                mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           # weights=np.concatenate([mag, mag, mag]).ravel(),
                                           r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                           method='leastsq', fit_kws={'maxfev': 10000})
            else:
                mag = 1/np.sqrt(Br**2+Bz**2+Bphi**2)
                self.result = self.mod.fit(np.concatenate([Br, Bz, Bphi]).ravel(),
                                           # weights=np.concatenate([mag, mag, mag]).ravel(),
                                           r=RR, z=ZZ, phi=PP, x=XX, y=YY, params=self.params,
                                           method='least_squares', fit_kws={'verbose': 0,
                                                                            'gtol': 1e-12,
                                                                            'ftol': 1e-12,
                                                                            'xtol': 1e-12})

        self.params = self.result.params
        end_time = time()
        print(("Elapsed time was %g seconds" % (end_time - start_time)))
        report_fit(self.result, show_correl=False)
        if cfg_pickle.save_pickle:  # and not cfg_pickle.recreate:
            self.pickle_results(self.pickle_path+cfg_pickle.save_name)

    def fit_external(self, cfg_params, cfg_pickle, profile=False):
        raise NotImplementedError('Oh no! you got lazy during refactoring')

    def pickle_results(self, pickle_name='default'):
        """Pickle the resulting Parameters after a fit is performed."""

        pkl.dump(self.result.params, open(pickle_name+'_results.p', "wb"), pkl.HIGHEST_PROTOCOL)

    def merge_data_fit_res(self):
        """Combine the fit results and the input data into one dataframe for easier
        comparison of results.

        Adds three columns to input_data: `Br_fit, Bphi_fit, Bz_fit` or `Bx_fit, By_fit, Bz_fit`,
        depending on the geometry.
        """
        bf = self.result.best_fit

        self.input_data.loc[:, 'Br_fit'] = bf[0:len(bf)//3]
        self.input_data.loc[:, 'Bz_fit'] = bf[len(bf)//3:2*len(bf)//3]
        self.input_data.loc[:, 'Bphi_fit'] = bf[2*len(bf)//3:]

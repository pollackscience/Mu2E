#! /usr/bin/env python
"""Module for generating a synthetic magnetic field using field parameterizations.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
import six.moves.cPickle as pkl
import numpy as np
from scipy import special
import pandas as pd
from mu2e.dataframeprod import DataFrameMaker
from mu2e import mu2e_ext_path


def field_generator(input_xs, input_ys, input_zs, L, ns, ks, seed, file_name, ABs=None):
    """Input desired field dimensions, type of parameterization, get a field.

    The workflow should mimic what recieving a real field would be like:

    1) Determine the x,y,z coords.
    2) Determine the field function to be used.
    3) Freeze a bunch of free params to set up field function.
    4) Generate a df of `x,y,z,Bx,By,Bz`.
    5) Pass it through `dataframeprod` to get the cylindrical coords.
    6) Pickle the new df, ready for use in main workflow.
    """

    np.random.seed(seed)

    xx, yy, zz = np.meshgrid(input_xs, input_ys, input_zs)
    xx = xx.flatten().astype(float)
    yy = yy.flatten().astype(float)
    zz = zz.flatten().astype(float)

    doABs = False
    if not ABs:
        ABs = {}
        doABs = True
    for n in range(ns):
        for k in range(ks):
            if doABs:
                if n == 0:
                    ABs[f'A_{n}_{k}'] = 0
                    ABs[f'B_{n}_{k}'] = 0
                else:
                    ABs[f'A_{n}_{k}'] = np.random.uniform(-500, 500)
                    ABs[f'B_{n}_{k}'] = np.random.uniform(-500, 500)

    bx, by, bz = synth_3d_producer_hel(xx, yy, zz, L, ns, ks, ABs)

    df = pd.DataFrame({'X': xx, 'Y': yy, 'Z': zz, 'Bx': bx, 'By': by, 'Bz': bz})

    data_maker = DataFrameMaker(mu2e_ext_path+'datafiles/synth/'+file_name,
                                field_map_version='synth', input_type='df', input_df=df)
    data_maker.do_basic_modifications()
    data_maker.make_dump()


def synth_3d_producer_hel(x, y, z, L, ns, ks, ABs):
    '''
    Function used for generating a synthetic field.  Should not be used in fitting routines.
    This function is not parallelized.
    '''

    P = L/(2*np.pi)

    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)

    model_r = np.zeros(r.shape)
    model_z = np.zeros(r.shape)
    model_phi = np.zeros(r.shape)

    for n in range(ns):
        print('n:', n)
        for k in range(ks):
            iv = special.iv(k, (n/P)*np.abs(r))
            ivp = 0.5*(special.iv(-1+k, (n/P)*np.abs(r)) +
                       special.iv(1+k, (n/P)*np.abs(r)))

            A = ABs[f'A_{n}_{k}']
            B = ABs[f'B_{n}_{k}']

            model_r += (n/P)*(ivp*(A*np.cos(n*z/P+k*phi) + B*np.sin(n*z/P+k*phi)))

            model_z += (n/P)*(iv*(-A*np.sin(n*z/P+k*phi) + B*np.cos(n*z/P+k*phi)))

            model_phi += (1.0/r)*(-k*iv*(A*np.sin(n*z/P+k*phi)-B*np.cos(n*z/P+k*phi)))

    model_x = model_r*np.cos(phi)-model_phi*np.sin(phi)
    model_y = model_r*np.sin(phi)+model_phi*np.cos(phi)

    return (model_x, model_y, model_z)


if __name__ == "__main__":
    x = range(-1000, 1025, 25)
    y = range(-1000, 1025, 25)
    z = range(-4500, 4525, 25)
    ns = 10
    ks = 6
    L = 25000
    seed = 111
    file_name = 'synth3'
    params = pkl.load(open(mu2e_ext_path+'fit_params/Glass_Hel_Test_results.p', "rb"))
    ABs = {}
    for name in params:
        ABs[name] = params[name].value

    field_generator(x, y, z, L, ns, ks, seed, file_name, ABs)

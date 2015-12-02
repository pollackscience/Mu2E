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

    def __init__(self, input_data, z_steps = 15, x_steps = 10, y_steps = 10,r_steps=None,phi_steps=(np.pi/2,)):
        self.full_field = input_data
        self.sparse_field = self.full_field

        self.apply_selection('Z',z_steps)
        if r_steps:
            self.apply_selection('R',r_steps)
            self.apply_selection('Phi',phi_steps)
        else:
            self.apply_selection('X',x_steps)
            self.apply_selection('Y',y_steps)

        for mag in ['Bz','Br','Bphi','Bx','By','Bz']:
            self.sparse_field.eval('{0}err = 0.0001*{0}'.format(mag))
            #self.sparse_field[self.sparse_field.Z > 8000][self.sparse_field.Z < 13000].eval('{0}err = 0.0000001*{0}'.format(mag))

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
            if coord =='Phi':
                coord_vals=[]
                for step in steps:
                    coord_vals.append(step)
                    if step!=0: coord_vals.append(step-np.pi)
                    else: coord_vals.append(step+np.pi)
            elif coord =='R':
                if isinstance(steps[0], collections.Sequence):
                    coord_vals = np.sort(np.unique([val for sublist in steps for val in sublist]))
                else:
                    coord_vals = steps
        elif steps=='all':
                coord_vals = np.sort(self.full_field[coord].unique())
        else:
            raise TypeError(coord+" steps must be scalar or list of values!")

        if coord=='R':
            self.sparse_field = self.sparse_field.query('|'.join(['(-1e-6<R-'+str(i)+'<1e-6)' for i in coord_vals]))
        else:
            self.sparse_field = self.sparse_field[self.sparse_field[coord].isin(coord_vals)]
        if len(self.sparse_field[coord].unique()) != len(coord_vals):
            print 'Warning!:',set(coord_vals)-set(self.sparse_field[coord].unique()), 'not valid input_data',coord

    def get_toy(self):
        return self.sparse_field

    def bad_calibration(self,measure = True, position=False):
        measure_sf = [1-2.03e-4, 1+1.48e-4, 1-0.81e-4, 1-1.46e-4, 1-0.47e-4]
        pos_offset = [-1.5, 0.23, -0.62, 0.12, -0.18]
        probes = abs(self.sparse_field.Y).unique()
        if measure:
            if len(probes)<len(measure_sf): raise IndexError('need more measure_sf, too many probes')
            for i,probe in enumerate(probes):
                self.sparse_field.ix[abs(self.sparse_field.Y)==probe, 'Bz'] *= measure_sf[i]
                self.sparse_field.ix[abs(self.sparse_field.Y)==probe, 'By'] *= measure_sf[i]

        if position:
            if len(probes)<len(pos_offset): raise IndexError('need more pos_offset, too many probes')
            for i,probe in enumerate(probes):
                if probe==0:
                    self.sparse_field.ix[self.sparse_field.Y==probe, 'Y'] += pos_offset[i]
                else:
                    self.sparse_field.ix[self.sparse_field.Y==probe, 'Y'] += pos_offset[i]
                    self.sparse_field.ix[self.sparse_field.Y==-probe, 'Y'] -= pos_offset[i]




if __name__=="__main__":
    data_maker1=DataFileMaker('../FieldMapData_1760_v5/Mu2e_DSmap',use_pickle = True)
    hpg = HallProbeGenerator(data_maker1.data_frame)
    hall_toy = hpg.get_toy()

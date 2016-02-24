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
from mu2e.fieldfitter import FieldFitter
from mu2e.plotter import Plotter
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
            self.phi_steps = phi_steps
        else:
            self.apply_selection('X',x_steps)
            self.apply_selection('Y',y_steps)

        for mag in ['Bz','Br','Bphi','Bx','By','Bz']:
            self.sparse_field.eval('{0}err = abs(0.0001*{0}+1e-15)'.format(mag))

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
            elif coord in ['Z','X','Y']:
                coord_vals = steps
        elif steps=='all':
                coord_vals = np.sort(self.full_field[coord].unique())
        else:
            raise TypeError(coord+" steps must be scalar or list of values!")

        if coord=='R' or coord=='Phi':
            self.sparse_field = self.sparse_field.query('|'.join(['(-1e-6<'+coord+'-'+str(i)+'<1e-6)' for i in coord_vals]))
        else:
            self.sparse_field = self.sparse_field[self.sparse_field[coord].isin(coord_vals)]
        if len(self.sparse_field[coord].unique()) != len(coord_vals):
            print 'Warning!: specified vals:'
            print np.sort(coord_vals)
            print 'remaining vals:'
            print np.sort(self.sparse_field[coord].unique())

    def get_toy(self):
        return self.sparse_field

    def bad_calibration(self,measure = False, position=False, rotation=False):
        measure_sf = [1-2.03e-4, 1+1.48e-4, 1-0.81e-4, 1-1.46e-4, 1-0.47e-4]
        pos_offset = [-1.5, 0.23, -0.62, 0.12, -0.18]
        #rotation_angle = [ 0.00047985,  0.00011275,  0.00055975, -0.00112114,  0.00051197]
        rotation_angle = [ 0.0005,  0.0004,  0.0005, 0.0003,  0.0004]
        #rotation_angle = [ 0.4,  0.35,  0.55, 0.22,  0.18]
        for phi in self.phi_steps:
            probes = self.sparse_field[np.isclose(self.sparse_field.Phi,phi)].R.unique()
            if measure:
                if len(probes)>len(measure_sf): raise IndexError('need more measure_sf, too many probes')
                for i,probe in enumerate(probes):
                    self.sparse_field.ix[(abs(self.sparse_field.R)==probe), 'Bz'] *= measure_sf[i]
                    self.sparse_field.ix[(abs(self.sparse_field.R)==probe), 'Br'] *= measure_sf[i]
                    self.sparse_field.ix[(abs(self.sparse_field.R)==probe), 'Bphi'] *= measure_sf[i]

            if position:
                if len(probes)>len(pos_offset): raise IndexError('need more pos_offset, too many probes')
                for i,probe in enumerate(probes):
                    if probe==0:
                        self.sparse_field.ix[abs(self.sparse_field.R)==probe, 'R'] += pos_offset[i]
                    else:
                        self.sparse_field.ix[abs(self.sparse_field.R)==probe, 'R'] += pos_offset[i]
                        self.sparse_field.ix[abs(self.sparse_field.R)==-probe, 'R'] -= pos_offset[i]

            if rotation:
                if len(probes)>len(rotation_angle): raise IndexError('need more rotation_angle, too many probes')
                for i,probe in enumerate(probes):
                    tmp_Bz = self.sparse_field[self.sparse_field.R==probe].Bz
                    tmp_Br = self.sparse_field[self.sparse_field.R==probe].Br
                    self.sparse_field.ix[(abs(self.sparse_field.R)==probe), 'Bz'] = tmp_Br*np.sin(rotation_angle[i])+tmp_Bz*np.cos(rotation_angle[i])
                    self.sparse_field.ix[(abs(self.sparse_field.R)==probe), 'Br'] = tmp_Br*np.cos(rotation_angle[i])-tmp_Bz*np.sin(rotation_angle[i])


def make_fit_plots(plot_maker, cfg_data, cfg_geom, cfg_plot):
    '''make a series of fit plots, given a plotter class and some information
        on the kind of plots you want. must specify phi steps or xy_steps'''

    geom = cfg_geom.geom
    plot_type = cfg_plot.plot_type
    if geom == 'cyl': steps = cfg_geom.phi_steps
    if geom == 'cart': steps = cfg_geom.xy_steps
    conditions = cfg_data.conditions
    zlims = cfg_plot.zlims

    ABC_geom = {'cyl':[['Bz','R','Z'],['Br','R','Z'],['Bphi','R','Z']],
                'cart':[['Bx','Y','Z'],['By','Y','Z'],['Bz','Y','Z'],
                       ['Bx','X','Z'],['By','X','Z'],['Bz','X','Z']]}

    if cfg_plot.plot_type == 'mpl':
        if cfg_geom.geom == 'cart':
            for ABC in ABC_geom[geom]:
                plot_maker.plot_A_v_B_and_C_fit_ext(ABC[0],ABC[1],ABC[2],steps,zlims,False,*conditions)
        elif geom == 'cyl':
            for ABC in ABC_geom[geom]:
                plot_maker.plot_A_v_B_and_C_fit_cyl(ABC[0],ABC[1],ABC[2],steps,zlims,False,*conditions)
    elif plot_type == 'plotly':
        if geom == 'cart':
            for ABC in ABC_geom[geom]:
                plot_maker.plot_A_v_B_and_C_fit_ext_plotly(ABC[0],ABC[1],ABC[2],steps,zlims,False,*conditions)
        elif geom == 'cyl':
            for ABC in ABC_geom[geom]:
                plot_maker.plot_A_v_B_and_C_fit_cyl_plotly(ABC[0],ABC[1],ABC[2],steps,zlims,False,*conditions)
    if cfg_plot.plot_type=='mpl':plt.show()


def field_map_analysis(suffix, cfg_data, cfg_geom, cfg_params, cfg_pickle, cfg_plot, profile=False):
    '''Universal function to perform all types of hall probe measurements, plots,
    and further analysis.  Takes input cfg namedtuples to determine analysis'''

    plt.close('all')
    input_data = DataFileMaker(cfg_data.path, use_pickle=True).data_frame
    input_data.query(' and '.join(cfg_data.conditions))
    hpg = HallProbeGenerator(input_data, z_steps = cfg_geom.z_steps,
            r_steps = cfg_geom.r_steps, phi_steps = cfg_geom.phi_steps,
            x_steps = cfg_geom.xy_steps, y_steps = cfg_geom.xy_steps)

    if cfg_geom.bad_calibration[0]:
        hpg.bad_calibration(measure = True, position = False, rotation = False)
    if cfg_geom.bad_calibration[1]:
        hpg.bad_calibration(measure = False, position = True, rotation=False)
    if cfg_geom.bad_calibration[2]:
        hpg.bad_calibration(measure = False, position = False, rotation = True)

    hall_measure_data = hpg.get_toy()

    ff = FieldFitter(hall_measure_data, cfg_geom)
    if profile:
        ZZ,RR,PP,Bz,Br,Bphi = ff.fit(cfg_geom.geom, cfg_params, cfg_pickle, profile = profile)
        return ZZ,RR,PP,Bz,Br,Bphi
    else:
        ff.fit(cfg_geom.geom, cfg_params, cfg_pickle, profile = profile)

    plot_maker = Plotter.from_hall_study({'_'.join([cfg_data.magnet,cfg_data.datatype]):hall_measure_data},fit_result = ff.result, use_html_dir = cfg_plot.html_loc)
    plot_maker.extra_suffix=suffix

    make_fit_plots(plot_maker, cfg_data, cfg_geom, cfg_plot)
    return hall_measure_data, ff, plot_maker


if __name__=="__main__":
    data_maker1=DataFileMaker('../FieldMapData_1760_v5/Mu2e_DSmap',use_pickle = True)
    hpg = HallProbeGenerator(data_maker1.data_frame)
    hall_toy = hpg.get_toy()

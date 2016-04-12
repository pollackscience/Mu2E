#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
from mu2e.tools.fiteval import get_mag_field_function
from numba import jit
import odespy


#units conversion:
q = 1.60217662e-19 #C
me = 9.10938356e-31 #kg
q_o_me = 175882002272 #C/kg
c = 299792458000 #mm/s


def gamma(v):
    '''calculate gamma factor for a given velocity'''
    beta = v/c
    return 1/np.sqrt(1-np.dot(beta,beta))

def calc_lorentz_accel(v_vec,b_vec):
    '''Calculate lorentz acceleration on a charged particle.
    Currently just accel due to magnetic field on an electron'''
    a = -1*q_o_me*np.cross(v_vec,b_vec)/(gamma(v_vec))
    return a

def add_vel(u,v):
    '''relativistic velocity addition in 3D'''
    return 1/(1+np.dot(u,v))*(u+v/gamma(u)+(gamma(u)/(1+gamma(u)))*(np.dot(u,v)*u))

def lorentz_force(state,time,mag_field):
    '''
    Calculate the velocity and acceleration on a particle due to
    lorentz force for a magnetic field as a function of time.

    state = [x,y,z,vx,vy,vz]
    time = array of time values
    '''
    f = np.empty(6)
    f[:3] = state[3:]
    f[3:] = calc_lorentz_accel(np.asarray(state[3:]),mag_field(state[0],state[1],state[2],True))
    return f

def terminate(state,time,step_no):
    '''terminate clause to end ode solver once the electron reaches detector material'''
    radius = 700
    return ((np.sqrt(state[step_no][0]**2+state[step_no][1]**2)>radius)
            or (state[step_no][2]>12000)
            or (state[step_no][2]<5000))

class ElectronSwimmer:
    '''Wrapper class for odespy ode solver, specifically for simulating
    the path of an electron moving through the DS magnetic field.'''
    def __init__(self, init_mom, init_pos, b_field, time_steps, ode_method):
        '''Give an electron initial momentum and position, b_field function, time_steps, and ode.
        Momentum must be in units of MeV, position must be in mm'''

        self.init_mom = init_mom
        self.init_v = self.init_mom/(0.511*np.sqrt(1+np.dot(self.init_mom,self.init_mom)/0.511**2))*c
        self.init_pos = init_pos
        self.b_field = b_field
        self.time_steps = time_steps
        self.solver = getattr(odespy,ode_method)(lorentz_force,f_args=(self.b_field,))
        self.solver.set_initial_condition([self.init_pos[0], self.init_pos[1], self.init_pos[2],
            self.init_v[0], self.init_v[1], self.init_v[2]])
        self.init_E = gamma(self.init_v)*0.511

    def solve(self,verbose=True,timer=False):
        '''Run the ode solver and output the (state,time) tuple'''
        if verbose:
            print 'swimming electron with {0} MeV, starting at {1} mm, for {2} s'.format(self.init_mom, self.init_pos, self.time_steps[-1])
        X,t = self.solver.solve(self.time_steps,terminate)
        self.final_v =  np.asarray([X[-1,3],X[-1,4],X[-1,5]])
        self.final_E = gamma(self.final_v)*0.511
        if verbose:
            print 'init energy:', self.init_E, 'MeV'
            print 'final energy:', self.final_E, 'MeV'
            e_diff = self.final_E - self.init_E
            print 'energy difference: {0} MeV ({1:.4}%)'.format(e_diff,(e_diff*100)/self.init_E)

        return X,t

    def get_init_v(self):
        return self.init_v
    def get_final_v(self):
        return self.final_v
    def get_init_E(self):
        return self.init_E
    def get_final_E(self):
        return self.final_E




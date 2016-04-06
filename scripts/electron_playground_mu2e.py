#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14, 10
from fiteval import get_mag_field_function
from time import time
from numba import jit
from scipy.integrate import odeint, ode
import odespy

plt.close('all')
mag_field_function = get_mag_field_function()

#generate a uniform cube grid, and a uniform mag field in the z direction
x = y = np.linspace(-700,700,9)
z = np.linspace(5500,12000,9)
xx,yy,zz = np.meshgrid(x,y,z)

df = pd.DataFrame(np.array([xx,yy,zz]).reshape(3,-1).T,columns=['X','Y','Z'])
print df
print mag_field_function(df['X'][0],df['Y'][0],df['Z'][0],cart=True)
df['Bx'],df['By'],df['Bz']= zip(*df.apply(lambda row: mag_field_function(row['X'],row['Y'],row['Z'],cart=True),axis=1))

#load the field into a dataframe
#df = pd.DataFrame(np.array([xx,yy,zz,bxx,byy,bzz]).reshape(6,-1).T,columns = ['X','Y','Z','Bx','By','Bz'])

#reduce the number of datapoints for appropriate quiver plotting:
#df_quiver = df.query('(X+5)%10==0 and (Y+5)%10==0 and Z%10==0')
#recreate 3d meshgrid by reshaping the df back into six 3d arrays
quiver_size = int(round(df.shape[0]**(1./3.)))
#print 'quiver_size', quiver_size
qxx,qyy,qzz,qbxx,qbyy,qbzz = df.values.T.reshape(6,quiver_size,quiver_size,quiver_size)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Z (length)')
ax.set_ylabel('X (length)')
ax.set_zlabel('Y (length)')
ax.quiver(qzz,qxx,qyy, qbzz,qbxx,qbyy, length=400,linewidths=(2,),arrow_length_ratio=0.2,alpha=0.6,colors='r')
plt.show()

#now lets assume we have an electron v = 1 unit/s in z dir
#it starts at 0,0,0
#only affected by F=qvXb
#x = x0 + vt + 1/2at^2
#vf = vi+at

#units conversion:
q = 1.60217662e-19 #C
me = 9.10938356e-31 #kg
q_o_me = 175882002272 #C/kg
c = 299792458000 #mm/s


def gamma(v):
    beta = v/c
    return 1/np.sqrt(1-np.dot(beta,beta))
def calc_lorentz_accel(v_vec,b_vec):
    a = -1*q_o_me*np.cross(v_vec,b_vec)/(gamma(v_vec))
    return a
def add_vel(u,v):
    return 1/(1+np.dot(u,v))*(u+v/gamma(u)+(gamma(u)/(1+gamma(u)))*(np.dot(u,v)*u))

def update_kinematics(p_vec_0,v_vec_0,dt):
# RK4
    k1 = dt*calc_lorentz_accel(v_vec_0,mag_field_function(p_vec_0[0],p_vec_0[1],p_vec_0[2],True))
    l1 = dt*v_vec_0
    x1 = p_vec_0+l1*0.5
    v1 = v_vec_0+k1*0.5
    k2 = dt*calc_lorentz_accel(v1,mag_field_function(x1[0],x1[1],x1[2],True))
    l2 = dt*(v_vec_0+k1*0.5)
    x2 = p_vec_0+l2*0.5
    v2 = v_vec_0+k2*0.5
    k3 = dt*calc_lorentz_accel(v2,mag_field_function(x2[0],x2[1],x2[2],True))
    l3 = dt*(v_vec_0+k2*0.5)
    x3 = p_vec_0+l3
    v3 = v_vec_0+k3
    k4 = dt*calc_lorentz_accel(v3,mag_field_function(x3[0],x3[1],x3[2],True))
    l4 = dt*(v_vec_0+k3)
    v_vec_1 = v_vec_0+(1/6.0)*(k1+2*k2+2*k3+k4)
    p_vec_1 = p_vec_0+(1/6.0)*(l1+2*l2+2*l3+l4)*1e3
    return (p_vec_1,v_vec_1)

def lorentz_force(state,time):
    '''
    Calculate the velocity and acceleration on a particle due to
    lorentz force for a magnetic field as a function of time.

    state = [x,y,z,vx,vy,vz]
    time = array of time values
    '''
    f = np.empty(6)
    f[:3] = state[3:]
    f[3:] = calc_lorentz_accel(np.asarray(state[3:]),mag_field_function(state[0],state[1],state[2],True))
    return f


pos = np.array([0,0,6000])
init_pos = pos
mom = np.array([10,60,10]) #in MeV
init_mom = mom
v = mom/(0.511*np.sqrt(1+np.dot(mom,mom)/0.511**2))*c
init_v = v
path_x = [pos[0]]
path_y = [pos[1]]
path_z = [pos[2]]
path = [pos]
dt = 5e-13
total_time = 0
#while (x[0]<=pos[0]<=x[-1] and y[0]<=pos[1]<=y[-1] and z[0]<=pos[2]<=z[-1] and total_time<1e12):
start_time=time()
#while (x[0]<=pos[0]<=x[-1] and y[0]<=pos[1]<=y[-1] and z[0]<=pos[2]<=z[-1] and total_time<dt*1e5):
#    pos,v = update_kinematics(pos,v,dt)
#    path.append(pos)
#    total_time+=dt
#print total_time
t_steps = np.linspace(0,4e-8,1e5)
init_state = [init_pos[0],init_pos[1],init_pos[2],init_v[0],init_v[1],init_v[2]]
#X = odeint(lorentz_force,init_state,t)
solver = odespy.Dop853(lorentz_force)
solver.set_initial_condition(init_state)
X,t = solver.solve(t_steps)
end_time=time()
v_final = np.asarray([X[-1,3],X[-1,4],X[-1,5]])
print("Elapsed time was %g seconds" % (end_time - start_time))

#ax.plot(path_z,path_x,zs=path_y,linewidth=2)
#path = np.asarray(path)
#ax.plot(path[:,2],path[:,0],zs=path[:,1],linewidth=2)
ax.plot(X[:,2],X[:,0],zs=X[:,1],linewidth=2)
ax.set_title('Path of electron through magnetic field')


# these are matplotlib.patch.Patch properties
textstr = 'init pos={0}\ninit mom={1} (MeV)\nB={2}'.format(init_pos, init_mom, 'ideal DS field map')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
print 'init_E', gamma(init_v)*0.511, 'MeV'
#print 'final_E', gamma(v)*0.511, 'MeV'
print 'final_E', gamma(v_final)*0.511, 'MeV'
#print 'energy diff', gamma(v)*0.511 - gamma(init_v)*0.511, 'MeV'
print 'energy diff', gamma(v_final)*0.511 - gamma(init_v)*0.511, 'MeV'

# place a text box in upper left in axes coords
ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
plt.show()
plt.savefig('../plots/anim/electron_path_DS.pdf')

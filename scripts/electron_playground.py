#! /usr/bin/env python

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14, 10
from time import time

plt.close('all')

#generate a uniform cube grid, and a uniform mag field in the z direction
x = y = np.linspace(-25,25,51)
z = np.linspace(0,50,51)
bx=by=np.zeros(len(x))
bz = np.full(z.shape,3.0)
#bx=bz=np.zeros(len(x))
#by = np.full(z.shape,3.0)
xx,yy,zz = np.meshgrid(x,y,z)
bxx,byy,bzz = np.meshgrid(bx,by,bz)

#load the field into a dataframe
df = pd.DataFrame(np.array([xx,yy,zz,bxx,byy,bzz]).reshape(6,-1).T,columns = ['X','Y','Z','Bx','By','Bz'])

#reduce the number of datapoints for appropriate quiver plotting:
df_quiver = df.query('(X+5)%10==0 and (Y+5)%10==0 and Z%10==0')
#recreate 3d meshgrid by reshaping the df back into six 3d arrays
quiver_size = int(round(df_quiver.shape[0]**(1./3.)))
print 'quiver_size', quiver_size
qxx,qyy,qzz,qbxx,qbyy,qbzz = df_quiver.values.T.reshape(6,quiver_size,quiver_size,quiver_size)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X (length)')
ax.set_ylabel('Y (length)')
ax.set_zlabel('Z (length)')
ax.quiver(qxx,qyy,qzz, qbxx,qbyy,qbzz, length=3,linewidths=(2,),arrow_length_ratio=0.6,colors='r')

#now lets assume we have an electron v = 1 unit/s in z dir
#it starts at 0,0,0
#only affected by F=qvXb
#x = x0 + vt + 1/2at^2
#vf = vi+at

#natural units conversion:
# B: 1 MeV^2 = 1.4440271e9 T
# L: 1/MeV = 1.9732705e-7 m
# s: 1/MeV = 6.582122e-22 s
q = 1.60217662e-19 #kg
me = 9.10938356e-31 #C
c = 299792458 #m/s

def gamma(v):
    v=v/c
    return 1/np.sqrt(1-np.dot(v,v))
def calc_lorentz_accel(v_vec,b_vec):
    #a = -1*np.cross(v_vec,b_vec/1.4440271e-3)/(gamma(v_vec)*511e3)
    a = -1*q*np.cross(v_vec,b_vec)/(gamma(v_vec)*me)
    #print a
    return a
    #return -1*np.cross(v_vec,b_vec/1.4440271e-3)/511e3
def add_vel(u,v):
    #print u,v
    return 1/(1+np.dot(u,v))*(u+v/gamma(u)+(gamma(u)/(1+gamma(u)))*(np.dot(u,v)*u))

def update_kinematics(p_vec_0,v_vec_0,b_vec_0,dt):
    #p_vec_1 = p_vec_0+(dt*v_vec_0)*1e3
    #v_vec_1 = add_vel(v_vec_0, dt*calc_lorentz_accel(v_vec_0,b_vec_0))
    #v_vec_1 = v_vec_0 +dt*calc_lorentz_accel(v_vec_0,b_vec_0)
    k1 = dt*calc_lorentz_accel(v_vec_0,b_vec_0)
    l1 = dt*v_vec_0
    k2 = dt*calc_lorentz_accel(v_vec_0+k1*0.5,b_vec_0)
    l2 = dt*(v_vec_0+k1*0.5)
    k3 = dt*calc_lorentz_accel(v_vec_0+k2*0.5,b_vec_0)
    l3 = dt*(v_vec_0+k2*0.5)
    k4 = dt*calc_lorentz_accel(v_vec_0+k3,b_vec_0)
    l4 = dt*(v_vec_0+k3)
    v_vec_1 = v_vec_0+(1/6.0)*(k1+2*k2+2*k3+k4)
    p_vec_1 = p_vec_0+(1/6.0)*(l1+2*l2+2*l3+l4)*1e3
    return (p_vec_1,v_vec_1)

pos = np.array([10,-10,25])
init_pos = pos
mom = np.array([0,8e6,1e6]) #in eV
init_mom = mom
v = mom/(511e3*np.sqrt(1+np.dot(mom,mom)/511e3**2))*c
init_v = v
path = [pos]
dt = 1e-14
total_time = 0
start_time=time()
while (x[0]<=pos[0]<=x[-1] and y[0]<=pos[1]<=y[-1] and z[0]<=pos[2]<=z[-1] and total_time<dt*1e5):
    #pos,v = update_kinematics(pos,v,np.array(mag_field_function(pos[0],pos[1],pos[2],True)),dt)
    pos,v = update_kinematics(pos,v,np.array([0,0,3]),dt)
    #print pos
    path.append(pos)
    total_time+=dt
print total_time
end_time=time()
#if not cfg_pickle.recreate:
print("Elapsed time was %g seconds" % (end_time - start_time))


#ax.plot(path_z,path_x,zs=path_y,linewidth=2)
path = np.asarray(path)
ax.plot(path[:,0],path[:,1],zs=path[:,2],linewidth=2)
ax.set_title('Path of electron through magnetic field')


# these are matplotlib.patch.Patch properties
textstr = 'init pos={0}\ninit mom={1} (eV)\nB={2}'.format(init_pos, init_mom, 'ideal DS field map')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
print 'init_E', gamma(init_v)*0.511, 'MeV'
print 'final_E', gamma(v)*0.511, 'MeV'
print 'init_v', init_v/c, 'c'
print 'final_v', v/c, 'c'
print 'energy diff', gamma(v)*0.511 - gamma(init_v)*0.511, 'MeV'
print 'radius', (np.max(path[:,1])-np.min(path[:,1]))/2

# place a text box in upper left in axes coords
ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
plt.show()
plt.savefig('../plots/anim/electron_path_toy.pdf')

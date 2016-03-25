#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14, 10
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from fiteval import get_mag_field_function
from ROOT import TLorentzVector

plt.close('all')
mag_field_function = get_mag_field_function()

#generate a uniform cube grid, and a uniform mag field in the z direction
x = y = np.linspace(-700,700,9)
z = np.linspace(6000,12000,9)
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

def calc_lorentz_accel(v_vec,b_vec):
    return -1*np.cross(v_vec,b_vec)

def update_kinematics(p_vec,v_vec,b_vec,dt):
#not sure how to approach this in incremental steps
    a_vec = calc_lorentz_accel(v_vec,b_vec)
    p_vec_new = p_vec+(v_vec*dt+0.5*a_vec*dt**2)
    v_vec_new = v_vec+a_vec*dt
    return (p_vec_new,v_vec_new)

p = np.array([1,1,8000])
v = np.array([0.5,0.5,0.5])
init_v = v
path_x = [p[0]]
path_y = [p[1]]
path_z = [p[2]]
dt = 1e-2
while (p[0]<=x[-1] and p[1]<=y[-1] and p[2]<=z[-1]):
    print p
    p,v = update_kinematics(p,v,np.array(mag_field_function(p[0],p[1],p[2],True)),dt)
    path_x.append(p[0])
    path_y.append(p[1])
    path_z.append(p[2])


ax.plot(path_z,path_x,zs=path_y,linewidth=2)
ax.set_title('Path of electron through magnetic field')


# these are matplotlib.patch.Patch properties
textstr = 'init pos={0}\ninit v={1}\nB={2}'.format([0,0,8000], init_v, 'ideal DS field map')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
plt.show()

#! /usr/bin/env python

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

def calc_lorentz_accel(v_vec,b_vec):
    return -1*np.cross(v_vec,b_vec)

def update_kinematics(p_vec,v_vec,b_vec,dt):
#not sure how to approach this in incremental steps
    a_vec = calc_lorentz_accel(v_vec,b_vec)
    p_vec_new = p_vec+v_vec*dt+0.5*a_vec*dt**2
    v_vec_new = v_vec+a_vec*dt
    return (p_vec_new,v_vec_new)

p = np.array([0,0,0])
v = np.array([0,8,1])
path_x = [p[0]]
path_y = [p[1]]
path_z = [p[2]]
dt = 0.001
while (p[0]<=x[-1] and p[1]<=y[-1] and p[2]<=z[-1]):
    p,v = update_kinematics(p,v,np.array([0,0,3]),dt)
    path_x.append(p[0])
    path_y.append(p[1])
    path_z.append(p[2])


ax.plot(path_x,path_y,zs=path_z,linewidth=2)
ax.set_title('Path of electron through magnetic field')


# these are matplotlib.patch.Patch properties
textstr = 'init pos={0}\ninit v={1}\nB={2}'.format([0,0,0], [0,8,1], [0,0,3])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
plt.show()

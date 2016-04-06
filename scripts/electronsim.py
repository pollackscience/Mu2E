#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14, 10
from mu2e.tools.fiteval import get_mag_field_function
import mu2e.tools.particletransport as patr
from numba import jit
from time import time

plt.close('all')
mag_field_function = get_mag_field_function()

######################################
# Add this to the plotter class soon #
######################################

#generate a regular grid for plotting the mag field of the DS in quiver form
x = y = np.linspace(-700,700,9)
z = np.linspace(5500,12000,9)
xx,yy,zz = np.meshgrid(x,y,z)

df = pd.DataFrame(np.array([xx,yy,zz]).reshape(3,-1).T,columns=['X','Y','Z'])
print df
print mag_field_function(df['X'][0],df['Y'][0],df['Z'][0],cart=True)
df['Bx'],df['By'],df['Bz']= zip(*df.apply(lambda row: mag_field_function(row['X'],row['Y'],row['Z'],cart=True),axis=1))

#recreate 3d meshgrid by reshaping the df back into six 3d arrays
quiver_size = int(round(df.shape[0]**(1./3.)))
#print 'quiver_size', quiver_size
qxx,qyy,qzz,qbxx,qbyy,qbzz = df.values.T.reshape(6,quiver_size,quiver_size,quiver_size)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Z (mm)')
ax.set_ylabel('X (mm)')
ax.set_zlabel('Y (mm)')
ax.quiver(qzz,qxx,qyy, qbzz,qbxx,qbyy, length=400,linewidths=(2,),arrow_length_ratio=0.2,alpha=0.6,colors='r')
plt.show()

#############################
# Now lets swim an electron #
#############################

init_pos = np.array([0,0,6000]) #in mm
init_mom = np.array([10,90,10]) #in MeV
t_steps = np.linspace(0,4e-8,1e4)
solver = patr.ElectronSwimmer(init_mom, init_pos, mag_field_function, t_steps, 'Dop853')
start_time=time()
X,t = solver.solve()
end_time=time()
print("Elapsed time was %g seconds" % (end_time - start_time))

#ax.plot(path_z,path_x,zs=path_y,linewidth=2)
#path = np.asarray(path)
#ax.plot(path[:,2],path[:,0],zs=path[:,1],linewidth=2)
ax.plot(X[:,2],X[:,0],zs=X[:,1],linewidth=2)
ax.set_title('Path of electron through magnetic field')


# these are matplotlib.patch.Patch properties
textstr = 'init pos={0}\ninit mom={1} (MeV)\nB={2}'.format(init_pos, init_mom, 'ideal DS field map')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
plt.show()
plt.savefig('../plots/anim/electron_path_DS.pdf')

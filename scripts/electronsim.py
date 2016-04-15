#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14, 10
from mu2e.tools.fiteval import get_mag_field_function
from mu2e.src.fiteval_c import FitFunctionMaker
import mu2e.tools.particletransport as patr
from numba import jit
from time import time

plt.close('all')
mag_field_function_ideal = get_mag_field_function('Mau10_825mm_v1')
mag_field_function_bad_m = get_mag_field_function('Mau10_bad_m_test_v1')
mag_field_function_bad_p = get_mag_field_function('Mau10_bad_p_test_v1')
mag_field_function_bad_r = get_mag_field_function('Mau10_bad_r_test_v1')

######################################
# Add this to the plotter class soon #
######################################

#generate a regular grid for plotting the mag field of the DS in quiver form
x = y = np.linspace(-700,700,6)
z = np.linspace(5500,12000,6)
xx,yy,zz = np.meshgrid(x,y,z)

df = pd.DataFrame(np.array([xx,yy,zz]).reshape(3,-1).T,columns=['X','Y','Z'])
print df
print mag_field_function_ideal(df['X'][0],df['Y'][0],df['Z'][0],cart=True)
df['Bx'],df['By'],df['Bz']= zip(*df.apply(lambda row: mag_field_function_ideal(row['X'],row['Y'],row['Z'],cart=True),axis=1))

#recreate 3d meshgrid by reshaping the df back into six 3d arrays
quiver_size = int(round(df.shape[0]**(1./3.)))
#print 'quiver_size', quiver_size
qxx,qyy,qzz,qbxx,qbyy,qbzz = df.values.T.reshape(6,quiver_size,quiver_size,quiver_size)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Z (mm)')
ax.set_ylabel('X (mm)')
ax.set_zlabel('Y (mm)')
ax.quiver(qzz,qxx,qyy, qbzz,qbxx,qbyy, length=600,linewidths=(2,),arrow_length_ratio=0.2, pivot='middle',alpha=0.6,colors='r')
plt.show()

#############################
# Now lets swim an electron #
#############################

init_pos = np.array([0,0,6000]) #in mm
init_mom = np.array([0,100,-20]) #in MeV
t_steps = np.linspace(0,8e-8,2e4)
solver_ideal = patr.ElectronSwimmer(init_mom, init_pos, mag_field_function_ideal, t_steps, 'Dop853')
solver_bad_m = patr.ElectronSwimmer(init_mom, init_pos, mag_field_function_bad_m, t_steps, 'Dop853')
solver_bad_p = patr.ElectronSwimmer(init_mom, init_pos, mag_field_function_bad_p, t_steps, 'Dop853')
solver_bad_r = patr.ElectronSwimmer(init_mom, init_pos, mag_field_function_bad_r, t_steps, 'Dop853')
start_time=time()
X,t = solver_ideal.solve()
X_bm= solver_bad_m.solve()[0]
X_bp= solver_bad_p.solve()[0]
X_br= solver_bad_r.solve()[0]
end_time=time()
print("Elapsed time was %g seconds" % (end_time - start_time))

#ax.plot(path_z,path_x,zs=path_y,linewidth=2)
#path = np.asarray(path)
#ax.plot(path[:,2],path[:,0],zs=path[:,1],linewidth=2)
ax.plot(X[:,2],X[:,0],zs=X[:,1],linewidth=2,label='ideal path')
ax.plot(X_bm[:,2],X_bm[:,0],zs=X_bm[:,1],linewidth=2,linestyle='--',label='measurement syst')
ax.plot(X_bp[:,2],X_bp[:,0],zs=X_bp[:,1],linewidth=2,linestyle=':',label='position syst')
ax.plot(X_br[:,2],X_br[:,0],zs=X_br[:,1],linewidth=2,linestyle='-.', label='rotation syst')
ax.set_title('Path of electron through magnetic field')


# these are matplotlib.patch.Patch properties
textstr = 'init pos={0}\ninit mom={1} ({2:.3} MeV)\nB={3}'.format(init_pos, init_mom, solver_ideal.get_init_E(), 'ideal DS field map')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
ax.legend()
plt.show()
plt.savefig('../plots/anim/electron_path_DS.pdf')

print 'ideal xyz       :',X[-1,0:3], 'ideal-{this}:', X[-1,0:3] - X[-1,0:3]
print 'bad measure xyz :',X_bm[-1,0:3], 'ideal-{this}:', X[-1,0:3] - X_bm[-1,0:3]
print 'bad position xyz:',X_bp[-1,0:3], 'ideal-{this}:', X[-1,0:3] - X_bp[-1,0:3]
print 'bad rotation xyz:',X_br[-1,0:3], 'ideal-{this}:', X[-1,0:3] - X_br[-1,0:3]

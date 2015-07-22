#! /usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import cPickle as pkl
import os
from math import *
from datetime import datetime

startTime = datetime.now()

def makeR(row):
  return sqrt(row['X']**2+row['Y']**2)
def centerX(row,offset):
  if offset:
    return row['X']-3900
  else:
    return row['X']-3904
def makeBr(row):
  return sqrt(row['Bx']**2+row['By']**2)
def makeTheta(row):
  return np.arctan2(row['Y'],row['X'])

offset = False
if offset:
  fileName = 'FieldMapData_1760_v5/Mu2e_PSMapOffset.p'
else:
  fileName = 'FieldMapData_1760_v5/Mu2e_PSMap.p'


if not os.path.isfile(fileName):
  data_file = pd.read_csv('FieldMapData_1760_v5/Mu2e_PSMap.txt', header=None, names = ['X','Y','Z','Bx','By','Bz'], delim_whitespace=True)
  data_file['X'] = data_file.apply(centerX, args = (offset,), axis=1)
  data_file['R'] = data_file.apply(makeR, axis=1)
  data_file['Theta'] = data_file.apply(makeTheta, axis=1)
  data_file['Br'] = data_file.apply(makeBr, axis=1)
  pkl.dump( data_file, open( fileName, "wb" ),pkl.HIGHEST_PROTOCOL )
else:
  data_file = pkl.load(open(fileName,"rb"))


#data_file.info()

print data_file.head(10)
zvals = data_file.Z.value_counts()[data_file.Z[0]]

print datetime.now() - startTime

print data_file.head(10)
print data_file['Bz'].max(), data_file['Bz'].min()

init_num = 200
#data_file.plot()
#data_file = data_file.sort(['Z'])
#dfs = data_file.sort(['Z','R','Theta'])
dfs = data_file.sort(['Z','R'])
#dfs = data_file.sort(['X','Y','Z'])
data_file_z0 = dfs[(zvals*init_num):zvals*(init_num+1)]
#bins =  np.arange(0,4200,35)
#ind = np.digitize(data_file_z0['R'],bins)
#print data_file_z0[222:257]
data_file_z0_constR = data_file_z0[800:820]
print data_file_z0_constR.head()
#raw_input()
#data_file_z0_constR = data_file_z0
print data_file_z0_constR['X'].mean()
#raw_input()
#raw_input()
#dfs_z0 = data_file_z0
#print data_file_z0.tail()
#fig = plt.figure().gca()
fig, ax = plt.subplots(1)
X = data_file_z0.X[::10]
Y = data_file_z0.Y[::10]
Bx = data_file_z0.Bx[::10]
By = data_file_z0.By[::10]
Bz = data_file_z0.Bz[::10]
R = data_file_z0_constR.R
Theta = data_file_z0_constR.Theta
Br = data_file_z0_constR.Br
plt.figure(1)
plt.plot(Theta,Br,'ro')
plt.xlabel(r'$\theta$ (Radians)')
plt.ylabel('Br (T)')
if offset:
  plt.title('Radial Magnetic Field vs Theta at Z={0}, 4mm offset'.format(-4929))
else:
  plt.title('Radial Magnetic Field vs Theta at Z={0}, R=555.09 mm'.format(-4929))
plt.axis([-0.1, 3.24,0.22,0.26])
plt.grid(True)
if offset:
  fig.savefig('Br1_offset.png')
else:
  fig.savefig('Br1.png')



#plt.figure(2)
fig, ax = plt.subplots(1,1)
#plt.axis(Theta,Br,'ro')
#quiv = ax.quiver(X, Y, Bx, By)
quiv = ax.quiver(X,Y,Bx,By,pivot='mid')
plt.quiverkey(quiv, 1400, -130, 0.1, '0.1 T', coordinates='data',clim=[-1.1,5])
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title('Radial Magnetic Components, Z = {0}'.format(-4929))
#fig.set_zlabel('Z (mm)')
#cb = plt.colorbar(quiv)
#cb.set_label('mean value')
if offset:
  fig.savefig('PsField1_offset.png')
else:
  fig.savefig('PsField1.png')

def update_quiver(num, data_file, quiv):
  #print num
  data_file_z0= data_file[zvals*num:zvals*(num+1)]
  #print data_file_z0.tail()
  #X = data_file_z0.X[::10]
  #Y = data_file_z0.Y[::10]
  Bx = data_file_z0.Bx[::10]
  By = data_file_z0.By[::10]
  Bz = data_file_z0.Bz[::10]
  #quiv = ax.quiver(X,Y,Bx,By,Bz,pivot='mid')
  quiv.set_UVC(Bx, By,Bz)
  return quiv

dfs = data_file.sort(['X','Y','Z'])
dfs = dfs.query('X == 0 and Y==0')
print dfs.head()


plt.figure(3)
plt.plot(dfs.Z,dfs.Bz,'ro')
plt.xlabel('Z (mm)')
plt.ylabel('Bz (T)')
plt.title('Bz vs Z at X=0, Y=0')
#plt.axis([-0.1, 3.24,0.22,0.26])
plt.grid(True)
plt.savefig('Bz.png')

data_file_z0 = data_file_z0.query('X==0')
plt.figure(4)
br_v_R, = plt.plot(data_file_z0.R,data_file_z0.Br,'ro', label='Br')
by_v_R, = plt.plot(data_file_z0.R,data_file_z0.By,'g^', label='By')
plt.legend(handles=[br_v_R,by_v_R],loc=2)
plt.xlabel('r (mm)')
plt.ylabel('Br (T)')
plt.title('Br and By vs r at X = 0, Z={0}'.format(-4929))
#plt.axis([-0.1, 3.24,0.22,0.26])
plt.grid(True)
if offset:
  plt.savefig('Br_vs_R_offset.png')
else:
  plt.savefig('Br_vs_R.png')

#ani = animation.FuncAnimation(fig, update_quiver, fargs=(dfs, quiv), blit=False, interval=100)


plt.show()
#fig.savefig('.png')



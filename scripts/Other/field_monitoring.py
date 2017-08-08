#! /usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mu2e.src.fiteval_c2 import FitFunctionMaker2
plt.close('all')

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

ffm = FitFunctionMaker2('../mu2e/src/Mau10_800mm_long.csv')

nominal = ffm.mag_field_function(700,700,5000,True)

xs = np.linspace(690,710,200)
zs = np.linspace(4990,5010,200)

def bz_func(x,y,z):
    return ffm.mag_field_function(x,y,z,True)[2]
vbz_func = np.vectorize(bz_func,excluded=['y'])

def by_func(x,y,z):
    return ffm.mag_field_function(x,y,z,True)[1]
vby_func = np.vectorize(by_func,excluded=['y'])

def bx_func(x,y,z):
    return ffm.mag_field_function(x,y,z,True)[0]
vbx_func = np.vectorize(bx_func,excluded=['y'])


xxs,zzs = np.meshgrid(xs,zs)
bxs = vbx_func(x=xxs, y=700, z=zzs)
bzs = vbz_func(x=xxs, y=700, z=zzs)

pcm = plt.pcolormesh(xxs,zzs,(bxs-nominal[0])*1000,cmap='viridis')
plt.contour(xxs, zzs, bxs, 6, colors='k', linewidth=2)
plt.title('$\Delta$Bx vs X and Z')
plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
cb = plt.colorbar(pcm)
cb.set_label('$\Delta$Bx (Gauss)')
plt.plot(700,5000,'o',markersize=10, color='k')
plt.annotate('Nominal\nPosition', xy=(700, 5000), xytext=(703, 5004),
            #arrowprops=dict(facecolor='black', shrink=0.05, shrinkA=200),
            arrowprops=dict(facecolor='black', shrink=0.08),
	    size='large',
            )

plt.figure()

pcm = plt.pcolormesh(xxs,zzs,(bzs-nominal[2])*1000,cmap='viridis')
plt.contour(xxs, zzs, bzs, 6, colors='k', linewidth=2)
plt.title('$\Delta$Bz vs X and Z')
plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
cb = plt.colorbar(pcm)
cb.set_label('$\Delta$Bz (Gauss)')
plt.plot(700,5000,'o',markersize=10, color='k')
plt.annotate('Nominal\nPosition', xy=(700, 5000), xytext=(703, 5004),
            #arrowprops=dict(facecolor='black', shrink=0.05, shrinkA=200),
            arrowprops=dict(facecolor='black', shrink=0.08),
	    size='large',
            )
plt.show()

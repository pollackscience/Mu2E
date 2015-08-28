#! /usr/bin/env python

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

class FieldFitter:
  """Input hall probe measurements, perform semi-analytical fit, return fit function and other stuff."""


def legendre_2d((x,y), a,b,c,d):
  leg = a*x*y + b*(x*y**2) + c*(y*x**2) + d*(x**2*y**2)
  return leg.ravel()


plt.close('all')
plt.rc('font', family='serif')
fig = plt.figure()
plt.hold(True)
ax = fig.gca(projection='3d')
X1 = np.arange(-2, 2, 0.1)
Y1 = np.arange(-2, 2, 0.1)
Z1 = X1*(Y1**2+1)
X, Y = np.meshgrid(X1, Y1)
Z = X*(Y**2+1)
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
scat = ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), color='black')
#ax.set_zlim(-1.01, 1.01)

#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(r'Z=X$\times$(Y+1)$^2$')

#fig.colorbar(surf, shrink=0.5, aspect=5)


#Z_noisy = Z.ravel() + 0.5*np.random.normal(size=Z.ravel().shape)
popt,pcov = curve_fit(legendre_2d, (X,Y), Z.ravel(), p0=(1,1,1,1))

Z_fitted = legendre_2d((X,Y),*popt).reshape(40,40)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(X, Y, Z_fitted, rstride=1, cstride=1, cmap=cm.coolwarm, alpha = 0.5)
surf = ax.plot_wireframe(X, Y, Z_fitted,color='green')

#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
#plt.get_current_fig_manager().window.wm_geometry("-2600+1300")

plt.show()

plt.get_current_fig_manager().window.wm_geometry("-2600-600")

print popt




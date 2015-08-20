#! /usr/bin/env python

import os
import mu2e
import numpy as np
import pandas as pd
from datafileprod import DataFileMaker
import src.RowTransformations as rt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import gridspec
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.formula.api import wls
from statsmodels.graphics.regressionplots import abline_plot
from collections import OrderedDict
import re


class Plotter:
  """Class that takes prepped datafile and produces all kinds of neat plots"""

  def __init__(self, data_frame_dict,main_suffix=None,alt_save_dir=None,clear=True):
    """Default constructor, takes a dict of pandas DataFrame.
    (optional suffix and save dir)"""
    if clear: plt.close('all')

    self.markers = ['o','v','^','s']
    self.colors = ['blue','darksalmon','lightgreen','plum']

    if type(data_frame_dict) != dict: raise TypeError('data_frame_dict must be dict')
    if len(data_frame_dict)==0: raise Exception('data_frame_dict must have at least one entry')

    if main_suffix and main_suffix not in data_frame_dict.keys():
      raise KeyError('main_suffix: '+main_suffix+'not in keys: '+data_frame_dict.keys())

    if not main_suffix and len(data_frame_dict)>1:
      raise Exception('must specify main_suffix if len(dict)>1')

    if not alt_save_dir:
      save_dir = os.path.abspath(os.path.dirname(mu2e.__file__))+'/../plots'
    else:
      save_dir = alt_save_dir

    try: self.plot_count = plt.get_fignums()[-1]
    except: self.plot_count = 0

    if len(data_frame_dict)==1:
      self.data_frame_dict = OrderedDict(data_frame_dict)
      self.suffix = self.data_frame_dict.keys()[0]
    else:
      self.suffix = main_suffix
      keys = sorted([key for key in data_frame_dict.keys() if key not in main_suffix])
      keys.insert(0,main_suffix)
      self.data_frame_dict = OrderedDict()
      for key in keys:
        self.data_frame_dict[key] = data_frame_dict[key]
      self.suffix_extra = '-'.join(self.data_frame_dict.keys())
      self.init_save_dir(save_dir,extra=True)

    self.init_save_dir(save_dir)


  def plot_wrapper(func):
    def inner(self,*args,**kwargs):
      self.plot_count+=1
      print'Plot {0} is: {1} {2}'.format(self.plot_count,args,kwargs)
      return func(self,*args)
    return inner

  def init_save_dir(self,save_dir,extra=False):
    if extra:
      self.save_dir_extra = save_dir+'/'+self.suffix_extra

      if not os.path.exists(self.save_dir_extra):
            os.makedirs(self.save_dir_extra)
    else:
      if self.suffix!='':
        self.save_dir=save_dir+'/'+self.suffix

      if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

  def polar_to_cart(self,r,theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return(x, y)

  def make_r(self,row):
    return np.sqrt(row['X']**2+row['Y']**2)
  def center_x(self,row,offset = None):
    if offset == None: offset = 3904
    return row['X']-offset
  def make_br(self,row):
    return np.sqrt(row['Bx']**2+row['By']**2)
  def make_theta(self,row):
    return np.arctan2(row['Y'],row['X'])
  def make_bottom_half(self,row):
    return (-row['Y'])

  def make_interp_grid(self, r=None, data_frame_orig = None, AB = None, val_name = None, interp_num=200):
    """Make a grid for interpolation, either for R vs Theta or A vs B, where
    A and B are some combination of X,Y,Z.  If A and B are used, they must be a pair of triples of the form:
    [['A',start_A,stop_A],['B',start_B,stop_B]].  dicts are not used as I want to preserve order."""

    if r==None and AB==None: raise Exception('No grid parameters specified')
    elif r!=None and AB!=None: raise Exception('Grid parameters over-specified')
    elif AB!=None and type(AB)!=dict and len(AB)!=2: raise Exception('AB is specified incorrectly')

    if r!=None:
      theta_array = np.linspace(-np.pi,np.pi,num=interp_num)
      r_array = np.full(interp_num,r)
      x_array = np.ones(interp_num)
      y_array = np.ones(interp_num)
      val_array = np.full(interp_num,np.nan)
      for i in np.arange(interp_num):
        x_array[i],y_array[i] = self.polar_to_cart(r_array[i],theta_array[i])
      data_frame_interp_grid = pd.DataFrame({'X':x_array,'Y':y_array,val_name:val_array})
      #data_frame_interp = pd.DataFrame({'X':x_array,'Y':y_array,val_name:val_array})
      #data_frame_interp_grid = pd.concat([data_frame_interp,data_frame_orig])
      #data_frame_interp_grid = data_frame_interp_grid[abs(data_frame_interp_grid.X)<(r+50)]
      #data_frame_interp_grid = data_frame_interp_grid[abs(data_frame_interp_grid.Y)<(r+50)]
      data_frame_interp_grid.sort(['X','Y'],inplace=True)
    else:
      a_array = np.linspace(AB[0][1],AB[0][2],interp_num)
      b_array = np.linspace(AB[1][1],AB[1][2],interp_num)
      val_array = np.full(interp_num,np.nan)
      data_frame_interp_grid = pd.DataFrame({AB[0][0]:a_array,AB[1][0]:b_array,val_name:val_array})
      data_frame_interp_grid.sort([AB[0][0],AB[1][0]],inplace=True)
    return data_frame_interp_grid



  def interpolate_data(self, old_frame, new_frame, field, x_ax='X', y_ax='Y', method='cubic'):
    """Interpolate B-field values given a new A-B grid.
    Currently only interpolates for a given field component, then recalculates R, Theta"""
    old_piv = old_frame.pivot(x_ax,y_ax,field)
    old_x = old_piv.index.values
    old_y = old_piv.columns.values
    old_vals = old_piv.values
    new_piv = new_frame.pivot(x_ax,y_ax,field)
    new_x = new_piv.index.values
    new_y = new_piv.columns.values
    new_xx,new_yy = np.meshgrid(new_x,new_y)

    print 'interpolating', field
    interp_function = interpolate.interp2d(old_x,old_y,old_vals.T,kind=method)
    new_vals = interp_function(new_x,new_y)
    data_interp = np.array([new_xx, new_yy, new_vals]).reshape(3, -1).T
    interp_frame = pd.DataFrame(data_interp, columns=[x_ax,y_ax,field])

    C = [c for c in ['X','Y','Z'] if c not in [x_ax,y_ax]][0]
    print 'making new',C
    #interp_frame[C] = old_frame[C].unique()[0]
    interp_frame.eval('{0}={1}'.format(C,old_frame[C].unique()[0]))
    print 'making new Theta'
    #interp_frame['Theta'] = interp_frame.apply(self.make_theta,axis=1)
    interp_frame['Theta'] = rt.apply_make_theta(interp_frame['X'].values, interp_frame['Y'].values)
    print 'making new R'
    #interp_frame['R'] = interp_frame.apply(self.make_r,axis=1)
    interp_frame['R'] = rt.apply_make_r(interp_frame['X'].values, interp_frame['Y'].values)
    interp_frame = interp_frame[['X','Y','Z','R','Theta',field]]
    return interp_frame


  @plot_wrapper
  def plot_A_v_B(self,A,B,*conditions):
    """Plot A vs B given some set of comma seperated boolean conditions"""
    data_frame = self.data_frame_dict[self.suffix].query(' and '.join(conditions))
    print data_frame.head()

    fig = plt.figure(self.plot_count)
    data_frame.eval('{0}err = 0.0001*{0}'.format(A))
    #plt.plot(data_frame[B],data_frame[A],'ro')
    plt.errorbar(data_frame[B],data_frame[A],yerr=data_frame[A+'err'],fmt='ro')
    plt.xlabel(B)
    plt.ylabel(A)
    plt.title('{0} vs {1} at {2}'.format(A,B,conditions))
    #plt.axis([-0.1, 3.24,0.22,0.26])
    plt.grid(True)
    plt.savefig(self.save_dir+'/{0}_v_{1}_at_{2}{3}.png'.format(A,B,'_'.join(conditions),self.suffix))
    return data_frame, fig

  @plot_wrapper
  def plot_A_v_B_ratio(self,A,B,*conditions):
    """Plot A vs B given some set of comma seperated boolean conditions, use multiple dataframes"""
    fig = plt.figure(self.plot_count)
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    ax1=fig.add_subplot(gs[0])
    ax2=fig.add_subplot(gs[1],sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.subplots_adjust(hspace=0)


    data_frame_dict = OrderedDict()
    for key in self.data_frame_dict:
      data_frame_dict[key] = self.data_frame_dict[key].query(' and '.join(conditions))
      data_frame_dict[key].eval('{0}err = 0.0001*{0}'.format(A))

    for i,key in enumerate(data_frame_dict):
      ax1.errorbar(data_frame_dict[key][B],data_frame_dict[key][A],yerr=data_frame_dict[key][A+'err'],
          linestyle='None',marker=self.markers[i], color = self.colors[i], markersize=7,label=key)
      if i>0:
        ax2.plot(data_frame_dict[key][B],data_frame_dict.values()[0][A]/data_frame_dict[key][A],
            linestyle='None',marker=self.markers[i], color= self.colors[i], markersize = 7,label=key)

    ax2.set_xlabel(B)
    ax2.axhline(1,linewidth=2,color='r')
    ax1.set_ylabel(A)
    labels = data_frame_dict.keys()
    labels = [re.sub('_','\_',i) for i in labels]

    if len(data_frame_dict)==2:
      ax2.set_ylabel(r'$\frac{\mathrm{'+labels[0]+r'}}{\mathrm{'+labels[1]+r'}}$')
    ax1.set_title('{0} vs {1} at {2}'.format(A,B,conditions))
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend(loc='best')
    plt.setp(ax2.get_yticklabels()[-2:], visible=False)
    fig.savefig(self.save_dir_extra+'/{0}_v_{1}_at_{2}{3}.png'.format(A,B,'_'.join(conditions),self.suffix_extra))
    return data_frame_dict, fig

  @plot_wrapper
  def plot_A_v_B_and_fit(self,A,B,*conditions):
    """Plot A vs B given some set of comma seperated boolean conditions"""
    data_frame = self.data_frame_dict[self.suffix].query(' and '.join(conditions))
    print data_frame.head()

    fig = plt.figure(self.plot_count)
    data_frame.eval('{0}err = 0.0001*{0}'.format(A))
    #plt.plot(data_frame[B],data_frame[A],'ro')
    plt.errorbar(data_frame[B],data_frame[A],yerr=data_frame[A+'err'],fmt='ro')
    plt.xlabel(B)
    plt.ylabel(A)
    plt.title('{0} vs {1} at {2}'.format(A,B,conditions))
    #plt.axis([-0.1, 3.24,0.22,0.26])
    plt.grid(True)
    lm = self.fit_linear_regression(data_frame,A,B,fig)
    plt.savefig(self.save_dir+'/{0}_v_{1}_at_{2}{3}_fit.png'.format(A,B,'_'.join(conditions),self.suffix))
    return data_frame, fig, lm

  @plot_wrapper
  def plot_A_v_B_and_C(self,A='Bz',B='X',C='Z',interp=False,interp_num=300, *conditions):
    """Plot A vs B and C given some set of comma seperated boolean conditions.
    B and C are the independent, A is the dependent. A bit complicated right now to get
    proper setup for contour plotting."""
    data_frame = self.data_frame_dict[self.suffix].query(' and '.join(conditions))
    print data_frame.head()

    if interp:
      data_frame_interp_grid = self.make_interp_grid(AB=[[B,data_frame[B].min(),data_frame[B].max()],[C,data_frame[C].min(),data_frame[C].max()]],
          data_frame_orig=data_frame,val_name=A,interp_num=interp_num)
      data_frame = self.interpolate_data(data_frame, data_frame_interp_grid, field = A, x_ax = B, y_ax =C, method='cubic')

    data_frame = data_frame.reindex(columns=[A,B,C])
    piv = data_frame.pivot(B,C,A)
    X=piv.columns.values
    Y=piv.index.values
    Z=piv.values
    Xi,Yi = np.meshgrid(X, Y)

    fig = plt.figure(self.plot_count).gca(projection='3d')
    surf = fig.plot_surface(Xi, Yi, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
    fig.zaxis.set_major_locator(LinearLocator(10))
    fig.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.view_init(elev=20., azim=45)

    #cb = plt.colorbar(surf, shrink=0.5, aspect=5)
    #cb.set_label(A)

    plt.xlabel(C)
    plt.ylabel(B)
    fig.set_zlabel(A)
    plt.title('{0} vs {1} and {2}, {3}'.format(A,B,C,conditions[0]))
    #plt.axis([-0.1, 3.24,0.22,0.26])
    #plt.grid(True)
    if interp:
      plt.savefig(self.save_dir+'/{0}_v_{1}_and_{2}_at_{3}_cont_interp{4}.png'.format(A,B,C,'_'.join(conditions),self.suffix),bbox_inches='tight')
    else:
      plt.savefig(self.save_dir+'/{0}_v_{1}_and_{2}_at_{3}_cont{4}.png'.format(A,B,C,'_'.join(conditions),self.suffix),bbox_inches='tight')

    self.plot_count+=1
    fig = plt.figure(self.plot_count)
    heat = plt.pcolor(Xi,Yi,Z)

    cb = plt.colorbar(heat, shrink=0.5, aspect=5)
    cb.set_label(A)

    plt.xlabel(C)
    plt.ylabel(B)
    plt.title('{0} vs {1} and {2}, {3}'.format(A,B,C,conditions[0]))
    plt.grid(True)
    if interp:
      plt.savefig(self.save_dir+'/{0}_v_{1}_and_{2}_at_{3}_heat_interp{4}.png'.format(A,B,C,'_'.join(conditions),self.suffix),bbox_inches='tight')
    else:
      plt.savefig(self.save_dir+'/{0}_v_{1}_and_{2}_at_{3}_heat{4}.png'.format(A,B,C,'_'.join(conditions),self.suffix),bbox_inches='tight')
    return fig,data_frame

  @plot_wrapper
  def plot_A_v_B_and_C_ratio(self,A='Bz',B='X',C='Z',interp=False,interp_num=300, *conditions):
    """Plot A vs B and C given some set of comma seperated boolean conditions.
    B and C are the independent, A is the dependent. A bit complicated right now to get
    proper setup for contour plotting."""
    data_frame = self.data_frame_dict[self.suffix].query(' and '.join(conditions))
    print data_frame.head()

    if interp:
      data_frame_interp_grid = self.make_interp_grid(AB=[[B,data_frame[B].min(),data_frame[B].max()],[C,data_frame[C].min(),data_frame[C].max()]],
          data_frame_orig=data_frame,val_name=A,interp_num=interp_num)
      data_frame = self.interpolate_data(data_frame, data_frame_interp_grid, field = A, x_ax = B, y_ax =C, method='cubic')

    data_frame = data_frame.reindex(columns=[A,B,C])
    piv = data_frame.pivot(B,C,A)
    X=piv.columns.values
    Y=piv.index.values
    Z=piv.values
    Xi,Yi = np.meshgrid(X, Y)

    fig = plt.figure(self.plot_count).gca(projection='3d')
    surf = fig.plot_surface(Xi, Yi, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
    fig.zaxis.set_major_locator(LinearLocator(10))
    fig.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.view_init(elev=20., azim=45)

    #cb = plt.colorbar(surf, shrink=0.5, aspect=5)
    #cb.set_label(A)

    plt.xlabel(C)
    plt.ylabel(B)
    fig.set_zlabel(A)
    plt.title('{0} vs {1} and {2}, {3}'.format(A,B,C,conditions[0]))
    #plt.axis([-0.1, 3.24,0.22,0.26])
    #plt.grid(True)
    if interp:
      plt.savefig(self.save_dir+'/{0}_v_{1}_and_{2}_at_{3}_cont_interp{4}.png'.format(A,B,C,'_'.join(conditions),self.suffix),bbox_inches='tight')
    else:
      plt.savefig(self.save_dir+'/{0}_v_{1}_and_{2}_at_{3}_cont{4}.png'.format(A,B,C,'_'.join(conditions),self.suffix),bbox_inches='tight')

    self.plot_count+=1
    fig = plt.figure(self.plot_count)
    heat = plt.pcolor(Xi,Yi,Z)

    cb = plt.colorbar(heat, shrink=0.5, aspect=5)
    cb.set_label(A)

    plt.xlabel(C)
    plt.ylabel(B)
    plt.title('{0} vs {1} and {2}, {3}'.format(A,B,C,conditions[0]))
    plt.grid(True)
    if interp:
      plt.savefig(self.save_dir+'/{0}_v_{1}_and_{2}_at_{3}_heat_interp{4}.png'.format(A,B,C,'_'.join(conditions),self.suffix),bbox_inches='tight')
    else:
      plt.savefig(self.save_dir+'/{0}_v_{1}_and_{2}_at_{3}_heat{4}.png'.format(A,B,C,'_'.join(conditions),self.suffix),bbox_inches='tight')
    return fig,data_frame


  @plot_wrapper
  def plot_A_v_Theta(self,A,r,z_cond,interp_num=200,method='linear',do_fit = True):
    from scipy import interpolate
    """Plot A vs Theta for a given Z and R. The values are interpolated """

    print self.data_frame_dict[self.suffix].head()
    data_frame = self.data_frame_dict[self.suffix].query(z_cond)
    if method!=None:

      data_frame_interp_grid = self.make_interp_grid(r=r, data_frame_orig=data_frame,val_name=A,interp_num=interp_num)
      data_frame_interp = self.interpolate_data(data_frame, data_frame_interp_grid, A,method=method)
      #data_frame_interp = data_frame_interp.query('R=={0}'.format(r))
      data_frame_interp = data_frame_interp[(data_frame_interp.R-r).abs()<0.0005]

    data_frame = data_frame[(data_frame.R-r).abs()<0.05]

    #print data_frame.head()
    #print data_frame_interp.head()
    #print data_frame.head()
    #raw_input()

    fig = plt.figure(self.plot_count)
    if method!=None:
      data_frame_interp.eval('{0}err = 0.0001*{0}'.format(A))
      #plt.plot(data_frame_interp.Theta,data_frame_interp[A],'b^')
      plt.errorbar(data_frame_interp.Theta,data_frame_interp[A],yerr=data_frame_interp[A+'err'],fmt='b^')
    plt.plot(data_frame.Theta,data_frame[A],'ro')
    plt.xlabel('Theta')
    plt.ylabel(A)
    plt.title('{0} vs Theta at {1} for R=={2}'.format(A,z_cond,r))
    ###plt.axis([-0.1, 3.24,0.22,0.26])
    plt.grid(True)
    savename = self.save_dir+'/{0}_v_Theta_at_{1}_R=={2}{3}.png'.format(A,z_cond,r,self.suffix)
    if not do_fit:
      plt.savefig(savename,bbox_inches='tight')
    else:
      if method:
        popt,pcov = self.fit_radial_plot(data_frame_interp, A, savename=savename,fig=fig,p0=(-0.0001,0.0,0.05))
      else:
        popt,pcov = self.fit_radial_plot(data_frame, A, savename=savename,fig=fig,p0=(-0.0001,0.0,0.05))

    if (method and do_fit): return data_frame_interp,fig,popt,pcov
    elif (method and not do_fit): return data_frame_interp,fig
    elif (not method and do_fit): return data_frame,fig,popt,pcov
    else: return data_frame,fig

  @plot_wrapper
  def plot_mag_field(self,step_size = 1,*conditions):
    data_frame = self.data_frame_dict[self.suffix].query(' and '.join(conditions))
    fig, ax = plt.subplots(1,1)
    print 'len Y',len(np.unique(data_frame.Y.values))
    print 'len X',len(np.unique(data_frame.X.values))
    print data_frame.head()
    quiv = ax.quiver(data_frame.X[::step_size],data_frame.Y[::step_size],data_frame.Bx[::step_size],data_frame.By[::step_size],pivot='mid')
    plt.quiverkey(quiv, 1400, -1430, 0.5, '0.5 T', coordinates='data',clim=[-1.1,5])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Radial Magnetic Components, {0}'.format([x for x in conditions if 'Z' in x]))
    plt.grid(True)
    circle2=plt.Circle((0,0),831.038507,color='b',fill=False)
    fig.gca().add_artist(circle2)
    fig.savefig(self.save_dir+'/PsField_{0}{1}.png'.format('_'.join(conditions),self.suffix))

  @plot_wrapper
  def plot_mag_field2(self,A,B,density= 1,*conditions):
    data_frame = self.data_frame_dict[self.suffix].query(' and '.join(conditions))
    fig, ax = plt.subplots(1,1)
    piv = data_frame.pivot(A,B,'B'+A.lower())
    xax = piv.columns.values
    yax = piv.index.values
    V = piv.values
    U = data_frame.pivot(A,B,'B'+B.lower()).values

    mag = np.sqrt(V**2+U**2)


    plt.streamplot(xax, yax, U, V, color = mag, density=density,linewidth=2)

    plt.xlabel('{} (mm)'.format(B))
    plt.ylabel('{} (mm)'.format(A))
    cb = plt.colorbar()
    cb.set_label('Mag (T)')

    plt.title('Magnetic Field Lines in {0}-{1} plane for {2}'.format(A,B,conditions))
    fig.savefig(self.save_dir+'/Field_Lines_{0}{1}.png'.format('_'.join(conditions),self.suffix),bbox_inches='tight')

  def fit_radial_plot(self, df, mag, savename,fig=None,p0=(0.0001,0.0,0.05)):
    """Given a data_frame, fit the theta vs B(r)(z) plot and plot the result"""
    def cos_func(x, A,p1,p2):
      return A*np.cos(x+p1)+p2
    #popt, pcov = curve_fit(cos_func, df.Theta.values, df[mag].values, sigma=df[mag+'err'].values, absolute_sigma=True, p0=p0)
    popt, pcov = curve_fit(cos_func, df.Theta.values, df[mag].values, sigma=df[mag+'err'].values, p0=p0)
    try:
      std_devs = np.sqrt(np.diag(pcov))
    except:
      std_devs = [0,0,0]

    if fig==None:
      fig = plt.gcf()
    elif type(fig) == str and fig.lowercase() == 'new':
      self.plot_count+=1
      plt.figure(self.plot_count)

    curvex=np.linspace(-np.pi,np.pi,500)
    curvey=cos_func(curvex,popt[0],popt[1],popt[2])
    plt.plot(curvex,curvey,color='lawngreen',linestyle='--',linewidth=3)
    plt.figtext(0.33, 0.75,
        'Amplitude: {0:.2e}$\pm${1:.1e}\nPhase: {2:.2e}$\pm${3:.1e}\nY-Offset: {4:.2e}$\pm${5:.1e}'.format(
          popt[0],std_devs[0],popt[1],std_devs[1],popt[2],std_devs[2]),
        size='large')
    plt.draw()
    if '.png' in savename: savename = savename.split('.png')[0]
    plt.savefig(savename+'_fit.png',bbox_inches='tight')
    return popt,pcov

  def fit_linear_regression(self, df,A,B,fig=None,text_x=0.15,text_y=0.75):
    """Basic WLS for DataFrames"""
    lm = wls(formula = '{0} ~ {1}'.format(A,B), data=df, weights=df[A+'err']).fit()
    if fig==None:
      fig = plt.gcf()
    elif type(fig) == str and fig.lower() == 'new':
      self.plot_count+=1
      fig = plt.figure(self.plot_count)

    abline_plot(model_results=lm,ax=fig.axes[0])
    plt.figtext(text_x, text_y,
      'Intercept: {0:.3e}$\pm${1:.3e}\nSlope: {2:.3e}$\pm${3:.3e}'.format(lm.params[0], lm.bse[0], lm.params[1], lm.bse[1]),
      figure=fig, size='large',axes=fig.axes[0])
    plt.draw()
    return lm

if __name__=="__main__":
  data_maker=DataFileMaker('FieldMapData_1760_v5/Mu2e_PSMap_fastTest',use_pickle = True)
  plot_maker = Plotter(data_maker.data_frame)
  #fit_compare_sweep()


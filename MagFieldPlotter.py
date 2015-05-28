#! /usr/bin/env python

from DataFileProducer import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate


class Plotter:
  """Class that takes prepped datafile and produces all kinds of neat plots"""
  def __init__(self, data_frame):
    self.data_frame = data_frame
    self.plot_count = 0

  def plot_wrapper(func):
    def inner(self,*args,**kwargs):
      self.plot_count+=1
      print'Plot {0} is: {1} {2}'.format(self.plot_count,args,kwargs)
      return func(self,*args)
    return inner

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

  def interpolate_data(self, old_frame, new_frame, field, method='cubic'):
    """Interpolate B-field values given a new x-y grid.
    Currently only interpolates Bx, By, Bz, then recalculates R, Theta, Br"""
    old_piv = old_frame.pivot('X','Y',field)
    old_x = old_piv.index.values
    old_y = old_piv.columns.values
    old_vals = old_piv.values
    new_piv = new_frame.pivot('X','Y',field)
    new_x = new_piv.index.values
    new_y = new_piv.columns.values
    new_xx,new_yy = np.meshgrid(new_x,new_y)

    print 'interpolating', field
    interp_function = interpolate.interp2d(old_x,old_y,old_vals.T,kind=method)
    new_vals = interp_function(new_x,new_y)
    data_interp = np.array([new_xx, new_yy, new_vals]).reshape(3, -1).T
    interp_frame = pd.DataFrame(data_interp, columns=['X','Y',field])

    print 'making new theta'
    interp_frame['Theta'] = interp_frame.apply(self.make_theta,axis=1)
    print 'making new r'
    interp_frame['R'] = interp_frame.apply(self.make_r,axis=1)
    #print 'making new br'
    #interp_frame['Br'] = interp_frame.apply(self.make_br,axis=1)
    print 'making new z'
    interp_frame['Z'] = old_frame.Z.unique()[0]
    interp_frame = interp_frame[['X','Y','Z','R','Theta',field]]
    return interp_frame





  @plot_wrapper
  def plot_A_v_B(self,A,B,*conditions):
    """Plot A vs B given some set of comma seperated boolean conditions"""
    data_frame = self.data_frame.query(' and '.join(conditions))
    print data_frame.head()

    plt.figure(self.plot_count)
    plt.plot(data_frame[B],data_frame[A],'ro')
    plt.xlabel(B)
    plt.ylabel(A)
    plt.title('{0} vs {1} at {2}'.format(A,B,conditions))
    #plt.axis([-0.1, 3.24,0.22,0.26])
    plt.grid(True)
    plt.savefig('plots/{0}_v_{1}_at_{2}.png'.format(A,B,'_'.join(conditions)))

  @plot_wrapper
  def plot_A_v_B_and_C(self,A,B,C,*conditions):
    """Plot A vs B and C given some set of comma seperated boolean conditions.
    B and C are the independent, A is the dependent. A bit complicated right now to get
    proper setup for contour plotting."""
    data_frame = self.data_frame.query(' and '.join(conditions))
    data_frame = data_frame.reindex(columns=[A,B,C])
    print data_frame.head()

    data_frame = data_frame.pivot(B,C)
    X=data_frame.columns.levels[1].values
    Y=data_frame.index.values
    Z=data_frame.values
    Xi,Yi = np.meshgrid(X, Y)

    fig = plt.figure(self.plot_count).gca(projection='3d')
    surf = fig.plot_surface(Xi, Yi, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
    fig.zaxis.set_major_locator(LinearLocator(10))
    fig.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.view_init(elev=20., azim=45)

    cb = plt.colorbar(surf, shrink=0.5, aspect=5)
    cb.set_label(A)

    plt.xlabel(C)
    plt.ylabel(B)
    fig.set_zlabel(A)
    plt.title('{0} vs {1} and {2}, {3}'.format(A,B,C,conditions[0]))
    #plt.axis([-0.1, 3.24,0.22,0.26])
    #plt.grid(True)
    plt.savefig('plots/{0}_v_{1}_and_{2}_at_{3}.png'.format(A,B,C,'_'.join(conditions)))


  @plot_wrapper
  def plot_Bz_v_Theta(self,r,z_cond,interp_num=200,method='linear',order=None):
    from scipy import interpolate
    """Plot Bz vs Theta for a given Z and R. The values are interpolated """

    print self.data_frame.head()
    data_frame = self.data_frame.query(z_cond)
    #f = interpolate.interp2d(data_frame.X, data_frame.Y, data_frame.Br, kind='cubic')

    theta_array = np.linspace(-np.pi,np.pi,num=interp_num)
    r_array = np.full(interp_num,r)
    #z_array = np.full(interp_num,data_frame.Z.unique()[0])
    #bz_array = np.full(interp_num,np.nan)
    x_array = np.ones(interp_num)
    y_array = np.ones(interp_num)
    val_array = np.full(interp_num,np.nan)
    for i in np.arange(interp_num):
      x_array[i],y_array[i] = self.polar_to_cart(r_array[i],theta_array[i])
    data_frame_interp = pd.DataFrame({'X':x_array,'Y':y_array,'Bz':val_array})
    data_frame_interp_grid = pd.concat([data_frame_interp,data_frame])
    data_frame_interp_grid = data_frame_interp_grid[abs(data_frame_interp_grid.X)<(r+50)]
    data_frame_interp_grid = data_frame_interp_grid[abs(data_frame_interp_grid.Y)<(r+50)]
    data_frame_interp_grid.sort(['X','Y'],inplace=True)
    #df_tmp = pd.DataFrame({'X':x_array,'Y':y_array})
    #df_tmp = pd.concat([df_tmp,data_frame])
    #df_tmp = df_tmp[(abs(df_tmp.X)<r+50) & (abs(df_tmp.Y)<r+50)]
    #df_tmp.sort(['X','Y'],inplace=True)

    data_frame_interp = self.interpolate_data(data_frame, data_frame_interp_grid, 'Bz')
    #data_frame_interp = pd.DataFrame({'X':x_array,'Y':y_array,'Z':z_array,'Theta':theta_array,'R':r_array,'Bz':bz_array})
    #data_frame = pd.concat([data_frame,data_frame_interp])
    #data_frame.sort(['R','Theta'],inplace=True)
    #data_frame.sort('R',inplace=True)
    #data_frame.Bx.interpolate(inplace=True,method=method,order=order)
    #data_frame.By.interpolate(inplace=True,method=method,order=order)
    #data_frame.Bz.interpolate(inplace=True,method=method,order=order)
    #data_frame.Br = data_frame.apply(self.make_br, axis=1)
    #data_frame.interpolate(inplace=True,method=method)

    data_frame = data_frame[(data_frame.R-r).abs()<0.05]

    print data_frame.head()
    data_frame_interp = data_frame_interp.query('R=={0}'.format(r))
    print data_frame_interp.head()
    #print data_frame.head()
    #raw_input()

    plt.figure(self.plot_count)
    plt.plot(data_frame_interp.Theta,data_frame_interp.Bz,'b^')
    plt.plot(data_frame.Theta,data_frame.Bz,'ro')
    plt.xlabel('Theta')
    plt.ylabel('Bz')
    plt.title('Bz vs Theta at {0} for R=={1}'.format(z_cond,r))
    ###plt.axis([-0.1, 3.24,0.22,0.26])
    plt.grid(True)
    plt.savefig('plots/Bz_v_Theta_at_{0}_R=={1}.png'.format(z_cond,r))
    #data_frame_interp_grid = pd.concat([data_frame_interp,data_frame])
    #data_frame_interp_grid = data_frame_interp_grid[abs(data_frame_interp_grid.X)<(r+50)]
    #data_frame_interp_grid = data_frame_interp_grid[abs(data_frame_interp_grid.Y)<(r+50)]
    #data_frame_interp_grid.sort(['X','Y'],inplace=True)
    #return data_frame, data_frame_interp,data_frame_interp_grid

  @plot_wrapper
  def plot_mag_field(self,step_size = 1,*conditions):
    data_frame = self.data_frame.query(' and '.join(conditions))
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
    fig.savefig('plots/PsField_{0}.png'.format('_'.join(conditions)))




if __name__=="__main__":
  data_maker=DataFileMaker('FieldMapData_1760_v5/Mu2e_PSMap',use_pickle = True)
  plot_maker = Plotter(data_maker.data_frame)
  #plot_maker.plot_A_v_B('Br','Y','Z==-4929','X==0')
  #plot_maker.plot_A_v_B('Br','Y','Z==-4929','X==400')
  #plot_maker.plot_mag_field(5,'Z==-4929','Y<1200','X<1075','Y>-1200','X>-1075')
  #print plot_maker.data_frame.head()
  #plot_maker.plot_Br_v_Theta(831.038507,'Z==-4929',method='polynomial',order=2)
  #plot_maker.plot_A_v_B('Bz','Theta','Z==-4929','R>200','R<202')
  #plot_maker.plot_A_v_B('Bz','X','Z==-4929','Y==0')
  #plot_maker.plot_mag_field(1,'Z==-4929')
  #plot_maker.plot_A_v_B_and_C('Bz','X','Z','Y==0','Z>-5000','Z<-4000','X>500')
  #data_frame, data_frame_interp,data_frame_grid = plot_maker.plot_Br_v_Theta(201.556444,'Z==-4929',300)
  plot_maker.plot_Bz_v_Theta(201.556444,'Z==-4929',300)


  plt.show()

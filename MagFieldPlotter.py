#! /usr/bin/env python

from DataFileProducer import *
import matplotlib.pyplot as plt


class Plotter:
  """Class that takes prepped datafile and produces all kinds of neat plots"""
  def __init__(self, data_file):
    self.data_file = data_file
    self.plot_count = 0

  def plot_wrapper(func):
    def inner(self,*args):
      self.plot_count+=1
      print'Plot {0} is: {1}'.format(self.plot_count,args)
      return func(self,*args)
    return inner

  @plot_wrapper
  def plot_A_v_B(self,A,B,*conditions):
    """Plot A vs B given some set of comma seperated boolean conditions"""
    print self.data_file.head()
    print ' and '.join(conditions)
    data_file = self.data_file.query(' and '.join(conditions))

    plt.figure(self.plot_count)
    plt.plot(data_file[B],data_file[A],'ro')
    plt.xlabel(B)
    plt.ylabel(A)
    plt.title('{0} vs {1} at {2}'.format(A,B,conditions))
    #plt.axis([-0.1, 3.24,0.22,0.26])
    plt.grid(True)
    plt.savefig('{0}_v_{1}_at_{2}.png'.format(A,B,'_'.join(conditions)))

  @plot_wrapper
  def plot_mag_field(self,step_size = 1,*conditions):
    data_file = self.data_file.query(' and '.join(conditions))
    fig, ax = plt.subplots(1,1)
    print 'len Y',len(np.unique(data_file.Y.values))
    print 'len X',len(np.unique(data_file.X.values))
    print data_file.head()
    quiv = ax.quiver(data_file.X[::step_size],data_file.Y[::step_size],data_file.Bx[::step_size],data_file.By[::step_size],pivot='mid')
    plt.quiverkey(quiv, 1400, -1430, 0.5, '0.5 T', coordinates='data',clim=[-1.1,5])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Radial Magnetic Components, {0}'.format([x for x in conditions if 'Z' in x]))
    plt.grid(True)
    circle2=plt.Circle((0,0),831.038507,color='b',fill=False)
    fig.gca().add_artist(circle2)
    fig.savefig('PsField_{0}.png'.format('_'.join(conditions)))



if __name__=="__main__":
  data_maker=DataFileMaker('FieldMapData_1760_v5/Mu2e_PSMap',use_pickle = True)
  plot_maker = Plotter(data_maker.data_file)
  #plot_maker.plot_A_v_B('Br','Y','Z==-4929','X==0')
  #plot_maker.plot_A_v_B('Br','Y','Z==-4929','X==400')
  plot_maker.plot_mag_field(5,'Z==-4929','Y<1200','X<1075','Y>-1200','X>-1075')
  print plot_maker.data_file.head()
  #plot_maker.plot_A_v_B('Br','Theta','Z==-4929','R>831','R<832')
  #plot_maker.plot_mag_field(1,'Z==-4929')


  plt.show()

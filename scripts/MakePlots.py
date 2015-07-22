#! /usr/bin/env python

from mu2e.datafileprod import DataFileMaker
from mu2e.plotter import Plotter
import matplotlib.pyplot as plt

def hall_probe_sweep():
  hall_probe_distances = [40,80,120,160]
  #z_regions = [-6129,-6154,-6179,-6204,-6229]
  z_regions = range(-6229,-4004,500)
  df_fit_values = pd.DataFrame()
  for z in z_regions:
    df,fig,lm = plot_maker.plot_A_v_B_and_fit('Br','X','Z=={}'.format(z),'Y==0','X>=40','X<=160')
    for r in hall_probe_distances:
      df,fig,popt,pcov = plot_maker.plot_A_v_Theta('Br',r,'Z=={}'.format(z),18,'cubic')
      Aerr = 0
      try: Aerr = np.sqrt(np.diag(pcov)[0])
      except: Aerr = df_fit_values.tail(1)['A err'].values[0]
      offset = popt[0]/lm.params[1]
      offset_err = abs(offset)*np.sqrt((Aerr/popt[0])**2+(lm.bse[1]/lm.params[1])**2)
      df_fit_values = df_fit_values.append({'Z':z,'R':r,'A':popt[0],'A err':Aerr,'dBr/dr':lm.params[1],'dBr/dr err':lm.bse[1],
        'Offset (mm)':offset, 'Offset err':offset_err},ignore_index=True)
      #df,fig,popt,pcov = plot_maker.plot_A_v_Theta('Bz',r,'Z=={}'.format(z),300,'cubic')
      #popt,pcov = plot_maker.fit_radial_plot(df,'Bz')
      plt.close('all')

  df_fit_values_indexed = df_fit_values.set_index(['Z','R'])
  ax = df_fit_values_indexed.unstack().plot(y='Offset (mm)',yerr='Offset err',kind='line', style='^',linewidth=2)
  ax.set_ylabel('Offset (mm)')
  ax.set_xlabel('Z Position (mm)')
  ax.set_title('Offset Calculation')
  ax.set_xlim(ax.get_xlim()[0]-100, ax.get_xlim()[1]+100)
  plt.savefig(plot_maker.save_dir+'/offsets.png')


  ax = df_fit_values_indexed.unstack().plot(y='A',yerr='A err',kind='line', style='^',linewidth=2)
  ax.set_ylabel('Br Amplitude (T)')
  ax.set_xlabel('Z Position (mm)')
  ax.set_title('Amplitude of Sinusoid Fit')
  ax.set_xlim(ax.get_xlim()[0]-100, ax.get_xlim()[1]+100)
  plt.savefig(plot_maker.save_dir+'/amplitude.png')

  ax = df_fit_values_indexed.iloc[df_fit_values_indexed.index.get_level_values('R') == 40].unstack().plot(y='dBr/dr',yerr='dBr/dr err',kind='line', style='^',linewidth=2,legend=False)
  ax.set_ylabel('dBr/dr')
  ax.set_xlabel('Z Position (mm)')
  ax.set_title('Change in Br as a Function of r')
  ax.set_xlim(ax.get_xlim()[0]-100, ax.get_xlim()[1]+100)
  plt.savefig(plot_maker.save_dir+'/slope.png')


def fit_compare_sweep():
  z_regions = range(-6229,-4004,500)
  for z in z_regions:
    df_left, fig_left = plot_maker.plot_A_v_B('Br','X','Z=={}'.format(z),'Y==0','X>-300','X<-100')
    df_right, fig_right = plot_maker.plot_A_v_B('Br','X','Z=={}'.format(z),'Y==0','X<300','X>100')
    df_full, fig_full = plot_maker.plot_A_v_B('Br','X','Z=={}'.format(z),'Y==0','((X<300&X>100)|(X>-300&X<-100))')
    lm_right = plot_maker.fit_linear_regression(df_right, 'Br','X',fig=fig_full,text_x=0.44,text_y=0.75)
    lm_left = plot_maker.fit_linear_regression(df_left, 'Br','X',fig=fig_full,text_x=0.25,text_y=0.55)
    plt.savefig(plot_maker.save_dir+'/Br_v_X_at_Z=={}_fit_comp.png'.format(z))

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


if __name__=="__main__":
  #import sys
  #print sys.path
  data_maker=DataFileMaker('../FieldMapData_1760_v5/Mu2e_PSMap_fastTest',use_pickle = True)
  plot_maker = Plotter(data_maker.data_frame)
  fit_compare_sweep()


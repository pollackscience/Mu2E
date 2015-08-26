#! /usr/bin/env python

from mu2e.datafileprod import DataFileMaker
from mu2e.plotter import *
import matplotlib.pyplot as plt
import pandas as pd

def hall_probe_sweep(z_regions = None, probes = None):
  if z_regions == None:
    z_regions = range(-6229,-4004,500)
  if probes == None:
    hall_probe_distances = [40,80,120,160]
  else:
    hall_probe_distances = [probes] if type(probes)==int else probes
  #z_regions = [-6129,-6154,-6179,-6204,-6229]
  df_fit_values = pd.DataFrame()
  for z in z_regions:
    print 'Z ==',z
    df,fig,lm = plot_maker.plot_A_v_B_and_fit('Br','X','Z=={}'.format(z),'Y==0','X>=40','X<=160')
    for r in hall_probe_distances:
      print 'R ==',r
      df,fig,popt,pcov = plot_maker.plot_A_v_Theta('Br',r,'Z=={}'.format(z),18,'cubic')
      Aerr = 0
      try: Aerr = np.sqrt(np.diag(pcov)[0])
      except: Aerr = df_fit_values.tail(1)['A err'].values[0]
      offset = popt[0]/lm.params[1]
      offset_err = abs(offset)*np.sqrt((Aerr/popt[0])**2+(lm.bse[1]/lm.params[1])**2)
      df_fit_values = df_fit_values.append({'Z':z,'R':r,'A':popt[0],'A err':Aerr,'dBr/dr':lm.params[1],'dBr/dr err':lm.bse[1],
        'Offset (mm)':offset, 'Offset err':offset_err},ignore_index=True)
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

  ax = df_fit_values_indexed.iloc[df_fit_values_indexed.index.get_level_values('R') == hall_probe_distances[0]].unstack().plot(y='dBr/dr',yerr='dBr/dr err',kind='line', style='^',linewidth=2,legend=False)
  ax.set_ylabel('dBr/dr')
  ax.set_xlabel('Z Position (mm)')
  ax.set_title('Change in Br as a Function of r')
  ax.set_xlim(ax.get_xlim()[0]-100, ax.get_xlim()[1]+100)
  plt.savefig(plot_maker.save_dir+'/slope.png')


def fit_compare_sweep(z_regions = None):
  if z_regions == None:
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

def do_sym():
  plot_maker = Plotter({'DS_Mau':data_maker1.data_frame})
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','Z<4500')
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','4500<Z<8000')
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','8000<Z<13000')
  plot_maker = Plotter({'DS_GA01':data_maker2.data_frame},clear=False)
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','Z<4500')
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','4500<Z<8000')
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','8000<Z<13000')
  plot_maker = Plotter({'DS_GA02':data_maker3.data_frame},clear=False)
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','Z<4500')
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','4500<Z<8000')
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','8000<Z<13000')
  plot_maker.plot_symmetry('Y','Z',False,0,'X==0','-300<Y<300','Z<4500')
  plot_maker.plot_symmetry('Y','Z',False,0,'X==0','-300<Y<300','4500<Z<8000')
  plot_maker.plot_symmetry('Y','Z',False,0,'X==0','-300<Y<300','8000<Z<13000')
  plot_maker = Plotter({'DS_GA03':data_maker4.data_frame},clear=False)
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','Z<4500')
  plot_maker.plot_symmetry('X','Z',True,300,'Y==0','-300<X<300','4500<Z<8000')
  plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-300<X<300','8000<Z<13000')
  plot_maker.plot_symmetry('Y','Z',False,0,'X==0','-300<Y<300','Z<4500')
  plot_maker.plot_symmetry('Y','Z',False,0,'X==0','-300<Y<300','4500<Z<8000')
  plot_maker.plot_symmetry('Y','Z',False,0,'X==0','-300<Y<300','8000<Z<13000')

  plt.show()

def do_field_comps():
  #plotter = Plotter({'DS_Mau':data_maker1.data_frame,'DS_GA01':data_maker2.data_frame},'DS_Mau')
  #field_comps_set1D(plotter)
  #plotter = Plotter({'DS_GA01':data_maker2.data_frame,'DS_GA02':data_maker3.data_frame},'DS_GA01')
  #field_comps_set1D(plotter)
  #plotter = Plotter({'DS_GA02':data_maker3.data_frame,'DS_GA03':data_maker4.data_frame},'DS_GA02')
  #field_comps_set1D(plotter)
  plotter = Plotter({'DS_Mau':data_maker1.data_frame,'DS_GA03':data_maker4.data_frame},'DS_Mau')
  field_comps_set1D(plotter)

def field_comps_set1D(plotter):
  #plotter.plot_A_v_B_ratio('Bz','Z','Y==0','X==0')
  #plotter.plot_A_v_B_ratio('Bz','Z','Y==0','X==300')
  #plotter.plot_A_v_B_ratio('Bz','Z','Y==0','X==-300')
  #plotter.plot_A_v_B_ratio('Br','X','Y==0','Z==5171')
  #plotter.plot_A_v_B_ratio('Br','X','Y==0','Z==5171','-400<X<400')
  #plotter.plot_A_v_B_ratio('Br','Y','X==0','Z==5171')
  #plotter.plot_A_v_B_ratio('Br','Y','X==0','Z==5171','-400<Y<400')
  #plotter.plot_A_v_B_and_C_ratio('Br','X','Y','Z==5671')
  #plotter.plot_A_v_B_and_C_ratio('Br','X','Y','Z==7671')
  #plotter.plot_A_v_B_and_C_ratio('Br','X','Y','Z==9671')
  #plotter.plot_A_v_B_and_C_ratio('Br','X','Y','Z==11671')
  #plotter.plot_A_v_B_and_C_ratio('Br','X','Y','Z==5671','-300<X<300','-300<Y<300')
  plotter.plot_A_v_B_and_C_ratio('Br','X','Y','Z==4146','-300<X<300','-300<Y<300')


if __name__=="__main__":
  data_maker1=DataFileMaker('../FieldMapData_1760_v5/Mu2e_DSmap',use_pickle = True)
  data_maker2=DataFileMaker('../FieldMapsGA01/Mu2e_DS_GA0',use_pickle = True)
  data_maker3=DataFileMaker('../FieldMapsGA02/Mu2e_DS_GA0',use_pickle = True)
  data_maker4=DataFileMaker('../FieldMapsGA03/Mu2e_DS_GA0',use_pickle = True)
  plot_maker = Plotter({'DS_Mau':data_maker1.data_frame,'DS_GA01':data_maker2.data_frame,'DS_GA02':data_maker3.data_frame,'DS_GA03':data_maker4.data_frame},'DS_Mau')
  #do_sym()
  #plot_maker = Plotter(data_maker1.data_frame,suffix='DS_Mau',data_frame_dict={'DS_GA01':data_maker2.data_frame})
  #plot_maker.plot_symmetry('X','Z','Y==0','X<300','X>-300','Z<4500')
  #plot_maker.plot_symmetry('X','Z','Y==0','X<300','X>-300','4500<Z<8000')
  #plot_maker.plot_symmetry('X','Z','Y==0','X<300','X>-300','8000<Z<13000')
  #data_maker=DataFileMaker('../FieldMapData_1760_v5/Mu2e_PSMap_fastTest',use_pickle = True)
  #plot_maker = Plotter(data_maker.data_frame,suffix='PS')
  #fit_compare_sweep()

  #data_maker_offset=DataFileMaker('FieldMapData_1760_v5/Mu2e_PSMap_-2.52mmOffset',use_pickle = True)
  #plot_maker_offset = Plotter(data_maker_offset.data_frame,'-2.52mmOffset')
  #data_maker_offset2=DataFileMaker('FieldMapData_1760_v5/Mu2e_PSMap_-1.5mmOffset',use_pickle = True)
  #plot_maker_offset2 = Plotter(data_maker_offset2.data_frame,'-1.5mmOffset')
  #plot_maker.plot_A_v_B('Br','Y','Z==-4929','X==0')
  #plot_maker.plot_A_v_B('Br','Y','Z==-4929','X==400')
  #plot_maker.plot_mag_field(5,'Z==-4929','Y<1200','X<1075','Y>-1200','X>-1075')
  #print plot_maker.data_frame.head()
  #plot_maker.plot_Br_v_Theta(831.038507,'Z==-4929',method='polynomial',order=2)
  #plot_maker.plot_A_v_B('Bz','Theta','Z==-4929','R>200','R<202')
  #plot_maker.plot_A_v_B('Bz','X','Z==-4929','Y==0')
  #plot_maker.plot_mag_field(1,'Z==-4929')
  #plot_maker.plot_A_v_B_and_C('Bz','X','Z',True,300, 'Y==0','Z>-5000','Z<-4000','X>500')
  #plot_maker.plot_A_v_B_and_C('Bz','X','Z',False,0, 'Y==0')
  #plot_maker.plot_A_v_B_and_C('Bz','X','Z',True,300, 'Y==0','Z>-6200','Z<-5700','X>500','X<1000')
  #plot_maker.plot_A_v_B_and_C('Bz','X','Z',False,0, 'Y==0','Z>-6200','Z<-5700','X>500','X<1000')
  #data_frame, data_frame_interp,data_frame_grid = plot_maker.plot_Br_v_Theta(201.556444,'Z==-4929',300)
  #plot_maker.plot_A_v_Theta('Bz',500,'Z==-4929',300,'cubic')
  #plot_maker.plot_A_v_Theta('Br',150,'Z==-6179',300,'cubic')
  #plot_maker.plot_A_v_B_and_C('Br','X','Y',False,0, 'Z==-6179')
  #plot_maker.plot_mag_field2('X','Z',1.3,'Y==0')

  #plt.show()


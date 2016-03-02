#! /usr/bin/env python

from mu2e.datafileprod import DataFileMaker
from mu2e.plotter import *
from mu2e.tools.physics_funcs import *
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

def scalar_field_plotter(plotter,z_low_cond,z_high_cond,*other_conds):
    '''Only use with single df'''
    y_cond1 = 'Y==0'
    y_cond2 = 'Y==-600'
    y_cond3 = 'Y==600'
    z_cond = 'Z==5021'
    plotter.set_df(calc_scalar_field(plotter.get_df(),z_low_cond,z_high_cond,*other_conds))
    xz1_conditions = (y_cond1,z_low_cond,z_high_cond,)+other_conds
    xz2_conditions = (y_cond2,z_low_cond,z_high_cond,)+other_conds
    xz3_conditions = (y_cond3, z_low_cond,z_high_cond,)+other_conds
    xy_conditions = (z_cond,z_low_cond,z_high_cond,)+other_conds
    plotter.plot_A_v_B_and_C('Scalar','X','Z',False,0,*xz1_conditions)
    plotter.plot_A_v_B_and_C('Scalar','X','Z',False,0,*xz2_conditions)
    plotter.plot_A_v_B_and_C('Scalar','X','Z',False,0,*xz3_conditions)
    plotter.plot_A_v_B_and_C('Scalar','X','Y',False,0,*xy_conditions)


if __name__=="__main__":
    #data_maker0=DataFileMaker('../datafiles/FieldMapData_1760_v5/Mu2e_PSmap',use_pickle = True)
    #data_maker1=DataFileMaker('../datafiles/FieldMapData_1760_v5/Mu2e_DSmap',use_pickle = True)
    #data_maker2=DataFileMaker('../datafiles/FieldMapsGA01/Mu2e_DS_GA0',use_pickle = True)
    #data_maker3=DataFileMaker('../datafiles/FieldMapsGA02/Mu2e_DS_GA0',use_pickle = True)
    #data_maker4=DataFileMaker('../datafiles/FieldMapsGA03/Mu2e_DS_GA0',use_pickle = True)
    #data_maker5=DataFileMaker('../datafiles/FieldMapsGA04/Mu2e_DS_GA0',use_pickle = True)
    #data_maker6 = DataFileMaker('../datafiles/Mau10/Standard_Maps/Mu2e_DSMap',use_pickle = True)
    #data_maker7 = DataFileMaker('../datafiles/Mau10/TS_and_PS_OFF/Mu2e_DSMap',use_pickle = True)
    data_maker8 = DataFileMaker('../datafiles/Mau10/DS_OFF/Mu2e_DSMap',use_pickle = True)
    #plot_maker = Plotter({'DS_Mau':data_maker1.data_frame,'DS_GA01':data_maker2.data_frame,'DS_GA02':data_maker3.data_frame,'DS_GA03':data_maker4.data_frame},'DS_Mau')
    #plot_maker = Plotter({'DS_Mau9':data_maker1.data_frame,'DS_Mau10':data_maker6.data_frame},'DS_Mau10')
    #plot_maker = Plotter({'DS_Mau10':data_maker6.data_frame})
    #plot_maker = Plotter({'DS_GA04':data_maker5.data_frame})
    #plot_maker = Plotter({'DS_Mau':data_maker1.data_frame})
    #plot_maker = Plotter({'DS_Mau10_DS_only':data_maker7.data_frame})
    plot_maker = Plotter({'DS_Mau10_DS_off':data_maker8.data_frame})
    #plot_maker_ps = Plotter({'PS_Mau':data_maker0.data_frame})
    #do_sym()
    #plot_maker = Plotter(data_maker1.data_frame,suffix='DS_Mau',data_frame_dict={'DS_GA01':data_maker2.data_frame})
    #plot_maker.plot_symmetry('X','Z','Y==0','X<300','X>-300','Z<4500')
    #plot_maker.plot_symmetry('X','Z','Y==0','X<300','X>-300','4500<Z<8000')
    #plot_maker.plot_symmetry('X','Z','Y==0','X<300','X>-300','8000<Z<13000')
    #data_maker=DataFileMaker('../datafiles/FieldMapData_1760_v5/Mu2e_PSMap_fastTest',use_pickle = True)
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
    #plot_maker.plot_A_v_B('By','Y','Z==5021','X==0','Y<851','Y>-851')
    #plot_maker.plot_A_v_B('By','Y','Z==9521','X==0','Y<851','Y>-851')
    #plot_maker_ps.plot_A_v_B('By','Y','Z==-6129','X==0','Y<251','Y>-251')
    #plot_maker.plot_A_v_B('By','Z','Z>8400','Z<10000','X==0','Y==25')
    #plot_maker.plot_A_v_B('Bz','Z','Z>8400','Z<10000','X==0','Y==50')
    #plot_maker.plot_A_v_B('Bz','Z','Z>8400','Z<10000','X==0','Y==75')
    #plot_maker.plot_mag_field(1,'Z==-4929')
    #plot_maker.plot_A_v_B_and_C('Bz','X','Z',True,300, 'Y==0','Z>-5000','Z<-4000','X>500')
    #plot_maker.plot_A_v_B_and_C('Bz','X','Z',False,0, 'Y==0')
    #plot_maker.plot_A_v_B_and_C('Bz','X','Z',True,300, 'Y==0','Z>-6200','Z<-5700','X>500','X<1000')
    #plot_maker.plot_A_v_B_and_C('Bz','X','Z',False,0, 'Y==0','Z>-6200','Z<-5700','X>500','X<1000')
    #data_frame, data_frame_interp,data_frame_grid = plot_maker.plot_Br_v_Theta(201.556444,'Z==-4929',300)
    #plot_maker.plot_A_v_Theta('Bz',500,'Z==-4929',300,'cubic')
    #plot_maker.plot_A_v_Theta('Br',150,'Z==-6179',300,'cubic')
    #plot_maker.plot_A_v_B_and_C('Br','X','Y',False,0, 'Z==-6179')
    #plot_maker.plot_mag_field2('Y','Z',1.3,'X==0','Z>14000')

    #plot_maker.plot_A_v_B_and_C('Bphi','Y','Z',False,0,'X==0','R<651','Z>4000','Z<14000')
    #plot_maker.plot_A_v_B_and_C('Bphi','Y','X',False,0,'Z==9171','-700<X<700','-700<Y<700')
    #plot_maker.plot_A_v_Theta('Bphi',500,'Z==9171',300,'cubic')
    #plot_maker_ps.plot_mag_field2('Y','Z',1.3,'X==0','Z<-8500')
    #plot_maker.plot_symmetry('Y','Z',False,0,'X==0','-600<Y<600','4000<Z<13000')
    #plot_maker.plot_A_v_B_and_C('Br','Y','Z',False,0,'X==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_symmetry('X','Z',False,0,'Y==0','-600<X<600','5000<Z<13000')
    #plot_maker.plot_symmetry('X','Z',False,0,'Y==100','-600<X<600','5000<Z<13000')
    #plot_maker.plot_symmetry('X','Z',False,0,'Y==200','-600<X<600','5000<Z<13000')
    #plot_maker.plot_symmetry('X','Z',False,0,'Y==300','-600<X<600','5000<Z<13000')
    #plot_maker.plot_symmetry('X','Z',False,0,'Y==400','-600<X<600','5000<Z<13000')
    #plot_maker.plot_symmetry('X','Z',False,0,'Y==500','-600<X<600','5000<Z<13000')
    #plot_maker.plot_A_v_B_and_C_plotly('Bz','Y','Z',False,0,'X==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_A_v_B_and_C_plotly('Br','Y','Z',False,0,'X==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_A_v_B_and_C_plotly('Bphi','Y','Z',False,0,'X==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_A_v_B_and_C_plotly('Bx','Y','Z',False,0,'X==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_A_v_B_and_C_ratio_plotly('Bz','Y','Z','X==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_A_v_B_and_C_ratio_plotly('Bx','Y','Z','X==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_A_v_B_and_C_ratio_plotly('By','Y','Z','X==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_symmetry('X','Z',False,0,'Y==0','R<651','5000<Z<13000')
    #plot_maker.plot_A_v_B_and_C_plotly('Bx','Y','Z',False,0,'X==0','R<651','Z>5000','Z<13000')
    #fig, df_int = plot_maker.plot_A_v_B_and_C('Br','X','Y',True,1000,'Z==9021','-501<X<501','-501<Y<501')
    plot_maker.plot_A_v_B_and_C_plotly_v2('Br','X','Y',True,300,'Z==9521','-301<X<301','-301<Y<301')
    plot_maker.plot_A_v_B_and_C_plotly('Br','X','Y',True,300,'Z==9521','-301<X<301','-301<Y<301')
    #scalar_field_plotter(plot_maker,'Z>5000','Z<13000','X>-650','X<650','Y>-650','Y<650')
    #plot_maker.plot_A_v_B_and_C('Bx','Y','Z',False,0,'X==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_A_v_B_and_C('By','Y','Z',False,0,'X==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_A_v_B_and_C('Bx','X','Z',False,0,'Y==0','R<651','Z>5000','Z<13000')
    #plot_maker.plot_A_v_B_and_C('By','X','Z',False,0,'Y==0','R<651','Z>5000','Z<13000')
    plt.show()


#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import cPickle as pkl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14, 10
from mu2e.tools.fiteval import get_mag_field_function
from mu2e.src.fiteval_c import FitFunctionMaker
import mu2e.tools.particletransport as patr
from time import time
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

del Axes3D

def interp(df,row_index):
    anchor = df.ix[row_index][np.isfinite(df.ix[row_index])].keys()[0]
    step_size = ((df.ix[row_index][anchor]-df.ix[row_index-1][anchor])/
            (df.ix[row_index+1][anchor]-df.ix[row_index-1][anchor]))
    nan_cols = df.ix[row_index][np.isnan(df.ix[row_index])].keys()
    for col in nan_cols:
        new_val = df.ix[row_index-1][col]+step_size*(df.ix[row_index+1][col]-df.ix[row_index-1][col])
        df.set_value(row_index,col,new_val)

def get_close_rows(df, val):
    min_row = df['Z'] <= val
    max_row = df['Z'] >= val
    idx_Min = df.ix[min_row, 'A'].idxmax()
    idx_Max = df.ix[max_row, 'A'].idxmin()
    return df.ix[idx_Min:idx_Max].copy()

if __name__ == "__main__":
    use_pickle = True
    plt.close('all')

    mag_field_function_ideal = get_mag_field_function('Mau10_825mm_v1')
    ffm= FitFunctionMaker("../mu2e/src/param_825.csv")
    mag_field_function_ideal_c = ffm.mag_field_function
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
    print df.head()
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

    init_pos = np.array([0,0,6000]) #in mm
    init_mom = np.array([0,100,-20]) #in MeV
    t_steps = np.linspace(0,8e-8,2e4)
    solver_ideal = patr.ElectronSwimmer(init_mom, init_pos, mag_field_function_ideal_c, t_steps, 'Dop853')
    solver_bad_m = patr.ElectronSwimmer(init_mom, init_pos, mag_field_function_bad_m, t_steps, 'Dop853')
    solver_bad_p = patr.ElectronSwimmer(init_mom, init_pos, mag_field_function_bad_p, t_steps, 'Dop853')
    solver_bad_r = patr.ElectronSwimmer(init_mom, init_pos, mag_field_function_bad_r, t_steps, 'Dop853')

    if not use_pickle:

#############################
# Now lets swim an electron #
#############################

        start_time=time()
        X,t = solver_ideal.solve()
        X_bm= solver_bad_m.solve()[0]
        X_bp= solver_bad_p.solve()[0]
        X_br= solver_bad_r.solve()[0]
        end_time=time()
        print("Elapsed time was %g seconds" % (end_time - start_time))


        print 'ideal xyz       :',X[-1,0:3], 'ideal-{this}:', X[-1,0:3] - X[-1,0:3]
        print 'bad measure xyz :',X_bm[-1,0:3], 'ideal-{this}:', X[-1,0:3] - X_bm[-1,0:3]
        print 'bad position xyz:',X_bp[-1,0:3], 'ideal-{this}:', X[-1,0:3] - X_bp[-1,0:3]
        print 'bad rotation xyz:',X_br[-1,0:3], 'ideal-{this}:', X[-1,0:3] - X_br[-1,0:3]

        df_ideal = pd.DataFrame(X[:,0:3], columns=['X','Y','Z'])
        df_bm = pd.DataFrame(X_bm[:,0:3], columns=['X','Y','Z'])
        df_br = pd.DataFrame(X_br[:,0:3], columns=['X','Y','Z'])
        df_bp = pd.DataFrame(X_bp[:,0:3], columns=['X','Y','Z'])

        pkl.dump([df_ideal, df_bm, df_br, df_bp], open('electrons.p', 'wb'), protocol = pkl.HIGHEST_PROTOCOL)

    else:
        df_ideal, df_bm, df_br, df_bp = pkl.load(open('electrons.p', 'rb'))

    ax.plot(df_ideal['Z'], df_ideal['X'], zs=df_ideal['Y'], linewidth=2, color='k', label='ideal path')
    ax.plot(df_bm['Z'], df_bm['X'], zs=df_bm['Y'], linewidth=2, linestyle='--', color = 'b', label='measurement syst')
    ax.plot(df_bp['Z'], df_bp['X'], zs=df_bp['Y'], linewidth=2, linestyle=':', color = 'g', label='position syst')
    ax.plot(df_br['Z'], df_br['X'], zs=df_br['Y'], linewidth=2, linestyle='-.', color = 'r', label='rotation syst')
    ax.set_title('Path of electron through magnetic field')


# these are matplotlib.patch.Patch properties
    textstr = 'init pos={0}\ninit mom={1} ({2:.3} MeV)\nB={3}'.format(init_pos, init_mom, solver_ideal.get_init_E(), 'ideal DS field map')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
    ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax.legend()
    plt.show()
    plt.savefig('../plots/anim/electron_path_DS.pdf')

    zs_trk = np.linspace(8410,8410+3051, 20)
    df_dict = {'df_ideal':df_ideal, 'df_bm':df_bm, 'df_br': df_br, 'df_bp':df_bp}
    df_tmp = pd.DataFrame({'X':[np.nan]*len(zs_trk), 'Y':[np.nan]*len(zs_trk), 'Z':zs_trk})
    for df_name in df_dict:
        df_dict[df_name] = df_dict[df_name].append(df_tmp)
        df_field = df_dict[df_name]
        df_field.sort_values('Z', inplace=True)
        df_field.reset_index(inplace=True, drop=True)
        for z in zs_trk:
            row_index = df_field[np.isclose(df_field.Z, z)].index.values[0]
            interp(df_field, row_index)

    df_ideal = df_dict['df_ideal']
    df_bm=df_dict['df_bm']
    df_br=df_dict['df_br']
    df_bp = df_dict['df_bp']
    for i,z in enumerate(zs_trk):
        x = df_ideal[np.isclose(df_ideal.Z ,z)].X.values[0]
        y = df_ideal[np.isclose(df_ideal.Z ,z)].Y.values[0]
        xbm = df_bm[np.isclose(df_bm.Z ,z)].X.values[0]
        ybm = df_bm[np.isclose(df_bm.Z ,z)].Y.values[0]
        xbr = df_br[np.isclose(df_br.Z ,z)].X.values[0]
        ybr = df_br[np.isclose(df_br.Z ,z)].Y.values[0]
        xbp = df_bp[np.isclose(df_bp.Z ,z)].X.values[0]
        ybp = df_bp[np.isclose(df_bp.Z ,z)].Y.values[0]
        plt.figure()
        p = plt.plot(x,y,'ok',label='ideal pos')
        plt.title('Electron XY displacement for Z={}'.format(z))
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        ax =plt.gca()
        ax.set_ylim(y-10,y+10)
        ax.set_xlim(x-10,x+10)
        a1 = ax.arrow(x,y,x-xbm,y-ybm,head_width=0.4, head_length=1, linewidth=2, length_includes_head=True, fc='g', ec='g',label='meas syst')
        a2 = ax.arrow(x,y,x-xbr,y-ybr,head_width=0.4, head_length=1, linewidth=2, length_includes_head=True, fc='b', ec='b')
        a3 = ax.arrow(x,y,x-xbp,y-ybp,head_width=0.4, head_length=1, linewidth=2, length_includes_head=True, fc='r', ec='r')
        plt.legend([p[0],a1,a2,a3], ['ideal position', 'meas syst', 'rot syst', 'pos syst'],
                handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow),})
        plt.savefig('../plots/anim/electron_path_displacements_{0}.png'.format(i))
        print np.sqrt((x-xbm)**2+(y-ybm)**2),  np.sqrt((x-xbr)**2+(y-ybr)**2), np.sqrt((x-xbp)**2+(y-ybp)**2)
    plt.show()


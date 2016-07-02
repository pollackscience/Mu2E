#! /usr/bin/env python

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.tools as tls
import plotly.graph_objs as go


# Definitions of mu2e-specific plotting functions.
# Plots can be displayed in matplotlib, plotly (offline mode), or plotly (html only)
# Plot functions all take a dataframe input

def mu2e_plot(df, x, y, conditions = None, mode = 'mpl', info = None, savename = None):
    '''Currently plotly can convert simple mpl plots directly into plotly.
    Therefor, generate the mpl first, then convert to plotly if necessary.'''

    _modes = ['mpl', 'plotly', 'plotly_html', 'plotly_nb']

    if conditions:
        df = df.query(conditions)

    if mode not in _modes:
        raise ValueError(mode+' not in '+_modes)

    ax = df.plot(x, y, kind='line')
    ax.grid(True)
    plt.ylabel(y)
    plt.title(' '.join(filter(lambda x:x, [info, x, 'v', y, conditions])))
    if mode == 'mpl':
        plt.legend()

    elif 'plotly' in mode:
        fig = ax.get_figure()
        py_fig = tls.mpl_to_plotly(fig)
        py_fig['layout']['showlegend'] = True

        if mode == 'plotly_nb':
            init_notebook_mode()
            iplot(py_fig)
        elif mode == 'plotly_html':
            plot(py_fig)

    if savename:
        plt.savefig(savename)

def mu2e_plot3d(df, x, y, z, conditions = None, mode = 'mpl', info = None, savename = None, df_fit = None):
    from mpl_toolkits.mplot3d import Axes3D
    del Axes3D
    '''Currently, plotly cannot convert 3D mpl plots directly into plotly (without a lot of work).
    For now, the mpl and plotly generation are seperate (may converge in future if necessary).'''

    _modes = ['mpl', 'plotly', 'plotly_html', 'plotly_nb']

    if conditions:

        # Special treatment to detect cylindrical coordinates:
        # Some regex matching to find any 'Phi' condition and treat it as (Phi & Phi-Pi).
        # This is done because when specifying R-Z coordinates, we almost always want the
        # 2-D plane that corresponds to that (Phi & Phi-Pi) pair.  In order to plot this,
        # we assign a value of -R to all R points that correspond to the 'Phi-Pi' half-plane

        p = re.compile(r'(?:and)*\s*Phi\s*==\s*([-+]?(?:[0-9]*\.[0-9]+|[0-9]+))')
        phi_str = p.search(conditions)
        conditions_nophi = p.sub('', conditions)
        conditions_nophi = re.sub(r'^\s*and\s*','',conditions_nophi)
        try:
            phi = float(phi_str.group(1))
        except:
            phi = None

        df = df.query(conditions_nophi)
        if phi != None:
            isc = np.isclose
            if isc(phi,0):
                nphi = np.pi
            else:
                nphi = phi-np.pi
            df = df[(isc(phi,df.Phi)) | (isc(nphi,df.Phi))]
            df.ix[isc(nphi,df.Phi), 'R']*=-1

    if mode not in _modes:
        raise ValueError(mode+' not in '+_modes)

    piv = df.pivot(x, y, z)
    X=piv.index.values
    Y=piv.columns.values
    Z=np.transpose(piv.values)
    Xi,Yi = np.meshgrid(X, Y)
    if df_fit:
        piv_fit = df.pivot(x, y, z+'_fit')
        Z_fit=np.transpose(piv_fit.values)

    if mode == 'mpl':
        fig = plt.figure().gca(projection='3d')

        if df_fit:
            fig.plot(Xi.ravel(), Yi.ravel(), Z.ravel(), 'ko',markersize=2 )
            fig.plot_wireframe(Xi, Yi, Z_fit,color='green')
        else:
            fig.plot_surface(Xi, Yi, Z, rstride=1, cstride=1, cmap=plt.get_cmap('viridis'), linewidth=0, antialiased=False)
        fig.zaxis.set_major_locator(LinearLocator(10))
        fig.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        plt.xlabel(x)
        plt.ylabel(y)
        fig.set_zlabel(z)
        fig.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        fig.zaxis.labelpad=20
        fig.zaxis.set_tick_params(direction='out',pad=10)
        plt.title(' '.join(filter(lambda x:x, [info, x, y, 'v', z, conditions])))
        plt.title('{0} vs {1} and {2}, {3}'.format(z,x,y,conditions[0]))
        fig.view_init(elev=35., azim=30)

    elif 'plotly' in mode:
        layout = go.Layout(
                        title='Plot of {0} vs {1} and {2} for DS, {3}'.format(z,x,y,conditions),
                        autosize=False,
                        width=675,
                        height=650,
                        scene=dict(
                                xaxis=dict(
                                        title='{} (mm)'.format(x),
                                        gridcolor='rgb(255, 255, 255)',
                                        zerolinecolor='rgb(255, 255, 255)',
                                        showbackground=True,
                                        backgroundcolor='rgb(230, 230,230)'
                                        ),
                                yaxis=dict(
                                        title='{} (mm)'.format(y),
                                        gridcolor='rgb(255, 255, 255)',
                                        zerolinecolor='rgb(255, 255, 255)',
                                        showbackground=True,
                                        backgroundcolor='rgb(230, 230,230)'
                                        ),
                                zaxis=dict(
                                        title='{} (T)'.format(z),
                                        gridcolor='rgb(255, 255, 255)',
                                        zerolinecolor='rgb(255, 255, 255)',
                                        showbackground=True,
                                        backgroundcolor='rgb(230, 230,230)'
                                        ),
                                cameraposition=[[-0.1, 0.5, -0.7, -0.2], [0.0, 0, 0.0], 2.8]
                                ),
                        showlegend=True,
                        )
        surface = go.Surface(x=Xi, y=Yi, z=Z, colorbar = go.ColorBar(title='Tesla',titleside='right'), colorscale = 'Viridis')
        data = [surface]
        fig = go.Figure(data=data, layout=layout)

        if mode == 'plotly_nb':
            init_notebook_mode()
            iplot(fig)
        elif mode == 'plotly_html':
            plot(fig)



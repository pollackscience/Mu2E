#! /usr/bin/env python
"""Module for generating Mu2e plots.

This module defines a host of useful functions for generating plots for various Mu2E datasets.  It
is expected that these datasets be in the form of :class:`pandas.DataFrame` objects, and the
independent and dependent variables are strings that correspond to DF columns.  These functions wrap
calls to either :mod:`matplotlib` or :mod:`plotly`.  The latter has numerous modes, depending on how
the user wants to share the plots.


Example:
    Plotting in a `jupyter notebook`:

    .. code-block:: python

        In [1]: import os
        ...     from mu2e import mu2e_ext_path
        ...     from mu2e.dataframeprod import DataFrameMaker
        ...     from mu2e.mu2eplots import mu2e_plot, mu2e_plot3d
        ...     %matplotlib inline # for embedded mpl plots
        ...     plt.rcParams['figure.figsize'] = (12,8) # make them bigger

        In [2]: df = DataFrameMaker(
        ...         mu2e_ext_path + 'path/to/Mu2e_DSMap',
        ...         use_pickle = True
        ...     ).data_frame

        In [3]: mu2e_plot(df, 'Z', 'Bz', 'Y==0 and X==0', mode='mpl')
        ...     #static image of 2D matplotlib plot

        In [4]: mu2e_plot3d(
        ...         df, 'R', 'Z', 'Bz',
        ...         'R<=800 and Phi==0 and 12700<Z<12900',
        ...         mode='plotly_nb')
        ...     #interactive 3D plot in plotly for jupyter nb



Todo:
    * Allow for \*\*kwargs inheriting from mpl, map them for plotly as well.
    * Build in more flexibility for particle trapping plots.
    * Particle animation GraphWidget will not work with updated jupyter, keep an eye on dev.


*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
from mpldatacursor import datacursor
import ipywidgets as widgets
from IPython.display import display
from plotly.widgets import GraphWidget
from offline import init_notebook_mode, iplot, plot
from mu2e import mu2e_ext_path


def mu2e_plot(df, x, y, conditions=None, mode='mpl', info=None, savename=None, ax=None):
    """Generate 2D plots, x vs y.

    Generate a 2D plot for a given DF and two columns. An optional selection string is applied to
    the data via the :func:`pandas.DataFrame.query` interface, and is added to the title.  This
    function supports matplotlib and various plotly plotting modes.

    Args:
        df (pandas.DataFrame): The input DF, must contain columns with labels corresponding to the
            'x' and 'y' args.
        x (str): Name of the independent variable.
        y (str): Name of the dependent variable.
        conditions (str, optional): A string adhering to the :mod:`numexpr` syntax used in
            :func:`pandas.DataFrame.query`.
        mode (str, optional): A string indicating which plotting package and method should be used.
            Default is 'mpl'. Valid values: ['mpl', 'plotly', 'plotly_html', 'plotly_nb']
        info (str, optional): Extra information to add to the legend.
        savename (str, optional): If not `None`, the plot will be saved to the indicated path and
            file name.
        ax (matplotlib.axis, optional): Use existing mpl axis object.

    Returns:
        axis object if 'mpl', else `None`.
    """

    _modes = ['mpl', 'plotly', 'plotly_html', 'plotly_nb']

    if conditions:
        df = df.query(conditions)

    if mode not in _modes:
        raise ValueError(mode+' not in '+_modes)

    leg_label = y+' '+info if info else y
    ax = df.plot(x, y, ax=ax, kind='line', label=leg_label, legend=True, linewidth=2)
    ax.grid(True)
    plt.ylabel(y)
    plt.title(' '.join(filter(lambda x: x, [x, 'v', y, conditions])))

    if 'plotly' in mode:
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
    if mode == 'mpl':
        return ax
    else:
        return None


def mu2e_plot3d(df, x, y, z, conditions=None, mode='mpl', info=None, save_dir=None, df_fit=None,
                ptype='3D'):
    """Generate 3D plots, x and y vs z.

    Generate a 3D surface plot for a given DF and three columns. An optional selection string is
    applied to the data via the :func:`pandas.DataFrame.query` interface, and is added to the title.
    Extra processing is done to plot cylindrical coordinates if that is detected from `conditions`.
    This function supports matplotlib and various plotly plotting modes.  If `df_fit` is specified,
    `df` is plotted as a a scatter plot, and `df_fit` as a wireframe plot.

    Args:
        df (pandas.DataFrame): The input DF, must contain columns with labels corresponding to the
            'x', 'y', and 'z' args.
        x (str): Name of the first independent variable.
        y (str): Name of the second independent variable.
        z (str): Name of the dependent variable.
        conditions (str, optional): A string adhering to the :mod:`numexpr` syntax used in
            :func:`pandas.DataFrame.query`.
        mode (str, optional): A string indicating which plotting package and method should be used.
            Default is 'mpl'. Valid values: ['mpl', 'plotly', 'plotly_html', 'plotly_nb']
        info (str, optional): Extra information to add to the title.
        save_dir (str, optional): If not `None`, the plot will be saved to the indicated path. The
            file name is automated, based on the input args.
        df_fit (bool, optional): If the input df contains columns with the suffix '_fit', plot a
            scatter plot using the normal columns, and overlay a wireframe plot using the '_fit'
            columns.  Also generate a heatmap showing the residual difference between the two plots.
        ptype (str, optional): Choose between '3d' and 'heat'.  Default is '3d'.

    Returns:
        Name of saved image/plot, or None.
    """

    _modes = ['mpl', 'plotly', 'plotly_html', 'plotly_nb']
    save_name = None

    if conditions:

        # Special treatment to detect cylindrical coordinates:
        # Some regex matching to find any 'Phi' condition and treat it as (Phi & Phi-Pi).
        # This is done because when specifying R-Z coordinates, we almost always want the
        # 2-D plane that corresponds to that (Phi & Phi-Pi) pair.  In order to plot this,
        # we assign a value of -R to all R points that correspond to the 'Phi-Pi' half-plane

        p = re.compile(r'(?:and)*\s*Phi\s*==\s*([-+]?(?:[0-9]*\.[0-9]+|[0-9]+))')
        phi_str = p.search(conditions)
        conditions_nophi = p.sub('', conditions)
        conditions_nophi = re.sub(r'^\s*and\s*', '', conditions_nophi)
        conditions_nophi = conditions_nophi.strip()
        try:
            phi = float(phi_str.group(1))
        except:
            phi = None

        df = df.query(conditions_nophi)

        # Make radii negative for negative phi values (for plotting purposes)
        if phi is not None:
            isc = np.isclose
            if isc(phi, 0):
                nphi = np.pi
            else:
                nphi = phi-np.pi
            df = df[(isc(phi, df.Phi)) | (isc(nphi, df.Phi))]
            df.ix[isc(nphi, df.Phi), 'R'] *= -1

        conditions_title = conditions_nophi.replace(' and ', ', ')
        conditions_title = conditions_title.replace('R!=0', '')
        conditions_title = conditions_title.strip()
        conditions_title = conditions_title.strip(',')
        if phi is not None:
            conditions_title += ', Phi=={0:.2f}'.format(phi)

    if mode not in _modes:
        raise ValueError(mode+' not in '+_modes)

    # Format the coordinates
    piv = df.pivot(x, y, z)
    X = piv.index.values
    Y = piv.columns.values
    Z = np.transpose(piv.values)
    Xi, Yi = np.meshgrid(X, Y)
    if df_fit:
        piv_fit = df.pivot(x, y, z+'_fit')
        Z_fit = np.transpose(piv_fit.values)
        data_fit_diff = (Z - Z_fit)*10000
        Xa = np.concatenate(([X[0]], 0.5*(X[1:]+X[:-1]), [X[-1]]))
        Ya = np.concatenate(([Y[0]], 0.5*(Y[1:]+Y[:-1]), [Y[-1]]))

    # Prep save area
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = '{0}_{1}{2}_{3}'.format(
            z, x, y, '_'.join([i for i in conditions_title.split(', ') if i != 'and']))
        save_name = re.sub(r'[<>=!\s]', '', save_name)

        if df_fit:
            save_name += '_fit'

    # Start plotting
    if mode == 'mpl':
        if ptype.lower() == '3d':
            fig = plt.figure().gca(projection='3d')
        else:
            fig = plt.figure()

        if df_fit:
            fig.plot(Xi.ravel(), Yi.ravel(), Z.ravel(), 'ko', markersize=2)
            fig.plot_wireframe(Xi, Yi, Z_fit, color='green')
        elif ptype.lower() == '3d':
            fig.plot_surface(Xi, Yi, Z, rstride=1, cstride=1, cmap=plt.get_cmap('viridis'),
                             linewidth=0, antialiased=False)
        elif ptype.lower() == 'heat':
            plt.pcolormesh(Xi, Yi, Z, cmap=plt.get_cmap('viridis'))
        else:
            raise KeyError(ptype+' is an invalid type!, must be "heat" or "3D"')

        plt.xlabel(x+' (mm)', fontsize=18)
        plt.ylabel(y+' (mm)', fontsize=18)
        if ptype.lower() == '3d':
            fig.set_zlabel(z+' (T)', fontsize=18)
            fig.ticklabel_format(style='sci', axis='z')
            fig.zaxis.labelpad = 20
            fig.zaxis.set_tick_params(direction='out', pad=10)
            fig.xaxis.labelpad = 20
            fig.yaxis.labelpad = 20
        else:
            cb = plt.colorbar()
            cb.set_label(z+' (T)', fontsize=18)
        if info is not None:
            plt.title('{0} {1} vs {2} and {3} for DS\n{4}'.format(info, z, x, y, conditions_title),
                      fontsize=20)
        else:
            plt.title('{0} vs {1} and {2} for DS\n{3}'.format(z, x, y, conditions_title),
                      fontsize=20)
        if ptype.lower() == '3d':
            fig.view_init(elev=35., azim=30)
        if save_dir:
            plt.savefig(save_dir+'/'+save_name+'.png')

        if df_fit:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            max_val = np.max(data_fit_diff)
            min_val = np.min(data_fit_diff)
            abs_max_val = max(abs(max_val), abs(min_val))
            if (abs_max_val) > 5:
                heat = ax2.pcolormesh(Xa, Ya, data_fit_diff, vmin=-5, vmax=5,
                                      cmap=plt.get_cmap('viridis'))
                cb = plt.colorbar(heat, aspect=7)
                cb_ticks = cb.ax.get_yticklabels()
                cb_ticks[0] = '< -5'
                cb_ticks[-1] = '> 5'
                cb_ticks = cb.ax.set_yticklabels(cb_ticks)
            else:
                heat = ax2.pcolormesh(Xa, Ya, data_fit_diff, cmap=plt.get_cmap('viridis'),
                                      vmin=-abs_max_val, vmax=abs_max_val)
                cb = plt.colorbar(heat, aspect=7)
            plt.title('{0} vs {1} and {2} for DS\n{3}'.format(z, x, y, conditions_title),
                      fontsize=20)
            cb.set_label('Data-Fit (G)', fontsize=18)
            ax2.set_xlabel(x+' (mm)', fontsize=18)
            ax2.set_ylabel(y+' (mm)', fontsize=18)
            datacursor(heat, hover=True, bbox=dict(alpha=1, fc='w'))
            if save_dir:
                plt.savefig(save_dir+'/'+save_name+'_heat.pdf')

    elif 'plotly' in mode:
        axis_title_size = 18
        axis_tick_size = 14
        if info is not None:
            title = '{0} {1} vs {2} and {3} for DS<br>{4}'.format(info, z, x, y, conditions_title)
        else:
            title = '{0} vs {1} and {2} for DS<br>{3}'.format(z, x, y, conditions_title)

        if ptype == 'heat':
            layout = go.Layout(
                title=title,
                # ticksuffix is a workaround to add a bit of padding
                width=800,
                height=650,
                autosize=False,
                xaxis=dict(
                    title='{} (mm)'.format(x),
                    titlefont=dict(size=axis_title_size, family='Arial Black'),
                    tickfont=dict(size=axis_tick_size),
                ),
                yaxis=dict(
                    title='{} (mm)'.format(y),
                    titlefont=dict(size=axis_title_size, family='Arial Black'),
                    tickfont=dict(size=axis_tick_size),
                ),
            )

        elif ptype == '3d':
            layout = go.Layout(
                title=title,
                titlefont=dict(size=30),
                autosize=False,
                width=800,
                height=650,
                scene=dict(
                    xaxis=dict(
                        title='{} (mm)'.format(x),
                        titlefont=dict(size=axis_title_size, family='Arial Black'),
                        tickfont=dict(size=axis_tick_size),
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)',
                    ),
                    yaxis=dict(
                        title='{} (mm)'.format(y),
                        titlefont=dict(size=axis_title_size, family='Arial Black'),
                        tickfont=dict(size=axis_tick_size),
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)',
                    ),
                    zaxis=dict(
                        title='{} (T)'.format(z),
                        titlefont=dict(size=axis_title_size, family='Arial Black'),
                        tickfont=dict(size=axis_tick_size),
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)',
                    ),
                ),
                showlegend=True,
                legend=dict(x=0.8, y=0.9, font=dict(size=18, family='Overpass')),
            )
        if df_fit:
            scat = go.Scatter3d(
                x=Xi.ravel(), y=Yi.ravel(), z=Z.ravel(), mode='markers',
                marker=dict(size=3, color='rgb(0, 0, 0)',
                            line=dict(color='rgb(0, 0, 0)'), opacity=1),
                name='Data')
            lines = [scat]
            line_marker = dict(color='green', width=2)
            do_leg = True
            for i, j, k in zip(Xi, Yi, Z_fit):
                if do_leg:
                    lines.append(
                        go.Scatter3d(x=i, y=j, z=k, mode='lines',
                                     line=line_marker, name='Fit',
                                     legendgroup='fitgroup')
                    )
                else:
                    lines.append(
                        go.Scatter3d(x=i, y=j, z=k, mode='lines',
                                     line=line_marker, name='Fit',
                                     legendgroup='fitgroup', showlegend=False)
                    )
                do_leg = False

            # For hovertext
            z_offset = (np.min(Z)-abs(np.min(Z)*0.3))*np.ones(Z.shape)
            textz = [['x: '+'{:0.5f}'.format(Xi[i][j])+'<br>y: '+'{:0.5f}'.format(Yi[i][j]) +
                      '<br>z: ' + '{:0.5f}'.format(data_fit_diff[i][j]) for j in
                      range(data_fit_diff.shape[1])] for i in range(data_fit_diff.shape[0])]
            # For heatmap

            # projection in the z-direction
            def proj_z(x, y, z):
                return z
            colorsurfz = proj_z(Xi, Yi, data_fit_diff)
            tracez = go.Surface(
                z=z_offset, x=Xi, y=Yi, colorscale='Viridis', zmin=-2, zmax=2, name='residual',
                showlegend=True, showscale=True, surfacecolor=colorsurfz, text=textz,
                hoverinfo='text',
                colorbar=dict(
                    title='Data-Fit (G)',
                    titlefont=dict(size=18),
                    tickfont=dict(size=20),
                    xanchor='center'),
            )
            lines.append(tracez)

        else:
            if ptype == '3d':
                surface = go.Surface(
                    x=Xi, y=Yi, z=Z,
                    colorbar=go.ColorBar(title='Tesla', titleside='right'),
                    colorscale='Viridis')
                lines = [surface]
            elif ptype == 'heat':
                heat = go.Heatmap(x=X, y=Y, z=Z,
                                  colorbar=go.ColorBar(
                                      title='{0} (T)'.format(z),
                                      titleside='top',
                                      titlefont=dict(size=18),
                                      tickfont=dict(size=20),
                                  ),
                                  colorscale='Viridis', showscale=True)
                lines = [heat]

        fig = go.Figure(data=lines, layout=layout)

        # Generate Plot
        if mode == 'plotly_nb':
            init_notebook_mode()
            iplot(fig)
        elif mode == 'plotly_html_img':
            if save_dir:
                plot(fig, filename=save_dir+'/'+save_name+'.html', image='jpeg',
                     image_filename=save_name)
            else:
                plot(fig)
        elif mode == 'plotly_html':
            if save_dir:
                plot(fig, filename=save_dir+'/'+save_name+'.html', auto_open=False)
            else:
                plot(fig)

    return save_name


def mu2e_plot3d_ptrap(df, x, y, z, mode='plotly_nb', save_name=None, color=None,
                      df_xray=None, x_range=None, y_range=None, z_range=None, title=None):
    """Generate 3D scatter plots, typically for visualizing 3D positions of charged particles.

    Generate a 3D scatter plot for a given DF and three columns. Due to the large number of points
    that are typically plotted, :mod:`matplotlib` is not supported.

    Args:
        df (pandas.DataFrame): The input DF, must contain columns with labels corresponding to the
            'x', 'y', and 'z' args.
        x (str): Name of the first variable.
        y (str): Name of the second variable.
        z (str): Name of the third variable.
        mode (str, optional): A string indicating which plotting package and method should be used.
            Default is 'mpl'. Valid values: ['mpl', 'plotly', 'plotly_html', 'plotly_nb']
        save_name (str, optional): If not `None`, the plot will be saved to
            `mu2e_ext_path+ptrap/save_name.html` (or `.jpeg`)
        color: (str, optional): Name of fourth varible, represented by color of marker.
        df_xray: (:class:`pandas.DataFrame`, optional): A seperate DF, representing the geometry of
            the material that is typically included during particle simulation.

    Returns:
        Name of saved image/plot, or None.

    Notes:
        Growing necessity for many input args, should implement `kwargs` in future.
    """
    _modes = ['plotly', 'plotly_html', 'plotly_html_img', 'plotly_nb']

    if mode not in _modes:
        raise ValueError(mode+' not in '+_modes)

    if mode == 'mpl':
        pass

    elif 'plotly' in mode:
        axis_title_size = 18
        axis_tick_size = 14
        layout = go.Layout(
            title=title if title else 'Particle Trapping Exercise',
            titlefont=dict(size=30),
            autosize=False,
            width=900,
            height=650,
            scene=dict(
                xaxis=dict(
                    title='{} (mm)'.format(x),
                    titlefont=dict(size=axis_title_size, family='Arial Black'),
                    tickfont=dict(size=axis_tick_size),
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    range=x_range,
                ),
                yaxis=dict(
                    title='{} (mm)'.format(y),
                    titlefont=dict(size=axis_title_size, family='Arial Black'),
                    tickfont=dict(size=axis_tick_size),
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    range=y_range,
                ),
                zaxis=dict(
                    title='{} (mm)'.format(z),
                    titlefont=dict(size=axis_title_size, family='Arial Black'),
                    tickfont=dict(size=axis_tick_size),
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    range=z_range,
                ),
                aspectmode='manual' if (x_range or y_range or z_range) else 'data',
                aspectratio=dict(x=6, y=1, z=1) if (x_range or y_range or z_range) else dict(),
                camera=dict(
                    eye=dict(x=1.99, y=-2, z=2)
                ),
            ),
            showlegend=True,
            legend=dict(x=0.8, y=0.9, font=dict(size=18, family='Overpass')),
        )
        scat_plots = []

        # Set the xray image if available
        if isinstance(df_xray, pd.DataFrame):
            print 'binning...'
            xray_query = 'xstop<1000 and tstop< 200 and sqrt(xstop*xstop+ystop*ystop)<900'
            df_xray.query(xray_query, inplace=True)

            h, edges = np.histogramdd(np.asarray([df_xray.zstop, df_xray.xstop, df_xray.ystop]).T,
                                      bins=(100, 25, 25))
            centers = [(e[1:]+e[:-1])/2.0 for e in edges]
            hadj = np.rot90(h).flatten()/h.max()
            cx, cy, cz = np.meshgrid(*centers)
            df_binned = pd.DataFrame(dict(x=cx.flatten(), y=cy.flatten(), z=cz.flatten(), h=hadj))
            df_binned = df_binned.query('h>0')
            df_binned['cats'] = pd.cut(df_binned.h, 200)
            print 'binned'
            groups = np.sort(df_binned.cats.unique())
            for i, group in enumerate(groups[0:]):
                df_tmp = df_binned[df_binned.cats == group]
                if i == 0:
                    sl = True
                else:
                    sl = False
                scat_plots.append(
                    go.Scatter3d(
                        x=df_tmp.x, y=df_tmp.y, z=df_tmp.z, mode='markers',
                        marker=dict(sizemode='diameter', size=4,
                                    color='black', opacity=0.03*(i+1), symbol='square'),
                        name='X-ray', showlegend=sl, legendgroup='xray'))

            df_newmat = df_binned.query('12800<x<13000 and sqrt(y**2+z**2)<800')
            if len(df_newmat) > 0:
                scat_plots.append(
                    go.Scatter3d(
                        x=df_newmat.x, y=df_newmat.y, z=df_newmat.z, mode='markers',
                        marker=dict(sizemode='diameter', size=6,
                                    color='red', opacity=0.4, symbol='square'),
                        name='X-ray', showlegend=False, legendgroup='xray'))

        if color:
            scat_plots.append(
                go.Scatter3d(
                    x=df[x], y=df[y], z=df[z], mode='markers',
                    marker=dict(size=4, color=df[color], colorscale='Viridis', opacity=1,
                                line=dict(color='black', width=1),
                                showscale=True, cmin=0, cmax=110,
                                colorbar=dict(title='Momentum (MeV)')),
                    text=df[color].astype(int),
                    name='Muons'))

        # If there is an xray, treat this as main
        elif isinstance(df_xray, pd.DataFrame):
            scat_plots.append(
                go.Scatter3d(
                    x=df[x], y=df[y], z=df[z], mode='markers',
                    marker=dict(size=3, color='blue', opacity=0.5),
                    name='Muons'))
        # If there's no xray, then treat this as an xray
        else:
            # This is a complicated method of generating the x-ray images
            #   * The xray hits are binned in 3D.
            #   * A new df is generated, based on the bin centers and occupancy of each bin.
            #   * The df is binned into categories, based on occupancy.
            #   * Each category is plotted, with an opacity based on the occupancy
            #   * The result is a lego-style plot of the xray hits

            print 'binning...'
            h, edges = np.histogramdd(np.asarray([df[x], df[y], df[z]]).T, bins=(200, 20, 20))
            centers = [(e[1:]+e[:-1])/2.0 for e in edges]
            hadj = np.rot90(h).flatten()/h.max()
            cx, cy, cz = np.meshgrid(*centers)
            df_binned = pd.DataFrame(dict(x=cx.flatten(), y=cy.flatten(), z=cz.flatten(), h=hadj))

            df_binned['cats'] = pd.cut(df_binned.h, 200)
            print 'binned'
            groups = np.sort(df_binned.cats.unique())
            for i, group in enumerate(groups[1:]):
                df_tmp = df_binned[df_binned.cats == group]
                if i == 0:
                    sl = True
                else:
                    sl = False
                scat_plots.append(
                    go.Scatter3d(
                        x=df_tmp.x, y=df_tmp.y, z=df_tmp.z, mode='markers',
                        marker=dict(sizemode='diameter', size=4,
                                    color='black', opacity=0.03*(i+1), symbol='square'),
                        name='X-ray', showlegend=sl, legendgroup='xray'))

        fig = go.Figure(data=scat_plots, layout=layout)

        if mode == 'plotly_nb':
            init_notebook_mode()
            iplot(fig)
        elif mode == 'plotly_html_img':
            if save_name:
                plot(fig, filename=mu2e_ext_path+'ptrap/'+save_name+'.html', image='jpeg',
                     image_filename=save_name)
            else:
                plot(fig)
        elif mode == 'plotly_html':
            if save_name:
                plot(fig, filename=mu2e_ext_path+'ptrap/'+save_name+'.html', auto_open=False)
            else:
                plot(fig)
        elif mode == 'plotly':
            py.iplot(fig)

    return save_name


def mu2e_plot3d_ptrap_anim(df_group1, x, y, z, df_xray, df_group2=None, color=None, title=None):
    """Generate 3D scatter plots, with a slider widget typically for visualizing 3D positions of
    charged particles.

    Generate a 3D scatter plot for a given DF and three columns. Due to the large number of points
    that are typically plotted, :mod:`matplotlib` is not supported. A slider widget is generated,
    corresponding to the evolution in time.

    Notes:
        * Currently many options are hard-coded.
        * Use `g.plot(fig)` after executing this function.
    Args:
        df_group1 (pandas.DataFrame): The input DF, must contain columns with labels corresponding
        to the 'x', 'y', and 'z' args.
        x (str): Name of the first variable.
        y (str): Name of the second variable.
        z (str): Name of the third variable.
        df_xray: (:class:`pandas.DataFrame`): A seperate DF, representing the geometry of the
            material that is typically included during particle simulation.
        df_group2: (:class:`pandas.DataFrame`, optional): A seperate DF, representing the geometry
            of the material that is typically included during particle simulation.
        color (optional): Being implemented...
        title (str, optional): Title.

    Returns:
        (g, fig) for plotting.
    """
    init_notebook_mode()
    axis_title_size = 18
    axis_tick_size = 14
    layout = go.Layout(
        title=title if title else 'Particle Trapping Time Exercise',
        titlefont=dict(size=30),
        autosize=False,
        width=900,
        height=650,
        scene=dict(
            xaxis=dict(
                title='{} (mm)'.format(x),
                titlefont=dict(size=axis_title_size, family='Arial Black'),
                tickfont=dict(size=axis_tick_size),
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=[3700, 17500],
            ),
            yaxis=dict(
                title='{} (mm)'.format(y),
                titlefont=dict(size=axis_title_size, family='Arial Black'),
                tickfont=dict(size=axis_tick_size),
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=[-1000, 1000],
            ),
            zaxis=dict(
                title='{} (mm)'.format(z),
                titlefont=dict(size=axis_title_size, family='Arial Black'),
                tickfont=dict(size=axis_tick_size),
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=[-1000, 1000],
            ),
            aspectratio=dict(x=6, y=1, z=1),
            aspectmode='manual',
            camera=dict(
                eye=dict(x=1.99, y=-2, z=2)
            ),
        ),
        showlegend=True,
        legend=dict(x=0.7, y=0.9, font=dict(size=18, family='Overpass')),
    )

    class time_shifter:
        def __init__(self, group2=False, color=False):
            self.x = df_group1[df_group1.sid == sids[0]][x]
            self.y = df_group1[df_group1.sid == sids[0]][y]
            self.z = df_group1[df_group1.sid == sids[0]][z]
            self.group2 = group2
            self.docolor = color
            if self.docolor:
                self.color = df_group1[df_group1.sid == sids[0]].p

            if self.group2:
                self.x2 = df_group2[df_group2.sid == sids[0]][x]
                self.y2 = df_group2[df_group2.sid == sids[0]][y]
                self.z2 = df_group2[df_group2.sid == sids[0]][z]
                if self.docolor:
                    self.color2 = df_group2[df_group2.sid == sids[0]].p

        def on_time_change(self, name, old_value, new_value):
            try:
                self.x = df_group1[df_group1.sid == sids[new_value]][x]
                self.y = df_group1[df_group1.sid == sids[new_value]][y]
                self.z = df_group1[df_group1.sid == sids[new_value]][z]
                if self.docolor:
                    self.color = df_group1[df_group1.sid == sids[new_value]].p
            except:
                self.x = []
                self.y = []
                self.z = []
                if self.docolor:
                    self.color = []

            if self.group2:
                try:
                    self.x2 = df_group2[df_group2.sid == sids[new_value]][x]
                    self.y2 = df_group2[df_group2.sid == sids[new_value]][y]
                    self.z2 = df_group2[df_group2.sid == sids[new_value]][z]
                    if self.docolor:
                        self.color2 = df_group2[df_group2.sid == sids[new_value]].p
                except:
                    self.x2 = []
                    self.y2 = []
                    self.z2 = []
                    if self.docolor:
                        self.color2 = []
            self.replot()

        def replot(self):
            if self.docolor:
                g.restyle({'x': [self.x], 'y': [self.y], 'z': [self.z],
                           'marker': dict(size=5, color=self.color,
                                          opacity=1, colorscale='Viridis', cmin=0, cmax=100,
                                          showscale=True,
                                          line=dict(color='black', width=1),
                                          colorbar=dict(title='Momentum (MeV)', xanchor='left'))
                           },
                          indices=[0])
            else:
                g.restyle({'x': [self.x], 'y': [self.y], 'z': [self.z]}, indices=[0])
            if self.group2:
                if self.docolor:
                    g.restyle({'x': [self.x2], 'y': [self.y2], 'z': [self.z2],
                               'marker': dict(size=5, color=self.color2,
                                              opacity=1, colorscale='Viridis', cmin=0, cmax=100,
                                              showscale=True,
                                              line=dict(color='black', width=1),
                                              colorbar=dict(title='Momentum (MeV)',
                                                            xanchor='left')),
                               },
                              indices=[1])
                else:
                    g.restyle({'x': [self.x2], 'y': [self.y2], 'z': [self.z2]}, indices=[1])
    try:
        g1_name = df_group1.name
    except:
        g1_name = 'Group 1'

    group2 = False
    if isinstance(df_group2, pd.DataFrame):
        group2 = True
        try:
            g2_name = df_group2.name
        except:
            g2_name = 'Group 1'

    if group2:
        if len(df_group1.sid.unique()) > len(df_group2.sid.unique()):
            sids = np.sort(df_group1.sid.unique())
        else:
            sids = np.sort(df_group2.sid.unique())
    else:
        sids = np.sort(df_group1.sid.unique())

    scats = []
    if color:
        mdict = dict(size=5, color=df_group1[df_group1.sid == sids[0]].p, opacity=1,
                     colorscale='Viridis', cmin=0, cmax=100,
                     showscale=True,
                     line=dict(color='black', width=1),
                     colorbar=dict(title='Momentum (MeV)', xanchor='left'))
    else:
        mdict = dict(size=5, color='red', opacity=0.7)

    init_scat = go.Scatter3d(
        x=df_group1[df_group1.sid == sids[0]][x],
        y=df_group1[df_group1.sid == sids[0]][y],
        z=df_group1[df_group1.sid == sids[0]][z],
        mode='markers',
        # marker=dict(size=5, color='red', opacity=0.7),
        marker=mdict,
        name=g1_name,
    )
    scats.append(init_scat)

    if group2:
        if color:
            mdict2 = dict(size=5, color=df_group2[df_group2.sid == sids[0]].p, opacity=1,
                          colorscale='Viridis', cmin=0, cmax=100,
                          showscale=True,
                          line=dict(color='black', width=1),
                          colorbar=dict(title='Momentum (MeV)', xanchor='left'))
        else:
            mdict2 = dict(size=5, color='blue', opacity=0.7)
        init_scat2 = go.Scatter3d(
            x=df_group2[df_group2.sid == sids[0]][x],
            y=df_group2[df_group2.sid == sids[0]][y],
            z=df_group2[df_group2.sid == sids[0]][z],
            mode='markers',
            marker=mdict2,
            name=g2_name)
        scats.append(init_scat2)

    xray_query = 'xstop<1000 and tstop<200 and sqrt(xstop*xstop+ystop*ystop)<900'
    df_xray.query(xray_query, inplace=True)
    print 'binning...'
    h, edges = np.histogramdd(np.asarray([df_xray.zstop, df_xray.xstop, df_xray.ystop]).T,
                              bins=(100, 25, 25))
    centers = [(e[1:]+e[:-1])/2.0 for e in edges]
    hadj = np.rot90(h).flatten()/h.max()
    cx, cy, cz = np.meshgrid(*centers)
    df_binned = pd.DataFrame(dict(x=cx.flatten(), y=cy.flatten(), z=cz.flatten(), h=hadj))
    df_binned = df_binned.query('h>0')
    df_binned['cats'] = pd.cut(df_binned.h, 200)
    print 'binned'
    groups = np.sort(df_binned.cats.unique())
    for i, group in enumerate(groups[0:]):
        df_tmp = df_binned[df_binned.cats == group]
        if i == 0:
            sl = True
        else:
            sl = False
        scats.append(
            go.Scatter3d(
                x=df_tmp.x, y=df_tmp.y, z=df_tmp.z, mode='markers',
                marker=dict(size=4, color='black', opacity=0.03*(i+1), symbol='square'),
                name='X-ray', showlegend=sl, legendgroup='xray'))

    df_newmat = df_binned.query('12800<x<13000 and sqrt(y**2+z**2)<800')
    if len(df_newmat) > 0:
        scats.append(
            go.Scatter3d(
                x=df_newmat.x, y=df_newmat.y, z=df_newmat.z, mode='markers',
                marker=dict(sizemode='diameter', size=6,
                            color='red', opacity=0.4, symbol='square'),
                name='X-ray', showlegend=False, legendgroup='xray'))

    p_slider = widgets.IntSlider(min=0, max=130, value=0, step=1, continuous_update=False)
    p_slider.description = 'Time Shift'
    p_slider.value = 0
    p_state = time_shifter(group2, color)
    p_slider.on_trait_change(p_state.on_time_change, 'value')

    fig = go.Figure(data=scats, layout=layout)
    g = GraphWidget('https://plot.ly/~BroverCleveland/70/')
    display(g)
    display(p_slider)
    return g, fig

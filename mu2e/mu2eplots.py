#! /usr/bin/env python

import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
import plotly.tools as tls


# Definitions of mu2e-specific plotting functions.
# Plots can be displayed in matplotlib, plotly (offline mode), or plotly (html only)

def mu2e_plot(df, x, y, conditions = None, mode = 'mpl', info = None, savename = None):
    _modes = ['mpl', 'plotly', 'plotly_html', 'plotly_offline']
    if conditions:
        df = df.query(conditions)

    if mode not in _modes:
        raise ValueError(mode+' not in '+_modes)

    #plt.plot(df[x], df[y], label=y)
    ax = df.plot(x, y, kind='line')
    ax.grid(True)
    plt.ylabel(y)
    plt.title(' '.join(filter(lambda x:x, [info, x, 'v', y, conditions])))
    if mode == 'mpl':
        plt.legend()

    if mode == 'plotly_offline':
        init_notebook_mode()
        fig = ax.get_figure()
        py_fig = tls.mpl_to_plotly(fig)
        iplot(py_fig)


    if savename:
        plt.savefig(savename)





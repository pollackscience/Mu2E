#! /usr/bin/env python

from __future__ import division
import plotly.graph_objs as go
from mu2e.offline import init_notebook_mode, iplot
import numpy as np


def interp_phi(df, x, y, z, df_alt=None, plot=True):
    """First thing's first. Plot the xyz point, and the immediate cube around it."""

    df_trimmed = df.query('{0}<=X<={1} and {2}<=Y<={3} and {4}<=Z<={5}'.format(
        x-25, x+25, y-25, y+25, z-25, z+25))
    df_trimmed = df_trimmed[['X', 'Y', 'Z', 'Bx', 'By', 'Bz']]

    df_true = df_trimmed.query('X=={0} and Y=={1} and Z=={2}'.format(x, y, z))
    if len(df_true) == 1:
        bx_interp = df_true.Bx
        by_interp = df_true.By
        bz_interp = df_true.Bz

    else:
        if len(df_trimmed.X.unique()) == len(df_trimmed.Y.unique()) == len(df_trimmed.Z.unique()):
            df_trimmed = df_trimmed.query('X!={0} and Y!={1} and Z!={2}'.format(
                x, y, z))
        else:
            xs = np.sort(df_trimmed.X.unique())
            ys = np.sort(df_trimmed.Y.unique())
            zs = np.sort(df_trimmed.Z.unique())
            df_trimmed = df_trimmed.query(
                '(X=={0} or X=={1}) and (Y=={2} or Y=={3}) and (Z=={4} or Z=={5})'.format(
                    xs[0], xs[1], ys[0], ys[1], zs[0], zs[1]))

        df_trimmed = df_trimmed.sort_values(['X', 'Y', 'Z']).reset_index(drop=True)
        # indices:                 0       1       2       3       4       5       6       7
        df_trimmed['Vertex'] = ['p000', 'p001', 'p010', 'p011', 'p100', 'p101', 'p110', 'p111']

        x_rel = (x - df_trimmed.ix[0].X) / (df_trimmed.ix[4].X-df_trimmed.ix[0].X)
        y_rel = (y - df_trimmed.ix[0].Y) / (df_trimmed.ix[2].Y-df_trimmed.ix[0].Y)
        z_rel = (z - df_trimmed.ix[0].Z) / (df_trimmed.ix[1].Z-df_trimmed.ix[0].Z)

        # print x_rel
        bx_interp = ((y_rel)*(z_rel)*(x_rel*df_trimmed.ix[7].Bx + (1-x_rel)*df_trimmed.ix[3].Bx) +
                     (1-y_rel) * (z_rel) * (x_rel*df_trimmed.ix[5].Bx + (1-x_rel)*df_trimmed.ix[1].Bx) +
                     (y_rel) * (1-z_rel) * (x_rel*df_trimmed.ix[6].Bx + (1-x_rel)*df_trimmed.ix[2].Bx) +
                     (1-y_rel) * (1-z_rel)*(x_rel*df_trimmed.ix[4].Bx + (1-x_rel)*df_trimmed.ix[0].Bx))

        by_interp = ((x_rel)*(z_rel)*(y_rel*df_trimmed.ix[7].By + (1-y_rel)*df_trimmed.ix[5].By) +
                     (1-x_rel) * (z_rel) * (y_rel*df_trimmed.ix[3].By + (1-y_rel)*df_trimmed.ix[1].By) +
                     (x_rel) * (1-z_rel) * (y_rel*df_trimmed.ix[6].By + (1-y_rel)*df_trimmed.ix[4].By) +
                     (1-x_rel) * (1-z_rel)*(y_rel*df_trimmed.ix[2].By + (1-y_rel)*df_trimmed.ix[0].By))

        bz_interp = ((x_rel)*(y_rel)*(z_rel*df_trimmed.ix[7].Bz + (1-z_rel)*df_trimmed.ix[6].Bz) +
                     (1-x_rel) * (y_rel) * (z_rel*df_trimmed.ix[3].Bz + (1-z_rel)*df_trimmed.ix[2].Bz) +
                     (x_rel) * (1-y_rel) * (z_rel*df_trimmed.ix[5].Bz + (1-z_rel)*df_trimmed.ix[4].Bz) +
                     (1-x_rel) * (1-y_rel)*(z_rel*df_trimmed.ix[1].Bz + (1-z_rel)*df_trimmed.ix[0].Bz))

    if plot:
        init_notebook_mode()

        trace1 = go.Scatter3d(
            x=df_trimmed.X,
            y=df_trimmed.Y,
            z=df_trimmed.Z,
            mode='markers',
            marker=dict(
                size=12,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5
                ),
                opacity=0.8,
            ),
            text=df_trimmed.Vertex,
        )

        trace2 = go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode='markers',
            marker=dict(
                color='rgb(127, 127, 127)',
                size=12,
                symbol='circle',
                line=dict(
                    color='rgb(204, 204, 204)',
                    width=1
                ),
                opacity=0.9
            )
        )
        data = [trace1, trace2]
        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
        )
        fig = go.Figure(data=data, layout=layout)
        iplot(fig)

    return df_trimmed, [bx_interp, by_interp, bz_interp]
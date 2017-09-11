#! /usr/bin/env python

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import collections
import plotly.graph_objs as go
from mu2e.offline import init_notebook_mode, iplot
import numpy as np
from six.moves import range
from six.moves import zip


def interp_phi_cubic(df, x, y, z, plot=False, mode='lacey', shift=0):
    if shift == 0:
        df_trimmed = df.query('{0}<=X<{1} and {2}<=Y<{3} and {4}<=Z<{5}'.format(
            x-50, x+50, y-50, y+50, z-50, z+50))
        df_trimmed = df_trimmed[['X', 'Y', 'Z', 'Bx', 'By', 'Bz']]
        df_trimmed = df_trimmed.sort_values(['X', 'Y', 'Z']).reset_index(drop=True)

    elif shift == -1:
        df_trimmed = df.query('{0}<=X<{1} and {2}<=Y<{3} and {4}<=Z<{5}'.format(
            x-25, x+75, y-50, y+50, z-50, z+50))
        df_trimmed = df_trimmed[['X', 'Y', 'Z', 'Bx', 'By', 'Bz']]
        df_trimmed = df_trimmed.sort_values(['X', 'Y', 'Z']).reset_index(drop=True)

    elif shift == 1:
        df_trimmed = df.query('{0}<=X<{1} and {2}<=Y<{3} and {4}<=Z<{5}'.format(
            x-75, x+25, y-50, y+50, z-50, z+50))
        df_trimmed = df_trimmed[['X', 'Y', 'Z', 'Bx', 'By', 'Bz']]
        df_trimmed = df_trimmed.sort_values(['X', 'Y', 'Z']).reset_index(drop=True)

    else:
        raise KeyError('wrong shift')

    #df_true = df_trimmed.query('X=={0} and Y=={1} and Z=={2}'.format(x, y, z))
    #if len(df_true) == 1:
        #pass
        # bx_interp = df_true.Bx
        # by_interp = df_true.By
        # bz_interp = df_true.Bz
    if False:
        pass

    else:
        df_trimmed['Vertex'] = [
            # 0           1         2         3         4         5        6        7
            '-1-1-1' , '-1-10' , '-1-11' , '-1-12' , '-10-1' , '-100' , '-101' , '-102' ,
            # 8           9         10        11        12        13       14       15
            '-11-1' ,  '-110' ,  '-111' ,  '-112' ,  '-12-1' , '-120' , '-121' , '-122' ,
            # 16         17        18        19        20        21       22       23
            '0-1-1' ,  '0-10' ,  '0-11' ,  '0-12' ,  '00-1' ,  '000' ,  '001' ,  '002' ,
            # 24         25        26        27        28        29       30       31
            '01-1' ,   '010' ,   '011' ,   '012' ,   '02-1' ,  '020' ,  '021' ,  '022' ,
            # 32         33        34        35        36        37       38       39
            '1-1-1' ,  '1-10' ,  '1-11' ,  '1-12' ,  '10-1' ,  '100' ,  '101' ,  '102' ,
            # 40         41        42        43        44        45       46       47
            '11-1' ,   '110' ,   '111' ,   '112' ,   '12-1' ,  '120' ,  '121' ,  '122' ,
            # 48         49        50        51        52        53       54       55
            '2-1-1' ,  '2-10' ,  '2-11' ,  '2-12' ,  '20-1' ,  '200' ,  '201' ,  '202' ,
            # 56         57        58        59        60        61       62       63
            '21-1' ,   '210' ,   '211' ,   '212' ,   '22-1' ,  '220' ,  '221' ,  '222' ,
        ]
        index_list = list(range(0,64))
        vertex_dict = dict(list(zip(df_trimmed.Vertex, index_list)))

        # Define some lagrange polynomials
        def lg_mone(n):
            return -(n**3-3*n**2+2*n)/6.0

        def lg_zero(n):
            return (n**3-2*n**2-n+2)/2.0

        def lg_one(n):
            return -(n**3-n**2-2*n)/2.0

        def lg_two(n):
            return (n**3-n)/6.0

        # Numerical integration

        def _int_methods(b, rel, *verts):
            if len(verts) == 2:
                return (b.ix[vertex_dict[verts[0]]] + b.ix[vertex_dict[verts[1]]])/2.0
            elif len(verts) == 3:
                return (b.ix[vertex_dict[verts[0]]] + 4*b.ix[vertex_dict[verts[1]]] +
                        b.ix[vertex_dict[verts[2]]])/3.0
            else:
                if mode == 'lacey':
                    return (b.ix[vertex_dict[verts[0]]] + 3*b.ix[vertex_dict[verts[1]]] +
                            3*b.ix[vertex_dict[verts[2]]] + b.ix[vertex_dict[verts[3]]])*(3.0/8.0)

                else:
                    A = (-3*b.ix[vertex_dict[verts[2]]] - b.ix[vertex_dict[verts[0]]] +
                         3*b.ix[vertex_dict[verts[1]]] + b.ix[vertex_dict[verts[3]]])/6.0
                    B = (b.ix[vertex_dict[verts[0]]] + b.ix[vertex_dict[verts[2]]] -
                         2*b.ix[vertex_dict[verts[1]]])/2.0
                    C = (6*b.ix[vertex_dict[verts[2]]] - 2*b.ix[vertex_dict[verts[0]]] -
                         3*b.ix[vertex_dict[verts[1]]] - b.ix[vertex_dict[verts[3]]])/6.0
                    D = b.ix[vertex_dict[verts[1]]]

                    return (A*rel**3 + B*rel**2 + C*rel + D)

        def num_int(b, rel, x, y, z):
            if isinstance(x, collections.Sequence):
                verts = [str(i)+str(y)+str(z) for i in range(x[0], x[-1]+1)]
            elif isinstance(y, collections.Sequence):
                verts = [str(x)+str(i)+str(z) for i in range(y[0], y[-1]+1)]
            elif isinstance(z, collections.Sequence):
                verts = [str(x)+str(y)+str(i) for i in range(z[0], z[-1]+1)]
            else:
                raise AttributeError('exactly one arg in num_int must be a collection')

            return _int_methods(b, rel, *verts)

        def int_collection(b, rel, x, y, z):
            if x == 'var':
                int1 = num_int(b, rel, (-1, 2), y, z)
                int2 = num_int(b, rel, (0, 1), y, z)
                int3 = num_int(b, rel, (-1, 0), y, z)
                int4 = int2
                int5 = num_int(b, rel, (-1, 1), y, z)
                int6 = int2
                int7 = num_int(b, rel, (1, 2), y, z)
            elif y == 'var':
                int1 = num_int(b, rel, x, (-1, 2), z)
                int2 = num_int(b, rel, x, (0, 1), z)
                int3 = num_int(b, rel, x, (-1, 0), z)
                int4 = int2
                int5 = num_int(b, rel, x, (-1, 1), z)
                int6 = int2
                int7 = num_int(b, rel, x, (1, 2), z)
            else:
                int1 = num_int(b, rel, x, y, (-1, 2))
                int2 = num_int(b, rel, x, y, (0, 1))
                int3 = num_int(b, rel, x, y, (-1, 0))
                int4 = int2
                int5 = num_int(b, rel, x, y, (-1, 1))
                int6 = int2
                int7 = num_int(b, rel, x, y, (1, 2))

            if mode == 'lacey':
                return ((-rel**2/2.0)*int1 + (3*rel**2/2.0)*int2 + rel*int3 - rel*int4 -
                        (1/3.0)*int5 - 0.5*int6 + (1/6.0)*int7)
            else:
                return -int1

        x_rel = (x - df_trimmed.ix[21].X) / (25.0)
        y_rel = (y - df_trimmed.ix[21].Y) / (25.0)
        z_rel = (z - df_trimmed.ix[21].Z) / (25.0)
        # print x_rel, y_rel, z_rel
        bxs = df_trimmed.Bx
        bys = df_trimmed.By
        bzs = df_trimmed.Bz

        bx_interp = -(lg_mone(z_rel)*(lg_mone(y_rel)*(int_collection(bxs, x_rel, 'var', -1, -1)) +
                                      lg_zero(y_rel)*(int_collection(bxs, x_rel, 'var', 0, -1)) +
                                      lg_one(y_rel)*(int_collection(bxs, x_rel, 'var', 1, -1)) +
                                      lg_two(y_rel)*(int_collection(bxs, x_rel, 'var', 2, -1))) +
                      lg_zero(z_rel)*(lg_mone(y_rel)*(int_collection(bxs, x_rel, 'var', -1, 0)) +
                                      lg_zero(y_rel)*(int_collection(bxs, x_rel, 'var', 0, 0)) +
                                      lg_one(y_rel)*(int_collection(bxs, x_rel, 'var', 1, 0)) +
                                      lg_two(y_rel)*(int_collection(bxs, x_rel, 'var', 2, 0))) +
                      lg_one(z_rel)*(lg_mone(y_rel)*(int_collection(bxs, x_rel, 'var', -1, 1)) +
                                     lg_zero(y_rel)*(int_collection(bxs, x_rel, 'var', 0, 1)) +
                                     lg_one(y_rel)*(int_collection(bxs, x_rel, 'var', 1, 1)) +
                                     lg_two(y_rel)*(int_collection(bxs, x_rel, 'var', 2, 1))) +
                      lg_two(z_rel)*(lg_mone(y_rel)*(int_collection(bxs, x_rel, 'var', -1, 2)) +
                                     lg_zero(y_rel)*(int_collection(bxs, x_rel, 'var', 0, 2)) +
                                     lg_one(y_rel)*(int_collection(bxs, x_rel, 'var', 1, 2)) +
                                     lg_two(y_rel)*(int_collection(bxs, x_rel, 'var', 2, 2)))
                      )

        by_interp = -(lg_mone(z_rel)*(lg_mone(x_rel)*(int_collection(bys, y_rel, -1, 'var', -1)) +
                                      lg_zero(x_rel)*(int_collection(bys, y_rel, 0, 'var', -1)) +
                                      lg_one(x_rel)*(int_collection(bys, y_rel, 1, 'var', -1)) +
                                      lg_two(x_rel)*(int_collection(bys, y_rel, 2, 'var', -1))) +
                      lg_zero(z_rel)*(lg_mone(x_rel)*(int_collection(bys, y_rel, -1, 'var', 0)) +
                                      lg_zero(x_rel)*(int_collection(bys, y_rel, 0, 'var', 0)) +
                                      lg_one(x_rel)*(int_collection(bys, y_rel, 1, 'var', 0)) +
                                      lg_two(x_rel)*(int_collection(bys, y_rel, 2, 'var', 0))) +
                      lg_one(z_rel)*(lg_mone(x_rel)*(int_collection(bys, y_rel, -1, 'var', 1)) +
                                     lg_zero(x_rel)*(int_collection(bys, y_rel, 0, 'var', 1)) +
                                     lg_one(x_rel)*(int_collection(bys, y_rel, 1, 'var', 1)) +
                                     lg_two(x_rel)*(int_collection(bys, y_rel, 2, 'var', 1))) +
                      lg_two(z_rel)*(lg_mone(x_rel)*(int_collection(bys, y_rel, -1, 'var', 2)) +
                                     lg_zero(x_rel)*(int_collection(bys, y_rel, 0, 'var', 2)) +
                                     lg_one(x_rel)*(int_collection(bys, y_rel, 1, 'var', 2)) +
                                     lg_two(x_rel)*(int_collection(bys, y_rel, 2, 'var', 2)))
                      )

        bz_interp = -(lg_mone(y_rel)*(lg_mone(x_rel)*(int_collection(bzs, z_rel, -1, -1, 'var')) +
                                      lg_zero(x_rel)*(int_collection(bzs, z_rel, 0, -1, 'var')) +
                                      lg_one(x_rel)*(int_collection(bzs, z_rel, 1, -1, 'var')) +
                                      lg_two(x_rel)*(int_collection(bzs, z_rel, 2, -1, 'var'))) +
                      lg_zero(y_rel)*(lg_mone(x_rel)*(int_collection(bzs, z_rel, -1, 0, 'var')) +
                                      lg_zero(x_rel)*(int_collection(bzs, z_rel, 0, 0, 'var')) +
                                      lg_one(x_rel)*(int_collection(bzs, z_rel, 1, 0, 'var')) +
                                      lg_two(x_rel)*(int_collection(bzs, z_rel, 2, 0, 'var'))) +
                      lg_one(y_rel)*(lg_mone(x_rel)*(int_collection(bzs, z_rel, -1, 1, 'var')) +
                                     lg_zero(x_rel)*(int_collection(bzs, z_rel, 0, 1, 'var')) +
                                     lg_one(x_rel)*(int_collection(bzs, z_rel, 1, 1, 'var')) +
                                     lg_two(x_rel)*(int_collection(bzs, z_rel, 2, 1, 'var'))) +
                      lg_two(y_rel)*(lg_mone(x_rel)*(int_collection(bzs, z_rel, -1, 2, 'var')) +
                                     lg_zero(x_rel)*(int_collection(bzs, z_rel, 0, 2, 'var')) +
                                     lg_one(x_rel)*(int_collection(bzs, z_rel, 1, 2, 'var')) +
                                     lg_two(x_rel)*(int_collection(bzs, z_rel, 2, 2, 'var')))
                      )

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
        print(x_rel)

    return df_trimmed, [bx_interp, by_interp, bz_interp]
    #return df_trimmed, vertex_dict

def interp_phi_quad(df, x, y, z, plot=False):
    df_trimmed = df.query('{0}<=X<={1} and {2}<=Y<={3} and {4}<=Z<={5}'.format(
        x-37.5, x+37.5, y-37.5, y+37.5, z-37.5, z+37.5))
    df_trimmed = df_trimmed[['X', 'Y', 'Z', 'Bx', 'By', 'Bz']]
    df_trimmed = df_trimmed.sort_values(['X', 'Y', 'Z']).reset_index(drop=True)

    #df_true = df_trimmed.query('X=={0} and Y=={1} and Z=={2}'.format(x, y, z))
    #if len(df_true) == 1:
        #pass
        # bx_interp = df_true.Bx
        # by_interp = df_true.By
        # bz_interp = df_true.Bz
    if False:
        pass

    else:
        df_trimmed['Vertex'] = [
            #   0         1         2         3         4        5        6        7        8
            '-1-1-1' , '-1-10' , '-1-11' , '-10-1' , '-100' , '-101' , '-11-1' , '-110' , '-111' ,
            #   9        10        11        12        13       14       15       16       17
            '0-1-1'  , '0-10'  , '0-11'  , '00-1'  , '000'  , '001'  , '01-1'  , '010'  , '011'  ,
            #  18        19        20        21        22       23       24       25       26
            '1-1-1'  , '1-10'  , '1-11'  , '10-1'  , '100'  , '101'  , '11-1'  , '110'  , '111'  ,
        ]

        x_rel = (x - df_trimmed.ix[13].X) / (25.0)
        y_rel = (y - df_trimmed.ix[13].Y) / (25.0)
        z_rel = (z - df_trimmed.ix[13].Z) / (25.0)
        # print x_rel, y_rel, z_rel
        bxs = df_trimmed.Bx
        bys = df_trimmed.By
        bzs = df_trimmed.Bz

        tuning = 0.02
        bx_interp = -(
            tuning*(x_rel*((x_rel+1)*bxs[13]-x_rel*bxs[4]) -
                    x_rel*((1-x_rel)*bxs[13]+x_rel*bxs[22]))+
            -(((bxs[4]+bxs[22])/2.0-bxs[13])*x_rel**2+((bxs[22]-bxs[4])/2.0)*x_rel+bxs[13])
        )
        #bx_interp = -(
        #    x_rel*0.5*(bxs[4]+bxs[13]) - x_rel*0.5*(bxs[13]+bxs[22]) -
        #    (1.0/6.0)*(bxs[4]+4*bxs[13]+bxs[22])
        #)
        #bx_interp = (
        #    ((
            #  #(min(0, x_rel)*(abs(x_rel)*bxs[9]-(1-abs(x_rel))*bxs[18])) -
            #  (((bxs[0]+bxs[18])/2.0-bxs[9])*x_rel**2+((bxs[18]-bxs[0])/2.0)*x_rel+bxs[9])) *
            # 0.5*(y_rel**2-y_rel) +
            # (
            #  #(min(0, x_rel)*(abs(x_rel)*bxs[12]-(1-abs(x_rel))*bxs[21])) -
            #     (((bxs[3]+bxs[21])/2.0-bxs[12])*x_rel**2+((bxs[21]-bxs[3])/2.0)*x_rel+bxs[12])) *
            # (-y_rel**2+1) +
            # (
            #  #(min(0, x_rel)*(abs(x_rel)*bxs[15]-(1-abs(x_rel))*bxs[24])) -
            #     (((bxs[6]+bxs[24])/2.0-bxs[15])*x_rel**2+((bxs[24]-bxs[6])/2.0)*x_rel+bxs[15])) *
            # 0.5*(y_rel**2+y_rel))*0.5*(z_rel**2-z_rel) +
            #((
            #  #(min(0, x_rel)*(abs(x_rel)*bxs[10]-(1-abs(x_rel))*bxs[19])) -
            #    (((bxs[1]+bxs[19])/2.0-bxs[10])*x_rel**2+((bxs[19]-bxs[1])/2.0)*x_rel+bxs[10])) *
            # 0.5*(y_rel**2-y_rel) +
            # (
              #(min(0, x_rel)*(abs(x_rel)*bxs[13]-(1-abs(x_rel))*bxs[22])) -
            #     (((bxs[4]+bxs[22])/2.0-bxs[13])*x_rel**2+((bxs[22]-bxs[4])/2.0)*x_rel+bxs[13])) *
            # (-y_rel**2+1) #+
            # (
            #  #(min(0, x_rel)*(abs(x_rel)*bxs[16]-(1-abs(x_rel))*bxs[25])) -
            #     (((bxs[7]+bxs[25])/2.0-bxs[16])*x_rel**2+((bxs[25]-bxs[7])/2.0)*x_rel+bxs[16])) *
            # 0.5*(y_rel**2+y_rel))*(-z_rel**2+1) +
            #((
            #  #(min(0, x_rel)*(abs(x_rel)*bxs[11]-(1-abs(x_rel))*bxs[20])) -
            #    (((bxs[2]+bxs[20])/2.0-bxs[11])*x_rel**2+((bxs[20]-bxs[2])/2.0)*x_rel+bxs[11])) *
            # 0.5*(y_rel**2-y_rel) +
            # (
            #  #(min(0, x_rel)*(abs(x_rel)*bxs[14]-(1-abs(x_rel))*bxs[23])) -
            #     (((bxs[5]+bxs[23])/2.0-bxs[14])*x_rel**2+((bxs[23]-bxs[5])/2.0)*x_rel+bxs[14])) *
            # (-y_rel**2+1) +
            # (
            #  #(min(0, x_rel)*(abs(x_rel)*bxs[17]-(1-abs(x_rel))*bxs[26])) -
            #     (((bxs[8]+bxs[26])/2.0-bxs[17])*x_rel**2+((bxs[26]-bxs[8])/2.0)*x_rel+bxs[17])) *
            # 0.5*(y_rel**2+y_rel))*0.5*(z_rel**2+z_rel)
        #)

        # bx_interp = -(
        #     ((-(((bxs[0]+bxs[18])/2.0-bxs[9])*x_rel**2+((bxs[18]-bxs[0])/2.0)*x_rel+bxs[9])) *
        #      0.5*(y_rel**2-y_rel) +
        #      (-(((bxs[3]+bxs[21])/2.0-bxs[12])*x_rel**2+((bxs[21]-bxs[3])/2.0)*x_rel+bxs[12])) *
        #      (-y_rel**2+1) +
        #      (-(((bxs[6]+bxs[24])/2.0-bxs[15])*x_rel**2+((bxs[24]-bxs[6])/2.0)*x_rel+bxs[15])) *
        #      0.5*(y_rel**2+y_rel))*0.5*(z_rel**2-z_rel) +
        #     ((-(((bxs[1]+bxs[19])/2.0-bxs[10])*x_rel**2+((bxs[19]-bxs[1])/2.0)*x_rel+bxs[10])) *
        #      0.5*(y_rel**2-y_rel) +
        #      (-(((bxs[4]+bxs[22])/2.0-bxs[13])*x_rel**2+((bxs[22]-bxs[4])/2.0)*x_rel+bxs[13])) *
        #      (-y_rel**2+1) +
        #      (-(((bxs[7]+bxs[25])/2.0-bxs[16])*x_rel**2+((bxs[25]-bxs[7])/2.0)*x_rel+bxs[16])) *
        #      0.5*(y_rel**2+y_rel))*(-z_rel**2+1) +
        #     ((-(((bxs[2]+bxs[20])/2.0-bxs[11])*x_rel**2+((bxs[20]-bxs[2])/2.0)*x_rel+bxs[11])) *
        #      0.5*(y_rel**2-y_rel) +
        #      (-(((bxs[5]+bxs[23])/2.0-bxs[14])*x_rel**2+((bxs[23]-bxs[5])/2.0)*x_rel+bxs[14])) *
        #      (-y_rel**2+1) +
        #      (-(((bxs[8]+bxs[26])/2.0-bxs[17])*x_rel**2+((bxs[26]-bxs[8])/2.0)*x_rel+bxs[17])) *
        #      0.5*(y_rel**2+y_rel))*0.5*(z_rel**2+z_rel)
        # )

        by_interp = -(
            ((-(((bys[0]+bys[6])/2.0-bys[3])*y_rel**2+((bys[6]-bys[0])/2.0)*y_rel+bys[3])) *
             0.5*(x_rel**2-x_rel) +
             (-(((bys[9]+bys[15])/2.0-bys[12])*y_rel**2+((bys[15]-bys[9])/2.0)*y_rel+bys[12])) *
             (-x_rel**2+1) +
             (-(((bys[18]+bys[24])/2.0-bys[21])*y_rel**2+((bys[24]-bys[18])/2.0)*y_rel+bys[21])) *
             0.5*(x_rel**2+x_rel))*0.5*(z_rel**2-z_rel) +
            ((-(((bys[1]+bys[7])/2.0-bys[4])*y_rel**2+((bys[7]-bys[1])/2.0)*y_rel+bys[4])) *
             0.5*(x_rel**2-x_rel) +
             (-(((bys[10]+bys[16])/2.0-bys[13])*y_rel**2+((bys[16]-bys[10])/2.0)*y_rel+bys[13])) *
             (-x_rel**2+1) +
             (-(((bys[19]+bys[25])/2.0-bys[22])*y_rel**2+((bys[25]-bys[19])/2.0)*y_rel+bys[22])) *
             0.5*(x_rel**2+x_rel))*(-z_rel**2+1) +
            ((-(((bys[2]+bys[8])/2.0-bys[5])*y_rel**2+((bys[8]-bys[2])/2.0)*y_rel+bys[5])) *
             0.5*(x_rel**2-x_rel) +
             (-(((bys[11]+bys[17])/2.0-bys[14])*y_rel**2+((bys[17]-bys[11])/2.0)*y_rel+bys[14])) *
             (-x_rel**2+1) +
             (-(((bys[20]+bys[26])/2.0-bys[23])*y_rel**2+((bys[26]-bys[20])/2.0)*y_rel+bys[23])) *
             0.5*(x_rel**2+x_rel))*0.5*(z_rel**2+z_rel)
        )

        bz_interp = -(
            ((-(((bzs[0]+bzs[2])/2.0-bzs[1])*z_rel**2+((bzs[2]-bzs[0])/2.0)*z_rel+bzs[1])) *
             0.5*(x_rel**2-x_rel) +
             (-(((bzs[9]+bzs[11])/2.0-bzs[10])*z_rel**2+((bzs[11]-bzs[9])/2.0)*z_rel+bzs[10])) *
             (-x_rel**2+1) +
             (-(((bzs[18]+bzs[20])/2.0-bzs[19])*z_rel**2+((bzs[20]-bzs[18])/2.0)*z_rel+bzs[19])) *
             0.5*(x_rel**2+x_rel))*0.5*(y_rel**2-y_rel) +
            ((-(((bzs[3]+bzs[5])/2.0-bzs[4])*z_rel**2+((bzs[5]-bzs[3])/2.0)*z_rel+bzs[4])) *
             0.5*(x_rel**2-x_rel) +
             (-(((bzs[12]+bzs[14])/2.0-bzs[13])*z_rel**2+((bzs[14]-bzs[12])/2.0)*z_rel+bzs[13])) *
             (-x_rel**2+1) +
             (-(((bzs[21]+bzs[23])/2.0-bzs[22])*z_rel**2+((bzs[23]-bzs[21])/2.0)*z_rel+bzs[22])) *
             0.5*(x_rel**2+x_rel))*(-y_rel**2+1) +
            ((-(((bzs[6]+bzs[8])/2.0-bzs[7])*z_rel**2+((bzs[8]-bzs[6])/2.0)*z_rel+bzs[7])) *
             0.5*(x_rel**2-x_rel) +
             (-(((bzs[15]+bzs[17])/2.0-bzs[16])*z_rel**2+((bzs[17]-bzs[15])/2.0)*z_rel+bzs[16])) *
             (-x_rel**2+1) +
             (-(((bzs[24]+bzs[26])/2.0-bzs[25])*z_rel**2+((bzs[26]-bzs[24])/2.0)*z_rel+bzs[25])) *
             0.5*(x_rel**2+x_rel))*0.5*(y_rel**2+y_rel)
        )

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

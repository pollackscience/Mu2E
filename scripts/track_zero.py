#! /usr/bin/env python

from mu2e.datafileprod import DataFileMaker
from mu2e.plotter import *
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import plotly.plotly as py
import plotly.graph_objs as go
from mu2e.tools.new_iplot import new_iplot, get_plotlyjs



'''
plt.close('all')
data_maker1=DataFileMaker('../datafiles/FieldMapData_1760_v5/Mu2e_DSmap',use_pickle = True)
data_maker5=DataFileMaker('../datafiles/FieldMapsGA04/Mu2e_DS_GA0',use_pickle = True)
plot_maker = Plotter({'DS_GA04':data_maker5.data_frame},no_show=True)
#plot_maker = Plotter({'DS_Mau':data_maker1.data_frame},no_show=True)

min_br = []
min_x = []
min_y = []
min_z = []
for z in range(4321,14022,50):
    print 'doing z = ',z
    fig, df_int = plot_maker.plot_A_v_B_and_C('Br','X','Y',True,800,'Z=={}'.format(z),'-501<X<501','-501<Y<501')
    my_mins = df_int[df_int.Br == df_int.Br.min()]
    print my_mins
    min_br.append(my_mins.Br.values[0])
    min_x.append(my_mins.X.values[0])
    min_y.append(my_mins.Y.values[0])
    min_z.append(z)
plt.close('all')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs=min_z,ys=min_x,zs=min_y)
ax.set_xlabel('Z (mm)')
ax.set_ylabel('X (mm)')
ax.set_zlabel('Y (mm)')

plt.show()

plt.savefig(plot_maker.save_dir+'/track_zero2.png')
'''

trace_data = go.Scatter3d(
    x=min_z,
    y=min_x,
    z=min_y,
    mode='lines',
    marker=go.Marker(
        color='#1f77b4',
        size=12,
        symbol='circle',
        line=go.Line(
            color='rgb(0,0,0)',
            width=0
        )
    ),
    line=go.Line(
        color='#1f77b4',
        width=2
    ),
    name='min(Br) location'
)

trace_zero = go.Scatter3d(
    x=min_z,
    y=[0]*len(min_x),
    z=[0]*len(min_y),
    mode='lines',
    marker=go.Marker(
        color='#1f77b4',
        size=12,
        symbol='circle',
        line=go.Line(
            color='rgb(0,0,0)',
            width=0
        )
    ),
    line=go.Line(
        color='red',
        width=2
    ),
    name = '0 line',
)

data = [trace_data, trace_zero]
layout = go.Layout(
        title='Plot of X,Y location of min(Br) as function of Z for DS',
        autosize=False,
        width=675,
        height=650,
        scene=dict(
            xaxis=dict(
                title='Z (mm)',
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
                ),
            yaxis=dict(
                title='X (mm)',
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
                ),
            zaxis=dict(
                title='Y (mm)',
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
                ),
            ),
        showlegend=True,
    )
fig = go.Figure(data=data, layout=layout)
plot_html = new_iplot(fig,show_link=False)
savename = plot_maker.html_dir+'/track_zero.html'
with open(savename,'w') as html_file:
    html_file.write(plot_html)

#! /usr/bin/env python

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import numpy as np

init_notebook_mode()

# Creating the data
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
xGrid, yGrid = np.meshgrid(y, x)
R = np.sqrt(xGrid ** 2 + yGrid ** 2)
z = np.sin(R)
z2 = np.cos(R)

# Creating the plot
lines = []
line_marker = dict(color='#0066FF', width=2)
line_marker2 = dict(color='green', width=2)

scene=dict(
    xaxis=dict(
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)'
    ),
    yaxis=dict(
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)'
    ),
    zaxis=dict(
        gridcolor='rgb(255, 255, 255)',
        zerolinecolor='rgb(255, 255, 255)',
        showbackground=True,
        backgroundcolor='rgb(230, 230,230)'
    )
)

fig = tools.make_subplots(rows=1,cols=2,specs=[[{'is_3d':True},{'is_3d':False}]])
for i, j, k in zip(xGrid, yGrid, z): #comment this out and subplot 2 works
    fig.append_trace(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker,name='wire1'),1,1) #comment this out and subplot 2 works
for i, j, k in zip(xGrid, yGrid, z2): #comment this out and subplot 2 works
    fig.append_trace(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker2,name='wire2'),1,1) #comment this out and subplot 2 works
fig.append_trace(go.Heatmap(x=x, y=y, z=np.sqrt((z-z2)**2)),1,2) #comment this out and subplot 1 works
fig['layout'].update(title='3d with heatmap',height=600,width=1000,showlegend=False)
plot_url = iplot(fig)

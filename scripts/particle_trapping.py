import cPickle as pkl
from mu2e.mu2eplots import mu2e_plot3d_ptrap
from mu2e.datafileprod import g4root_to_df


g4root_to_df('../datafiles/G4ParticleSim/low_e_ele',True)
#df_nttvd, df_ntpart = pkl.load(open('../datafiles/G4ParticleSim/GA04_gridtest.p','rb'))
#mu2e_plot3d_ptrap(df_ntpart.query('xstop<-2500').ix[0:30000],'zstop','xstop','ystop',mode='plotly_html_img')

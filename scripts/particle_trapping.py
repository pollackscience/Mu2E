import cPickle as pkl
import mu2e
from mu2e.mu2eplots import mu2e_plot3d_ptrap
from mu2e.dataframeprod import g4root_to_df


g4root_to_df(mu2e.mu2e_ext_path+'datafiles/G4ParticleSim/z13k_ft_muons_GA05',True)
#df_nttvd, df_ntpart = pkl.load(open('../datafiles/G4ParticleSim/GA04_gridtest.p','rb'))
#mu2e_plot3d_ptrap(df_ntpart.query('xstop<-2500').ix[0:30000],'zstop','xstop','ystop',mode='plotly_html_img')

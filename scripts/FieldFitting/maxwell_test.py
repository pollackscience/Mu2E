#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from mu2e.datafileprod import DataFileMaker
from collections import OrderedDict
import pandas as pd

df_GA05= DataFileMaker('../datafiles/FieldMapsGA05/DSMap',use_pickle = True).data_frame
df_Mau10 = DataFileMaker('../datafiles/Mau10/Standard_Maps/Mu2e_DSMap',use_pickle = True).data_frame

# Make some large rectangle

dfm = df_Mau10.query('5000<Z<5100 and 0<=X<=150 and -25<=Y<=25')[['X','Y','Z','Bx','By','Bz']]
dfg = df_GA05.query('5000<Z<5100 and 0<=X<=150 and -25<=Y<=25')[['X','Y','Z','Bx','By','Bz']]
df_combo = pd.merge(dfm,dfg,on=['X','Y','Z'],suffixes=('m','g'))

# make a bunch of 3x3x3 cubes, use center points on each face to calc flux
# triple nested loop

mau_div = OrderedDict()
mau_curl_xy = OrderedDict()
mau_curl_xz = OrderedDict()
mau_curl_yz = OrderedDict()
ga_div = OrderedDict()
ga_curl_xy = OrderedDict()
ga_curl_xz = OrderedDict()
ga_curl_yz = OrderedDict()
for x in np.sort(dfm.X.unique())[:-2]:
    print(x)
    df_x = df_combo.query('{0}<=X<={1}'.format(x,x+50))
    for y in np.sort(df_x.Y.unique())[:-2]:
        df_xy = df_x.query('{0}<=Y<={1}'.format(y,y+50))
        for z in np.sort(df_xy.Z.unique())[:-2]:
            z_faces       = df_xy[(df_xy.X==x+25) & (df_xy.Y==y+25)]
            face1         = z_faces[df_xy.Z==z]
            bzm_div       = face1.Bzm.values[0]
            bxm_curl_xz   = face1.Bxm.values[0]
            bym_curl_yz   = face1.Bym.values[0]
            bzg_div       = face1.Bzg.values[0]
            bxg_curl_xz   = face1.Bxg.values[0]
            byg_curl_yz   = face1.Byg.values[0]

            face2         = z_faces[df_xy.Z==z+50]
            bzm_div      -= face2.Bzm.values[0]
            bxm_curl_xz  -= face2.Bxm.values[0]
            bym_curl_yz  -= face2.Bym.values[0]
            bzg_div      -= face2.Bzg.values[0]
            bxg_curl_xz  -= face2.Bxg.values[0]
            byg_curl_yz  -= face2.Byg.values[0]

            x_faces       = df_xy[(df_xy.Z==z+25) & (df_xy.Y==y+25)]
            face3         = x_faces[df_xy.X==x]
            bxm_div       = face3.Bxm.values[0]
            bym_curl_xy   = face3.Bym.values[0]
            bzm_curl_xz   = face3.Bzm.values[0]
            bxg_div       = face3.Bxg.values[0]
            byg_curl_xy   = face3.Byg.values[0]
            bzg_curl_xz   = face3.Bzg.values[0]

            face4         = x_faces[df_xy.X==x+50]
            bxm_div      -= face4.Bxm.values[0]
            bym_curl_xy  -= face4.Bym.values[0]
            bzm_curl_xz  -= face4.Bzm.values[0]
            bxg_div      -= face4.Bxg.values[0]
            byg_curl_xy  -= face4.Byg.values[0]
            bzg_curl_xz  -= face4.Bzg.values[0]

            y_faces       = df_xy[(df_xy.Z==z+25) & (df_xy.X==x+25)]
            face5         = y_faces[df_xy.Y==y]
            bym_div       = face5.Bym.values[0]
            bxm_curl_xy   = face5.Bxm.values[0]
            bzm_curl_yz   = face5.Bzm.values[0]
            byg_div       = face5.Byg.values[0]
            bxg_curl_xy   = face5.Bxg.values[0]
            bzg_curl_yz   = face5.Bzg.values[0]

            face6         = y_faces[df_xy.Y==y+50]
            bym_div      -= face6.Bym.values[0]
            bxm_curl_xy  -= face6.Bxm.values[0]
            bzm_curl_yz  -= face6.Bzm.values[0]
            byg_div      -= face6.Byg.values[0]
            bxg_curl_xy  -= face6.Bxg.values[0]
            bzg_curl_yz  -= face6.Bzg.values[0]

            btotm_div = bzm_div+bxm_div+bym_div
            btotm_curl_xy = bxm_curl_xy+bym_curl_xy
            btotm_curl_xz = bxm_curl_xz+bzm_curl_xz
            btotm_curl_yz = bzm_curl_yz+bym_curl_yz
            mau_div['x{0}_y{1}_z{2}'.format(x,y,z)]=btotm_div
            print('x{0}_y{1}_z{2}'.format(x,y,z))
            mau_curl_xy['x{0}_y{1}_z{2}'.format(x,y,z)]=btotm_curl_xy
            mau_curl_xz['x{0}_y{1}_z{2}'.format(x,y,z)]=btotm_curl_xz
            mau_curl_yz['x{0}_y{1}_z{2}'.format(x,y,z)]=btotm_curl_yz

            btotg_div = bzg_div+bxg_div+byg_div
            btotg_curl_xy = bxg_curl_xy+byg_curl_xy
            btotg_curl_xz = bxg_curl_xz+bzg_curl_xz
            btotg_curl_yz = bzg_curl_yz+byg_curl_yz
            ga_div['x{0}_y{1}_z{2}'.format(x,y,z)]=btotg_div
            ga_curl_xy['x{0}_y{1}_z{2}'.format(x,y,z)]=btotg_curl_xy
            ga_curl_xz['x{0}_y{1}_z{2}'.format(x,y,z)]=btotg_curl_xz
            ga_curl_yz['x{0}_y{1}_z{2}'.format(x,y,z)]=btotg_curl_yz

df_maxwell = pd.DataFrame({'mau_div':mau_div, 'ga_div':ga_div,
    'mau_curl_xy':mau_curl_xy, 'ga_curl_xy':ga_curl_xy,
    'mau_curl_xz':mau_curl_xz, 'ga_curl_xz':ga_curl_xz,
    'mau_curl_yz':mau_curl_yz, 'ga_curl_yz':ga_curl_yz},index=list(mau_div.keys()))

df_maxwell['mau_minus_ga_div']=np.abs(df_maxwell.mau_div)-np.abs(df_maxwell.ga_div)
df_maxwell['mau_minus_ga_curl_xy']=np.abs(df_maxwell.mau_curl_xy)-np.abs(df_maxwell.ga_curl_xy)
df_maxwell['mau_minus_ga_curl_xz']=np.abs(df_maxwell.mau_curl_xz)-np.abs(df_maxwell.ga_curl_xz)
df_maxwell['mau_minus_ga_curl_yz']=np.abs(df_maxwell.mau_curl_yz)-np.abs(df_maxwell.ga_curl_yz)



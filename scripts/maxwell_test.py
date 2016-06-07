#! /usr/bin/env python
import numpy as np
from mu2e.datafileprod import DataFileMaker
from collections import OrderedDict
import pandas as pd

df_GA05= DataFileMaker('../datafiles/FieldMapsGA05/DSMap',use_pickle = True).data_frame
df_Mau10 = DataFileMaker('../datafiles/Mau10/Standard_Maps/Mu2e_DSMap',use_pickle = True).data_frame

# Make some large rectangle

dfm = df_Mau10.query('5000<Z<6000 and -800<X<800 and -800<Y<800')[['X','Y','Z','Bx','By','Bz']]
dfg = df_GA05.query('5000<Z<6000 and -800<X<800 and -800<Y<800')[['X','Y','Z','Bx','By','Bz']]
df_combo = pd.merge(dfm,dfg,on=['X','Y','Z'],suffixes=('m','g'))

# make a bunch of 3x3x3 cubes, use center points on each face to calc flux
# triple nested loop

mau_cubes = OrderedDict()
ga_cubes = OrderedDict()
for z in np.sort(dfm.Z.unique())[:-2]:
    print z
    df_z = df_combo.query('{0}<=Z<={1}'.format(z,z+50))
    for x in np.sort(df_z.X.unique())[:-2]:
        df_zx = df_z.query('{0}<=X<={1}'.format(x,x+50))
        for y in np.sort(df_zx.Y.unique())[:-2]:
            z_faces = df_zx[(df_zx.X==x+25) & (df_zx.Y==y+25)]
            face1 = z_faces[df_zx.Z==z]
            bzm   = face1.Bzm.values[0]
            bzg   = face1.Bzg.values[0]
            face2 = z_faces[df_zx.Z==z+50]
            bzm  -= face2.Bzm.values[0]
            bzg  -= face2.Bzg.values[0]

            x_faces = df_zx[(df_zx.Z==z+25) & (df_zx.Y==y+25)]
            face3 = x_faces[df_zx.X==x]
            bxm   = face3.Bxm.values[0]
            bxg   = face3.Bxg.values[0]
            face4 = x_faces[df_zx.X==x+50]
            bxm  -= face4.Bxm.values[0]
            bxg  -= face4.Bxg.values[0]

            y_faces = df_zx[(df_zx.Z==z+25) & (df_zx.X==x+25)]
            face5 = y_faces[df_zx.Y==y]
            bym   = face5.Bym.values[0]
            byg   = face5.Byg.values[0]
            face6 = y_faces[df_zx.Y==y+50]
            bym  -= face6.Bym.values[0]
            byg  -= face6.Byg.values[0]

            btotm = bzm+bxm+bym
            mau_cubes['x{0}_y{1}_z{2}'.format(x,y,z)]=btotm

            btotg = bzg+bxg+byg
            ga_cubes['x{0}_y{1}_z{2}'.format(x,y,z)]=btotg

df_cubes = pd.DataFrame({'mau_cubes':mau_cubes, 'ga_cubes':ga_cubes})
df_cubes['mau_minus_ga']=np.abs(df_cubes.mau_cubes)-np.abs(df_cubes.ga_cubes)



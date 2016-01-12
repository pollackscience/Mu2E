#! /usr/bin/env python

from __future__ import division
import os
import mu2e
import numpy as np
import pandas as pd
import mu2e.src.RowTransformations as rt
import re

"""
This module is for calculating physics-related features based on an input magnetic field
dataframe.  This will either append a new column to the given dataframe, or return a
standalone object.
"""

def calc_scalar_field(df,z_low_cond,z_high_cond,*other_conds):
    """
    Calculate the magnetic scalar potential, given the vector potential, for a 2D slice of
    the magnetic field dataframe. For now, we're looking at a single 2d slice.
    This is for DS only, assuming that the radial center of the tail catcher region is the
    starting point.
    """

    conditions = (z_low_cond,z_high_cond,)+other_conds
    z_val = df.query(' and '.join(conditions)).Z.unique().max()

    #ready the y-values
    y_conds = conditions+('Z=={}'.format(z_val),'X==0')
    print y_conds
    df_y = df.query(' and '.join(y_conds)).sort_values('Y')
    neg_y = df_y[df_y.Y<=0].sort_values('Y', ascending=False).By.cumsum()[::-1]
    pos_y = df_y[df_y.Y>=0].sort_values('Y').By.cumsum()
    new_ys = pd.concat([neg_y,pos_y[1:]],ignore_index=True)
    df_y['By']=np.asarray(pd.concat([neg_y,pos_y[1:]],ignore_index=True))

    #ready the pivot, starting with the tail catcher
    mini_dfs = df.query(' and '.join(conditions)).reindex(columns=['X','Y','Z']).sort_values(['Y','X','Z']).reset_index(drop=True)
    mini_dfs['Scalar']=pd.Series(np.full(len(mini_dfs['Z']),0),index=mini_dfs.index)
    for i,y in enumerate(df_y.Y.unique()):
        piv = df.query(' and '.join(conditions)+' and Y=={}'.format(y)).pivot('X','Z','Bz')
        piv_x = df.query(' and '.join(conditions)+' and Y=={}'.format(y)).pivot('X','Z','Bx')
        piv.sort_index(axis=1,ascending=False, inplace=True)
        piv.sort_index(inplace=True)
        piv_x.sort_index(axis=1,ascending=False, inplace=True)
        piv_x.sort_index(inplace=True)

        #cumsum out from the R=0 in the +/- R directions for the initial column
        neg_piv = piv_x.iloc[:,0].iloc[0:len(piv_x)//2+1]
        neg_piv = neg_piv.sort_index(ascending=False).cumsum().sort_index()
        piv.iloc[:,0].iloc[0:len(piv)//2+1] += neg_piv

        pos_piv = piv_x.iloc[:,0].iloc[len(piv_x)//2:]
        pos_piv = pos_piv.cumsum()
        piv.iloc[:,0].iloc[len(piv)//2:] += pos_piv

        #cumsum the rest of the 2d plane in the Z direction
        piv = piv.cumsum(axis=1)
        piv = piv + df_y[df_y.Y==y].By.values

        #attach the output as a column 'Scalar' to the original df
        mini_df = piv.stack().reset_index().rename(columns={0:'Scalar'})
        mini_df.loc[:,'Y'] = pd.Series(np.full(len(mini_df['Z']),y),index=mini_df.index)
        mini_df = mini_df.sort_values(['Y','X','Z']).reset_index(drop=True)
        mini_dfs.loc[i*len(mini_df.Z):(i+1)*(len(mini_df.Z))-1,'Scalar'] = mini_df.Scalar.values

    #df = pd.merge(df, mini_dfs,  on=['Z','X','Y'], how='outer')
    return mini_dfs






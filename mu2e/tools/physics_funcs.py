#! /usr/bin/env python

from __future__ import division
import os
import mu2e
import numpy as np
import pandas as pd
import mu2e.src.RowTransformations as rt

"""
This module is for calculating physics-related features based on an input magnetic field
dataframe.  This will either append a new column to the given dataframe, or return a
standalone object.
"""

def calc_scalar_field(df,conditions):
    """
    Calculate the magnetic scalar potential, given the vector potential, for a 2D slice of
    the magnetic field dataframe. For now, we're looking at a single 2d slice.
    This is for DS only, assuming that the radial center of the tail catcher region is the
    starting point.
    """

    #ready the pivot, starting with the tail catcher
    piv = df.query(' and '.join(conditions)).pivot('X','Z','Bz')
    piv_x = df.query(' and '.join(conditions)).pivot('X','Z','Bx')
    piv.sort_index(axis=1,ascending=False, inplace=True)
    piv.sort_index(inplace=True)
    piv_x.sort_index(axis=1,ascending=False, inplace=True)
    piv_x.sort_index(inplace=True)

    #cumsum out from the R=0 in the +/- R directions for the initial column
    neg_piv = piv_x.iloc[:,0].iloc[0:len(piv_x)//2+1]
    neg_piv = neg_piv.sort_index(ascending=False).cumsum().sort_index()
    piv.iloc[:,0].iloc[0:len(piv)//2+1] = neg_piv

    pos_piv = piv_x.iloc[:,0].iloc[len(piv_x)//2:]
    pos_piv = pos_piv.cumsum()
    piv.iloc[:,0].iloc[len(piv)//2:] = pos_piv

    #cumsum the rest of the 2d plane in the Z direction
    piv = piv.cumsum(axis=1)

    #attach the output as a column 'Scalar' to the original df
    mini_df = piv.stack().reset_index().rename(columns={0:'Scalar'})
    mini_df.loc[:,'Y'] = pd.Series(np.full(len(mini_df['Z']),0),index=mini_df.index)
    new_df = pd.merge(df, mini_df,  on=['Z','X','Y'])
    return new_df






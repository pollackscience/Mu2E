#! /usr/bin/env python

from mu2e import mu2e_ext_path
from mu2e.dataframeprod import g4root_to_df


# g4root_to_df(mu2e_ext_path+'datafiles/G4ParticleSim/z13k_ft_muons_GA05', True, True)
# g4root_to_df(mu2e_ext_path+'datafiles/G4ParticleSim/low_e_ele', True, True)
g4root_to_df(mu2e_ext_path+'datafiles/G4ParticleSim/z13k_muons_nomat_GA05', True, True)

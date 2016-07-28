#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14, 10
from mu2e.tools.fiteval import get_mag_field_function
from mu2e.tools.fiteval import get_mag_field_function2
from mu2e.src.fiteval_c import FitFunctionMaker
from mu2e.src.fiteval_c2 import FitFunctionMaker2
import mu2e.tools.particletransport as patr
from numba import jit
from time import time

#mag_field_function_ideal1 = get_mag_field_function('Mau10_825mm_v1')
#mag_field_function_bad_m = get_mag_field_function2('Mau10_bad_m_test_req')
ffm = FitFunctionMaker("../mu2e/src/param_825.csv")
ffm2  = FitFunctionMaker2('../mu2e/src/Mau10_800mm_long.csv')

out = ffm.mag_field_function(600,-100,7000,True)
out2 = ffm2.mag_field_function(600,-100,7000,True)
print out[0],out[1],out[2]
print out2[0],out2[1],out2[2]
#print mag_field_function_ideal(0,0,9000,True)
#print mag_field_function_bad_m(0,0,9000,True)



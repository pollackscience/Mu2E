#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14, 10
from mu2e.tools.fiteval import get_mag_field_function
from mu2e.src.fiteval_c import FitFunctionMaker
import mu2e.tools.particletransport as patr
from numba import jit
from time import time

mag_field_function_ideal = get_mag_field_function('Mau10_825mm_v1')
ffm= FitFunctionMaker("../mu2e/src/param_825.csv")

print mag_field_function_ideal(-100,500,9000,True)
out = ffm.mag_field_function(-100,500,9000,True)
print out[0],out[1],out[2]


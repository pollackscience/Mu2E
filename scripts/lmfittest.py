#!/usr/bin/env python
#<examples/doc_basic.py>
from lmfit import minimize, Parameters, Parameter, report_fit
import numpy as np

# create data to be fitted
x = np.linspace(0, 15, 301)
data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
        np.random.normal(size=len(x), scale=0.2) )

# define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """ model decaying sine wave, subtract data"""
    amp = params['amp'].value
    shift = params['shift'].value
    omega = params['omega'].value
    decay = params['decay'].value

    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
    return model - data

# create a set of Parameters
params = Parameters()
params.add('amp',   value= 10,  min=0)
params.add('decay', value= 0.1)
params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
params.add('omega', value= 3.0)


# do fit, here with leastsq model
result = minimize(fcn2min, params, args=(x, data))

# calculate final result
final = data + result.residual

# write error report
report_fit(params)

# try to plot results
try:
    import pylab
    pylab.plot(x, data, 'k+')
    pylab.plot(x, final, 'r')
    pylab.show()
except:
    pass

import numpy as np

from lmfit import Model, Parameters

def my_poly(x, **params):
    val= 0.0
    parnames = sorted(params.keys())
    for i, pname in enumerate(parnames):
        val += params[pname]*x**i
    return val

my_model = Model(my_poly)

# Parameter names and starting values
params = Parameters()
params.add('C00', value=-10)
params.add('C01', value=  5)
params.add('C02', value=  1)
params.add('C03', value=  0)
params.add('C04', value=  0)

x = np.linspace(-20, 20, 101)
y = -30.4 + 7.8*x - 0.5*x*x + 0.03 * x**3 + 0.009*x**4
y = y + np.random.normal(size=len(y), scale=0.2)

out = my_model.fit(y, params, x=x)
print(out.fit_report())

Fitting Data
================================

The module that handles the interface between the underlying fit functions, the hall probe data, and
the `lmfit.py` package is mainly handled by the `fieldfitter.py` module.
Typically, the parametric fitting function has ~400 free parameters, so care has been taken to
optimize the function calls.  The outputs of the fitting procedure are appended to the dataframe of
the input data, to assist in visualization and further analysis.  The fit parameters are saved in a
pickle file, and can be used to recreate the fit, act as initial values for further refinement of
the fit, or used in a standalone plugin to generate field values based on the fit.

An example of a fit to a particular data set is shown below:

.. raw:: html

    <iframe width=900 height=700 frameborder="0" seamless=”seamless” scrolling="no" src="_static/Bz_RZ_Z4200_Z13900_Phi0.00_fit.html"></iframe>

**************************
The `fieldfitter` Module
**************************
.. automodule:: mu2e.fieldfitter
    :members:


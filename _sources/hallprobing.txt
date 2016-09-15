Hall Probe Measurements
=========================

The FMS hardware will only take a sparse set of measurements of the DS and PS during calibration.
These measurements will also be subject to systematic errors that are inherent in all real-world
data taking exercises: there are physical limitations on the probes themselves.  In order to mimic
the placement, granularity, and accuracy of these probes, a dedicated module was produced to convert
a default magnetic field grid into simulated hall probe measurements.  For the most basic case, a
simple subset of the field is selected, assuming no positional or measurement errors.  These results
are then fed to the fitting software for the next stage of the analysis.

**************************
The `hallprober` Module
**************************
.. automodule:: mu2e.hallprober
    :members:


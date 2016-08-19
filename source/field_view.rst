########################
The Mu2E Magnetic Fields
########################

We'll be using a pandas DataFrame object to store the magnetic field information for the DS. The
Mau10 map has been processed such that cylindrical coordinates and a cylindrical vector field were
generated from the initial Cartesian information. We've also recentered the X coordinate, such that
there is an equal number of steps above and below the X=0 line. This may not be equivalent to
recentering the DS on the geometric center, however (we will revisit this issue later, and the
complications that arise because of it).

.. raw:: html

    <iframe width="900" height="800" frameborder="0" scrolling="no" src="_static/Bz_RZ_R1000_3200Z14000_Phi0.00.html"></iframe>

There should be an iframe above.

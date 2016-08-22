Prepping Data
================================

The data used in this mu2e package must be formatted as a valid pandas DataFrame. To accomplish this,
the `dataframeprod.py` module has a useful class for taking a csv-like magnetic field simulation
and converting it to a DF, along with some additional column creation.

There is also a small helper function to convert a ROOT file to a DF, for particle trapping simulation studies.

.. automodule:: mu2e.dataframeprod
    :members:
    :undoc-members:
    :show-inheritance:

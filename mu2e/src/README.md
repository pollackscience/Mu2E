# Magnetic Field Retrieving Function
This directory contains a standalone c++ class for evaluating the magnetic field of the
DS in the Mu2E experiment at any arbitrary position.  This class loads in a plaintext csv that
contains the parameter values for a given field map.  The class has a single function, `get_field`, that takes 3 position arguments (and a boolean).  This function returns a `vector<double>` of size 3, containing the 3 field components.

The function currently works in cartesian coordinates or cylindrical coordinates.

*Input units:*
* Cartesian: mm, mm, mm
* Cylindrical: mm, rads, mm

The coordinate system is centered around the center of the DS, as defined by the Mau and GA maps.  This corresponds to a shift in 'X' of 3896 mm (not 3904).  If using Mu2E coordinates, the input X must be offset by this shift.

*CSV Files:*
* Mau10_800mm_long.csv: ideal map
* Mau10_bad_m_test_req.csv: map based on measurement offset systematics
* Mau10_bad_p_test_req.csv: map based on position offsets (in radial direction)
* Mau10_bad_r_test_req.csv: map based on orientation offset (probe rotated in R-Z plane)

## Compiling
### C++ only
Run `make`.  Note that the default compiler is clang, you may have to edit `Makefile` to use gcc.  GSL libraries must be available.

### Python
Run `make -f Makefile.bpy`.  This uses boost-python to generate a python-compatible shared object.  This step can be **ignored** if you are using C++ only.

## Usage
Please see `main.cc`.

```c++
    FitFunctionMaker* ffm       = new FitFunctionMaker("Mau10_800mm_long.csv");
    FitFunctionMaker* ffm_bad_m = new FitFunctionMaker("Mau10_bad_m_test_req.csv");
    vector<double> bxyz         = ffm->get_field(100,-500,10000, true);
    vector<double> bxyz_bad_m   = ffm_bad_m->get_field(100,-500,10000, true);
```

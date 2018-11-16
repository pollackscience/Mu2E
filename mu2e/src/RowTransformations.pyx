from libc.math cimport sqrt
from libc.math cimport atan2
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport pow
from libc.math cimport M_PI
cimport numpy as np
import numpy as np

#usage: makecython3 RowTransformations.pyx "" "-I`python -c 'import numpy; print(numpy.get_include())'`"


cdef double make_zeta(double z, double theta, double pitch):
    return z - (pitch/(2*M_PI))*theta

cpdef np.ndarray[double] apply_make_zeta(np.ndarray col_z, np.ndarray col_theta, np.float pitch):
    assert (col_z.dtype == np.float and col_theta.dtype == np.float)
    cdef Py_ssize_t i, n = len(col_z)
    assert (len(col_z) == len(col_theta))
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(col_z)):
        res[i] = make_zeta(col_z[i], col_theta[i], pitch)
    return res

cdef double make_r(double x, double y):
    return sqrt(pow(x,2)+pow(y,2))

cpdef np.ndarray[double] apply_make_r(np.ndarray col_x, np.ndarray col_y):
    assert (col_x.dtype == np.float and col_y.dtype == np.float)
    cdef Py_ssize_t i, n = len(col_x)
    assert (len(col_x) == len(col_y))
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(col_x)):
        res[i] = make_r(col_x[i], col_y[i])
    return res

cdef double make_theta(double x, double y):
    return atan2(y,x)

cpdef np.ndarray[double] apply_make_theta(np.ndarray col_x, np.ndarray col_y):
    assert (col_x.dtype == np.float and col_y.dtype == np.float)
    cdef Py_ssize_t i, n = len(col_x)
    assert (len(col_x) == len(col_y))
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(col_x)):
        res[i] = make_theta(col_x[i], col_y[i])
    return res

cdef double make_bphi(double phi, double bx, double by):
    return -bx*sin(phi)+by*cos(phi)

cpdef np.ndarray[double] apply_make_bphi(np.ndarray col_phi, np.ndarray col_bx, np.ndarray col_by):
    assert (col_bx.dtype == np.float and col_by.dtype == np.float and col_phi.dtype == np.float)
    cdef Py_ssize_t i, n = len(col_bx)
    assert (len(col_bx) == len(col_by)==len(col_phi))
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(col_bx)):
        res[i] = make_bphi(col_phi[i], col_bx[i], col_by[i])
    return res

cdef double make_br(double phi, double bx, double by):
    return bx*cos(phi)+by*sin(phi)

cpdef np.ndarray[double] apply_make_br(np.ndarray col_phi, np.ndarray col_bx, np.ndarray col_by):
    assert (col_bx.dtype == np.float and col_by.dtype == np.float and col_phi.dtype == np.float)
    cdef Py_ssize_t i, n = len(col_bx)
    assert (len(col_bx) == len(col_by)==len(col_phi))
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(col_bx)):
        res[i] = make_br(col_phi[i], col_bx[i], col_by[i])
    return res

cdef double make_bphi_wald(double phi, double rho, double bx, double by, double pitch):
    cdef double ca = 1.0
    if rho>0:
        ca = rho/(sqrt(pow(rho,2) + pow(pitch/(2*M_PI),2)))
    return (-bx*sin(phi) + by*cos(phi))/ca

cpdef np.ndarray[double] apply_make_bphi_wald(np.ndarray col_phi, np.ndarray col_rho, np.ndarray
                                              col_bx, np.ndarray col_by, np.float pitch):
    assert (col_bx.dtype == np.float and col_by.dtype == np.float and col_phi.dtype == np.float and
            col_rho.dtype == np.float)
    cdef Py_ssize_t i, n = len(col_bx)
    assert (len(col_bx) == len(col_by) == len(col_phi) == len(col_rho))
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(col_bx)):
        res[i] = make_bphi_wald(col_phi[i], col_rho[i], col_bx[i], col_by[i], pitch)
    return res

cdef double make_bzeta(double phi, double rho, double bx, double by, double bz, double pitch):
    cdef double ta = 0
    if rho>0:
        ta = pitch/(2*M_PI*rho)
    return (bx*sin(phi) - by*cos(phi))*ta + bz

cpdef np.ndarray[double] apply_make_bzeta(np.ndarray col_phi, np.ndarray col_rho, np.ndarray col_bx,
                                          np.ndarray col_by, np.ndarray col_bz, np.float pitch):
    assert (col_bx.dtype == np.float and col_by.dtype == np.float and col_bz.dtype == np.float and
            col_phi.dtype == np.float and col_rho.dtype == np.float)
    cdef Py_ssize_t i, n = len(col_bx)
    assert (len(col_bx) == len(col_by) == len(col_phi) == len(col_rho) == len(col_bz))
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(col_bx)):
        res[i] = make_bzeta(col_phi[i], col_rho[i], col_bx[i], col_by[i], col_bz[i], pitch)
    return res

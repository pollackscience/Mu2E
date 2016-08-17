from libc.math cimport sqrt
from libc.math cimport atan2
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport pow
cimport numpy as np
import numpy as np

#usage: makecython RowTransformations.pyx "" "-I`python -c 'import numpy; print(numpy.get_include())'`"

cdef double make_r(double x, double y):
  return sqrt(pow(x,2)+pow(y,2))

cpdef np.ndarray[double] apply_make_r(np.ndarray col_a, np.ndarray col_b):
  assert (col_a.dtype == np.float and col_b.dtype == np.float)
  cdef Py_ssize_t i, n = len(col_a)
  assert (len(col_a) == len(col_b))
  cdef np.ndarray[double] res = np.empty(n)
  for i in range(len(col_a)):
    res[i] = make_r(col_a[i], col_b[i])
  return res

cdef double make_theta(double x, double y):
  return atan2(y,x)


cpdef np.ndarray[double] apply_make_theta(np.ndarray col_a, np.ndarray col_b):
  assert (col_a.dtype == np.float and col_b.dtype == np.float)
  cdef Py_ssize_t i, n = len(col_a)
  assert (len(col_a) == len(col_b))
  cdef np.ndarray[double] res = np.empty(n)
  for i in range(len(col_a)):
    res[i] = make_theta(col_a[i], col_b[i])
  return res

cdef double make_bphi(double phi, double x, double y):
  return -x*sin(phi)+y*cos(phi)

cpdef np.ndarray[double] apply_make_bphi(np.ndarray col_phi, np.ndarray col_a, np.ndarray col_b):
  assert (col_a.dtype == np.float and col_b.dtype == np.float and col_phi.dtype == np.float)
  cdef Py_ssize_t i, n = len(col_a)
  assert (len(col_a) == len(col_b)==len(col_phi))
  cdef np.ndarray[double] res = np.empty(n)
  for i in range(len(col_a)):
    res[i] = make_bphi(col_phi[i], col_a[i], col_b[i])
  return res

cdef double make_br(double phi, double x, double y):
  return x*cos(phi)+y*sin(phi)

cpdef np.ndarray[double] apply_make_br(np.ndarray col_phi, np.ndarray col_a, np.ndarray col_b):
  assert (col_a.dtype == np.float and col_b.dtype == np.float and col_phi.dtype == np.float)
  cdef Py_ssize_t i, n = len(col_a)
  assert (len(col_a) == len(col_b)==len(col_phi))
  cdef np.ndarray[double] res = np.empty(n)
  for i in range(len(col_a)):
    res[i] = make_br(col_phi[i], col_a[i], col_b[i])
  return res

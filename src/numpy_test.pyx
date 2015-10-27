import numpy as np
cimport numpy as np

def myfunc(np.ndarray[np.float64_t] A):
  cdef int i
  for i in xrange(10):
    A[i]=i
  return A

def main():
  a = np.asarray([0.0]*10)
  b = myfunc(a)
  print b


from libc.math cimport exp, log
import numpy as np
cimport numpy as np
import cython
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )

def logsumexp64( np.ndarray[DTYPE_t, ndim = 1] x ):
    cdef int N = x.shape[0]
    cdef int i
    cdef double xmax
    cdef double tmp = 0
    cdef double output

    xmax = np.max(x)

    for i in xrange(0,N):
        tmp += exp( x[i] - xmax )

    return log(tmp) + xmax

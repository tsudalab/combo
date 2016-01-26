import numpy as np
cimport numpy as np
import cython
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
def diagAB_64( np.ndarray[DTYPE_t, ndim = 2] A, np.ndarray[DTYPE_t, ndim = 2] B ):
    cdef int N = A.shape[0]
    cdef int M = A.shape[1]

    cdef np.ndarray[DTYPE_t, ndim = 1] diagAB = np.zeros( N, dtype=DTYPE )
    cdef int i, j

    for i in xrange( N ):
        for j in xrange( M ):
            diagAB[i] += A[i,j]*B[j,i]

    return diagAB

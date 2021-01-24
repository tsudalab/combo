import numpy as np
cimport numpy as np
import cython
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
def traceAB3_64( np.ndarray[DTYPE_t, ndim = 2] A, np.ndarray[DTYPE_t, ndim = 3] B ):
    cdef int N = A.shape[0]
    cdef int M = A.shape[1]
    cdef int D = B.shape[0]

    cdef np.ndarray[DTYPE_t, ndim = 1] traceAB = np.zeros( D, dtype = DTYPE )
    cdef int i, j, d

    for d in xrange( D ):
        traceAB[d] = 0
        for i in xrange( N ):
            for j in xrange( M ):
                traceAB[d] += A[i,j]*B[d,j,i]
    return traceAB

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
def traceAB2_64( np.ndarray[DTYPE_t, ndim = 2] A, np.ndarray[DTYPE_t, ndim = 2] B ):
    cdef int N = A.shape[0]
    cdef int M = A.shape[1]

    cdef DTYPE_t traceAB = 0
    cdef int i, j, d

    for i in xrange( N ):
        for j in xrange( M ):
            traceAB += A[i,j]*B[j,i]
    return traceAB

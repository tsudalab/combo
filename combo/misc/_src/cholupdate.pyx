from libc.math cimport sqrt
from libc.math cimport abs as cabs
import numpy as np
cimport numpy as np
import cython
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
cdef inline double hypot(double x, double y):
    cdef double t
    x = cabs(x)
    y = cabs(y)
    t = x if x < y else y
    x = x if x > y else y
    t = t/x
    return x*sqrt(1+t*t)

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
def cholupdate64( np.ndarray[DTYPE_t, ndim = 2] L, np.ndarray[DTYPE_t, ndim = 1] x ):
    cdef int N = x.shape[0]
    cdef double c, r, s, eps
    cdef int k, i
    cdef np.ndarray[DTYPE_t, ndim = 1] x2 = x

    for k in xrange( 0, N ):
        r = hypot(L[k,k], x2[k])
        c = r /  L[ k, k ]
        s = x2[ k ] /  L[ k, k ]
        L[ k, k ] = r

        for i in xrange( k+1, N ):
            L[ k, i ] = ( L[ k, i ] + s * x2[ i ] ) /  c
            x2[ i ] = c * x2[ i ] - s  * L[ k, i ]

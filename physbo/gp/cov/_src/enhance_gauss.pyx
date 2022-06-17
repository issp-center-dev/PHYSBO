import numpy as np
cimport numpy as np
import cython
cimport cython
from libc.math cimport exp

DTYPE64 = np.float64
ctypedef np.float64_t DTYPE64_t
DTYPE32 = np.float32
ctypedef np.float32_t DTYPE32_t

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )

def grad_width64( np.ndarray[DTYPE64_t, ndim = 2] X, np.ndarray[DTYPE64_t, ndim = 1] width, np.ndarray[DTYPE64_t, ndim = 2] G ):
    """
    Gradiant along width direction (64bit).

    Parameters
    ----------
    X: numpy.ndarray[numpy.float64_t, ndim = 2]

    width: numpy.ndarray[numpy.float64_t, ndim = 1]
        The grid width
    G: numpy.ndarray[numpy.float64_t, ndim = 2]
        The gram matrix
    Returns
    -------
    numpy.ndarray
    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]

    cdef np.ndarray[DTYPE64_t, ndim = 3] gradG = np.zeros([D,N,N], dtype = DTYPE64 )
    cdef int i, j, d

    for d in xrange( D ):
        for i in xrange( N ):
            for j in xrange( i+1, N ):
                gradG[d,i,j]= ( X[i,d] - X[j,d] )/width[d]
                gradG[d,i,j]= gradG[d,i,j]**2 * G[i,j]
                gradG[ d, j, i ] = gradG[ d, i, j ]
    return gradG

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )

def grad_width32( np.ndarray[DTYPE32_t, ndim = 2] X, np.ndarray[DTYPE32_t, ndim = 1] width, np.ndarray[DTYPE32_t, ndim = 2] G ):
    """

    Gradiant along width direction (32bit).

    Parameters
    ----------
    X: numpy.ndarray[numpy.float32_t, ndim = 2]

    width: numpy.ndarray[numpy.float32_t, ndim = 1]
        The grid width
    G: numpy.ndarray[numpy.float32_t, ndim = 2]
        The gram matrix
    Returns
    -------
    numpy.ndarray
    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]

    cdef np.ndarray[DTYPE32_t, ndim = 3] gradG = np.zeros([D,N,N], dtype = DTYPE32 )
    cdef int i, j, d

    for d in xrange( D ):
        for i in xrange( N ):
            for j in xrange( i+1, N ):
                gradG[d,i,j]= ( X[i,d] - X[j,d] )/width[d]
                gradG[d,i,j]= gradG[d,i,j]**2 * G[i,j]
                gradG[ d, j, i ] = gradG[ d, i, j ]
    return gradG

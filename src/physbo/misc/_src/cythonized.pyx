# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Merged from traceAB.pyx, cholupdate.pyx, diagAB.pyx, and logsumexp.pyx

from libc.math cimport sqrt, exp, log
from libc.math cimport abs as cabs
import numpy as np
cimport numpy as np
import cython
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# ==============================================================================
# Functions from cholupdate.pyx
# ==============================================================================

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

# ==============================================================================
# Functions from diagAB.pyx
# ==============================================================================

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
def diagAB_64( np.ndarray[DTYPE_t, ndim = 2] A, np.ndarray[DTYPE_t, ndim = 2] B ):
    """ Return diagonal part of AB

    Parameters
    ==========
    A: np.ndarray
        NxM matrix
    B: np.ndarray
        MxN matrix

    Returns
    =======
    d: np.ndarray
        Diagonal part of the matrix AB
    """
    cdef int N = A.shape[0]
    cdef int M = A.shape[1]

    cdef np.ndarray[DTYPE_t, ndim = 1] diagAB = np.zeros( N, dtype=DTYPE )
    cdef int i, j

    for i in xrange( N ):
        for j in xrange( M ):
            diagAB[i] += A[i,j]*B[j,i]

    return diagAB

# ==============================================================================
# Functions from logsumexp.pyx
# ==============================================================================

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
def logsumexp64( np.ndarray[DTYPE_t, ndim = 1] x ):
    """ Calculate log(sum(exp(x)))

    Parameters
    ==========
    x: np.ndarray
    """
    cdef int N = x.shape[0]
    cdef int i
    cdef double xmax
    cdef double tmp = 0
    cdef double output

    xmax = np.max(x)

    for i in xrange(0,N):
        tmp += exp( x[i] - xmax )

    return log(tmp) + xmax

# ==============================================================================
# Functions from traceAB.pyx
# ==============================================================================

@cython.boundscheck( False )
@cython.wraparound( False )
@cython.nonecheck( False )
def traceAB3_64( np.ndarray[DTYPE_t, ndim = 2] A, np.ndarray[DTYPE_t, ndim = 3] B ):
    """ Calculates vector of trace of AB[i], where i is the first axis of 3-rank tensor B

    Parameters
    ==========
    A: np.ndarray
        NxM matrix
    B: np.ndarray
        dxMxN tensor

    Returns
    =======
    traceAB: np.ndarray
    """
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
    """ Calculates trace of AB

    Parameters
    ==========
    A: np.ndarray
        NxM matrix
    B: np.ndarray
        MxN matrix

    Returns
    =======
    traceAB: float
        trace of the matrix AB
    """
    cdef int N = A.shape[0]
    cdef int M = A.shape[1]

    cdef DTYPE_t traceAB = 0
    cdef int i, j, d

    for i in xrange( N ):
        for j in xrange( M ):
            traceAB += A[i,j]*B[j,i]
    return traceAB

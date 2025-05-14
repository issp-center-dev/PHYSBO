# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Pure Python version of merged Cython functions
# Converted from traceAB.pyx, cholupdate.pyx, diagAB.pyx, and logsumexp.pyx

import math
import numpy as np

DTYPE = np.float64

# ==============================================================================
# Functions from cholupdate.pyx
# ==============================================================================


def hypot(x, y):
    """Pure Python version of hypot function"""
    x = abs(x)
    y = abs(y)
    t = min(x, y)
    x = max(x, y)
    if x == 0:
        return 0
    t = t / x
    return x * math.sqrt(1 + t * t)


def cholupdate64(L, x):
    """Pure Python version of Cholesky update"""
    N = x.shape[0]
    x2 = x.copy()

    for k in range(N):
        r = hypot(L[k, k], x2[k])
        c = r / L[k, k]
        s = x2[k] / L[k, k]
        L[k, k] = r

        for i in range(k + 1, N):
            L[k, i] = (L[k, i] + s * x2[i]) / c
            x2[i] = c * x2[i] - s * L[k, i]


# ==============================================================================
# Functions from diagAB.pyx
# ==============================================================================


def diagAB_64(A, B):
    """Return diagonal part of AB

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
    N = A.shape[0]
    M = A.shape[1]

    diagAB = np.zeros(N, dtype=DTYPE)

    for i in range(N):
        for j in range(M):
            diagAB[i] += A[i, j] * B[j, i]

    return diagAB


# ==============================================================================
# Functions from logsumexp.pyx
# ==============================================================================


def logsumexp64(x):
    """Calculate log(sum(exp(x)))

    Parameters
    ==========
    x: np.ndarray
    """
    N = x.shape[0]
    tmp = 0.0

    xmax = np.max(x)

    for i in range(N):
        tmp += math.exp(x[i] - xmax)

    return math.log(tmp) + xmax


# ==============================================================================
# Functions from traceAB.pyx
# ==============================================================================


def traceAB3_64(A, B):
    """Calculates vector of trace of AB[i], where i is the first axis of 3-rank tensor B

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
    N = A.shape[0]
    M = A.shape[1]
    D = B.shape[0]

    traceAB = np.zeros(D, dtype=DTYPE)

    for d in range(D):
        for i in range(N):
            for j in range(M):
                traceAB[d] += A[i, j] * B[d, j, i]
    return traceAB


def traceAB2_64(A, B):
    """Calculates trace of AB

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
    N = A.shape[0]
    M = A.shape[1]

    traceAB = 0.0

    for i in range(N):
        for j in range(M):
            traceAB += A[i, j] * B[j, i]
    return traceAB

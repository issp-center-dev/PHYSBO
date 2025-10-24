# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

def diagAB(A, B):
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

    return np.einsum("ij,ji->i", A, B)


def logsumexp(x):
    """Calculate log(sum(exp(x)))

    Parameters
    ==========
    x: np.ndarray
    """
    xmax = np.max(x)
    return np.log(np.sum(np.exp(x - xmax))) + xmax


def traceAB3(A, B):
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

    return np.einsum("ij,kji->k", A, B)


def traceAB2(A, B):
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

    return np.einsum("ij,ji->", A, B)

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

DTYPE64 = np.float64
DTYPE32 = np.float32


def grad_width64(X, width, G):
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
    N = X.shape[0]
    D = X.shape[1]

    gradG = np.zeros([D, N, N], dtype=DTYPE64)

    for d in range(D):
        for i in range(N):
            for j in range(i + 1, N):
                gradG[d, i, j] = (X[i, d] - X[j, d]) / width[d]
                gradG[d, i, j] = gradG[d, i, j] ** 2 * G[i, j]
                gradG[d, j, i] = gradG[d, i, j]
    return gradG


def grad_width32(X, width, G):
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
    N = X.shape[0]
    D = X.shape[1]

    gradG = np.zeros([D, N, N], dtype=DTYPE32)

    for d in range(D):
        for i in range(N):
            for j in range(i + 1, N):
                gradG[d, i, j] = (X[i, d] - X[j, d]) / width[d]
                gradG[d, i, j] = gradG[d, i, j] ** 2 * G[i, j]
                gradG[d, j, i] = gradG[d, i, j]
    return gradG

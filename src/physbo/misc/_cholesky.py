# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import scipy


def _cholupdate_fastupdate(U, x):
    """Cholesky update using fast update method."""

    N = x.shape[0]
    x2 = x.copy()

    for k in range(N):
        r = np.hypot(U[k, k], x2[k])
        c = r / U[k, k]
        s = x2[k] / U[k, k]
        U[k, k] = r

        U[k, k + 1 :] += s * x2[k + 1 :]
        U[k, k + 1 :] /= c
        x2[k + 1 :] *= c
        x2[k + 1 :] -= s * U[k, k + 1 :]


def _cholupdate_direct(U, x):
    """Cholesky update just by calling scipy.linalg.cholesky directly."""

    A = np.dot(U.T, U) + np.outer(x, x)
    U[:] = scipy.linalg.cholesky(A, check_finite=False)


def cholupdate(U, x, matrix_size_threshold=500):
    """Cholesky update

    This calculates the Cholesky decomposition of A = U.T @ U + x @ x.T.

    Parameters
    ----------
    U: numpy.ndarray
        Upper triangular matrix of the Cholesky decomposition of the original matrix.
        U is updated to U' of A in place.
    x: numpy.ndarray
        Vector to be added to the original matrix.
    matrix_size_threshold: int
        If the size of the matrix is larger than this threshold, the fast update method is used.
        Otherwise, the direct method is used.
        Default is 500.
    """

    N = U.shape[0]
    if N > matrix_size_threshold:
        _cholupdate_fastupdate(U, x)
    else:
        _cholupdate_direct(U, x)

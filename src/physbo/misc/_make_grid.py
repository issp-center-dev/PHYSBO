# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations
import itertools

import numpy as np


def make_grid(
    min_X: list[float] | np.ndarray,
    max_X: list[float] | np.ndarray,
    num_X: int | list[int] | np.ndarray,
) -> np.ndarray:
    """Make a grid of points in the search space.

    Arguments
    =========
    min_X: np.ndarray | list[float] | float
        Minimum value of search space for each dimension
    max_X: np.ndarray | list[float] | float
        Maximum value of search space for each dimension
    num_X: int | list[int] | np.ndarray
        Number of points in each dimension

    Returns
    =======
    np.ndarray
        The grid of points in the search space
        The output is a numpy array of shape (N, d), where N is the number of points and d is the dimension of the search space

    Raises
    ======
    ValueError
        If min_X and max_X have different number of dimensions
        If num_X has different number of dimensions from min_X and max_X
    """

    if isinstance(min_X, list):
        min_X = np.array(min_X)
    if isinstance(max_X, list):
        max_X = np.array(max_X)

    if min_X.ndim != 1:
        raise ValueError(
            f"ERROR: min_X must be a 1D array, but got {min_X.ndim}D array"
        )
    if max_X.ndim != 1:
        raise ValueError(
            f"ERROR: max_X must be a 1D array, but got {max_X.ndim}D array"
        )

    if min_X.shape[0] != max_X.shape[0]:
        raise ValueError(
            f"ERROR: min_X and max_X must have the same number of dimensions, but got {min_X.shape[0]} and {max_X.shape[0]}"
        )

    d = min_X.shape[0]

    if isinstance(num_X, int):
        num_X = np.full(d, num_X)
    elif isinstance(num_X, list):
        num_X = np.array(num_X)

    if num_X.ndim != 1:
        raise ValueError(
            f"ERROR: num_X must be a 1D array, but got {num_X.ndim}D array"
        )
    if num_X.shape[0] != d:
        raise ValueError(
            f"ERROR: num_X must have the same number of dimensions as min_X and max_X, but got {num_X.shape[0]} and {d}"
        )

    ls = [np.linspace(min_X[i], max_X[i], num_X[i]) for i in range(d)]

    N = np.prod(num_X)
    X = np.zeros((N, d))
    for i, x in enumerate(itertools.product(*ls)):
        X[i, :] = x

    return X

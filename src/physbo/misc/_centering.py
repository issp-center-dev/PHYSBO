# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Centering / normalization helpers for input and target arrays."""

import numpy as np


def centering(X):
    """
    Normalize the mean and standard deviation along the each column of X to 0 and 1, respectively

    Parameters
    ----------
    X: numpy array
        N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.

    Returns
    -------
    X_normalized: numpy array
        normalized N x d dimensional matrix.
    """
    stdX = np.std(X, 0)
    index = np.where(stdX != 0)
    X_normalized = (X[:, index[0]] - np.mean(X[:, index[0]], 0)) / stdX[index[0]]
    return X_normalized


def min_max_scaling(X, low=0.0, high=1.0):
    """
    Normalize the minimum and maximum along the each column of X to low and high, respectively

    if all the elements in a column are the same, return (low + high) / 2.0 for the column.

    Parameters
    ----------
    X: numpy array
        N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
    low: float
        Minimum value of the normalized result. Default is 0.0
    high: float
        Maximum value of the normalized result. Default is 1.0
    Returns
    -------
    X_normalized: numpy array
        Normalized N x d dimensional matrix.
    """
    min_vals, max_vals = np.min(X, 0), np.max(X, 0)
    diff = max_vals - min_vals
    d = high - low
    center = 0.5 * (low + high)
    index = np.where(diff != 0)
    res = np.ones_like(X) * center
    res[:, index] = (X[:, index] - min_vals[index]) / diff[index] * d + low
    return res

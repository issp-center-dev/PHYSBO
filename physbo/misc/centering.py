# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np


# TODO: move to base_model after base_model is implemented
def get_permutation_importance(
    model, X, t, n_perm: int, comm=None, split_features_parallel=False
):
    """
    Calculating permutation importance of model

    Parameters
    ==========
    X: numpy.ndarray
        inputs
    t: numpy.ndarray
        target (label)
    n_perm: int
        number of permutations
    comm: MPI.Comm
        MPI communicator

    Returns
    =======
    numpy.ndarray
        importance_mean
    numpy.ndarray
        importance_std
    """

    n_features = X.shape[1]

    if n_perm < 1:
        print("WARNING: n_perm is less than 1. Return 0.")
        return np.zeros(n_features), np.zeros(n_features)

    model.prepare(X, t)
    fmean = model.get_post_fmean(X, X)
    MSE_base = np.mean((fmean - t) ** 2)

    if comm is None:
        mpisize = 1
        mpirank = 0
    else:
        mpisize = comm.Get_size()
        mpirank = comm.Get_rank()

    features = np.arange(n_features)
    if split_features_parallel:
        features = np.array_split(features, mpisize)[mpirank]
    n_features_local = len(features)

    scores = np.zeros(n_features_local)
    scores_2 = np.zeros(n_features_local)

    for i_feature in features:
        X_perm = X.copy()
        for i_perm in range(n_perm):
            X_perm[:, i_feature] = np.random.permutation(X_perm[:, i_feature])
            fmean = model.get_post_fmean(X, X_perm)
            s = np.mean((fmean - t) ** 2) - MSE_base
            scores[i_feature] += s
            scores_2[i_feature] += s**2

    if comm is not None and mpisize > 1:
        res = np.zeros(n_features_local)
        res_2 = np.zeros(n_features_local)
        comm.Allreduce(scores, res)  # default of op is MPI.SUM
        comm.Allreduce(scores_2, res_2)
        scores[:] = res
        scores_2[:] = res_2

    if not split_features_parallel:
        n_perm *= mpisize

    importance_mean = scores / n_perm
    importance_std = np.sqrt(scores_2 / (n_perm - 1) - importance_mean**2)

    return importance_mean, importance_std

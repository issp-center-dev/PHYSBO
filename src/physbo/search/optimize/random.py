# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np


class Optimizer:
    def __init__(self, min_X, max_X, nsamples):
        min_X = np.array(min_X)
        max_X = np.array(max_X)
        if min_X.ndim > 1 or max_X.ndim > 1:
            raise ValueError("min_X and max_X must be 1-dimensional")
        min_X = min_X.reshape(-1)
        max_X = max_X.reshape(-1)

        if min_X.size != max_X.size:
            raise ValueError("min_X and max_X must have the same number of dimensions")

        self.dim = min_X.size
        self.min_X = min_X
        self.max_X = max_X

        if nsamples < 1:
            raise ValueError("nsamples must be greater than 0")
        self.nsamples = nsamples

    def __call__(self, fn, mpicomm=None):
        if mpicomm is not None:
            mpisize = mpicomm.Get_size()
            mpirank = mpicomm.Get_rank()
        else:
            mpisize = 1
            mpirank = 0

        nsamples_local = self.nsamples // mpisize
        if mpirank < self.nsamples % mpisize:
            nsamples_local += 1

        result_fx = -np.inf
        result_x = None
        for i in range(nsamples_local):
            x = self.min_X + (self.max_X - self.min_X) * np.random.rand(self.dim)
            fx = fn(x)
            if fx > result_fx:
                result_fx = fx
                result_x = x
        if mpisize > 1:
            result_fx_all = np.zeros(mpisize)
            mpicomm.Allgather(np.array([result_fx]), result_fx_all)
            best_rank = np.argmax(result_fx_all)
            mpicomm.Bcast(result_x, root=best_rank)

        return result_x

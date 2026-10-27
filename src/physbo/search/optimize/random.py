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
            # gathered as objects, so that scores of any type (e.g. float32,
            # or integers beyond 2**53) are compared as they are
            result_fx_all = mpicomm.allgather(result_fx)
            if all(fx == -np.inf for fx in result_fx_all):
                # no process kept a sample (all scores nan or -inf): the same
                # result as a single process, on every process
                return None
            # the first of equally good processes, like np.argmax
            best_rank = max(range(mpisize), key=lambda rank: result_fx_all[rank])
            # a process that kept no sample (it had none, or only scores of
            # nan or -inf) still needs a buffer to receive the point
            if result_x is None:
                # of the type of the samples, as on the sending process
                result_x = np.empty(
                    self.dim, dtype=np.result_type(self.min_X, self.max_X, float)
                )
            mpicomm.Bcast(result_x, root=best_rank)

        return result_x

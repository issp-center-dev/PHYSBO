# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import warnings

import numpy as np


class Optimizer:
    """Random search for the maximum of a function within bounds.

    Parameters
    ----------
    min_X, max_X : array_like
        Lower and upper bounds of each dimension.
    nsamples : int
        The number of points sampled uniformly within the bounds, divided
        among the processes in a parallel run.
    """

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
        """
        Return the sample with the largest value of ``fn``.

        Parameters
        ----------
        fn : callable
            The function to maximize. Takes a numpy array of shape (dim,) and
            returns a score; scores of ``nan`` are ignored.
        mpicomm : MPI.Comm, optional
            MPI communicator; the samples are divided among its processes,
            and every process returns the same point.

        Returns
        -------
        numpy.ndarray
            The first sample with the largest score, of shape (dim,); in a
            parallel run, of the process of the lowest rank among equally good
            ones. If no sample has a score larger than ``-inf`` (all scores
            are ``nan`` or ``-inf``), the first sample (of rank 0) is returned
            and a ``RuntimeWarning`` is issued on every process.
        """
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
            elif result_x is None:
                # kept in case no score is larger than -inf
                result_x = x
        if mpisize > 1:
            # gathered as objects, so that scores of any type (e.g. float32,
            # or integers beyond 2**53) are compared as they are
            result_fx_all = mpicomm.allgather(result_fx)
            # the first of equally good processes, like np.argmax; with no
            # score larger than -inf anywhere, the first process, which has
            # at least one sample
            best_rank = max(range(mpisize), key=lambda rank: result_fx_all[rank])
            result_fx = result_fx_all[best_rank]
            # a process without samples still needs a buffer to receive the
            # point
            if result_x is None:
                # of the type of the samples, as on the sending process
                result_x = np.empty(
                    self.dim, dtype=np.result_type(self.min_X, self.max_X, float)
                )
            mpicomm.Bcast(result_x, root=best_rank)

        if result_fx == -np.inf:
            # the same on every process, as result_fx is the best of all
            warnings.warn(
                "No sample of the acquisition function has a score larger than"
                " -inf (all scores are nan or -inf); the first sample is"
                " returned.",
                RuntimeWarning,
            )
        return result_x

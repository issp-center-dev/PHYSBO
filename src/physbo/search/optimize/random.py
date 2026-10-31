# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import warnings

import numpy as np

from ... import _mpi
from ..._rng import get_rng
from ..utility import sample_feasible


class Optimizer:
    """Random search for the maximum of a function within bounds.

    Parameters
    ----------
    min_X, max_X : array_like
        Lower and upper bounds of each dimension.
    nsamples : int
        The number of points sampled uniformly within the bounds, divided
        among the processes in a parallel run.
    rng : int, numpy.random.SeedSequence, numpy.random.Generator or None, optional
        Random number generator for the samples; None (default) uses the
        global ``numpy.random`` state. In a parallel run every process
        draws its own samples from its own generator.
    constraint : callable, optional
        A function from an array of shape (N, d) to a boolean array of N
        elements (True for feasible points). If given, the samples are
        feasible points drawn by rejection sampling (on every process for
        its own share of the samples).
    max_trials_per_point : int, optional
        The rejection sampling of N feasible points is given up after
        ``max_trials_per_point * N`` draws. Default is the class attribute
        of the same name (10000).
    """

    #: Rejection sampling of N feasible points is given up after max_trials_per_point * N samples.
    max_trials_per_point = 10000

    def __init__(
        self,
        min_X,
        max_X,
        nsamples,
        rng=None,
        constraint=None,
        max_trials_per_point=None,
    ):
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
        self.rng = get_rng(rng)
        self.constraint = constraint
        if max_trials_per_point is not None:
            self.max_trials_per_point = max_trials_per_point

    def _sample_uniform(self, N):
        """Draw N points uniformly from the box [min_X, max_X]."""
        return self.min_X + (self.max_X - self.min_X) * self.rng.random((N, self.dim))

    def __call__(self, fn, mpicomm=None):
        """
        Return the sample with the largest value of ``fn``.

        Parameters
        ----------
        fn : callable
            The function to maximize. Takes a numpy array of shape (dim,) and
            returns a score; scores of ``nan`` are ignored while some sample
            has a score larger than ``-inf`` (see Returns for the case
            without one).
        mpicomm : MPI.Comm, optional
            MPI communicator; the samples are divided among its processes,
            and every process returns the same point. An exception raised by
            ``fn`` on some processes is raised on every process.

        Returns
        -------
        numpy.ndarray or None
            The first sample with the largest score, of shape (dim,); in a
            parallel run, of the process of the lowest rank among equally good
            ones. If no sample has a score larger than ``-inf`` (all scores
            are ``nan`` or ``-inf``), the first sample (of the lowest rank
            with a sample) is returned and a ``RuntimeWarning`` is issued on
            every process. None, on every process, only with a
            ``constraint`` for which no process found a feasible sample
            within about ``max_trials_per_point`` draws per sample of its
            share.
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

        def sample():
            if self.constraint is None:
                samples = None
                nlocal = nsamples_local
            else:
                # feasible points are drawn beforehand
                samples = sample_feasible(
                    self._sample_uniform,
                    nsamples_local,
                    self.constraint,
                    max_trials_per_point=self.max_trials_per_point,
                )
                nlocal = 0 if samples is None else nsamples_local
            result_fx = -np.inf
            result_x = None
            for i in range(nlocal):
                if samples is None:
                    x = self._sample_uniform(1)[0]
                else:
                    x = samples[i]
                fx = fn(x)
                if fx > result_fx:
                    result_fx = fx
                    result_x = x
                elif result_x is None:
                    # kept in case no score is larger than -inf
                    result_x = x
            return result_fx, result_x

        result, error = _mpi.call(mpicomm if mpisize > 1 else None, sample)
        result_fx, result_x = (None, None) if error is not None else result
        if mpisize > 1:
            # gathered as objects, so that scores of any type (e.g. float32,
            # or integers beyond 2**53) are compared as they are; with the
            # scores, whether the process has a sample at all (it may have
            # none with a constraint)
            results_all = _mpi.allgather(
                mpicomm, (result_fx, result_x is not None), error
            )
            ranks_with_sample = [
                rank for rank in range(mpisize) if results_all[rank][1]
            ]
            if len(ranks_with_sample) == 0:
                return None
            # the first of equally good processes, like np.argmax; with no
            # score larger than -inf anywhere, the first process with a sample
            best_rank = max(ranks_with_sample, key=lambda rank: results_all[rank][0])
            result_fx = results_all[best_rank][0]
            # a process without samples still needs a buffer to receive the
            # point
            if result_x is None:
                # of the type of the samples, as on the sending process
                result_x = np.empty(
                    self.dim, dtype=np.result_type(self.min_X, self.max_X, float)
                )
            mpicomm.Bcast(result_x, root=best_rank)
        elif result_x is None:
            # no feasible sample (only with a constraint)
            return None

        if result_fx == -np.inf:
            # the same on every process, as result_fx is the best of all
            warnings.warn(
                "No sample of the acquisition function has a score larger than"
                " -inf (all scores are nan or -inf); the first sample is"
                " returned.",
                RuntimeWarning,
            )
        return result_x

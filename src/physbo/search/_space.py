# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The search spaces of the policies.

A policy searches either a list of candidates (``_DiscreteSpace``) or a box
with an optional constraint (``_RangeSpace``). The object holds what belongs
to the space (the candidates and those not searched yet; the bounds and the
constraint) and what is done with it: drawing random points, taking the
best candidate, maximizing a function over the box. The policies of the
three kinds of objective (single, multi, unified) of one search space share
it by delegation: a policy has one as ``policy.space``.

A space does not keep the MPI communicator or the random number generator
of its policy; the methods that need them take them as arguments
(``mpicomm`` is the communicator of the policy as it is: None, or a
communicator of one or more processes).
"""

import itertools
import warnings

import numpy as np

from . import utility
from .optimize.random import Optimizer as RandomOptimizer
from .. import _mpi


def _num_procs(mpicomm):
    return 1 if mpicomm is None else mpicomm.size


def _space_attribute(name, doc):
    """A property of a policy that reads and writes the attribute ``name``
    of its search space: the public attributes of the search space of a
    policy (``policy.test``, ``policy.min_X``, ...) stay where they were."""

    def space_of(policy):
        try:
            return policy.space
        except AttributeError:
            raise AttributeError(
                f"'{type(policy).__name__}' object has no attribute '{name}' "
                "yet: it belongs to the search space of the policy, which the "
                "constructor of the policy class makes. In a subclass, call "
                "super().__init__() before reading or setting it."
            ) from None

    def fget(self):
        return getattr(space_of(self), name)

    def fset(self, value):
        setattr(space_of(self), name, value)

    return property(fget, fset, doc=doc)


class _SearchSpace:
    """What the policies ask from a search space of either kind."""

    def split(self, mpisize, mpirank):
        """Keep the share of this process of the candidates. Nothing to do
        unless the search space is a list of candidates."""

    def random(self, N, rng, mpicomm):
        """``N`` points drawn at random, the same on every process."""
        raise NotImplementedError

    def local_state(self):
        """The part of the state that differs between the processes, for a
        checkpoint."""
        return {}

    def restore_local_state(self, state):
        """Take back what ``local_state`` returned."""


class _DiscreteSpace(_SearchSpace):
    """A list of candidates.

    Attributes
    ----------
    test: physbo.Variable
        The candidates; each row of ``test.X`` is the feature vector of one.
        With a Bayesian linear model, ``test.Z`` holds the values of its
        basis functions.
    actions: numpy.ndarray
        The indices of the candidates not searched yet. Under MPI, each
        process has its share of them.
    """

    def __init__(self, test):
        self.test = test
        self.actions = np.arange(0, self.test.X.shape[0])

    def split(self, mpisize, mpirank):
        self.actions = np.array_split(self.actions, mpisize)[mpirank]

    def remove(self, local_index):
        """Remove the candidates at the positions ``local_index`` of
        ``actions`` (of this process)."""
        self.actions = np.delete(self.actions, local_index)

    def exclude(self, action):
        """Remove the candidates ``action`` (indices of candidates) from
        those not searched yet; the ones this process does not have are
        skipped."""
        if len(self.actions) > 0:
            local_index = np.searchsorted(self.actions, action)
            local_index = local_index[
                np.take(self.actions, local_index, mode="clip") == action
            ]
            self.remove(local_index)

    def random(self, N, rng, mpicomm):
        """
        Getting indexes of actions randomly, and removing them from the
        candidates not searched yet.

        Parameters
        ----------
        N: int
            Total number of search candidates.
        rng:
            The random number generator of the policy.
        mpicomm: MPI.Comm or None

        Returns
        -------
        action: numpy.ndarray
            Indexes of actions selected randomly from search candidates
            (all of them if fewer than ``N`` are left).
        """
        if _num_procs(mpicomm) == 1:
            n = len(self.actions)
            if n <= N:
                index = np.arange(0, n)
            else:
                index = rng.choice(len(self.actions), N, replace=False)
            action = self.actions[index]
            self.remove(index)
        else:
            mpisize = mpicomm.size
            nactions = mpicomm.gather(len(self.actions), root=0)
            local_indices = [[] for _ in range(mpisize)]
            if mpicomm.rank == 0:
                hi = np.add.accumulate(nactions)
                lo = np.roll(hi, 1)
                lo[0] = 0
                if hi[-1] <= N:
                    index = np.arange(0, hi[-1])
                else:
                    index = rng.choice(hi[-1], N, replace=False)
                ranks = np.searchsorted(hi, index, side="right")
                for r, i in zip(ranks, index):
                    local_indices[r].append(i - lo[r])
            local_indices = mpicomm.scatter(local_indices, root=0)
            local_actions = self.actions[local_indices]
            self.remove(local_indices)
            action = mpicomm.allgather(local_actions)
            action = itertools.chain.from_iterable(action)
            action = np.array(list(action))
        return action

    def find_champion(self, f, error, mpicomm):
        """The best candidate over all processes.

        ``f`` are the scores of the candidates not searched yet of this
        process. ``error`` is an exception raised on this process while
        computing them (see :func:`physbo._mpi.call`), raised then on every
        process.

        Returns
        -------
        champion: int
            The index of the best candidate (-1 if no candidate is left).
        local_champion: int
            The index of the best candidate of this process.
        local_index: int
            The position of ``local_champion`` in ``actions``, to be given
            to :meth:`remove` by the process that has the champion.
        """
        mpisize = _num_procs(mpicomm)
        if error is not None and mpisize == 1:
            raise error
        if error is not None:
            local_fmax = None
            local_index = None
            local_champion = None
        elif len(f) == 0:
            local_fmax = -float("inf")
            local_index = -1
            local_champion = -1
        else:
            local_fmax = np.max(f)
            local_index = np.argmax(f)
            local_champion = self.actions[local_index]
        if mpisize == 1:
            champion = local_champion
        else:
            local_champions = _mpi.allgather(mpicomm, local_champion, error)
            local_fs = mpicomm.allgather(local_fmax)
            champion_rank = np.argmax(local_fs)
            champion = local_champions[champion_rank]
        return champion, local_champion, local_index

    def local_state(self):
        return {"actions": self.actions}

    def restore_local_state(self, state):
        self.actions = state["actions"]


class _RangeSpace(_SearchSpace):
    """A box ``[min_X, max_X]``, with an optional constraint.

    Attributes
    ----------
    min_X, max_X: numpy.ndarray
        The bounds of each dimension.
    L_X: numpy.ndarray
        ``max_X - min_X``.
    dim: int
        The number of dimensions.
    constraint: callable or None
        From an array of shape (N, d) to a boolean array of N elements
        (True for feasible points).
    """

    def __init__(self, min_X, max_X, constraint=None):
        if min_X is None or max_X is None:
            raise ValueError("min_X and max_X must be specified")
        self.min_X = np.array(min_X)
        self.max_X = np.array(max_X)
        self.dim = self.min_X.shape[0]
        assert self.dim == self.max_X.shape[0], (
            "The dimension of min_X and max_X must be the same"
        )
        assert np.all(self.min_X < self.max_X), (
            "min_X must be less than max_X for each dimension"
        )
        self.L_X = self.max_X - self.min_X
        self.constraint = constraint

    def sample_uniform(self, N, rng):
        """Draw N points uniformly from the box [min_X, max_X]."""
        return rng.random((N, self.dim)) * self.L_X.reshape(1, -1) + self.min_X.reshape(
            1, -1
        )

    def random(self, N, rng, mpicomm, max_trials_per_point=10000):
        """
        Drawing feasible points uniformly.

        Parameters
        ----------
        N: int
            The number of points.
        rng:
            The random number generator of the policy.
        mpicomm: MPI.Comm or None
        max_trials_per_point: int
            Rejection sampling under the constraint is given up after
            ``max_trials_per_point * N`` samples.

        Returns
        -------
        action: numpy.ndarray
            ``N`` points of the box that satisfy the constraint, of shape
            (N, dim).
        """
        mpisize = _num_procs(mpicomm)
        if self.constraint is None:
            action = self.sample_uniform(N, rng)
            if mpisize > 1:
                mpicomm.Bcast(action, root=0)
            return action

        # Only rank 0 samples, since the number of trials depends on the random
        # numbers; the other ranks then draw the same number of points, so that
        # the generators of all the ranks stay in step as without a constraint.
        num_drawn = 0

        def draw(n):
            nonlocal num_drawn
            num_drawn += n
            return self.sample_uniform(n, rng)

        comm = mpicomm if mpisize > 1 else None
        mpirank = 0 if mpicomm is None else mpicomm.rank
        action, error = None, None
        if comm is None or mpirank == 0:
            # an exception of the constraint is raised on every rank below
            action, error = _mpi.call(
                comm,
                utility.sample_feasible,
                draw,
                N,
                self.constraint,
                max_trials_per_point=max_trials_per_point,
            )
        if comm is not None:
            # all the ranks must raise the error together
            found, num_drawn = _mpi.bcast(
                comm, (action is not None, num_drawn), error, root=0
            )
            if found:
                if mpirank != 0:
                    self.sample_uniform(num_drawn, rng)
                    action = np.zeros((N, self.dim))
                comm.Bcast(action, root=0)
            else:
                action = None
        if action is None:
            raise RuntimeError(
                "ERROR: feasible points were not found in {} random samples. "
                "The feasible region may be empty or too small "
                "(equality constraints are not supported).".format(
                    max_trials_per_point * N
                )
            )
        return action

    def default_optimizer(self, rng, max_trials_per_point=10000):
        """Optimizer of the acquisition function used when none is given."""
        return RandomOptimizer(
            min_X=self.min_X,
            max_X=self.max_X,
            nsamples=1000,
            rng=rng,
            constraint=self.constraint,
            max_trials_per_point=max_trials_per_point,
        )

    def constrained(self, fn):
        """Make the acquisition function -inf at infeasible points."""
        if self.constraint is None:
            return fn

        def wrapped(x):
            x = np.asarray(x)
            if utility.is_feasible(self.constraint, x.reshape(1, -1))[0]:
                return fn(x)
            return -np.inf

        return wrapped

    def argmax(self, fn, optimizer, mpicomm, random_point):
        """
        Maximize the acquisition function under the constraint.

        Parameters
        ----------
        fn: callable
            Acquisition function, from a point of shape (dim,) to float.
        optimizer: Optimizer object
            Called as optimizer(fn, mpicomm=...).
        mpicomm: MPI.Comm or None
        random_point: callable
            Called without arguments for a feasible random point, the same
            on every process, when the optimizer did not return a feasible
            one.

        Returns
        -------
        X: numpy.ndarray
            The point to be searched next.

        Notes
        -----
        Infeasible points are hidden from the optimizer by -inf, but the optimizer can
        still fail to return a feasible point (no feasible sample, or a local search
        started from an infeasible point). The returned point is therefore checked,
        and replaced with a feasible random point if it is None, not finite or not
        feasible. A point of the wrong dimension is an error of the optimizer and
        raises ValueError, as without a constraint.
        """
        if isinstance(optimizer, RandomOptimizer) and (
            # == rather than "is": a bound method (fn.constraint) is a new
            # object at every attribute access
            optimizer.constraint == self.constraint
        ):
            # the optimizer draws feasible samples only; no need to judge
            # them one by one again
            X = optimizer(fn, mpicomm=mpicomm)
        else:
            X = optimizer(self.constrained(fn), mpicomm=mpicomm)
        if self.constraint is None:
            return X

        def check():
            if X is None:
                return False
            x = np.asarray(X, dtype=float).reshape(-1)
            if x.shape[0] != self.dim:
                raise ValueError(
                    f"ERROR: the optimizer returned a point of {x.shape[0]} "
                    f"dimension(s), but the search space has {self.dim}"
                )
            x = x.reshape(1, -1)
            return bool(
                np.all(np.isfinite(x)) and utility.is_feasible(self.constraint, x)[0]
            )

        # the verdict of rank 0 decides, so that all the ranks take the same
        # branch; an exception while judging is raised on every rank
        comm = mpicomm if _num_procs(mpicomm) > 1 else None
        mpirank = 0 if mpicomm is None else mpicomm.rank
        if comm is None or mpirank == 0:
            is_ok, error = _mpi.call(comm, check)
        else:
            is_ok, error = None, None
        if comm is not None:
            is_ok = _mpi.bcast(comm, is_ok, error, root=0)
        if is_ok:
            return X
        if mpirank == 0:
            warnings.warn(
                "The optimizer of the acquisition function did not return a feasible point. "
                "A feasible random point is proposed instead.",
                RuntimeWarning,
                # the caller of the policy method that delegates here
                stacklevel=3,
            )
        return random_point()

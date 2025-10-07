# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import itertools
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from .. import Variable


class SearchSpace(ABC):
    """
    Abstract base class for search spaces in Bayesian optimization.

    This class defines the interface for different types of search spaces,
    including discrete candidate sets and continuous parameter ranges.
    """

    @abstractmethod
    def sample_random(self, n_samples: int) -> np.ndarray:
        """
        Sample random points from the search space.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, dimension) containing random samples
        """
        pass

    @abstractmethod
    def optimize_acquisition(
        self,
        acquisition_fn: Callable[[np.ndarray], float],
        optimizer: Optional[Callable] = None,
        mpicomm=None,
    ) -> np.ndarray:
        """
        Optimize the acquisition function to find the best candidate.

        Parameters
        ----------
        acquisition_fn : Callable[[np.ndarray], float]
            Acquisition function that takes a candidate point and returns a score
        optimizer : Callable, optional
            Optimization algorithm for continuous spaces. If None, default optimizer is used.
        mpicomm : MPI.Comm, optional
            MPI communicator for parallel optimization

        Returns
        -------
        np.ndarray
            Optimal point of shape (1, dimension)
        """
        pass

    @abstractmethod
    def is_discrete(self) -> bool:
        """
        Check if the search space is discrete.

        Returns
        -------
        bool
            True if discrete, False if continuous
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of the search space.

        Returns
        -------
        int
            Number of dimensions
        """
        pass


class DiscreteSearchSpace(SearchSpace):
    """
    Discrete search space with predefined candidate points.

    This class manages a finite set of candidate points and tracks which
    candidates have been explored.
    """

    def __init__(self, candidates: Union[np.ndarray, Variable], mpicomm=None):
        """
        Initialize discrete search space.

        Parameters
        ----------
        candidates : np.ndarray
            Array of shape (n_candidates, n_dim) containing all candidate points
        mpicomm : MPI.Comm, optional
            MPI communicator for parallel sampling
            If local_indices is provided, mpicomm must be provided too
        """

        if mpicomm is None:
            self.mpisize = 1
            self.mpirank = 0
        else:
            self.mpisize = mpicomm.Get_size()
            self.mpirank = mpicomm.Get_rank()
        self.mpicomm = mpicomm

        if isinstance(candidates, Variable):
            self.candidates = candidates
        else:
            self.candidates = Variable(X=candidates)
        self.n_candidates = self.candidates.X.shape[0]
        self.dim = self.candidates.X.shape[1]
        self.available_indices = np.array_split(np.arange(self.n_candidates), self.mpisize)[self.mpirank]

    def sample_random(self, n_samples: int) -> np.ndarray:
        """
        Sample random indices from available candidates.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        np.ndarray
            Array of indices corresponding to selected candidates
        """

        nremains = len(self.available_indices)
        if self.mpisize == 1:
            if nremains < n_samples:
                return self.available_indices
            return np.random.choice(self.available_indices, n_samples, replace=False)

        buff_int = np.zeros(self.mpisize, dtype=int)
        self.mpicomm.Gather(np.array(nremains), buff_int, root=0)
        if self.mpirank == 0:
            hi = np.add.accumulate(buff_int)
            lo = np.roll(hi, 1)
            lo[0] = 0
            if hi[-1] < n_samples:
                index = np.arange(0, hi[-1])
            else:
                index = np.random.choice(hi[-1], n_samples, replace=False)
            ranks = np.searchsorted(hi, index, side="right")
            local_indices: list[list[int]] = [[] for _ in range(self.mpisize)]
            for r, i in zip(ranks, index):
                local_indices[r].append(i - lo[r])
        local_indices = self.mpicomm.scatter(local_indices, root=0)
        local_actions = self.available_indices[local_indices]
        actions = self.mpicomm.allgather(local_actions)
        actions = itertools.chain.from_iterable(actions)
        actions = np.array(list(actions))
        return actions

    def optimize_acquisition(
        self,
        acquisition_fn: Callable[[np.ndarray], float],
        optimizer: Optional[Callable] = None,
        mpicomm=None,
    ) -> np.ndarray:
        """
        Find the best candidate by evaluating all available candidates.

        For discrete spaces, we evaluate the acquisition function at all
        available candidates and select the one with the highest score.

        Parameters
        ----------
        acquisition_fn : Callable[[np.ndarray], float]
            Acquisition function that takes a candidate point and returns a score
        optimizer : Callable, optional
            Not used for discrete spaces
        mpicomm : MPI.Comm, optional
            MPI communicator for parallel evaluation

        Returns
        -------
        np.ndarray
            Best candidate point of shape (1, dimension)
        """

        nremains = len(self.available_indices)
        if nremains == 0:
            raise ValueError("No available candidates")

        # Find the best candidate for local indices
        local_best_idx = None
        local_best_score = -np.inf
        for idx in self.available_indices:
            candidate = self.X[idx, :]
            score = acquisition_fn(candidate)
            if score > local_best_score:
                local_best_score = score
                local_best_idx = idx

        # Find the globally best candidate
        if self.mpisize > 1:
            # Each process finds its best candidate, then select globally best
            scores_all = self.mpicomm.allgather(local_best_score)
            best_rank = np.argmax(scores_all)
            best_idx = local_best_idx
            self.mpicomm.Bcast(best_idx, root=best_rank)
        else:
            best_idx = local_best_idx

        return np.array([best_idx])

    def is_discrete(self) -> bool:
        """Return True for discrete search space."""
        return True

    def get_dimension(self) -> int:
        """Return the dimension of the search space."""
        return self.dim

    def remove_candidates(self, indices: np.ndarray):
        """
        Remove selected candidates from available candidates.

        Parameters
        ----------
        indices : np.ndarray
            Indices of candidates to remove
        """
        mask = np.isin(self.available_indices, indices, invert=True)
        self.available_indices = self.available_indices[mask]

    def get_candidates(self) -> Variable:
        """
        Get all candidate points.

        Returns
        -------
        Variable
            Array of all candidate points
        """
        return self.candidates

    @property
    def variable(self) -> Variable:
        """
        Get candidate as variable.
        """
        return self.candidates

    @property
    def X(self) -> np.ndarray:
        """
        Get candidate points.
        """
        return self.candidates.X

    @property
    def t(self) -> np.ndarray:
        """
        Get candidate targets.
        """
        return self.candidates.t

    @t.setter
    def t(self, t: np.ndarray):
        """
        Set candidate targets.
        """
        self.candidates.t = t

    @property
    def Z(self) -> np.ndarray:
        """
        Get candidate features.
        """
        return self.candidates.Z

    @Z.setter
    def Z(self, Z: np.ndarray):
        """
        Set candidate features.
        """
        self.candidates.Z = Z



class ContinuousSearchSpace(SearchSpace):
    """
    Continuous search space defined by bounds.

    This class represents a continuous search space bounded by minimum
    and maximum values for each dimension.
    """

    def __init__(self, min_bounds: np.ndarray, max_bounds: np.ndarray):
        """
        Initialize continuous search space.

        Parameters
        ----------
        min_bounds : np.ndarray
            Lower bounds for each dimension
        max_bounds : np.ndarray
            Upper bounds for each dimension

        Raises
        ------
        ValueError
            If bounds have different dimensions or min >= max for any dimension
        """
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.dim = len(min_bounds)

        if len(min_bounds) != len(max_bounds):
            raise ValueError("min_bounds and max_bounds must have the same length")
        if not np.all(min_bounds < max_bounds):
            raise ValueError(
                "min_bounds must be less than max_bounds for all dimensions"
            )

    def sample_random(self, n_samples: int) -> np.ndarray:
        """
        Sample random points from the continuous space using uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, dimension) containing random samples
        """
        return np.random.uniform(
            self.min_bounds, self.max_bounds, (n_samples, self.dim)
        )

    def optimize_acquisition(
        self,
        acquisition_fn: Callable[[np.ndarray], float],
        optimizer: Optional[Callable] = None,
        mpicomm=None,
    ) -> np.ndarray:
        """
        Optimize the acquisition function using the specified optimizer.

        Parameters
        ----------
        acquisition_fn : Callable[[np.ndarray], float]
            Acquisition function that takes a candidate point and returns a score
        optimizer : Callable, optional
            Optimization algorithm. If None, default random optimizer is used.
        mpicomm : MPI.Comm, optional
            MPI communicator for parallel optimization

        Returns
        -------
        np.ndarray
            Optimal point of shape (1, dimension)
        """
        if optimizer is None:
            # Use default random optimizer
            from .optimize.random import Optimizer

            optimizer = Optimizer(
                min_X=self.min_bounds, max_X=self.max_bounds, nsamples=1000
            )

        optimal_x = optimizer(acquisition_fn, mpicomm=mpicomm)
        return optimal_x.reshape(1, -1)

    def is_discrete(self) -> bool:
        """Return False for continuous search space."""
        return False

    def get_dimension(self) -> int:
        """Return the dimension of the search space."""
        return self.dim

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the bounds of the search space.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (min_bounds, max_bounds)
        """
        return self.min_bounds, self.max_bounds

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import numpy as np

class NDS:
    """
    Unified objective function based on the Non-Dominated Sorting (NDS) method.

    The unified objective function of original objectives t is defined as:
    t_unified = 1 / rank
    where rank is the Pareto rank of the solution.
    Solutions with the rank 1 are the Pareto solutions of the original set.
    Solutions with the rank 2 are the Pareto solutions of the set where the solutions with the rank 1 are removed.
    Solutions with the rank r are the Pareto solutions of the set where the solutions with the rank 1, 2, ..., r-1 are removed.

    Parameters
    ----------
    num_objectives: int
        Number of objectives
    """

    num_objectives: int
    """Number of objectives"""

    rank_max: int
    """Maximum rank of the NDS"""

    def __init__(self, num_objectives: int, rank_max: int = 10):
        """
        Initialize NDS unified objective function

        Parameters
        ----------
        num_objectives: int
            Number of objectives
        rank_max: int
            Maximum rank of the NDS
        """

        self.num_objectives = num_objectives
        self.rank_max = rank_max

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate unified objective function

        Parameters
        ----------
        t: np.ndarray
            Values of the original objective functions
            Shape: (N, num_objectives)

        Returns
        -------
        t_unified: np.ndarray
            Values of the unified objective function
            Shape: (N, 1)
        """
        return nds_impl(t, self.rank_max)


def nds_impl(t: np.ndarray, rank_max: int) -> np.ndarray:
    """
    NDS implementation

    Parameters
    ----------
    t: np.ndarray
        Training data
    rank_max: int
        Maximum rank of the NDS

    Returns
    -------
    t_unified: np.ndarray
        Values of the unified objective function
        Shape: (N, 1)
    """

    N, D = t.shape

    # diffs[i, j, :] = training.t[j, :] - training.t[i, :]
    diffs = np.zeros((N, N, D))
    for i in range(N):
        diffs[i, :, :] = t - t[i, :]

    # dominated[i, j] = True if training[i] is dominated by training[j]
    dominated = np.all(diffs >= 0, axis=2) & np.any(diffs > 0, axis=2)

    unified_t = np.zeros((N, 1))
    for rank in range(rank_max):
        # points in pareto are not dominated by any other solution
        pareto = np.all(~dominated, axis=1)
        unified_t[pareto] = 1.0 / (rank + 1)
        # points in pareto will not dominate any other solution after this rank
        dominated[:, pareto] = False # not dominates after this rank
        dominated[pareto, pareto] = True # marked as already processed
        # if all points are marked as already processed, break
        if np.all(np.diag(dominated)):
            break

    return unified_t.reshape(-1, 1)


def nds_impl_naive(t: np.ndarray, rank_max: int) -> np.ndarray:
    """
    Naive -- slow but simple -- implementation of NDS

    Parameters
    ----------
    t: np.ndarray
        Values of the original objective functions
        Shape: (N, num_objectives)
    rank_max: int
        Maximum rank of the NDS

    Returns
    -------
    t_unified: np.ndarray
        Values of the unified objective function
        Shape: (N, 1)
    """
    N = t.shape[0]
    unified_t = np.zeros(N)
    remains = list(range(N))

    for rank in range(rank_max):
        pareto, remains = _pareto_front_max(t, remains)
        unified_t[pareto] = 1.0 / (rank + 1)

    return unified_t.reshape(-1, 1)


def _is_dominated_max(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(b >= a) and np.any(b > a))

def _pareto_front_max(solutions, remains) -> tuple[list[int], list[int]]:
    pareto: list[int] = []
    new_remains: list[int] = []

    for i in remains:
        dominated = False
        for j in remains:
            if i != j and _is_dominated_max(solutions[i,:], solutions[j,:]):
                dominated = True
                break
        if dominated:
            new_remains.append(i)
        if not dominated:
            pareto.append(i)
    return pareto, new_remains


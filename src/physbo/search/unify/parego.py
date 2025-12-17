# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from typing import Optional

import numpy as np

from ...misc import min_max_scaling

class ParEGO:
    """
    ParEGO unified objective function

    The unified objective function of original objectives t is defined as:
    t_unified = weight_max * max(weights * t) + weight_sum * sum(weights * t)

    Before calculating the unified objective function, the original objectives are normalized to 0 and 1 using min-max scaling.
    """

    num_objectives: int
    """Number of objectives"""

    weight_sum: float
    """Weight of the sum of objectives"""

    weight_max: float
    """Weight of the max of objectives"""

    weights: Optional[np.ndarray]
    """Weights for the objectives.
    Weights are automatically normalized to sum to 1.
    If None, random weights are used for each call.
    """

    def __init__(
        self,
        num_objectives: int,
        weight_sum: float = 0.05,
        weight_max: float = 1.0,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize ParEGO unified objective function

        Parameters
        ----------
        num_objectives: int
            Number of objectives
        weight_sum: float
            Weight of the sum of objectives, default is 0.05
        weight_max: float
            Weight of the max of objectives, default is 1.0
        weights: np.ndarray
            Weights for the objectives.
            Weights are automatically normalized to sum to 1.
            If None (default), random weights are used for each call.
        """
        self.num_objectives = num_objectives
        self.weight_sum = weight_sum
        self.weight_max = weight_max
        if weights is not None:
            weights = np.array(weights)
            self.weights = weights / np.sum(weights)
        else:
            self.weights = None

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate unified objective function by ParEGO method

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
        s = 10
        if self.weights is None:
            weights = sample_weights(self.num_objectives, s)
            # weights = np.random.rand(self.num_objectives)
            # weights /= np.sum(weights)
        else:
            weights = self.weights

        if t.ndim == 1:
            t = t.reshape(1, -1)

        assert t.shape[1] == self.num_objectives, f"The number of objectives in t ({t.shape[1]}) must be the same as the number of objectives in the ParEGO ({self.num_objectives})"

        t_weighted = min_max_scaling(t, low=-1.0, high=0.0) * weights.reshape(1, -1)
        t_sum = np.sum(t_weighted, axis=1)
        t_max = np.max(t_weighted, axis=1)
        t_unified = self.weight_max * t_max + self.weight_sum * t_sum
        return t_unified.reshape(-1, 1)



def sample_weights(num_objectives: int, s: int) -> np.ndarray:
    """
    Sample weights for the objectives

    Parameters
    ----------
    num_objectives: int
        Number of objectives
    number_of_discrete_points: int
        Number of discrete points to sample weights
    """

    ls = np.zeros(num_objectives, dtype=int)
    for d in range(num_objectives - 1):
        ls[d] = np.random.randint(s + 1)
        s -= ls[d]
    ls[num_objectives - 1] = s

    return ls / np.sum(ls)



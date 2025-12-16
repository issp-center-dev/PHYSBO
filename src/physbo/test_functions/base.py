# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from abc import ABC, abstractmethod
import copy

import numpy as np


class TestFunction(ABC):
    """Abstract class for test functions.

    Test functions are used to evaluate the performance of the optimization algorithms.

    Note
    =====
    The test function f should be defined as a minimization problem.
    """

    def __init__(
        self,
        nobj: int,
        dim: int,
        min_X: np.ndarray | list[float] | float,
        max_X: np.ndarray | list[float] | float,
        test_maximizer: bool = True,
    ):
        """Initialize the test function.

        Arguments
        =========
        nobj: int
            Number of objectives.
        dim: int
            Number of dimensions.
        min_X: np.ndarray | list[float] | float
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """
        self._nobj = nobj
        self._dim = dim
        self._test_maximizer = test_maximizer

        if isinstance(min_X, float):
            self._min_X = np.full(dim, min_X)
        elif isinstance(min_X, list):
            self._min_X = np.array(min_X)
        else:
            self._min_X = copy.deepcopy(min_X)

        if isinstance(max_X, float):
            self._max_X = np.full(dim, max_X)
        elif isinstance(max_X, list):
            self._max_X = np.array(max_X)
        else:
            self._max_X = copy.deepcopy(max_X)

        if self._min_X.shape[0] != self._dim:
            raise ValueError(
                f"ERROR: dimension mismatch: self._min_X.shape[0] = {self._min_X.shape[0]}, self._dim = {self._dim}"
            )
        if self._max_X.shape[0] != self._dim:
            raise ValueError(
                f"ERROR: dimension mismatch: self._max_X.shape[0] = {self._max_X.shape[0]}, self._dim = {self._dim}"
            )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the test function at the given point.

        Arguments
        =========
        x: np.ndarray
            The point at which to evaluate the test function.
            x is a numpy array of shape (n, d), where n is the number of points and d is the dimension of the input space.

        Returns
        =======
        f: np.ndarray
            The value of the test function at the given point.
            The output value is a numpy array of shape (n, k), where k is the number of objectives.
        """
        if x.shape[1] != self._dim:
            raise ValueError(
                f"ERROR: dimension mismatch: x.shape[1] = {x.shape[1]}, self._dim = {self._dim}"
            )

        f = self.f(x)
        # This is assertion because it is the Developer's responsibility to ensure that the number of objectives is correct
        assert f.shape[1] == self._nobj

        if self._test_maximizer:
            return -f
        else:
            return f

    @property
    def dim(self) -> int:
        """Get the number of dimensions of the test function.

        Returns
        =======
        int
            The number of dimensions of the test function d.
        """
        return self._dim

    @property
    def nobj(self) -> int:
        """Get the number of objectives of the test function.

        Returns
        =======
        int
            The number of objectives of the test function k.
        """
        return self._nobj

    @property
    def min_X(self) -> np.ndarray:
        """Get the minimum values of the search space of the test function.

        Returns
        =======
        np.ndarray
            The minimum value of the test function for each dimension.
        """
        return copy.deepcopy(self._min_X)

    @property
    def max_X(self) -> np.ndarray:
        """Get the maximum values of the search space of the test function.

        Returns
        =======
        np.ndarray
            The maximum value of the test function for each dimension.
        """
        return copy.deepcopy(self._max_X)

    @abstractmethod
    def f(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the test function at the given point.

        Arguments
        =========
        x: np.ndarray
            The point at which to evaluate the test function.
            x is a numpy array of shape (n, d), where n is the number of points and d is the dimension of the input space.

        Returns
        =======
        f: np.ndarray
            The value of the test function at the given point.
            The output value is a numpy array of shape (n, k), where k is the number of objectives.
        """
        ...


    def constraint(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the constraint function at the given point.

        Arguments
        =========
        x: np.ndarray
            The point at which to evaluate the constraint function.
            x is a numpy array of shape (n, d), where n is the number of points and d is the dimension of the input space.

        Returns
        =======
        np.ndarray
            The boolean values indicating whether the point is valid or not.
            The output value is a numpy array of shape (n,), where n is the number of points.
        """
        # default implementation is that all points are valid
        return np.ones(x.shape[0], dtype=bool)


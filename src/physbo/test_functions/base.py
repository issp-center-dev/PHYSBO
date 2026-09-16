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

from ..search import utility as search_utility


class TestFunction(ABC):
    """Abstract class for test functions.

    Test functions are used to evaluate the performance of the optimization algorithms.

    Note
    =====
    Each test function is implemented in the same sense (minimization or
    maximization) as in the reference it follows, so that ``f`` and the
    docstring can be compared with the literature directly.
    A subclass declares its sense by the class attribute ``_is_maximization``
    (``False`` by default, i.e., minimization).

    ``test_maximizer`` selects the sense of the *returned* values:
    with ``test_maximizer=True`` (the default) the values returned by
    ``__call__`` and the reference box always describe a maximization
    problem (as PHYSBO maximizes objectives), and with ``test_maximizer=False``
    they always describe a minimization problem.
    The sign is flipped only when the declared sense and the requested sense
    differ.
    """

    # Sense in which ``f`` (and the reference box) is written.
    # Subclasses whose reference defines a maximization problem set this to True.
    _is_maximization: bool = False

    def __init__(
        self,
        nobj: int,
        dim: int,
        min_X: np.ndarray | list[float] | float,
        max_X: np.ndarray | list[float] | float,
        test_maximizer: bool = True,
        name: str | None = None,
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
            If True, the returned values describe a maximization problem
            (for testing a maximization problem solver such as PHYSBO).
            If False, they describe a minimization problem.
        """
        self._nobj = nobj
        self._dim = dim
        self._test_maximizer = test_maximizer
        self._name = name

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

        if self._needs_negation():
            return -f
        else:
            return f

    def _needs_negation(self) -> bool:
        """Whether the values of ``f`` must be negated to obtain the requested sense.

        The sign is flipped only when the sense in which ``f`` is written
        (``_is_maximization``) differs from the requested sense (``test_maximizer``).
        """
        return self._test_maximizer != self._is_maximization

    @property
    def is_maximization(self) -> bool:
        """Whether the test function is defined as a maximization problem in its reference.

        This describes how ``f`` is written, not the sense of the returned values
        (which is selected by ``test_maximizer``).

        Returns
        =======
        bool
            True if the original problem is a maximization problem.
        """
        return self._is_maximization

    @property
    def test_maximizer(self) -> bool:
        """Whether the returned values describe a maximization problem.

        Returns
        =======
        bool
            True if ``__call__`` returns values of a maximization problem.
        """
        return self._test_maximizer

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

        ``f`` is written in the sense of the reference (see ``_is_maximization``);
        the conversion to the requested sense is done by ``__call__``.

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

    def make_grid(self, num_X: int | list[int] | np.ndarray) -> np.ndarray:
        """Make a grid of points in the search space.

        Arguments
        =========
        num_X: int | list[int] | np.ndarray
            Number of points in each dimension.

        Returns
        =======
        np.ndarray
            The grid of points in the search space.
            The output is a numpy array of shape (N, d), where N is the number of points and d is the dimension of the search space.
        """
        return search_utility.make_grid(self.min_X, self.max_X, num_X, constraint=self.constraint)

    
    def set_name(self, name: str):
        """Set the name of the test function.

        Arguments
        =========
        name: str
            The name of the test function.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of the test function.

        Returns
        =======
        str
            The name of the test function.
        """
        if self._name is None:
            return self.__class__.__name__
        else:
            return self._name

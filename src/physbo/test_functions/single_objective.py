# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import numpy as np

from .base import TestFunction


class SingleTestFunction(TestFunction):
    def __init__(
        self,
        dim: int,
        min_X: np.ndarray | list[float] | float,
        max_X: np.ndarray | list[float] | float,
        test_maximizer: bool,
    ):
        super().__init__(
            nobj=1, dim=dim, min_X=min_X, max_X=max_X, test_maximizer=test_maximizer
        )

    def global_minimum_point(self) -> np.ndarray:
        raise NotImplementedError


class Sphere(SingleTestFunction):
    def __init__(
        self,
        dim: int = 2,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        test_maximizer: bool = True,
    ):
        r"""Sphere function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = \sum_{i=1}^d x_i^2

        Global minimum: :math:`f(\boldsymbol{0}) = 0`.

        Arguments
        =========
        dim: int, default=2
            Number of dimensions.
        min_X: np.ndarray | list[float] | float
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """
        super().__init__(
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x**2, axis=1, keepdims=True)

    def global_minimum_point(self) -> np.ndarray:
        return np.zeros((1, self.dim))


class Rastrigin(SingleTestFunction):
    def __init__(
        self,
        dim: int = 2,
        A: float = 10.0,
        min_X: np.ndarray | list[float] | float = -5.12,
        max_X: np.ndarray | list[float] | float = 5.12,
        test_maximizer: bool = True,
    ):
        r"""Rastrigin function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = A n + \sum_{i=1}^n (x_i^2 - A \cos(2 \pi x_i))

        Global minimum: :math:`f(\boldsymbol{0}) = 0`.

        Arguments
        =========
        dim: int, default=2
            Number of dimensions :math:`n`.
        A: float, default=10.0
            Amplitude parameter.
        min_X: np.ndarray | list[float] | float, default=-5.12
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=5.12
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Rastrigin, L. A. "Systems of extremal control." Mir, Moscow (1974).
        """
        super().__init__(
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )
        self._A = A

    def f(self, x: np.ndarray) -> np.ndarray:
        return self._A * self._dim + np.sum(
            x**2 - self._A * np.cos(2 * np.pi * x), axis=1, keepdims=True
        )

    def global_minimum_point(self) -> np.ndarray:
        return np.zeros((1, self.dim))


class Ackley(SingleTestFunction):
    def __init__(
        self,
        dim: int = 2,
        a: float = 20.0,
        b: float = 0.2,
        min_X: np.ndarray | list[float] | float = -32.768,
        max_X: np.ndarray | list[float] | float = 32.768,
        test_maximizer: bool = True,
    ):
        r"""Ackley function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = -a \exp \left( -b \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2} \right) - \exp \left( \frac{1}{n} \sum_{i=1}^n \cos(c x_i) \right) + a + \exp(1)

        Global minimum: :math:`f(\boldsymbol{0}) = 0`.

        Arguments
        =========
        dim: int, default=2
            Number of dimensions :math:`n`.
        a: float, default=20.0
            First parameter.
        b: float, default=0.2
            Second parameter.
        min_X: np.ndarray | list[float] | float, default=-32.768
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=32.768
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Ackley, D. H. (1987) "A connectionist machine for genetic hillclimbing", Kluwer Academic Publishers, Boston MA. p. 13-14.
        """
        super().__init__(
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )
        self._a = a
        self._b = b

    def f(self, x: np.ndarray) -> np.ndarray:
        mean_sq = np.mean(x**2, axis=1)
        mean_cos = np.mean(np.cos(2.0 * np.pi * x), axis=1)
        return (
            -self._a * np.exp(-self._b * np.sqrt(mean_sq))
            - np.exp(mean_cos)
            + self._a
            + np.exp(1.0)
        ).reshape(-1, 1)

    def global_minimum_point(self) -> np.ndarray:
        return np.zeros((1, self.dim))


class Rosenbrock(SingleTestFunction):
    def __init__(
        self,
        dim: int = 2,
        a: float = 100.0,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        test_maximizer: bool = True,
    ):
        r"""Rosenbrock function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = \sum_{i=1}^{n-1} \left( a (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right)

        Global minimum: :math:`f(1, \dots, 1) = 0`.

        Arguments
        =========
        dim: int, default=2
            Number of dimensions :math:`n`. Must be >= 2.
        a: float, default=100.0
            Amplitude parameter.
        min_X: np.ndarray | list[float] | float, default=-5.0
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=10.0
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        References
        ==========
        Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or least value of a function". The Computer Journal. 3 (3): 175-184.  https://doi.org/10.1093/comjnl/3.3.175
        """
        if dim < 2:
            raise ValueError(f"ERROR: dimension must be >= 2: dim = {dim}")
        super().__init__(
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )
        self._a = a

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.sum(
            self._a * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2,
            axis=1,
            keepdims=True,
        )

    def global_minimum_point(self) -> np.ndarray:
        return np.ones((1, self.dim))


class Beale(SingleTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -4.5,
        max_X: np.ndarray | list[float] | float = 4.5,
        test_maximizer: bool = True,
    ):
        r"""Beale function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = (1.5 - x_1 + x_1 x_2)^2 + (2.25 - x_1 + x_1 x_2^2)^2 + (2.625 - x_1 + x_1 x_2^3)^2

        Global minimum: :math:`f(3, 0.5) = 0`.

        Arguments
        =========
        min_X: np.ndarray | list[float] | float, default=-4.5
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=4.5
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """
        super().__init__(
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        xy = x_vals * y_vals
        xy2 = xy * y_vals
        xy3 = xy * y_vals * y_vals
        term1 = (1.5 - x_vals + xy) ** 2
        term2 = (2.25 - x_vals + xy2) ** 2
        term3 = (2.625 - x_vals + xy3) ** 2
        return (term1 + term2 + term3).reshape(-1, 1)

    def global_minimum_point(self) -> np.ndarray:
        return np.array([[3.0, 0.5]])


class Booth(SingleTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -10.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        test_maximizer: bool = True,
    ):
        r"""Booth function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = (x_1 + 2 x_2 - 7)^2 + (2 x_1 + x_2 - 5)^2

        Global minimum: :math:`f(1, 3) = 0`.

        Arguments
        =========
        min_X: np.ndarray | list[float] | float, default=-10.0
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=10.0
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """
        super().__init__(
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        term1 = (x_vals + 2.0 * y_vals - 7.0) ** 2
        term2 = (2.0 * x_vals + y_vals - 5.0) ** 2
        return (term1 + term2).reshape(-1, 1)

    def global_minimum_point(self) -> np.ndarray:
        return np.array([[1.0, 3.0]])


class Matyas(SingleTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -10.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        test_maximizer: bool = True,
    ):
        r"""Matyas function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = 0.26 (x_1^2 + x_2^2) - 0.48 x_1 x_2

        Global minimum: :math:`f(0, 0) = 0`.

        Arguments
        =========
        min_X: np.ndarray | list[float] | float, default=-10.0
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=10.0
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """
        super().__init__(
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        return (0.26 * (x_vals**2 + y_vals**2) - 0.48 * x_vals * y_vals).reshape(-1, 1)

    def global_minimum_point(self) -> np.ndarray:
        return np.array([[0.0, 0.0]])


class Himmelblau(SingleTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        test_maximizer: bool = True,
    ):
        r"""Himmelblau's function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2

        Global minimum: :math:`f(3, 2) = f(-2.805118, 3.131312) = f(-3.779310, -3.283186) = f(3.584428, -1.848126) = 0`.

        Arguments
        =========
        min_X: np.ndarray | list[float] | float, default=-5.0
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=5.0
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        
        References
        ==========
        Himmelblau, D. (1972). Applied Nonlinear Programming. McGraw-Hill.
        """
        super().__init__(
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        term1 = (x_vals**2 + y_vals - 11.0) ** 2
        term2 = (x_vals + y_vals**2 - 7.0) ** 2
        return (term1 + term2).reshape(-1, 1)

    def global_minimum_point(self) -> np.ndarray:
        return np.array(
            [
                [3.0, 2.0],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )


class ThreeHumpCamel(SingleTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        test_maximizer: bool = True,
    ):
        r"""Three-hump camel function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = 2 x_1^2 - 1.05 x_1^4 + x_1^6 / 6 + x_1 x_2 + x_2^2

        Global minimum: :math:`f(0, 0) = 0`.

        Arguments
        =========
        min_X: np.ndarray | list[float] | float, default=-5.0
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=5.0
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """
        super().__init__(
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        x2 = x_vals**2
        x4 = x2**2
        x6 = x4 * x2
        return (2.0 * x2 - 1.05 * x4 + x6 / 6.0 + x_vals * y_vals + y_vals**2).reshape(
            -1, 1
        )

    def global_minimum_point(self) -> np.ndarray:
        return np.array([[0.0, 0.0]])


class Easom(SingleTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -100.0,
        max_X: np.ndarray | list[float] | float = 100.0,
        test_maximizer: bool = True,
    ):
        r"""Easom function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = -\cos(x_1) \cos(x_2) \exp \left( -((x_1 - \pi)^2 + (x_2 - \pi)^2) \right) + 1

        Global minimum: :math:`f(\pi, \pi) = 0`.

        Arguments
        =========
        min_X: np.ndarray | list[float] | float, default=-100.0
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=100.0
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """
        super().__init__(
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        return (
            -np.cos(x_vals)
            * np.cos(y_vals)
            * np.exp(-((x_vals - np.pi) ** 2 + (y_vals - np.pi) ** 2))
            + 1.0
        ).reshape(-1, 1)

    def global_minimum_point(self) -> np.ndarray:
        return np.array([[np.pi, np.pi]])


class StyblinskiTang(SingleTestFunction):
    def __init__(
        self,
        dim: int = 2,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        test_maximizer: bool = True,
    ):
        r"""Styblinski-Tang function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = \sum_{i=1}^n \left( \frac{x_i^4 - 16 x_i^2 + 5 x_i}{2} \right)

        Global minimum: :math:`f(-2.903534, \dots, -2.903534) \approx -39.16617 n`.

        Arguments
        =========
        dim: int, default=2
            Number of dimensions.
        min_X: np.ndarray | list[float] | float, default=-5.0
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=5.0
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """
        super().__init__(
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.sum((x**4 - 16.0 * x**2 + 5.0 * x) / 2.0, axis=1, keepdims=True)

    def global_minimum_point(self) -> np.ndarray:
        # Approximately -2.903534 for each dimension
        return np.full((1, self.dim), -2.903534)


class Schaffer2(SingleTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -100.0,
        max_X: np.ndarray | list[float] | float = 100.0,
        test_maximizer: bool = True,
    ):
        r"""Schaffer's second function.

        .. math::

            \text{Minimize}\quad
            f(\boldsymbol{x}) = 0.5 + \frac{\sin^2(x_1^2 - x_2^2) - 0.5}{(1 + 0.001 (x_1^2 + x_2^2))^2}

        Global minimum: :math:`f(0, 0) = 0`.

        Arguments
        =========
        min_X: np.ndarray | list[float] | float, default=-100.0
            Minimum value of search space for each dimension.
        max_X: np.ndarray | list[float] | float, default=100.0
            Maximum value of search space for each dimension.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """
        super().__init__(
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        return (
            0.5 + (np.sin(x_vals**2 - y_vals**2)**2 - 0.5) / (1.0 + 0.001 * (x_vals**2 + y_vals**2))**2
        ).reshape(-1, 1)

    def global_minimum_point(self) -> np.ndarray:
        return np.array([[0.0, 0.0]])

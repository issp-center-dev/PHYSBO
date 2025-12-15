# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import numpy as np

from .base import TestFunction


class Sphere(TestFunction):
    """Sphere function.

    A simple unimodal function with a single global minimum at the origin.
    Formula: f(x) = sum(x_i^2)
    Global minimum: f(0, ..., 0) = 0
    """

    def __init__(
        self,
        dim: int = 2,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Sphere function.

        Args
        ======
            dim: Number of dimensions (default: 2)
            min_X: Minimum value of search space for each dimension (default: -5.0)
            max_X: Maximum value of search space for each dimension (default: 5.0)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        super().__init__(
            nobj=1,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x**2, axis=1, keepdims=True)

    def global_minimum_point(self) -> np.ndarray:
        return np.zeros((1, self.dim))


class Rastrigin(TestFunction):
    """Rastrigin function.

    A highly multimodal function with many local minima.
    Formula: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    where A = 10
    Global minimum: f(0, ..., 0) = 0
    """

    def __init__(
        self,
        dim: int = 2,
        A: float = 10.0,
        min_X: np.ndarray | list[float] | float = -5.12,
        max_X: np.ndarray | list[float] | float = 5.12,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Rastrigin function.

        Args
        ======
            dim: Number of dimensions (default: 2)
            A: Amplitude parameter (default: 10.0)
            min_X: Minimum value of search space for each dimension (default: -5.12)
            max_X: Maximum value of search space for each dimension (default: 5.12)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        super().__init__(
            nobj=1,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )
        self._A = A

    def f(self, x: np.ndarray) -> np.ndarray:
        return self._A * self._dim + np.sum(
            x**2 - self._A * np.cos(2 * np.pi * x), axis=1, keepdims=True
        )

    def global_minimum_point(self) -> np.ndarray:
        return np.zeros((1, self.dim))


class Ackley(TestFunction):
    """Ackley function.

    A multimodal function with many local minima.
    Formula: f(x) = -a*exp(-b*sqrt(sum(x_i^2)/n)) - exp(sum(cos(c*x_i))/n) + a + exp(1)
    where a = 20, b = 0.2, c = 2*pi
    Global minimum: f(0, ..., 0) = 0
    """

    def __init__(
        self,
        dim: int = 2,
        a: float = 20.0,
        b: float = 0.2,
        min_X: np.ndarray | list[float] | float = -32.768,
        max_X: np.ndarray | list[float] | float = 32.768,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Ackley function.

        Args
        ======
            dim: Number of dimensions (default: 2)
            a: First parameter (default: 20.0)
            b: Second parameter (default: 0.2)
            min_X: Minimum value of search space for each dimension (default: -32.768)
            max_X: Maximum value of search space for each dimension (default: 32.768)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        super().__init__(
            nobj=1,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
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


class Rosenbrock(TestFunction):
    """Rosenbrock function (Rosenbrock's valley or banana function).

    A unimodal function with a narrow curved valley.
    Formula: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum: f(1, ..., 1) = 0
    """

    def __init__(
        self,
        dim: int = 2,
        a: float = 100.0,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Rosenbrock function.

        Args
        ======
            dim: Number of dimensions (default: 2, must be >= 2)
            a: Amplitude parameter (default: 100.0)
            min_X: Minimum value of search space for each dimension (default: -5.0)
            max_X: Maximum value of search space for each dimension (default: 10.0)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        if dim < 2:
            raise ValueError(f"ERROR: dimension must be >= 2: dim = {dim}")
        super().__init__(
            nobj=1,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
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


class Beale(TestFunction):
    """Beale function.

    A 2D function with multiple local minima.
    Formula: f(x,y) = (1.5 - x + x*y)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2
    Global minimum: f(3, 0.5) = 0
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -4.5,
        max_X: np.ndarray | list[float] | float = 4.5,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Beale function.

        Args
        ======
            min_X: Minimum value of search space for each dimension (default: -4.5)
            max_X: Maximum value of search space for each dimension (default: 4.5)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        super().__init__(
            nobj=1,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
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


class Booth(TestFunction):
    """Booth function.

    A 2D function with a single global minimum.
    Formula: f(x,y) = (x + 2*y - 7)^2 + (2*x + y - 5)^2
    Global minimum: f(1, 3) = 0
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -10.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Booth function.

        Args
        ======
            min_X: Minimum value of search space for each dimension (default: -10.0)
            max_X: Maximum value of search space for each dimension (default: 10.0)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        super().__init__(
            nobj=1,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        term1 = (x_vals + 2.0 * y_vals - 7.0) ** 2
        term2 = (2.0 * x_vals + y_vals - 5.0) ** 2
        return (term1 + term2).reshape(-1, 1)

    def global_minimum_point(self) -> np.ndarray:
        return np.array([[1.0, 3.0]])


class Matyas(TestFunction):
    """Matyas function.

    A 2D function with a single global minimum.
    Formula: f(x,y) = 0.26*(x^2 + y^2) - 0.48*x*y
    Global minimum: f(0, 0) = 0
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -10.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Matyas function.

        Args
        ======
            min_X: Minimum value of search space for each dimension (default: -10.0)
            max_X: Maximum value of search space for each dimension (default: 10.0)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        super().__init__(
            nobj=1,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x_vals = x[:, 0]
        y_vals = x[:, 1]
        return (0.26 * (x_vals**2 + y_vals**2) - 0.48 * x_vals * y_vals).reshape(-1, 1)

    def global_minimum_point(self) -> np.ndarray:
        return np.array([[0.0, 0.0]])


class Himmelblau(TestFunction):
    """Himmelblau's function.

    A 2D function with four equal local minima.
    Formula: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    Global minima: f(3, 2) = f(-2.805118, 3.131312) = f(-3.779310, -3.283186) = f(3.584428, -1.848126) = 0
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize Himmelblau's function.

        Args
        ======
            min_X: Minimum value of search space for each dimension (default: -5.0)
            max_X: Maximum value of search space for each dimension (default: 5.0)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        super().__init__(
            nobj=1,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
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


class ThreeHumpCamel(TestFunction):
    """Three-hump camel function.

    A 2D function with three local minima, one of which is global.
    Formula: f(x,y) = 2*x^2 - 1.05*x^4 + x^6/6 + x*y + y^2
    Global minimum: f(0, 0) = 0
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Three-hump camel function.

        Args
        ======
            min_X: Minimum value of search space for each dimension (default: -5.0)
            max_X: Maximum value of search space for each dimension (default: 5.0)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        super().__init__(
            nobj=1,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
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


class Easom(TestFunction):
    """Easom function.

    A 2D function with a very narrow global minimum.
    Formula: f(x,y) = -cos(x)*cos(y)*exp(-((x-pi)^2 + (y-pi)^2)) + 1
    Global minimum: f(pi, pi) = 0
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -100.0,
        max_X: np.ndarray | list[float] | float = 100.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Easom function.

        Args
        ======
            min_X: Minimum value of search space for each dimension (default: -100.0)
            max_X: Maximum value of search space for each dimension (default: 100.0)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        super().__init__(
            nobj=1,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
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


class StyblinskiTang(TestFunction):
    """Styblinski-Tang function.

    A multimodal function with many local minima.
    Formula: f(x) = sum((x_i^4 - 16*x_i^2 + 5*x_i) / 2)
    Global minimum: f(-2.903534, ..., -2.903534) ≈ -39.16617*d
    """

    def __init__(
        self,
        dim: int = 2,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Styblinski-Tang function.

        Args
        ======
            dim: Number of dimensions (default: 2)
            min_X: Minimum value of search space for each dimension (default: -5.0)
            max_X: Maximum value of search space for each dimension (default: 5.0)
            optimizer_will_maximize: If True, the tested optimizer treats the maximization problem.
        """
        super().__init__(
            nobj=1,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.sum((x**4 - 16.0 * x**2 + 5.0 * x) / 2.0, axis=1, keepdims=True)

    def global_minimum_point(self) -> np.ndarray:
        # Approximately -2.903534 for each dimension
        return np.full((1, self.dim), -2.903534)

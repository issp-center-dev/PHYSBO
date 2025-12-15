# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import numpy as np
from .base import TestFunction


class Gaussian(TestFunction):
    def __init__(
        self,
        centers: np.ndarray,
        widths: np.ndarray | list[float] | float = 1.0,
        amplitudes: np.ndarray | list[float] | float = 1.0,
        min_X: np.ndarray | list[float] | float = -2.0,
        max_X: np.ndarray | list[float] | float = 2.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Gaussian function."""
        if centers.ndim != 2:
            raise ValueError(
                f"ERROR: centers must be a 2D array, but got {centers.ndim}D array"
            )
        nobj = centers.shape[0]
        dim = centers.shape[1]

        if isinstance(widths, float):
            widths = np.full(nobj, widths)
        elif isinstance(widths, list):
            widths = np.array(widths)
        if widths.shape[0] != nobj:
            raise ValueError(
                f"ERROR: widths must be a 1D array with length {nobj}, but got {widths.shape[0]}D array"
            )
        
        min_width = widths.min()
        if min_width <= 0.0:
            raise ValueError(
                f"ERROR: widths must be positive, but minimum value of widths is {min_width}"
            )
        
        if isinstance(amplitudes, float):
            amplitudes = np.full(nobj, amplitudes)
        elif isinstance(amplitudes, list):
            amplitudes = np.array(amplitudes)
        if amplitudes.shape[0] != nobj:
            raise ValueError(
                f"ERROR: amplitudes must be a 1D array with length {nobj}, but got {amplitudes.shape[0]}D array"
            )

        min_amplitude = amplitudes.min()
        if min_amplitude <= 0.0:
            raise ValueError(
                f"ERROR: amplitudes must be positive, but minimum value of amplitudes is {min_amplitude}"
            )

        max_amplitude = amplitudes.max()
        if max_amplitude != 1.0:
            print("INFO: amplitudes are normalized to have maximum value 1.0.")
            amplitudes = amplitudes / max_amplitude

        amplitudes = -1.0 * amplitudes

        self._centers = centers
        self._coeffs = (-0.5 / (widths**2)).reshape(1, nobj)
        self._amplitudes = amplitudes.reshape(1, nobj)

        super().__init__(
            nobj=nobj,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )
    
    def f(self, x: np.ndarray) -> np.ndarray:
        r = np.sum((x - self._centers) ** 2, axis=1).reshape(-1, self.nobj)
        return self._amplitudes * np.exp(self._coeffs * r)


class FonsecaFleming(TestFunction):
    def __init__(
        self,
        dim: int = 2,
        min_X: np.ndarray | list[float] | float = -2.0,
        max_X: np.ndarray | list[float] | float = 2.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the VLMOP2 function."""
        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[1]
        f1 = 1 - np.exp(-1 * np.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
        f2 = 1 - np.exp(-1 * np.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))
        return np.c_[f1, f2]


VLMOP2 = FonsecaFleming
"""Alias for FonsecaFleming"""


class BinhKorn(TestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = np.array([0.0, 0.0]),
        max_X: np.ndarray | list[float] | float = np.array([5.0, 3.0]),
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Binh-Korn function.

        Args
        ======
            min_X: Minimum value of search space for each dimension (default: [0.0, 0.0])
            max_X: Maximum value of search space for each dimension (default: [5.0, 3.0])
        """
        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = 4.0 * x1**2 + 4.0 * x2**2
        f2 = (x1 - 5.0) ** 2 + (x2 - 5.0) ** 2
        return np.c_[f1, f2]

    def constraint(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        g1 = (x1 - 5) ** 2 + x2**2 < 25.0
        g2 = (x1 - 8) ** 2 + (x2 + 3) ** 2 >= 7.7
        return np.logical_and(g1, g2)


class ChankongHaimes(TestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -20.0,
        max_X: np.ndarray | list[float] | float = 20.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Chankong-Haimes function."""
        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = 2.0 + (x1 - 2.0) ** 2 + (x2 - 1.0) ** 2
        f2 = 9.0 * x1 - (x2 - 1.0) ** 2
        return np.c_[f1, f2]

    def constraint(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        g1 = x1**2 + x2**2 <= 225.0
        g2 = x1 - 3 * x2 <= -10.0
        return np.logical_and(g1, g2)


class Binh4(TestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -7.0,
        max_X: np.ndarray | list[float] | float = 4.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Binh4 function."""
        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x1 * x1 - x2
        f2 = -0.5 * x1 - x2 - 1.0
        return np.c_[f1, f2]

    def constraint(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        g1 = 6.5 - x1 / 6.0 - x2 >= 0.0
        g2 = 7.5 - 0.5 * x1 - x2 >= 0.0
        g3 = 30.0 - 5 * x1 - x2 >= 0.0
        return np.logical_and(g1, g2, g3)


class Kursawe(TestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Kursawe function."""
        super().__init__(
            nobj=2,
            dim=3,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = -10.0 * np.exp(-0.2 * np.sqrt(np.sum(x[:, 0:2] ** 2, axis=1)))
        f1 += -10.0 * np.exp(-0.2 * np.sqrt(np.sum(x[:, 1:3] ** 2, axis=1)))

        f2 = np.sum(np.abs(x) ** 0.8 + 5.0 * np.sin(x**3), axis=1)

        return np.c_[f1, f2]


class Shaffer1(TestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -10.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Shaffer1 function."""
        super().__init__(
            nobj=2,
            dim=1,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x**2
        f2 = (x - 2) ** 2
        return np.c_[f1, f2]


class Shaffer2(TestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Shaffer2 function."""
        super().__init__(
            nobj=2,
            dim=1,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        if x <= 1.0:
            f1 = -x
        elif x <= 3.0:
            f1 = x - 2.0
        elif x <= 4.0:
            f1 = 4.0 - x
        else:
            f1 = x - 4.0
        f1 = x**2

        f2 = (x - 5) ** 2
        return np.c_[f1, f2]


class Poloni(TestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -np.pi,
        max_X: np.ndarray | list[float] | float = np.pi,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Poloni function."""
        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

        x_a = np.array([[1.0, 2.0]])
        self._a1 = self._b1(x_a)
        self._a2 = self._b2(x_a)

    def _b1(self, x: np.ndarray) -> float:
        X = x[:, 0]
        Y = x[:, 1]
        return 0.5 * np.sin(X) - 2.0 * np.cos(X) + np.sin(Y) - 1.5 * np.cos(Y)

    def _b2(self, x: np.ndarray) -> float:
        X = x[:, 0]
        Y = x[:, 1]
        return 1.5 * np.sin(X) - np.cos(X) + 2.0 * np.sin(Y) - 0.5 * np.cos(Y)

    def f(self, x: np.ndarray) -> np.ndarray:
        B1 = self._b1(x)
        B2 = self._b2(x)
        f1 = 1.0 + (self._a1 - B1) ** 2 + (self._a2 - B2) ** 2
        f2 = (x[:, 0] + 3) ** 2 + (x[:, 1] + 1) ** 2
        return np.c_[f1, f2]


class ZDT1(TestFunction):
    def __init__(
        self,
        dim: int = 30,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the ZDT1 function."""
        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self._dim - 1)
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.c_[f1, f2]


class ZDT2(TestFunction):
    def __init__(
        self,
        dim: int = 30,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the ZDT2 function."""
        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self._dim - 1)
        h = 1.0 - (f1 / g) ** 2
        f2 = g * h
        return np.c_[f1, f2]


class ZDT3(TestFunction):
    def __init__(
        self,
        dim: int = 30,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the ZDT3 function."""
        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self._dim - 1)
        f2 = g - np.sqrt(f1 * g) - f1 * np.sin(10.0 * np.pi * f1)
        return np.c_[f1, f2]


class ZDT4(TestFunction):
    def __init__(
        self,
        dim: int = 10,
        min_X: None | np.ndarray | list[float] | float = None,
        max_X: None | np.ndarray | list[float] | float = None,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the ZDT4 function."""
        if min_X is None:
            min_X = np.full(dim, -5.0)
            min_X[0] = 0.0
        if max_X is None:
            max_X = np.full(dim, 5.0)
            max_X[0] = 1.0
        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g = 91.0 + np.sum(x[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[:, 1:]), axis=1)
        f2 = g - np.sqrt(f1 * g)

        return np.c_[f1, f2]


class ZDT6(TestFunction):
    def __init__(
        self,
        dim: int = 10,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the ZDT6 function."""
        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = 1.0 - np.exp(-4.0 * x[:, 0]) * (np.sin(6.0 * np.pi * x[:, 0]) ** 6)
        g = 1.0 + 9.0 * (np.sum(x[:, 1:], axis=1) / (self._dim - 1)) ** 0.25
        h = 1.0 - (f1 / g) ** 2
        f2 = g * h
        return np.c_[f1, f2]


class OsyczkaKundu(TestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        max_X: np.ndarray | list[float] | float = [10.0, 10.0, 5.0, 6.0, 5.0, 10.0],
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Osyczka-Kundu function."""
        super().__init__(
            nobj=2,
            dim=6,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = (
            -25.0 * (x[:, 0] - 2.0) ** 2
            - (x[:, 1] - 2.0) ** 2
            - (x[:, 2] - 1.0) ** 2
            - (x[:, 3] - 4.0) ** 2
            - (x[:, 4] - 1.0) ** 2
        )
        f2 = np.sum(x**2, axis=1)
        return np.c_[f1, f2]

    def constraint(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]

        g1 = x1 + x2 - 2.0 >= 0.0
        g2 = 6.0 - x1 - x2 >= 0.0
        g3 = 2.0 - x2 + x1 >= 0.0
        g4 = 2.0 - x1 + 3.0 * x2 >= 0.0
        g5 = 4.0 - (x3 - 3.0) ** 2 - x4 >= 0.0
        g6 = (x5 - 3.0) ** 2 + x6 - 4.0 >= 0.0

        return np.logical_and(g1, g2, g3, g4, g5, g6)


class ConstrEX(TestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = [0.1, 0.0],
        max_X: np.ndarray | list[float] | float = [1.0, 5.0],
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the ConstrEX function."""
        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        f2 = (1.0 + x[:, 1]) / x[:, 0]
        return np.c_[f1, f2]

    def constraint(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        g1 = 9 * x1 + x2 >= 6.0
        g2 = 9 * x1 - x2 >= 1.0
        return np.logical_and(g1, g2)


class Viennet(TestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -3.0,
        max_X: np.ndarray | list[float] | float = 3.0,
        optimizer_will_maximize: bool = True,
    ):
        """Initialize the Viennet function."""
        super().__init__(
            nobj=3,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            optimizer_will_maximize=optimizer_will_maximize,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]

        r2 = x1**2 + x2**2

        f1 = 0.5 * r2 + np.sin(r2)
        f2 = (3 * x1 - 2 * x2 + 4) ** 2 / 8 + (x1 - x2 + 1) ** 2 / 27 + 15
        f3 = 1 / (r2 + 1) - 1.1 * np.exp(-r2)
        return np.c_[f1, f2, f3]

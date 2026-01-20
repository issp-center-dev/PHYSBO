# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from .base import TestFunction


class MultiTestFunction(TestFunction):
    def __init__(
        self,
        nobj: int,
        dim: int,
        min_X: np.ndarray | list[float] | float,
        max_X: np.ndarray | list[float] | float,
        test_maximizer: bool,
    ):
        super().__init__(
            nobj=nobj,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    @property
    def reference_min(self) -> np.ndarray:
        """Get the lower bound of the reference box.

        Reference box is a box that contains the entire non-dominated region.
        It is used to calculate the volume of the non-dominated region.

        Returns
        =======
        np.ndarray
            The reference minimum values of the test function.

        Note
        ====
        Reference box is calculated by using the default values of min_X and max_X.

        """
        if self._test_maximizer:
            return -self._ref_max()
        else:
            return self._ref_min()

    @property
    def reference_max(self) -> np.ndarray:
        """Get the upper bound of the reference box.

        Reference box is a box that contains the entire non-dominated region.
        It is used to calculate the volume of the non-dominated region.

        Returns
        =======
        np.ndarray
            The reference maximum values of the test function.

        Note
        ====
        Reference box is calculated by using the default values of min_X and max_X.
        """
        if self._test_maximizer:
            return -self._ref_min()
        else:
            return self._ref_max()

    @abstractmethod
    def _ref_min(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _ref_max(self) -> np.ndarray:
        raise NotImplementedError


class Gaussian(MultiTestFunction):
    def __init__(
        self,
        centers: np.ndarray,
        widths: np.ndarray | list[float] | float = 1.0,
        amplitudes: np.ndarray | list[float] | float = 1.0,
        min_X: np.ndarray | list[float] | float = -2.0,
        max_X: np.ndarray | list[float] | float = 2.0,
        test_maximizer: bool = True,
    ):
        r"""Gaussian function.

        .. math::

            \text{Minimize}\quad
            f_n(\boldsymbol{x}) = -A_n \exp \left( -\frac{\left|\boldsymbol{x} - \boldsymbol{c}_n\right|^2}{2 w_n^2} \right)

        Arguments
        =========
        centers : np.ndarray
            Centers of the Gaussian functions :math:`\boldsymbol{c}_n`.
        widths : np.ndarray | list[float] | float, default=1.0
            Widths of the Gaussian functions :math:`w_n`.
        amplitudes : np.ndarray | list[float] | float, default=1.0
            Amplitudes of the Gaussian functions :math:`A_n`.
        min_X : np.ndarray | list[float] | float, default=-2.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=2.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """

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

        self._centers = centers
        self._coeffs = (-0.5 / (widths**2)).reshape(1, nobj)
        self._amplitudes = amplitudes.reshape(1, nobj)

        super().__init__(
            nobj=nobj,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        # Ensure x is at least 2D: (n, dim)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Reshape for broadcasting: x (n, 1, dim), centers (1, nobj, dim)
        # Result: (n, nobj, dim)
        x_expanded = x[:, np.newaxis, :]  # (n, 1, dim)
        centers_expanded = self._centers[np.newaxis, :, :]  # (1, nobj, dim)

        # Compute squared distances: (n, nobj, dim) -> (n, nobj)
        r = np.sum((x_expanded - centers_expanded) ** 2, axis=2)

        return -self._amplitudes * np.exp(self._coeffs * r)

    def _ref_min(self) -> np.ndarray:
        return -1.0 * self._amplitudes.reshape(-1)

    def _ref_max(self) -> np.ndarray:
        return np.zeros(self.nobj)


class FonsecaFleming(MultiTestFunction):
    def __init__(
        self,
        dim: int = 2,
        min_X: np.ndarray | list[float] | float = -2.0,
        max_X: np.ndarray | list[float] | float = 2.0,
        test_maximizer: bool = True,
    ):
        r"""Fonseca and Fleming's function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = 1 - \exp \left( -\sum_{i=1}^N \left( x_i - \frac{1}{\sqrt{N}} \right)^2 \right) \\
            f_2(\boldsymbol{x}) = 1 - \exp \left( -\sum_{i=1}^N \left( x_i + \frac{1}{\sqrt{N}} \right)^2 \right)
            \end{cases}

        Arguments
        =========
        dim : int, default=2
            Number of dimensions :math:`N`.
        min_X : np.ndarray | list[float] | float, default=-2.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=2.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Carlos M. Fonseca, Peter J. Fleming; An Overview of Evolutionary Algorithms in Multiobjective Optimization. Evol Comput 1995; 3 (1): 1-16. doi: https://doi.org/10.1162/evco.1995.3.1.1

        """

        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[1]
        f1 = 1 - np.exp(-1 * np.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
        f2 = 1 - np.exp(-1 * np.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.zeros(self.nobj)

    def _ref_max(self) -> np.ndarray:
        return np.ones(self.nobj)


class Viennet(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -3.0,
        max_X: np.ndarray | list[float] | float = 3.0,
        test_maximizer: bool = True,
    ):
        r"""Viennet's function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = 0.5 (x_1^2 + x_2^2) + \sin(x_1^2 + x_2^2) \\
            f_2(\boldsymbol{x}) = (3 x_1 - 2 x_2 + 4)^2 / 8 + (x_1 - x_2 + 1)^2 / 27 + 15 \\
            f_3(\boldsymbol{x}) = 1 / (x_1^2 + x_2^2 + 1) - 1.1 \exp(-(x_1^2 + x_2^2))
            \end{cases}

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=-3.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=3.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Viennet, R., et al. "Multicriteria Optimization Using a Genetic Algorithm for Determining a Pareto Set," International Journal of Systems Science 27(2), 255-260 (1996).

        """
        super().__init__(
            nobj=3,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]

        r2 = x1**2 + x2**2

        f1 = 0.5 * r2 + np.sin(r2)
        f2 = (3 * x1 - 2 * x2 + 4) ** 2 / 8 + (x1 - x2 + 1) ** 2 / 27 + 15
        f3 = 1 / (r2 + 1) - 1.1 * np.exp(-r2)
        return np.c_[f1, f2, f3]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 15.0, -0.1])

    def _ref_max(self) -> np.ndarray:
        return np.array([10.0, 62.0, 0.2])


class BinhKorn(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = np.array([0.0, 0.0]),
        max_X: np.ndarray | list[float] | float = np.array([5.0, 3.0]),
        test_maximizer: bool = True,
    ):
        r"""Binh-Korn's function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = 4 x_1^2 + 4 x_2^2 \\
            f_2(\boldsymbol{x}) = (x_1 - 5)^2 + (x_2 - 5)^2
            \end{cases}

            \text{Subject to}
            \begin{cases}
            g_1(\boldsymbol{x}) = (x_1 - 5)^2 + x_2^2 \le 25 \\
            g_2(\boldsymbol{x}) = (x_1 - 8)^2 + (x_2 + 3)^2 \ge 7.7
            \end{cases}

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=np.array([0.0, 0.0])
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=np.array([5.0, 3.0])
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Binh, To Thanh, and Ulrich Korn. "MOBES: A multiobjective evolution strategy for constrained optimization problems." The third international conference on genetic algorithms (Mendel 97). Vol. 25. 1997.
        """
        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
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
        g1 = (x1 - 5) ** 2 + x2**2 <= 25.0
        g2 = (x1 - 8) ** 2 + (x2 + 3) ** 2 >= 7.7
        return np.logical_and(g1, g2)

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 4.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([140.0, 50.0])


class ChankongHaimes(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -20.0,
        max_X: np.ndarray | list[float] | float = 20.0,
        test_maximizer: bool = True,
    ):
        r"""Chankong-Haimes's function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = 2 + (x_1 - 2)^2 + (x_2 - 1)^2 \\
            f_2(\boldsymbol{x}) = 9 x_1 - (x_2 - 1)^2
            \end{cases}

            \text{Subject to}
            \begin{cases}
            g_1(\boldsymbol{x}) = x_1^2 + x_2^2 \le 225 \\
            g_2(\boldsymbol{x}) = x_1 - 3 x_2 \le -10
            \end{cases}

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=-20.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=20.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Chankong, V., and Haimes, Y. Y., "Multiobjective decision making: Theory and method", North-Holland series in system science and engineering, 1983.

        """

        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
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

    def _ref_min(self) -> np.ndarray:
        return np.array([2.0, -650.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([950.0, 180.0])


class KitaYabumotoMoriNishikawa(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -7.0,
        max_X: np.ndarray | list[float] | float = 4.0,
        test_maximizer: bool = True,
    ):
        r"""Kita-Yabumoto-Mori-Nishikawa's function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = x_1^2 - x_2 \\
            f_2(\boldsymbol{x}) = -\frac{1}{2}x_1 - x_2 - 1
            \end{cases}

            \text{Subject to}
            \begin{cases}
            g_1(\boldsymbol{x}) = 6.5 - \frac{x_1}{6} - x_2 \ge 0 \\
            g_2(\boldsymbol{x}) = 7.5 - \frac{x_1}{2} - x_2 \ge 0 \\
            g_3(\boldsymbol{x}) = 30 - 5 x_1 - x_2 \ge 0
            \end{cases}

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=-7.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=4.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Kita, H., Yabumoto, Y., Mori, N., Nishikawa, Y. (1996). Multi-objective optimization by means of the thermodynamical genetic algorithm. In: Voigt, HM., Ebeling, W., Rechenberg, I., Schwefel, HP. (eds) Parallel Problem Solving from Nature — PPSN IV. PPSN 1996. Lecture Notes in Computer Science, vol 1141. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-61723-X_1014
        """

        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
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
        return np.logical_and(np.logical_and(g1, g2), g3)

    def _ref_min(self) -> np.ndarray:
        return np.array([-4.0, -7.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([56.0, 9.5])


def Binh1(*args, **kwargs):
    """Binh's first function.

    This is an alias of :class:`BinhKorn`.

    References
    ==========
    To, Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm The Study Cases.
    """
    fn = BinhKorn(*args, **kwargs)
    fn.set_name("Binh1")
    return fn

def Binh2(*args, **kwargs):
    r"""Binh's second function.

    This is an alias of :class:`ChankongHaimes`.

    References
    ==========
    To, Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm The Study Cases.
    """
    fn = ChankongHaimes(*args, **kwargs)
    fn.set_name("Binh2")
    return fn


def Binh3(*args, **kwargs):
    """Binh's third function.

    This is an alias of :class:`FonsecaFleming`.

    References
    ==========
    To, Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm The Study Cases.
    """
    fn = FonsecaFleming(*args, **kwargs)
    fn.set_name("Binh3")
    return fn

def Binh4(*args, **kwargs):
    """Binh's fourth function.

    This is an alias of :class:`KitaYabumotoMoriNishikawa`.

    References
    ==========
    To, Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm The Study Cases.
    """
    fn = KitaYabumotoMoriNishikawa(*args, **kwargs)
    fn.set_name("Binh4")
    return fn


class Binh5(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = [0.1, 0.0],
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
        r"""Binh's fifth function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = x_1 \\
            f_2(\boldsymbol{x}) = \frac{g(x_2)}{x_1}
            \end{cases},\quad
            \text{where}\quad
            g(x) = 2 - \exp\left(-\left(\frac{x - 0.2}{0.004}\right)^2\right) - 0.8 \exp\left(-\left(\frac{x - 0.6}{0.4}\right)^2\right)


        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=[-0.1, 0.0]
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=1.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        To, Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm The Study Cases.
        """

        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x1
        g = (
            2.0
            - np.exp(-(((x2 - 0.2) / 0.004) ** 2))
            - 0.8 * np.exp(-(((x2 - 0.6) / 0.4) ** 2))
        )
        f2 = g / x1
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.1, 0.7])

    def _ref_max(self) -> np.ndarray:
        return np.array([1.0, 20.0])


class Binh6(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        test_maximizer: bool = True,
    ):
        r"""Binh's sixth function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = \sqrt{x_1^2 + x_2^2 + 1} \\
            f_2(\boldsymbol{x}) = \frac{g(x_4)}{f_1(\boldsymbol{x})}
            \end{cases},\quad
            \text{where}\quad
            g(x) = 100 (x_4 - x_3^2)^2 + (1 - x_3)^2 + 2

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=-5.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=5.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        To, Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm The Study Cases.
        """

        super().__init__(
            nobj=2,
            dim=4,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        f1 = np.sqrt(x1**2 + x2**2 + 1.0)
        g = 100.0 * (x4 - x3**2) ** 2 + (1 - x3) ** 2 + 2.0
        f2 = g / f1
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([1.0, 0.2])

    def _ref_max(self) -> np.ndarray:
        return np.array([8.9, 1.0e5])


class Binh8(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
        r"""Binh's eighth function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = x_1 + x_2 \\
            f_2(\boldsymbol{x}) = 1 - \exp(-4 x_1) \sin(5 \pi x_1)^4
            \end{cases}

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=0.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=1.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        To, Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm The Study Cases.
        """

        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x1 + x2
        f2 = 1.0 - np.exp(-4.0 * x1) * np.sin(5.0 * np.pi * x1) ** 4
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 0.3])

    def _ref_max(self) -> np.ndarray:
        return np.array([2.0, 1.0])


class Binh9(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
        r"""Binh's ninth function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = x_1 \\
            f_2(\boldsymbol{x}) = g(x_2) h(x_1, x_2)
            \end{cases}

            \text{where}
            \begin{cases}
            g(x_2) = 1 + 10 x_2 \\
            h(x_1, x_2) = 1 - \left( \frac{x_1}{g(x_2)} \right)^2 - \left( \frac{x_1}{g(x_2)} \right) \sin(8 \pi x_1)
            \end{cases}

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=0.0
                Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=1.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        To, Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm The Study Cases.
        """

        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x1
        g = 1 + 10 * x2
        h = 1.0 - (f1 / g) ** 2 - (f1 / g) * np.sin(8.0 * np.pi * f1)
        f2 = g * h
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, -0.5])

    def _ref_max(self) -> np.ndarray:
        return np.array([1.0, 12.0])


class Kursawe(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        test_maximizer: bool = True,
    ):
        r"""Kursawe's function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = \sum_{i=1}^{2} -10 \exp \left( -0.2 \sqrt{x_i^2 + x_{i+1}^2} \right) \\
            f_2(\boldsymbol{x}) = \sum_{i=1}^{3} \left( \left| x_i \right|^{0.8} + 5 \sin(x_i^3) \right)
            \end{cases}

        Arguments
        =========
        min_X: np.ndarray | list[float] | float, default=-5.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=5.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        F. Kursawe, "A variant of evolution strategies for vector optimization," in PPSN I, Vol 496 Lect Notes in Comput Sci. Springer-Verlag, 1991, pp. 193-197.
        """
        super().__init__(
            nobj=2,
            dim=3,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = -10.0 * np.exp(-0.2 * np.sqrt(np.sum(x[:, 0:2] ** 2, axis=1)))
        f1 += -10.0 * np.exp(-0.2 * np.sqrt(np.sum(x[:, 1:3] ** 2, axis=1)))

        f2 = np.sum(np.abs(x) ** 0.8 + 5.0 * np.sin(x**3), axis=1)

        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([-20.0, -12.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([-4.0, 26.0])


class Schaffer1(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -10.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        test_maximizer: bool = True,
    ):
        r"""Schaffer's first function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(x) = x^2 \\
            f_2(x) = (x - 2)^2
            \end{cases}

        Arguments
        =========
        min_X: np.ndarray | list[float] | float, default=-10.0
            Minimum value of the search space :math:`x_{\min}`.
        max_X : np.ndarray | list[float] | float, default=10.0
            Maximum value of the search space :math:`x_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Schaffer, J. David. "Multiple objective optimization with vector evaluated genetic algorithms." Proceedings of the first international conference on genetic algorithms and their applications. Psychology Press, 2014.

        """

        super().__init__(
            nobj=2,
            dim=1,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x**2
        f2 = (x - 2) ** 2
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([100.0, 144.0])


class Schaffer2(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        test_maximizer: bool = True,
    ):
        r"""Schaffer's second function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(x) = \begin{cases}
                -x & \text{if } x \leq 1.0 \\
                x - 2 & \text{if } 1.0 < x \leq 3.0 \\
                4 - x & \text{if } 3.0 < x \leq 4.0 \\
                x - 4 & \text{if } x > 4.0
            \end{cases} \\
            f_2(x) = (x - 5)^2
            \end{cases}

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=-5.0
                Minimum value of the search space :math:`x_{\min}`.
        max_X : np.ndarray | list[float] | float, default=10.0
            Maximum value of the search space :math:`x_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Schaffer, J. David. "Multiple objective optimization with vector evaluated genetic algorithms." Proceedings of the first international conference on genetic algorithms and their applications. Psychology Press, 2014.

        """

        super().__init__(
            nobj=2,
            dim=1,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        X = x.reshape(-1)
        f1 = -X
        f1[X > 1.0] = X[X > 1.0] - 2.0
        f1[X > 3.0] = 4.0 - X[X > 3.0]
        f1[X > 4.0] = X[X > 4.0] - 4.0

        f2 = (X - 5) ** 2
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([-1.0, 0.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([6.0, 100.0])


class Poloni(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -np.pi,
        max_X: np.ndarray | list[float] | float = np.pi,
        test_maximizer: bool = True,
    ):
        r"""Poloni's function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = 1.0 + (a_1 - b_1(\boldsymbol{x}))^2 + (a_2 - b_2(\boldsymbol{x}))^2 \\
            f_2(\boldsymbol{x}) = (x_1 + 3)^2 + (x_2 + 1)^2
            \end{cases}

            \text{where}
            \begin{cases}
            a_1 = 0.5 \sin(1) - 2 \cos(1) + \sin(2) - 1.5 \cos(2) \\
            a_2 = 1.5 \sin(1) - \cos(1) + 2 \sin(2) - 0.5 \cos(2) \\
            b_1(\boldsymbol{x}) = 0.5 \sin(x_1) - 2 \cos(x_1) + \sin(x_2) - 1.5 \cos(x_2) \\
            b_2(\boldsymbol{x}) = 1.5 \sin(x_1) - \cos(x_1) + 2 \sin(x_2) - 0.5 \cos(x_2) \\
            \end{cases}

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=-np.pi
                Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=np.pi
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.
        """

        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
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

    def _ref_min(self) -> np.ndarray:
        return np.array([1.0, 0.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([62.0, 55.0])


class ZDT1(MultiTestFunction):
    def __init__(
        self,
        dim: int = 30,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
        r"""ZDT's first function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = x_1 \\
            f_2(\boldsymbol{x}) = g(x_2) h(x_1, x_2)
            \end{cases}

            \text{where}
            \begin{cases}
            g(x_2) = 1 + 9 \sum_{i=2}^{N} x_i / (N - 1) \\
            h(x_1, x_2) = 1 - \sqrt{f_1(\boldsymbol{x}) / g(x_2)}
            \end{cases}

        Arguments
        =========
        dim: int, default=30
            Dimension of the problem :math:`N`.
        min_X : np.ndarray | list[float] | float, default=0.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=1.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        K. Deb, L. Thiele, M. Laumanns and E. Zitzler, "Scalable multi-objective optimization test problems," Proceedings of the 2002 Congress on Evolutionary Computation. CEC'02 (Cat. No.02TH8600), Honolulu, HI, USA, 2002, pp. 825-830 vol.1, doi: 10.1109/CEC.2002.1007032.
        """

        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self._dim - 1)
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 1.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([1.0, 7.2])


class ZDT2(MultiTestFunction):
    def __init__(
        self,
        dim: int = 30,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
        r"""ZDT's second function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = x_1 \\
            f_2(\boldsymbol{x}) = g(x_2) h(x_1, x_2) \\
            \end{cases}

            \text{where}
            \begin{cases}
            g(x_2) = 1 + 9 \sum_{i=2}^{N} x_i / (N - 1) \\
            h(x_1, x_2) = 1 - \left(f_1(\boldsymbol{x}) / g(x_2)\right)^2
            \end{cases}

        Arguments
        =========
        dim: int, default=30
            Dimension of the problem :math:`N`.
        min_X : np.ndarray | list[float] | float, default=0.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=1.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        K. Deb, L. Thiele, M. Laumanns and E. Zitzler, "Scalable multi-objective optimization test problems," Proceedings of the 2002 Congress on Evolutionary Computation. CEC'02 (Cat. No.02TH8600), Honolulu, HI, USA, 2002, pp. 825-830 vol.1, doi: 10.1109/CEC.2002.1007032.
        """
        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self._dim - 1)
        h = 1.0 - (f1 / g) ** 2
        f2 = g * h
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 1.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([1.0, 10.0])


class ZDT3(MultiTestFunction):
    def __init__(
        self,
        dim: int = 30,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
        r"""ZDT's third function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = x_1 \\
            f_2(\boldsymbol{x}) = g(\boldsymbol{x}) h(\boldsymbol{x})
            \end{cases}

            \text{where}
            \begin{cases}
            g(\boldsymbol{x}) = 1 + 9 \sum_{i=2}^{N} x_i / (N - 1) \\
            h(\boldsymbol{x}) = 1 - \sqrt{f_1(\boldsymbol{x}) / g(\boldsymbol{x})} - \frac{f_1(\boldsymbol{x})}{g(\boldsymbol{x})} \sin(10 \pi f_1(\boldsymbol{x}))
            \end{cases}

        Arguments
        =========
        dim: int, default=30
            Dimension of the problem :math:`N`.
        min_X : np.ndarray | list[float] | float, default=0.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=1.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        K. Deb, L. Thiele, M. Laumanns and E. Zitzler, "Scalable multi-objective optimization test problems," Proceedings of the 2002 Congress on Evolutionary Computation. CEC'02 (Cat. No.02TH8600), Honolulu, HI, USA, 2002, pp. 825-830 vol.1, doi: 10.1109/CEC.2002.1007032.
        """

        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self._dim - 1)
        f2 = g - np.sqrt(f1 * g) - f1 * np.sin(10.0 * np.pi * f1)
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 1.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([1.0, 7.2])


class ZDT4(MultiTestFunction):
    def __init__(
        self,
        dim: int = 10,
        min_X: None | np.ndarray | list[float] | float = None,
        max_X: None | np.ndarray | list[float] | float = None,
        test_maximizer: bool = True,
    ):
        r"""ZDT's fourth function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = x_1 \\
            f_2(\boldsymbol{x}) = g(\boldsymbol{x}) h(\boldsymbol{x})
            \end{cases}

            \text{where}
            \begin{cases}
            g(\boldsymbol{x}) = 1 + 10 (N - 1) + \sum_{i=2}^{N} \left(x_i^2 - 10 \cos(4 \pi x_i)\right) \\
            h(\boldsymbol{x}) = 1 - \sqrt{\frac{f_1(\boldsymbol{x})}{g(\boldsymbol{x})}}
            \end{cases}

        Arguments
        =========
        dim: int, default=10
            Dimension of the problem :math:`N`.
        min_X: np.ndarray | list[float] | float
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
            Default is x_1 = 0.0 and x_i = -5.0 for i = 2, ..., N.
        max_X: np.ndarray | list[float] | float
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
            Default is x_1 = 1.0 and x_i = 5.0 for i = 2, ..., N.
        test_maximizer: bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        K. Deb, L. Thiele, M. Laumanns and E. Zitzler, "Scalable multi-objective optimization test problems," Proceedings of the 2002 Congress on Evolutionary Computation. CEC'02 (Cat. No.02TH8600), Honolulu, HI, USA, 2002, pp. 825-830 vol.1, doi: 10.1109/CEC.2002.1007032.
        """

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
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x[:, 0]
        g = 1.0 + 10.0 * (self.dim - 1) + np.sum(x[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[:, 1:]), axis=1)
        f2 = g - np.sqrt(f1 * g)

        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 35.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([1.0, 305.0])


class ZDT6(MultiTestFunction):
    def __init__(
        self,
        dim: int = 10,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
        r"""ZDT's sixth function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = 1 - \exp(-4 x_1) \sin^6(6 \pi x_1) \\
            f_2(\boldsymbol{x}) = g(\boldsymbol{x}) h(\boldsymbol{x})
            \end{cases}

            \text{where}
            \begin{cases}
            g(\boldsymbol{x}) = 1 + 9 \left(\sum_{i=2}^{N} x_i / (N - 1)\right)^{0.25} \\
            h(\boldsymbol{x}) = 1 - \left(\frac{f_1(\boldsymbol{x})}{g(\boldsymbol{x})}\right)^2
            \end{cases}

        Arguments
        =========
        dim: int, default=10
            Dimension of the problem :math:`N`.
        min_X : np.ndarray | list[float] | float, default=0.0
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=1.0
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        K. Deb, L. Thiele, M. Laumanns and E. Zitzler, "Scalable multi-objective optimization test problems," Proceedings of the 2002 Congress on Evolutionary Computation. CEC'02 (Cat. No.02TH8600), Honolulu, HI, USA, 2002, pp. 825-830 vol.1, doi: 10.1109/CEC.2002.1007032.
        """
        super().__init__(
            nobj=2,
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = 1.0 - np.exp(-4.0 * x[:, 0]) * (np.sin(6.0 * np.pi * x[:, 0]) ** 6)
        g = 1.0 + 9.0 * (np.sum(x[:, 1:], axis=1) / (self._dim - 1)) ** 0.25
        h = 1.0 - (f1 / g) ** 2
        f2 = g * h
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.26, 0.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([1.0, 10.0])


class OsyczkaKundu(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        max_X: np.ndarray | list[float] | float = [10.0, 10.0, 5.0, 6.0, 5.0, 10.0],
        test_maximizer: bool = True,
    ):
        r"""Osyczka-Kundu's function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = -25 (x_1 - 2)^2 - (x_2 - 2)^2 - (x_3 - 1)^2 - (x_4 - 4)^2 - (x_5 - 1)^2 \\
            f_2(\boldsymbol{x}) = \sum_{i=1}^{6} x_i^2
            \end{cases}

            \text{Subject to}
            \begin{cases}
            g_1(\boldsymbol{x}) = x_1 + x_2 - 2 \ge 0 \\
            g_2(\boldsymbol{x}) = 6 - x_1 - x_2 \ge 0 \\
            g_3(\boldsymbol{x}) = 2 - x_2 + x_1 \ge 0 \\
            g_4(\boldsymbol{x}) = 2 - x_1 + 3 x_2 \ge 0 \\
            g_5(\boldsymbol{x}) = 4 - \left(x_3 - 3\right)^2 - x_4 \ge 0 \\
            g_6(\boldsymbol{x}) = \left(x_5 - 3\right)^2 + x_6 - 4 \ge 0 \\
            \end{cases}

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
            Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=[10.0, 10.0, 5.0, 6.0, 5.0, 10.0]
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Osyczka, A., Kundu, S. A new method to solve generalized multicriteria optimization problems using the simple genetic algorithm. Structural Optimization 10, 94-99 (1995). https://doi.org/10.1007/BF01743536
        """

        super().__init__(
            nobj=2,
            dim=6,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
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

        result = np.logical_and(g1, g2)
        result = np.logical_and(result, g3)
        result = np.logical_and(result, g4)
        result = np.logical_and(result, g5)
        result = np.logical_and(result, g6)
        return result

    def _ref_min(self) -> np.ndarray:
        return np.array([-1700.0, 2.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([0.0, 400.0])


class ConstrEX(MultiTestFunction):
    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = [0.1, 0.0],
        max_X: np.ndarray | list[float] | float = [1.0, 5.0],
        test_maximizer: bool = True,
    ):
        r"""ConstrEX function.

        .. math::

            \text{Minimize}
            \begin{cases}
            f_1(\boldsymbol{x}) = x_1 \\
            f_2(\boldsymbol{x}) = (1 + x_2) / x_1
            \end{cases}

            \text{Subject to}
            \begin{cases}
            g_1(\boldsymbol{x}) = 9 x_1 + x_2 \ge 6 \\
            g_2(\boldsymbol{x}) = 9 x_1 - x_2 \ge 1 \\
            \end{cases}

        Arguments
        =========
        min_X : np.ndarray | list[float] | float, default=[0.1, 0.0]
                Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        max_X : np.ndarray | list[float] | float, default=[1.0, 5.0]
            Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
        test_maximizer : bool, default=True
            If True, the test function is negated for testing a maximization problem solver.

        References
        ==========
        Deb, K. (2011). Multi-objective Optimisation Using Evolutionary Algorithms: An Introduction. In: Wang, L., Ng, A., Deb, K. (eds) Multi-objective Evolutionary Optimisation for Product Design and Manufacturing. Springer, London. https://doi.org/10.1007/978-0-85729-652-8_1
        """
        super().__init__(
            nobj=2,
            dim=2,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
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

    def _ref_min(self) -> np.ndarray:
        return np.array([0.1, 1.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([1.0, 60.0])


# VLMOP1 = Schaffer1
def VLMOP1(*args, **kwargs):
    r"""VL's first function (so-called VLMOP1).

    This is an alias of :class:`Schaffer1`.

    References
    ==========
    David A. van Veldhuizen and Gary B. Lamont. 1999. Multiobjective evolutionary algorithm test suites. In Proceedings of the 1999 ACM symposium on Applied computing (SAC '99). Association for Computing Machinery, New York, NY, USA, 351-357. https://doi.org/10.1145/298151.298382
    """
    fn = Schaffer1(*args, **kwargs)
    fn.set_name("VLMOP1")
    return fn


def VLMOP2(*args, **kwargs):
    r"""VL's second function (so-called VLMOP2).

    This is an alias of :class:`FonsecaFleming`.

    References
    ==========
    David A. van Veldhuizen and Gary B. Lamont. 1999. Multiobjective evolutionary algorithm test suites. In Proceedings of the 1999 ACM symposium on Applied computing (SAC '99). Association for Computing Machinery, New York, NY, USA, 351-357. https://doi.org/10.1145/298151.298382
    """
    fn = FonsecaFleming(*args, **kwargs)
    fn.set_name("VLMOP2")
    return fn


def VLMOP3(*args, **kwargs):
    r"""VL's third function (so-called VLMOP3).

    This is an alias of :class:`Viennet`.

    References
    ==========
    David A. van Veldhuizen and Gary B. Lamont. 1999. Multiobjective evolutionary algorithm test suites. In Proceedings of the 1999 ACM symposium on Applied computing (SAC '99). Association for Computing Machinery, New York, NY, USA, 351-357. https://doi.org/10.1145/298151.298382
    """
    fn = Viennet(*args, **kwargs)
    fn.set_name("VLMOP3")
    return fn


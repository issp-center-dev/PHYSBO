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
        The box is given in the same sense as the returned values
        (see ``test_maximizer``).

        Returns
        =======
        np.ndarray
            The reference minimum values of the test function.

        Note
        ====
        Unless stated otherwise in the docstring of each function,
        the reference box is calculated by using the default values of min_X and max_X.

        """
        if self._needs_negation():
            return -self._ref_max()
        else:
            return self._ref_min()

    @property
    def reference_max(self) -> np.ndarray:
        """Get the upper bound of the reference box.

        Reference box is a box that contains the entire non-dominated region.
        It is used to calculate the volume of the non-dominated region.
        The box is given in the same sense as the returned values
        (see ``test_maximizer``).

        Returns
        =======
        np.ndarray
            The reference maximum values of the test function.

        Note
        ====
        Unless stated otherwise in the docstring of each function,
        the reference box is calculated by using the default values of min_X and max_X.
        """
        if self._needs_negation():
            return -self._ref_min()
        else:
            return self._ref_max()

    @abstractmethod
    def _ref_min(self) -> np.ndarray:
        """Lower bound of the reference box, written in the same sense as ``f``."""
        raise NotImplementedError

    @abstractmethod
    def _ref_max(self) -> np.ndarray:
        """Upper bound of the reference box, written in the same sense as ``f``."""
        raise NotImplementedError


class Gaussian(MultiTestFunction):
    r"""Gaussian function.

    A sum of Gaussian peaks; each objective is maximal at its own center.

    .. math::

        \text{Maximize}\quad
        f_n(\boldsymbol{x}) = A_n \exp \left( -\frac{\left|\boldsymbol{x} - \boldsymbol{c}_n\right|^2}{2 w_n^2} \right)

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
        If True, the returned values describe a maximization problem (as defined above).
        If False, they are negated to describe a minimization problem.
    """

    _is_maximization = True

    def __init__(
        self,
        centers: np.ndarray,
        widths: np.ndarray | list[float] | float = 1.0,
        amplitudes: np.ndarray | list[float] | float = 1.0,
        min_X: np.ndarray | list[float] | float = -2.0,
        max_X: np.ndarray | list[float] | float = 2.0,
        test_maximizer: bool = True,
    ):
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

        return self._amplitudes * np.exp(self._coeffs * r)

    def _ref_min(self) -> np.ndarray:
        return np.zeros(self.nobj)

    def _ref_max(self) -> np.ndarray:
        return self._amplitudes.reshape(-1).copy()


class FonsecaFleming(MultiTestFunction):
    r"""Fonseca and Fleming's function (:math:`N`-variable form).

    .. math::

        \text{Minimize}
        \begin{cases}
        f_1(\boldsymbol{x}) = 1 - \exp \left( -\sum_{i=1}^N \left( x_i - \frac{1}{\sqrt{N}} \right)^2 \right) \\
        f_2(\boldsymbol{x}) = 1 - \exp \left( -\sum_{i=1}^N \left( x_i + \frac{1}{\sqrt{N}} \right)^2 \right)
        \end{cases}

    The Pareto-optimal set is the segment
    :math:`x_1 = \cdots = x_N \in [-1/\sqrt{N}, 1/\sqrt{N}]`.

    Arguments
    =========
    dim : int, default=2
        Number of dimensions :math:`N`.
    min_X : np.ndarray | list[float] | float, default=-4.0
        Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
    max_X : np.ndarray | list[float] | float, default=4.0
        Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
    test_maximizer : bool, default=True
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    The :math:`N`-variable form with the centers at :math:`\pm 1/\sqrt{N}`
    is given in Fonseca and Fleming (1995b); the search space
    :math:`-4 \le x_i \le 4` follows Van Veldhuizen (1999) and Deb (2001).
    Van Veldhuizen (1999) distinguishes two problems by Fonseca and Fleming:
    "Fonseca", the two-variable form of Fonseca and Fleming (1995a) with the
    centers :math:`(1, -1)` and :math:`(-1, 1)`, and "Fonseca (2)", the
    :math:`N`-variable form of Fonseca and Fleming (1995b) with
    :math:`-4 \le x_i \le 4`, which is the one implemented here (MOP2).
    The two-variable form is a different problem
    (the distance between the centers is :math:`2\sqrt{2}` instead of 2).
    :class:`VLMOP2` is the same function with the search space
    :math:`-2 \le x_i \le 2` used by Van Veldhuizen and Lamont (1999).

    References
    ==========
    Carlos M. Fonseca, Peter J. Fleming; Multiobjective Genetic Algorithms Made Easy: Selection, Sharing, and Mating Restriction. Proceedings of the 1st International Conference on Genetic Algorithms in Engineering Systems: Innovations and Applications (GALESIA), IEE, 1995, pp. 45-52. (1995b)

    Carlos M. Fonseca, Peter J. Fleming; An Overview of Evolutionary Algorithms in Multiobjective Optimization. Evol Comput 1995; 3 (1): 1-16. doi: https://doi.org/10.1162/evco.1995.3.1.1 (1995a; two-variable form)

    David A. Van Veldhuizen; Multiobjective Evolutionary Algorithms: Classifications, Analyses, and New Innovations. Ph.D. thesis, Air Force Institute of Technology, 1999. (MOP2)

    Kalyanmoy Deb; Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001.

    """

    def __init__(
        self,
        dim: int = 2,
        min_X: np.ndarray | list[float] | float = -4.0,
        max_X: np.ndarray | list[float] | float = 4.0,
        test_maximizer: bool = True,
    ):
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
    r"""Viennet's function (the third test problem of Viennet et al. (1996)).

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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    Viennet et al. (1996) propose several test problems; this is the third one.
    It is listed as MOP3 in Van Veldhuizen and Lamont (1999) (hence :class:`VLMOP3`)
    and as MOP5 in Van Veldhuizen (1999).
    The search space :math:`-3 \le x_i \le 3` agrees with Deb (2001).

    References
    ==========
    Viennet, R., et al. "Multicriteria Optimization Using a Genetic Algorithm for Determining a Pareto Set," International Journal of Systems Science 27(2), 255-260 (1996).

    Kalyanmoy Deb; Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001.

    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -3.0,
        max_X: np.ndarray | list[float] | float = 3.0,
        test_maximizer: bool = True,
    ):
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
    r"""Binh-Korn's function (test case 2 of Binh and Korn (1997)).

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

    The Pareto-optimal set consists of :math:`x_1 = x_2 \in [0, 3]` and
    :math:`x_1 \in [3, 5], x_2 = 3`.

    Arguments
    =========
    min_X : np.ndarray | list[float] | float, default=np.array([0.0, 0.0])
        Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
    max_X : np.ndarray | list[float] | float, default=np.array([5.0, 3.0])
        Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
    test_maximizer : bool, default=True
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    Binh and Korn (1997) present two test problems; this is test case 2
    (test case 1 is :class:`SRN`).
    :class:`Binh1` has the same objectives but no constraints and a
    different search space, and is therefore a different problem.

    References
    ==========
    To Thanh Binh and Ulrich Korn. "MOBES: A multiobjective evolution strategy for constrained optimization problems." The third international conference on genetic algorithms (Mendel 97). Vol. 25. 1997.
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = np.array([0.0, 0.0]),
        max_X: np.ndarray | list[float] | float = np.array([5.0, 3.0]),
        test_maximizer: bool = True,
    ):
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


class SRN(MultiTestFunction):
    r"""SRN (Srinivas and Deb's constrained test problem, also called Chankong-Haimes's function).

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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    The objectives originate from Chankong and Haimes (1983), who solved the
    unconstrained problem.
    Srinivas and Deb (1994) added the constraints :math:`g_1, g_2` to make the
    problem more difficult, and this constrained problem is known as SRN
    after them (see Deb (2001)).
    The problem is also often called Chankong-Haimes's function after the
    origin of the objectives; :class:`ChankongHaimes` is an alias.
    It is also test case 1 of Binh and Korn (1997) and
    the second study case of Binh (1999) (:class:`Binh2`).

    References
    ==========
    Srinivas, N. and Deb, K., "Multiobjective optimization using nondominated sorting in genetic algorithms," Evolutionary Computation 2(3), 221-248 (1994).

    Kalyanmoy Deb; Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001. (SRN)

    Chankong, V., and Haimes, Y. Y., "Multiobjective decision making: Theory and methodology", North-Holland series in system science and engineering, 1983. (Reprinted by Dover, 2008.)

    To Thanh Binh and Ulrich Korn. "MOBES: A multiobjective evolution strategy for constrained optimization problems." The third international conference on genetic algorithms (Mendel 97). Vol. 25. 1997.

    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -20.0,
        max_X: np.ndarray | list[float] | float = 20.0,
        test_maximizer: bool = True,
    ):
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
    r"""Kita-Yabumoto-Mori-Nishikawa's function.

    The original problem is a maximization problem:

    .. math::

        \text{Maximize}
        \begin{cases}
        f_1(\boldsymbol{x}) = -x_1^2 + x_2 \\
        f_2(\boldsymbol{x}) = \frac{1}{2}x_1 + x_2 + 1
        \end{cases}

        \text{Subject to}
        \begin{cases}
        g_1(\boldsymbol{x}) = \frac{x_1}{6} + x_2 \le \frac{13}{2} \\
        g_2(\boldsymbol{x}) = \frac{x_1}{2} + x_2 \le \frac{15}{2} \\
        g_3(\boldsymbol{x}) = 5 x_1 + x_2 \le 30 \\
        x_1 \ge 0, \quad x_2 \ge 0
        \end{cases}

    Both objectives increase with :math:`x_2`, so the Pareto-optimal set lies on
    the upper boundary of the feasible region:
    :math:`x_1 \in [0, 3],\ x_2 = 13/2 - x_1/6` (where :math:`g_1` is active).
    The feasible region is contained in :math:`0 \le x_1 \le 6, 0 \le x_2 \le 6.5`.

    Arguments
    =========
    min_X : np.ndarray | list[float] | float, default=0.0
        Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
        The non-negativity constraints :math:`x_1, x_2 \ge 0` are represented
        by this lower bound.
    max_X : np.ndarray | list[float] | float, default=7.0
        Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
    test_maximizer : bool, default=True
        If True, the returned values describe the maximization problem as defined above.
        If False, they are negated to describe a minimization problem.

    Note
    ====
    The search space is not stated explicitly in the references;
    :math:`[0, 7]^2` is chosen so that it contains the feasible region.

    A widely circulated variant (e.g., "Test function 4" in the Wikipedia
    article "Test functions for optimization") drops :math:`x_1, x_2 \ge 0`
    and uses :math:`-7 \le x_1, x_2 \le 4`.
    In that box none of the constraints is active and the Pareto-optimal set
    (:math:`x_2 \ge 6`) is outside the box, so it is a different problem.
    Versions of PHYSBO before this change implemented that variant.

    The reference box is the range of the objectives over the feasible region:
    :math:`f_1 \in [-36, 6.5]`, :math:`f_2 \in [1, 8.5]`.

    References
    ==========
    Kita, H., Yabumoto, Y., Mori, N., Nishikawa, Y. (1996). Multi-objective optimization by means of the thermodynamical genetic algorithm. In: Voigt, HM., Ebeling, W., Rechenberg, I., Schwefel, HP. (eds) Parallel Problem Solving from Nature — PPSN IV. PPSN 1996. Lecture Notes in Computer Science, vol 1141. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-61723-X_1014

    To Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm: The Study Cases. Technical report, Institute for Automation and Communication, Barleben, Germany. (study case 4)
    """

    _is_maximization = True

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 7.0,
        test_maximizer: bool = True,
    ):
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
        f1 = -x1 * x1 + x2
        f2 = 0.5 * x1 + x2 + 1.0
        return np.c_[f1, f2]

    def constraint(self, x: np.ndarray) -> np.ndarray:
        x1 = x[:, 0]
        x2 = x[:, 1]
        g1 = x1 / 6.0 + x2 <= 6.5
        g2 = 0.5 * x1 + x2 <= 7.5
        g3 = 5.0 * x1 + x2 <= 30.0
        return np.logical_and(np.logical_and(g1, g2), g3)

    def _ref_min(self) -> np.ndarray:
        return np.array([-36.0, 1.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([6.5, 8.5])


class Binh1(MultiTestFunction):
    r"""Binh's first function (the first study case of Binh (1999)).

    .. math::

        \text{Minimize}
        \begin{cases}
        f_1(\boldsymbol{x}) = 4 x_1^2 + 4 x_2^2 \\
        f_2(\boldsymbol{x}) = (x_1 - 5)^2 + (x_2 - 5)^2
        \end{cases}

    The Pareto-optimal set is the segment :math:`x_1 = x_2 \in [0, 5]`.

    Arguments
    =========
    min_X : np.ndarray | list[float] | float, default=-5.0
        Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
    max_X : np.ndarray | list[float] | float, default=10.0
        Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
    test_maximizer : bool, default=True
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    The objectives are the same as those of :class:`BinhKorn`, but this
    problem has no constraints and a different search space
    (:math:`-5 \le x_i \le 10`), so the Pareto-optimal set is different.
    Versions of PHYSBO before this change treated ``Binh1`` as an alias of
    :class:`BinhKorn`.

    References
    ==========
    To Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm: The Study Cases. Technical report, Institute for Automation and Communication, Barleben, Germany.
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        test_maximizer: bool = True,
    ):
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

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 0.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([800.0, 200.0])


class Binh2(SRN):
    r"""Binh's second function.

    This is an alias of :class:`SRN`; the objectives,
    constraints and search space of the second study case of Binh (1999)
    agree with it.
    See :class:`SRN` for the definition and the arguments.

    References
    ==========
    To Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm: The Study Cases. Technical report, Institute for Automation and Communication, Barleben, Germany.
    """


class Binh3(FonsecaFleming):
    r"""Binh's third function.

    This is an alias of :class:`FonsecaFleming` (the :math:`N`-variable form).
    Binh (1999) does not state the search space; the default of
    :class:`FonsecaFleming` is used.
    See :class:`FonsecaFleming` for the definition and the arguments.

    References
    ==========
    To Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm: The Study Cases. Technical report, Institute for Automation and Communication, Barleben, Germany.
    """


class Binh4(KitaYabumotoMoriNishikawa):
    r"""Binh's fourth function.

    This is an alias of :class:`KitaYabumotoMoriNishikawa`;
    Binh (1999) quotes the problem as a maximization problem as in Kita et al. (1996).
    See :class:`KitaYabumotoMoriNishikawa` for the definition and the arguments.

    References
    ==========
    To Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm: The Study Cases. Technical report, Institute for Automation and Communication, Barleben, Germany.
    """


class Binh5(MultiTestFunction):
    r"""Binh's fifth function (a multi-modal test problem of Deb (1999)).

    .. math::

        \text{Minimize}
        \begin{cases}
        f_1(\boldsymbol{x}) = x_1 \\
        f_2(\boldsymbol{x}) = \frac{g(x_2)}{x_1}
        \end{cases},\quad
        \text{where}\quad
        g(x) = 2 - \exp\left(-\left(\frac{x - 0.2}{0.004}\right)^2\right) - 0.8 \exp\left(-\left(\frac{x - 0.6}{0.4}\right)^2\right)

    :math:`g` has the global minimum at :math:`x_2 = 0.2` and a local minimum
    at :math:`x_2 = 0.6`.

    Arguments
    =========
    min_X : np.ndarray | list[float] | float, default=[0.1, 0.0]
        Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
    max_X : np.ndarray | list[float] | float, default=1.0
        Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
    test_maximizer : bool, default=True
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    The problem is one of the constructed test problems of Deb (1999)
    (also in Deb (2001)); Binh (1999) lists it as the fifth study case.

    References
    ==========
    Kalyanmoy Deb. "Multi-objective genetic algorithms: Problem difficulties and construction of test problems." Evolutionary Computation 7(3), 205-230 (1999). (Also: Technical Report CI-49/98, University of Dortmund, 1998.)

    Kalyanmoy Deb; Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001.

    To Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm: The Study Cases. Technical report, Institute for Automation and Communication, Barleben, Germany.
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = [0.1, 0.0],
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
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
    r"""Binh's sixth function (the sixth study case of Binh (1999)).

    .. math::

        \text{Minimize}
        \begin{cases}
        f_1(\boldsymbol{x}) = \sqrt{x_1^2 + x_2^2 + 1} \\
        f_2(\boldsymbol{x}) = \frac{g(x_3, x_4)}{f_1(\boldsymbol{x})}
        \end{cases},\quad
        \text{where}\quad
        g(x_3, x_4) = 100 (x_4 - x_3^2)^2 + (1 - x_3)^2 + 2

    Arguments
    =========
    min_X : np.ndarray | list[float] | float, default=-5.0
        Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
    max_X : np.ndarray | list[float] | float, default=5.0
        Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
    test_maximizer : bool, default=True
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    The implementation follows Binh (1999); the origin of the problem has not been identified.

    References
    ==========
    To Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm: The Study Cases. Technical report, Institute for Automation and Communication, Barleben, Germany.
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        test_maximizer: bool = True,
    ):
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
    r"""Binh's eighth function (the eighth study case of Binh (1999)).

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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    The implementation follows Binh (1999); the origin of the problem has not been identified.

    References
    ==========
    To Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm: The Study Cases. Technical report, Institute for Automation and Communication, Barleben, Germany.
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
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
    r"""Binh's ninth function (a discontinuous-front test problem of Deb (1999)).

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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    The problem is the discontinuous-front test problem of Deb (1999)
    :math:`h = 1 - (f_1/g)^\alpha - (f_1/g)\sin(2\pi q f_1)` with
    :math:`\alpha = 2` and :math:`q = 4` (also in Deb (2001)).
    It is listed as MOP6 in Van Veldhuizen (1999) and as the ninth study case in Binh (1999).

    References
    ==========
    Kalyanmoy Deb. "Multi-objective genetic algorithms: Problem difficulties and construction of test problems." Evolutionary Computation 7(3), 205-230 (1999). (Also: Technical Report CI-49/98, University of Dortmund, 1998.)

    Kalyanmoy Deb; Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001.

    David A. Van Veldhuizen; Multiobjective Evolutionary Algorithms: Classifications, Analyses, and New Innovations. Ph.D. thesis, Air Force Institute of Technology, 1999. (MOP6)

    To Thanh Binh. (1999). A Multiobjective Evolutionary Algorithm: The Study Cases. Technical report, Institute for Automation and Communication, Barleben, Germany.
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
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
    r"""Kursawe's function (the three-variable modified version, KUR in Deb (2001)).

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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    The implementation follows Deb (2001), which differs from the original
    problem of Kursawe (1991) in three respects (as noted by Deb):
    the original uses :math:`\sin^3(x_i)` instead of :math:`\sin(x_i^3)` in
    :math:`f_2`, is defined for any number :math:`n` of variables, and states
    no variable bounds.
    Van Veldhuizen (1999) reproduces the original form (general :math:`n`,
    :math:`\sin^3(x_i)`, no bounds) as MOP4 and notes that the problem was
    misprinted in the original paper.
    Because of the change from :math:`\sin^3(x_i)` to :math:`\sin(x_i^3)`,
    the Pareto-optimal set of this function differs from that of the original.

    References
    ==========
    Kursawe, F., "A variant of evolution strategies for vector optimization," in Parallel Problem Solving from Nature (PPSN I, 1990), Vol 496 Lect Notes in Comput Sci. Springer-Verlag, 1991, pp. 193-197. (Cited as Kursawe (1990) in Deb (2001) and Van Veldhuizen (1999).)

    Kalyanmoy Deb; Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001. (KUR)

    David A. Van Veldhuizen; Multiobjective Evolutionary Algorithms: Classifications, Analyses, and New Innovations. Ph.D. thesis, Air Force Institute of Technology, 1999. (MOP4)
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 5.0,
        test_maximizer: bool = True,
    ):
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
    r"""Schaffer's first function.

    .. math::

        \text{Minimize}
        \begin{cases}
        f_1(x) = x^2 \\
        f_2(x) = (x - 2)^2
        \end{cases}

    The Pareto-optimal set is :math:`x \in [0, 2]`.

    Arguments
    =========
    min_X: np.ndarray | list[float] | float, default=-10.0
        Minimum value of the search space :math:`x_{\min}`.
    max_X : np.ndarray | list[float] | float, default=10.0
        Maximum value of the search space :math:`x_{\max}`.
    test_maximizer : bool, default=True
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    Deb (2001) uses the search space :math:`-A \le x \le A` with
    :math:`A` from :math:`10` to :math:`10^5`; a larger :math:`A` makes the
    problem harder because the Pareto-optimal set becomes relatively smaller.
    The default corresponds to :math:`A = 10`.
    The reference box is computed from ``min_X`` and ``max_X``
    (the range of each objective over the search space), so it stays
    consistent for any :math:`A`.

    References
    ==========
    Schaffer, J. David. "Multiple objective optimization with vector evaluated genetic algorithms." Proceedings of the first international conference on genetic algorithms and their applications. Psychology Press, 2014.

    Kalyanmoy Deb; Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001.

    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -10.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        test_maximizer: bool = True,
    ):
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

    @staticmethod
    def _range_of_shifted_square(lo: float, hi: float, c: float) -> tuple[float, float]:
        """Range of (x - c)^2 over lo <= x <= hi."""
        ends = ((lo - c) ** 2, (hi - c) ** 2)
        vmin = 0.0 if lo <= c <= hi else min(ends)
        return vmin, max(ends)

    def _ref_min(self) -> np.ndarray:
        lo, hi = float(self._min_X[0]), float(self._max_X[0])
        return np.array(
            [
                self._range_of_shifted_square(lo, hi, 0.0)[0],
                self._range_of_shifted_square(lo, hi, 2.0)[0],
            ]
        )

    def _ref_max(self) -> np.ndarray:
        lo, hi = float(self._min_X[0]), float(self._max_X[0])
        return np.array(
            [
                self._range_of_shifted_square(lo, hi, 0.0)[1],
                self._range_of_shifted_square(lo, hi, 2.0)[1],
            ]
        )


class Schaffer2(MultiTestFunction):
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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    Note
    ====
    The search space :math:`-5 \le x \le 10` follows Deb (2001).

    References
    ==========
    Schaffer, J. David. "Multiple objective optimization with vector evaluated genetic algorithms." Proceedings of the first international conference on genetic algorithms and their applications. Psychology Press, 2014.

    Kalyanmoy Deb; Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001.

    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -5.0,
        max_X: np.ndarray | list[float] | float = 10.0,
        test_maximizer: bool = True,
    ):
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
    r"""Poloni's function.

    The original problem is a maximization problem:

    .. math::

        \text{Maximize}
        \begin{cases}
        f_1(\boldsymbol{x}) = -\left[1 + (a_1 - b_1(\boldsymbol{x}))^2 + (a_2 - b_2(\boldsymbol{x}))^2\right] \\
        f_2(\boldsymbol{x}) = -\left[(x_1 + 3)^2 + (x_2 + 1)^2\right]
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
        If True, the returned values describe the maximization problem as defined above.
        If False, they are negated to describe a minimization problem.

    Note
    ====
    Poloni et al. (2000) define the problem as a maximization problem as above.
    Deb (2001) (as POL) negates :math:`f_1` and :math:`f_2` and states it as a
    minimization problem with the same search space; that form is what
    ``test_maximizer=False`` returns.
    Listed as MOP3 in Van Veldhuizen (1999).

    References
    ==========
    Poloni, C., Giurgevich, A., Onesti, L., Pediroda, V., "Hybridization of a multi-objective genetic algorithm, a neural network and a classical optimizer for a complex design problem in fluid dynamics," Computer Methods in Applied Mechanics and Engineering 186(2-4), 403-420 (2000). https://doi.org/10.1016/S0045-7825(99)00394-1

    Kalyanmoy Deb; Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001. (POL)

    David A. Van Veldhuizen; Multiobjective Evolutionary Algorithms: Classifications, Analyses, and New Innovations. Ph.D. thesis, Air Force Institute of Technology, 1999. (MOP3)
    """

    _is_maximization = True

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = -np.pi,
        max_X: np.ndarray | list[float] | float = np.pi,
        test_maximizer: bool = True,
    ):
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
        f1 = -(1.0 + (self._a1 - B1) ** 2 + (self._a2 - B2) ** 2)
        f2 = -((x[:, 0] + 3) ** 2 + (x[:, 1] + 1) ** 2)
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([-62.0, -55.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([-1.0, 0.0])


class ZDT1(MultiTestFunction):
    r"""ZDT's first function.

    .. math::

        \text{Minimize}
        \begin{cases}
        f_1(\boldsymbol{x}) = x_1 \\
        f_2(\boldsymbol{x}) = g(\boldsymbol{x}) h(f_1(\boldsymbol{x}), g(\boldsymbol{x}))
        \end{cases}

        \text{where}
        \begin{cases}
        g(\boldsymbol{x}) = 1 + 9 \sum_{i=2}^{N} x_i / (N - 1) \\
        h(f1(\boldsymbol{x}), g(\boldsymbol{x})) = 1 - \sqrt{f_1(\boldsymbol{x}) / g(\boldsymbol{x})}
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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    References
    ==========
    Zitzler, E., Deb, K., and Thiele, L., "Comparison of Multiobjective Evolutionary Algorithms: Empirical Results," Evolutionary Computation 8(2), 173-195 (2000). doi: 10.1162/106365600568202.
    """

    def __init__(
        self,
        dim: int = 30,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
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
    r"""ZDT's second function.

    .. math::

        \text{Minimize}
        \begin{cases}
        f_1(\boldsymbol{x}) = x_1 \\
        f_2(\boldsymbol{x}) = g(\boldsymbol{x}) h(f_1(\boldsymbol{x}), g(\boldsymbol{x})) \\
        \end{cases}

        \text{where}
        \begin{cases}
        g(\boldsymbol{x}) = 1 + 9 \sum_{i=2}^{N} x_i / (N - 1) \\
        h(f_1(\boldsymbol{x}), g(\boldsymbol{x})) = 1 - \left(f_1(\boldsymbol{x}) / g(\boldsymbol{x})\right)^2
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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    References
    ==========
    Zitzler, E., Deb, K., and Thiele, L., "Comparison of Multiobjective Evolutionary Algorithms: Empirical Results," Evolutionary Computation 8(2), 173-195 (2000). doi: 10.1162/106365600568202.
    """

    def __init__(
        self,
        dim: int = 30,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
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
    r"""ZDT's third function.

    .. math::

        \text{Minimize}
        \begin{cases}
        f_1(\boldsymbol{x}) = x_1 \\
        f_2(\boldsymbol{x}) = g(\boldsymbol{x}) h(f_1(\boldsymbol{x}), g(\boldsymbol{x}))
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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    References
    ==========
    Zitzler, E., Deb, K., and Thiele, L., "Comparison of Multiobjective Evolutionary Algorithms: Empirical Results," Evolutionary Computation 8(2), 173-195 (2000). doi: 10.1162/106365600568202.
    """

    def __init__(
        self,
        dim: int = 30,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
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
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
        f2 = g * h
        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 1.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([1.0, 7.2])


class ZDT4(MultiTestFunction):
    r"""ZDT's fourth function.

    .. math::

        \text{Minimize}
        \begin{cases}
        f_1(\boldsymbol{x}) = x_1 \\
        f_2(\boldsymbol{x}) = g(\boldsymbol{x}) h(f_1(\boldsymbol{x}), g(\boldsymbol{x}))
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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    References
    ==========
    Zitzler, E., Deb, K., and Thiele, L., "Comparison of Multiobjective Evolutionary Algorithms: Empirical Results," Evolutionary Computation 8(2), 173-195 (2000). doi: 10.1162/106365600568202.
    """

    def __init__(
        self,
        dim: int = 10,
        min_X: None | np.ndarray | list[float] | float = None,
        max_X: None | np.ndarray | list[float] | float = None,
        test_maximizer: bool = True,
    ):
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
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h

        return np.c_[f1, f2]

    def _ref_min(self) -> np.ndarray:
        return np.array([0.0, 35.0])

    def _ref_max(self) -> np.ndarray:
        return np.array([1.0, 305.0])


class ZDT6(MultiTestFunction):
    r"""ZDT's sixth function.

    .. math::

        \text{Minimize}
        \begin{cases}
        f_1(\boldsymbol{x}) = 1 - \exp(-4 x_1) \sin^6(6 \pi x_1) \\
        f_2(\boldsymbol{x}) = g(\boldsymbol{x}) h(f_1(\boldsymbol{x}), g(\boldsymbol{x}))
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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    References
    ==========
    Zitzler, E., Deb, K., and Thiele, L., "Comparison of Multiobjective Evolutionary Algorithms: Empirical Results," Evolutionary Computation 8(2), 173-195 (2000). doi: 10.1162/106365600568202.
    """

    def __init__(
        self,
        dim: int = 10,
        min_X: np.ndarray | list[float] | float = 0.0,
        max_X: np.ndarray | list[float] | float = 1.0,
        test_maximizer: bool = True,
    ):
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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    References
    ==========
    Osyczka, A., Kundu, S. A new method to solve generalized multicriteria optimization problems using the simple genetic algorithm. Structural Optimization 10, 94-99 (1995). https://doi.org/10.1007/BF01743536
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        max_X: np.ndarray | list[float] | float = [10.0, 10.0, 5.0, 6.0, 5.0, 10.0],
        test_maximizer: bool = True,
    ):
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
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem as defined above.

    References
    ==========
    Kalyanmoy Deb; Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, 2001.
    """

    def __init__(
        self,
        min_X: np.ndarray | list[float] | float = [0.1, 0.0],
        max_X: np.ndarray | list[float] | float = [1.0, 5.0],
        test_maximizer: bool = True,
    ):
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


class ChankongHaimes(SRN):
    r"""Chankong-Haimes's function.

    This is an alias of :class:`SRN`, named after the origin of the objectives
    (Chankong and Haimes (1983) solved the unconstrained problem; the
    constraints were added by Srinivas and Deb (1994)).
    See :class:`SRN` for the definition and the arguments.

    References
    ==========
    Chankong, V., and Haimes, Y. Y., "Multiobjective decision making: Theory and methodology", North-Holland series in system science and engineering, 1983. (Reprinted by Dover, 2008.)
    """


# The MOP numbers below follow Van Veldhuizen and Lamont (SAC '99), which lists
# three problems (MOP1-3).  Note that Van Veldhuizen's Ph.D. thesis (1999)
# numbers the problems differently (MOP3 = Poloni, MOP5 = Viennet).


class VLMOP1(Schaffer1):
    r"""VL's first function (so-called VLMOP1).

    This is an alias of :class:`Schaffer1` (MOP1 in Van Veldhuizen and Lamont (1999)).
    The numbering follows Van Veldhuizen and Lamont (1999); Van Veldhuizen's
    Ph.D. thesis (1999) numbers the problems differently.
    See :class:`Schaffer1` for the definition and the arguments.

    References
    ==========
    David A. van Veldhuizen and Gary B. Lamont. 1999. Multiobjective evolutionary algorithm test suites. In Proceedings of the 1999 ACM symposium on Applied computing (SAC '99). Association for Computing Machinery, New York, NY, USA, 351-357. https://doi.org/10.1145/298151.298382
    """


class VLMOP2(FonsecaFleming):
    r"""VL's second function (so-called VLMOP2).

    This is an alias of :class:`FonsecaFleming` (MOP2 in Van Veldhuizen and Lamont (1999))
    with the search space :math:`-2 \le x_i \le 2` used there
    (:class:`FonsecaFleming` itself defaults to :math:`-4 \le x_i \le 4`).
    The numbering follows Van Veldhuizen and Lamont (1999); Van Veldhuizen's
    Ph.D. thesis (1999) numbers the problems differently.
    See :class:`FonsecaFleming` for the definition.

    Arguments
    =========
    dim : int, default=2
        Number of dimensions :math:`N`.
    min_X : np.ndarray | list[float] | float, default=-2.0
        Minimum value of the search space :math:`\boldsymbol{x}_{\min}`.
    max_X : np.ndarray | list[float] | float, default=2.0
        Maximum value of the search space :math:`\boldsymbol{x}_{\max}`.
    test_maximizer : bool, default=True
        If True, the returned values are negated to describe a maximization problem.
        If False, they describe the minimization problem.

    References
    ==========
    David A. van Veldhuizen and Gary B. Lamont. 1999. Multiobjective evolutionary algorithm test suites. In Proceedings of the 1999 ACM symposium on Applied computing (SAC '99). Association for Computing Machinery, New York, NY, USA, 351-357. https://doi.org/10.1145/298151.298382
    """

    def __init__(
        self,
        dim: int = 2,
        min_X: np.ndarray | list[float] | float = -2.0,
        max_X: np.ndarray | list[float] | float = 2.0,
        test_maximizer: bool = True,
    ):
        super().__init__(
            dim=dim,
            min_X=min_X,
            max_X=max_X,
            test_maximizer=test_maximizer,
        )


class VLMOP3(Viennet):
    r"""VL's third function (so-called VLMOP3).

    This is an alias of :class:`Viennet` (MOP3 in Van Veldhuizen and Lamont (1999)).
    The numbering follows Van Veldhuizen and Lamont (1999); Van Veldhuizen's
    Ph.D. thesis (1999) numbers the problems differently (this problem is MOP5 there).
    See :class:`Viennet` for the definition and the arguments.

    References
    ==========
    David A. van Veldhuizen and Gary B. Lamont. 1999. Multiobjective evolutionary algorithm test suites. In Proceedings of the 1999 ACM symposium on Applied computing (SAC '99). Association for Computing Machinery, New York, NY, USA, 351-357. https://doi.org/10.1145/298151.298382
    """


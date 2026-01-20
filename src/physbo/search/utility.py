# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import itertools
import numpy as np


class Simulator:
    """Simulator class wrapping the test function for discrete search space.

    Arguments
    =========
    test_X: numpy.ndarray
        The test points. Each row is a candidate.
    test_function: physbo.test_functions.base.TestFunction
        The test function.
    negate: bool, default=False
        If True, the test function value is negated before returning.
    """

    def __init__(self, test_X, test_function, negate=False):
        if test_X.ndim != 2:
            raise ValueError(
                f"ERROR: test_X must be a 2D array, but got {test_X.ndim}D array"
            )

        self.test_X = test_X
        self.test_function = test_function
        self.dim = test_X.shape[1]
        self.sign = -1.0 if negate else 1.0

    def __call__(self, action):
        X = self.test_X[action, :].reshape(-1, self.dim)
        f = self.test_function(X)
        return self.sign * f


def make_grid(
    min_X: list[float] | np.ndarray,
    max_X: list[float] | np.ndarray,
    num_X: int | list[int] | np.ndarray,
    constraint=None,
) -> np.ndarray:
    """Make a grid of points in the search space.

    Arguments
    =========
    min_X: np.ndarray | list[float] | float
        Minimum value of search space for each dimension
    max_X: np.ndarray | list[float] | float
        Maximum value of search space for each dimension
    num_X: int | list[int] | np.ndarray
        Number of points in each dimension

    Returns
    =======
    np.ndarray
        The grid of points in the search space
        The output is a numpy array of shape (N, d), where N is the number of points and d is the dimension of the search space

    Raises
    ======
    ValueError
        If min_X and max_X have different number of dimensions
        If num_X has different number of dimensions from min_X and max_X
    """

    if isinstance(min_X, list):
        min_X = np.array(min_X)
    if isinstance(max_X, list):
        max_X = np.array(max_X)

    if min_X.ndim != 1:
        raise ValueError(
            f"ERROR: min_X must be a 1D array, but got {min_X.ndim}D array"
        )
    if max_X.ndim != 1:
        raise ValueError(
            f"ERROR: max_X must be a 1D array, but got {max_X.ndim}D array"
        )

    if min_X.shape[0] != max_X.shape[0]:
        raise ValueError(
            f"ERROR: min_X and max_X must have the same number of dimensions, but got {min_X.shape[0]} and {max_X.shape[0]}"
        )

    d = min_X.shape[0]

    if isinstance(num_X, int):
        num_X = np.full(d, num_X)
    elif isinstance(num_X, list):
        num_X = np.array(num_X)

    if num_X.ndim != 1:
        raise ValueError(
            f"ERROR: num_X must be a 1D array, but got {num_X.ndim}D array"
        )
    if num_X.shape[0] != d:
        raise ValueError(
            f"ERROR: num_X must have the same number of dimensions as min_X and max_X, but got {num_X.shape[0]} and {d}"
        )

    if constraint is None:
        def constraint(x):
            return True

    ls = [np.linspace(min_X[i], max_X[i], num_X[i]) for i in range(d)]

    # N = np.prod(num_X)
    # X = np.zeros((N, d))
    X_list = []
    for i, x in enumerate(itertools.product(*ls)):
        # X[i, :] = x
        x_arr = np.array(x).reshape(1, -1)
        if constraint(x_arr):
            X_list.append(x)

    return np.array(X_list)


def plot_pareto_front(
    history,
    x=0,
    y=1,
    steps_begin=0,
    steps_end=None,
    ax=None,
    xlim=None,
    ylim=None,
    grid=True,
    style_common: dict={},
    style_pareto_front: dict={},
    style_dominated: dict={},
):
    r"""Plot the Pareto front of the history in the projection to the (x, y)-plane (objective x and y).

    Arguments
    =========
    history: History
        The history of the search.
    x: int, default=0
        The index of the objective to plot on the x-axis.
    y: int, default=1
        The index of the objective to plot on the y-axis.
    steps_begin: int, default=0
        The index (inclusive) of the step to begin plotting.
    steps_end: int, optional
        The index (exclusive) of the step to end plotting.
        If None, plot until the end.
    ax: matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created.
    xlim: tuple, optional
        The x-axis limits. If None, the limits are determined automatically.
    ylim: tuple, optional
        The y-axis limits. If None, the limits are determined automatically.
    grid: bool, default=True
        Whether to plot the grid.
    style_common: dict, optional
        The common setting for plotting the points.
    style_pareto_front: dict, optional
        The setting for plotting the Pareto front.
    style_dominated: dict, optional
        The setting for plotting the dominated points.

    Note
    ====
    - Items in style_* are passed to the `scatter` method of matplotlib.pyplot.
    - For each item, style_pareto_front and style_dominated are higher priority than style_common.
    - By default, the marker is "o".
    - By default, the color is "blue" for dominated points and "red" for Pareto front.
    """
    import matplotlib.pyplot as plt

    front, front_num = history.export_pareto_front()
    min_fx = np.full(2, np.inf)
    max_fx = np.full(2, -np.inf)

    undominated = []
    dominated = []
    if steps_end is None:
        steps_end = history.total_num_search
    for i in range(steps_begin, steps_end):
        if i in front_num:
            undominated.append(i)
        else:
            dominated.append(i)
        min_fx = np.minimum(min_fx, history.fx[i, [x, y]])
        max_fx = np.maximum(max_fx, history.fx[i, [x, y]])

    if ax is None:
        _, ax = plt.subplots()

    style = {"color": "blue", "marker": "o"}
    style.update(style_common)
    style.update(style_dominated)
    ax.scatter(
        history.fx[dominated, x],
        history.fx[dominated, y],
        **style,
    )

    style = {"color": "red", "marker": "o"}
    style.update(style_common)
    style.update(style_pareto_front)
    ax.scatter(
        history.fx[undominated, x],
        history.fx[undominated, y],
        **style,
    )

    ax.grid(grid, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(f"Objective {x + 1}")
    ax.set_ylabel(f"Objective {y + 1}")

    return ax


def plot_pareto_front_all(
    history,
    steps_begin=0,
    steps_end=None,
    ax=None,
    xlim=None,
    ylim=None,
    grid=True,
    style_common: dict={},
    style_pareto_front: dict={},
    style_dominated: dict={},
):
    """Plot the Pareto front of the history for all pairs of objectives.

    Arguments
    =========
    history: History
        The history of the search.
    steps_begin: int, optional
        The index (inclusive) of the step to begin plotting.
        If None, plot from the beginning.
    steps_end: int, optional
        The index (exclusive) of the step to end plotting.
        If None, plot until the end.
    ax: matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created.
    xlim: tuple, optional
        The x-axis limits. If None, the limits are determined automatically.
    ylim: tuple, optional
        The y-axis limits. If None, the limits are determined automatically.
    grid: bool, default=True
        Whether to plot the grid.
    style_common: dict, optional
        The common setting for plotting the points.
    style_pareto_front: dict, optional
        The setting for plotting the Pareto front.
    style_dominated: dict, optional
        The setting for plotting the dominated points.

    Note
    ====
    - Items in style_* are passed to the `scatter` method of matplotlib.pyplot.
    - For each item, style_pareto_front and style_dominated are higher priority than style_common.
    - By default, the marker is "o".
    - By default, the color is "blue" for dominated points and "red" for Pareto front.
    """
    import matplotlib.pyplot as plt

    nobj = history.fx.shape[1]
    nmatrix = nobj - 1
    if ax is None:
        _, ax = plt.subplots(
            nmatrix,
            nmatrix,
            figsize=(5 * nmatrix, 5 * nmatrix),
            sharex="col",
            sharey="row",
        )
    if not isinstance(ax, np.ndarray):
        ax = np.array([[ax]])
    for col in range(nmatrix):
        for row in range(nmatrix):
            i = col
            j = row + 1
            if i >= j:
                ax[row, col].set_visible(False)
                continue
            plot_pareto_front(
                history,
                x=i,
                y=j,
                steps_begin=steps_begin,
                steps_end=steps_end,
                ax=ax[row, col],
                xlim=xlim,
                ylim=ylim,
                grid=grid,
                style_common=style_common,
                style_pareto_front=style_pareto_front,
                style_dominated=style_dominated,
            )

    # Remove redundant labels
    for row in range(nmatrix):
        for col in range(nmatrix):
            if row < nmatrix - 1:
                ax[row, col].set_xlabel(None)
            if col > 0:
                ax[row, col].set_ylabel(None)
    return ax


def show_search_results(history, N):
    n = min(N, history.total_num_search)
    history.show_search_results(n)


def show_search_results_mo(history, N, disp_pareto_set=False):
    n = min(N, history.total_num_search)
    history.show_search_results_mo(n, disp_pareto_set)


def show_start_message_multi_search(N, score=None):
    if score is None:
        score = "random"
    print(f"{N + 1:04}-th multiple probe search ({score})")


def show_interactive_mode(simulator, history):
    if simulator is None and history.total_num_search == 0:
        print("interactive mode starts ... \n ")


def length_vector(t):
    N = len(t) if hasattr(t, "__len__") else 1
    return N


def is_learning(n, interval):
    if interval == 0:
        return n == 0
    elif interval > 0:
        return np.mod(n, interval) == 0
    else:
        return False

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
    """

    def __init__(self, test_X, test_function):
        if test_X.ndim != 2:
            raise ValueError(
                f"ERROR: test_X must be a 2D array, but got {test_X.ndim}D array"
            )

        self.test_X = test_X
        self.test_function = test_function
        self.dim = test_X.shape[1]

    def __call__(self, action):
        X = self.test_X[action, :].reshape(-1, self.dim)
        return self.test_function(X)


def make_grid(
    min_X: list[float] | np.ndarray,
    max_X: list[float] | np.ndarray,
    num_X: int | list[int] | np.ndarray,
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

    ls = [np.linspace(min_X[i], max_X[i], num_X[i]) for i in range(d)]

    N = np.prod(num_X)
    X = np.zeros((N, d))
    for i, x in enumerate(itertools.product(*ls)):
        X[i, :] = x

    return X


def plot_pareto_front(
    history,
    x=0,
    y=1,
    steps_begin=0,
    steps_end=None,
    ax=None,
    color=None,
    pareto_front_color=None,
    dominated_color=None,
    marker=None,
    pareto_front_marker=None,
    dominated_marker=None,
):
    """Plot the Pareto front of the history in the projection to the (x, y)-plane (objective x and y).

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
    color: str, optional
        The color of the points.
    pareto_front_color: str, optional
        The color of the Pareto front.
    dominated_color: str, optional
        The color of the dominated points.
    marker: str, optional
        The marker of the points.
    pareto_front_marker: str, optional
        The marker of the Pareto front.
    dominated_marker: str, optional
        The marker of the dominated points.

    Note
    ====
    - colors:
        - If both color and *_color are provided, *_color is used.
        - If only color is provided, the color is used for both the Pareto front and the dominated points.
        - If neither color nor *_color is provided, the color is set to "red" for the Pareto front and "blue" for the dominated points.
    - markers:
        - If both marker and *_marker are provided, *_marker is used.
        - If only marker is provided, the marker is used for both the Pareto front and the dominated points.
        - If neither marker nor *_marker is provided, the marker is set to "o" for the Pareto front and "o" for the dominated points.

    """
    import matplotlib.pyplot as plt

    if color is not None:
        if pareto_front_color is None:
            pareto_front_color = color
        if dominated_color is None:
            dominated_color = color
    else:
        if pareto_front_color is None:
            pareto_front_color = "red"
        if dominated_color is None:
            dominated_color = "blue"

    if marker is not None:
        if pareto_front_marker is None:
            pareto_front_marker = marker
        if dominated_marker is None:
            dominated_marker = marker
    else:
        if pareto_front_marker is None:
            pareto_front_marker = "o"
        if dominated_marker is None:
            dominated_marker = "o"

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

    ax.scatter(
        history.fx[dominated, x],
        history.fx[dominated, y],
        c=dominated_color,
        marker=dominated_marker,
    )
    ax.scatter(
        history.fx[undominated, x],
        history.fx[undominated, y],
        c=pareto_front_color,
        marker=pareto_front_marker,
    )
    ax.set_xlabel(f"Objective {x + 1}")
    ax.set_ylabel(f"Objective {y + 1}")

    xlim = [min_fx[0], max_fx[0]]
    ylim = [min_fx[1], max_fx[1]]
    return xlim, ylim


def plot_pareto_front_all(
    history,
    steps_begin=0,
    steps_end=None,
    ax=None,
    color=None,
    pareto_front_color=None,
    dominated_color=None,
    marker=None,
    pareto_front_marker=None,
    dominated_marker=None,
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
    color: str, optional
        The color of the points.
    pareto_front_color: str, optional
        The color of the Pareto front.
    dominated_color: str, optional
        The color of the dominated points.
    marker: str, optional
        The marker of the points.
    pareto_front_marker: str, optional
        The marker of the Pareto front.
    dominated_marker: str, optional
        The marker of the dominated points.

    Note
    ====
    - colors:
        - If both color and *_color are provided, *_color is used.
        - If only color is provided, the color is used for both the Pareto front and the dominated points.
        - If neither color nor *_color is provided, the color is set to "red" for the Pareto front and "blue" for the dominated points.
    - markers:
        - If both marker and *_marker are provided, *_marker is used.
        - If only marker is provided, the marker is used for both the Pareto front and the dominated points.
        - If neither marker nor *_marker is provided, the marker is set to "o" for the Pareto front and "o" for the dominated points.

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
                color=color,
                pareto_front_color=pareto_front_color,
                dominated_color=dominated_color,
                marker=marker,
                pareto_front_marker=pareto_front_marker,
                dominated_marker=dominated_marker,
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
    history.show_search_results(N)


def show_search_results_mo(history, N, disp_pareto_set=False):
    history.show_search_results_mo(N, disp_pareto_set)


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

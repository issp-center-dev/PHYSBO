# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np


class Simulator:
    """Simulator class wrapping the test function for discrete search space.

    Args
    ======
        test_X: The test points. Each row is a candidate.
        test_function: The test function.
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
        steps_end = history.num_runs
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

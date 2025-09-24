# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np


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

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The small search problems shared by the regression tests of the policies.

A 5 x 5 grid on [0, 1]^2 (discrete) or the box itself (range), a single
objective and a pair of objectives, a cheap learning configuration, and the
simulator of each kind of policy.
"""

from itertools import product

import numpy as np

import physbo

MIN_X = np.array([0.0, 0.0])
MAX_X = np.array([1.0, 1.0])


def grid_X():
    a = np.linspace(0.0, 1.0, 5)
    return np.array(list(product(a, a)))


def f_single(x):
    x = np.atleast_2d(x)
    return -np.sum((x - 0.6) ** 2, axis=-1)


def f_multi(x):
    x = np.atleast_2d(x)
    return np.stack(
        [-np.sum((x - 0.25) ** 2, axis=-1), -np.sum((x - 0.75) ** 2, axis=-1)],
        axis=-1,
    )


def make_config():
    """Default search config (alpha=1.0) with 3 virtual samples and a cheap fit."""
    config = physbo.misc.SetConfig()
    config.search.multi_probe_num_sampling = 3
    config.learning.is_disp = False
    config.learning.num_init_params_search = 5
    config.learning.max_epoch = 50
    config.learning.max_epoch_init_params_search = 10
    return config


def make_simulator(kind):
    X = grid_X()
    if kind == "discrete":
        return lambda action: f_single(X[action])
    if kind == "range":
        return lambda x: f_single(x)
    if kind in ("discrete_unified", "discrete_multi"):
        return lambda action: f_multi(X[action])
    return lambda x: f_multi(x)


def is_range(kind):
    return kind.startswith("range")

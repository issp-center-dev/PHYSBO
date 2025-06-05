# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function

from itertools import product

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")


def f(x):
    return -np.sum((x - 0.5) ** 2)


class simulator:
    def __init__(self):
        self.nslice = 11
        self.dim = 2
        self.N = self.nslice**self.dim
        self.X = np.zeros((self.N, self.dim))
        for i, x in enumerate(
            product(np.linspace(0.0, 1.0, self.nslice), repeat=self.dim)
        ):
            self.X[i, :] = list(x)

    def __call__(self, actions):
        return np.array([f(self.X[action, :]) for action in actions])


def test_interactive():
    sim = simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.discrete.Policy(test_X=sim.X)
    policy.set_seed(12345)

    actions = policy.random_search(
        max_num_probes=1, num_search_each_probe=nrand, simulator=None
    )
    targets = sim(actions)
    print(actions)
    print(targets)
    policy.write(actions, targets)
    physbo.search.utility.show_search_results(policy.history, nrand)

    actions = policy.bayes_search(
        max_num_probes=1, num_search_each_probe=nsearch, simulator=None, score="TS"
    )
    targets = sim(actions)
    print(actions)
    print(targets)
    policy.write(actions, targets)
    physbo.search.utility.show_search_results(policy.history, nsearch)

    res = policy.history
    best_fx, best_action = res.export_all_sequence_best_fx()
    print(best_fx)
    print(best_action)
    assert best_fx[-1] == pytest.approx(0.0, abs=0.001)
    assert best_action[-1] == 60

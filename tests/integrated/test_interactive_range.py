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


def sim(x):
    return -np.sum(x**2, axis=1)


def test_interactive():
    nrand = 10
    nsearch = 1
    min_X = np.array([-1.0, -1.0])
    max_X = np.array([1.0, 1.0])
    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
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
    ref_best_fx = -0.05486187932603481
    ref_best_action = np.array([0.1354500581633733, 0.1910894059585031])
    assert best_fx[-1] == pytest.approx(ref_best_fx, abs=0.001)
    assert np.allclose(best_action[-1], ref_best_action, atol=0.1)


def test_policy_with_initial_data():
    min_X = np.array([-1.0, -1.0])
    max_X = np.array([1.0, 1.0])
    solution = np.array([[0.0, 0.0]])
    solution_target = sim(solution)
    action = np.array([[1.0, 1.0], solution[0, :], [-1.0, -1.0]])
    target = sim(action)
    print(action)
    print(target)
    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X, initial_data=(action, target))
    best_fx, best_action = policy.history.export_all_sequence_best_fx()
    print(best_action)
    assert best_fx[-1] == pytest.approx(solution_target, abs=0.001)
    assert np.allclose(best_action[-1], solution, atol=0.1)

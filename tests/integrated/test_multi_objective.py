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


def vlmop2_minus(x):
    n = x.shape[1]
    y1 = 1 - np.exp(-1 * np.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
    y2 = 1 - np.exp(-1 * np.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))

    return np.c_[-y1, -y2]


class simulator:
    def __init__(self):
        a = np.linspace(-2, 2, 11)
        self.X = np.array(list(product(a, a)))
        self.t = vlmop2_minus(self.X)

    def __call__(self, action):
        return self.t[action]


def test_multi_objective_EHVI():
    sim = simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.discrete_multi.policy(test_X=sim.X, num_objectives=2)
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(max_num_probes=nsearch, simulator=sim, score="EHVI")
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.2392468337984477
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("EHVI", xs=sim.X)


def test_multi_objective_HVPI():
    sim = simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.discrete_multi.policy(test_X=sim.X, num_objectives=2)
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(max_num_probes=nsearch, simulator=sim, score="HVPI")
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.25322554948754283
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("HVPI", xs=sim.X)


def test_multi_objective_TS():
    sim = simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.discrete_multi.policy(test_X=sim.X, num_objectives=2)
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(max_num_probes=nsearch, simulator=sim, score="TS")
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.17724278568874974
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("TS", xs=sim.X)


def test_multi_objective_EHVI_rand():
    sim = simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.discrete_multi.policy(test_X=sim.X, num_objectives=2)
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(
        max_num_probes=nsearch, simulator=sim, score="EHVI", num_rand_basis=100
    )
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.08400891973743863
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("EHVI", xs=sim.X)


def test_multi_objective_HVPI_rand():
    sim = simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.discrete_multi.policy(test_X=sim.X, num_objectives=2)
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(
        max_num_probes=nsearch, simulator=sim, score="HVPI", num_rand_basis=100
    )
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.13374086517165518
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("HVPI", xs=sim.X)


def test_multi_objective_TS_rand():
    sim = simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.discrete_multi.policy(test_X=sim.X, num_objectives=2)
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(
        max_num_probes=nsearch, simulator=sim, score="TS", num_rand_basis=100
    )
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.134435814966692
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("TS", xs=sim.X)

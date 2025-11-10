# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")


def vlmop2_minus(x):
    n = x.shape[1]
    y1 = 1 - np.exp(-1 * np.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
    y2 = 1 - np.exp(-1 * np.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))

    return np.c_[-y1, -y2]


class Simulator:
    def __init__(self):
        pass

    def __call__(self, X):
        # X: (N, 2) or (2,) ndarray
        X = np.atleast_2d(X)
        return vlmop2_minus(X)


def test_multi_objective_EHVI():
    min_X = np.array([-2.0, -2.0])
    max_X = np.array([2.0, 2.0])
    sim = Simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.range_multi.Policy(
        min_X=min_X, max_X=max_X, num_objectives=2
    )
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(max_num_probes=nsearch, simulator=sim, score="EHVI")
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.21186716996705868
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("EHVI", xs=np.array([[0.0, 0.0]]))


def test_multi_objective_HVPI():
    min_X = np.array([-2.0, -2.0])
    max_X = np.array([2.0, 2.0])
    sim = Simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.range_multi.Policy(
        min_X=min_X, max_X=max_X, num_objectives=2
    )
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(max_num_probes=nsearch, simulator=sim, score="HVPI")
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.2517842814632759
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("HVPI", xs=np.array([[0.0, 0.0]]))


def test_multi_objective_TS():
    min_X = np.array([-2.0, -2.0])
    max_X = np.array([2.0, 2.0])
    sim = Simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.range_multi.Policy(
        min_X=min_X, max_X=max_X, num_objectives=2
    )
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(max_num_probes=nsearch, simulator=sim, score="TS")
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.13505409808944357
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("TS", xs=np.array([[0.0, 0.0]]))


def test_multi_objective_EHVI_rand():
    min_X = np.array([-2.0, -2.0])
    max_X = np.array([2.0, 2.0])
    sim = Simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.range_multi.Policy(
        min_X=min_X, max_X=max_X, num_objectives=2
    )
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(
        max_num_probes=nsearch, simulator=sim, score="EHVI", num_rand_basis=100
    )
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.22722097654091666
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("EHVI", xs=np.array([[0.0, 0.0]]))


def test_multi_objective_HVPI_rand():
    min_X = np.array([-2.0, -2.0])
    max_X = np.array([2.0, 2.0])
    sim = Simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.range_multi.Policy(
        min_X=min_X, max_X=max_X, num_objectives=2
    )
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(
        max_num_probes=nsearch, simulator=sim, score="HVPI", num_rand_basis=100
    )
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.22381132051342423
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("HVPI", xs=np.array([[0.0, 0.0]]))


def test_multi_objective_TS_rand():
    min_X = np.array([-2.0, -2.0])
    max_X = np.array([2.0, 2.0])
    sim = Simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.range_multi.Policy(
        min_X=min_X, max_X=max_X, num_objectives=2
    )
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(
        max_num_probes=nsearch, simulator=sim, score="TS", num_rand_basis=100
    )
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.16007470367651644
    assert vid == pytest.approx(vid_ref, rel=1e-3)
    policy.get_score("TS", xs=np.array([[0.0, 0.0]]))


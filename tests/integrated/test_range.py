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


def f(x):
    return -np.sum((x - 0.5) ** 2, axis=-1)


class Simulator:
    def __init__(self):
        pass

    def __call__(self, X):
        # X: (N, 2) or (2,) ndarray
        X = np.atleast_2d(X)
        return f(X)


def test_random_search():
    min_X = np.array([0.0, 0.0])
    max_X = np.array([1.0, 1.0])
    sim = Simulator()
    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
    policy.set_seed(12345)
    nsearch = 20
    res = policy.random_search(max_num_probes=nsearch, simulator=sim)
    best_fx, best_X = res.export_all_sequence_best_fx()
    print(best_fx[-1])
    print(best_X[-1])
    assert best_fx[-1] == pytest.approx(0.0, abs=0.01)
    assert np.allclose(best_X[-1], [0.5, 0.5], atol=0.1)


def test_bayes_search():
    min_X = np.array([0.0, 0.0])
    max_X = np.array([1.0, 1.0])
    sim = Simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(max_num_probes=nsearch, simulator=sim, score="EI")
    best_fx, best_X = res.export_all_sequence_best_fx()
    print(best_fx)
    print(best_X)
    assert best_fx[-1] == pytest.approx(0.0, abs=0.01)
    assert np.allclose(best_X[-1], [0.5, 0.5], atol=0.1)
    policy.get_score("EI", xs=np.array([[0.5, 0.5]]))


def test_bayes_search_rand():
    min_X = np.array([0.0, 0.0])
    max_X = np.array([1.0, 1.0])
    sim = Simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(
        max_num_probes=nsearch, simulator=sim, score="EI", num_rand_basis=100
    )
    best_fx, best_X = res.export_all_sequence_best_fx()
    print(best_fx)
    print(best_X)
    assert best_fx[-1] == pytest.approx(0.0, abs=0.02)
    assert np.allclose(best_X[-1], [0.5, 0.5], atol=0.1)
    policy.get_score("EI", xs=np.array([[0.5, 0.5]]))

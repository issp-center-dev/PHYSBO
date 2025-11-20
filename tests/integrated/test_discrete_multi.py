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


class TestDiscreteMulti:
    def setup_method(self):
        self.sim = simulator()
        self.nrand = 10
        self.nsearch = 5
        self.num_rand_basis = 100
        self.policy = physbo.search.discrete_multi.Policy(
            test_X=self.sim.X, num_objectives=2
        )
        self.policy.set_seed(12345)

    @pytest.mark.parametrize(
        "score, vid_ref",
        [
            ("EHVI", 0.2392468337984477),
            ("HVPI", 0.25322554948754283),
            ("TS", 0.17724278568874974),
        ],
    )
    def test_multi_objective(self, score, vid_ref):
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        res = self.policy.bayes_search(
            max_num_probes=self.nsearch, simulator=self.sim, score=score
        )
        vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
        assert vid == pytest.approx(vid_ref, rel=1e-3)
        # test to run without error
        self.policy.get_score(score, xs=self.sim.X)

    @pytest.mark.parametrize(
        "score, vid_ref",
        [
            ("EHVI", 0.23838772016010945),
            ("HVPI", 0.25322554948754283),
            ("TS", 0.20394729383806942),
        ],
    )
    def test_multi_objective_rand(self, score, vid_ref):
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        res = self.policy.bayes_search(
            max_num_probes=self.nsearch,
            simulator=self.sim,
            score=score,
            num_rand_basis=self.num_rand_basis,
        )
        vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
        assert vid == pytest.approx(vid_ref, rel=1e-3)
        # test to run without error
        self.policy.get_score(score, xs=self.sim.X)

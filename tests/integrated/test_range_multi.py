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


class TestRangeMulti:
    def setup_method(self):
        self.sim = Simulator()
        self.min_X = np.array([-2.0, -2.0])
        self.max_X = np.array([2.0, 2.0])
        self.nrand = 10
        self.nsearch = 5
        self.num_rand_basis = 100
        self.policy = physbo.search.range_multi.Policy(
            min_X=self.min_X, max_X=self.max_X, num_objectives=2
        )
        self.policy.set_seed(12345)


    @pytest.mark.parametrize(
        "score, vid_ref",
        [
            ("EHVI", 0.21186716996705868),
            ("HVPI", 0.2517842814632759),
            ("TS", 0.13505409808944357),
        ],
    )
    def test_multi_objective(self, score, vid_ref):
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        res = self.policy.bayes_search(max_num_probes=self.nsearch, simulator=self.sim, score=score)
        vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
        assert vid == pytest.approx(vid_ref, rel=1e-3)
        self.policy.get_score(score, xs=np.array([[0.0, 0.0]]))


    @pytest.mark.parametrize(
        "score, vid_ref",
        [
            ("EHVI", 0.22722097654091666),
            ("HVPI", 0.22381132051342423),
            ("TS", 0.16007470367651644),
        ],
    )
    def test_multi_objective_rand(self, score, vid_ref):
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        res = self.policy.bayes_search(max_num_probes=self.nsearch, simulator=self.sim, score=score, num_rand_basis=self.num_rand_basis)
        vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
        assert vid == pytest.approx(vid_ref, rel=1e-3)
        self.policy.get_score(score, xs=np.array([[0.0, 0.0]]))

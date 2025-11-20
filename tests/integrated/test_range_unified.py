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


class TestRangeUnified:
    def setup_method(self):
        self.sim = Simulator()
        self.min_X = np.array([-2.0, -2.0])
        self.max_X = np.array([2.0, 2.0])
        self.nrand = 10
        self.nsearch = 5
        self.num_rand_basis = 100
        self.unify_method = {
            "ParEGO": physbo.search.unify.ParEGO(
                num_objectives=2, weight_sum=0.5, weight_max=0.5, weights=None
            ),
            "NDS": physbo.search.unify.NDS(num_objectives=2, rank_max=10),
        }
        self.policy = physbo.search.range_unified.Policy(
            min_X=self.min_X, max_X=self.max_X, num_objectives=2
        )
        self.policy.set_seed(12345)

        self.vid_ref = {
            ("PI", "ParEGO", 0): 0.0999442245812544,
            ("EI", "ParEGO", 0): 0.10316945167256875,
            ("TS", "ParEGO", 0): 0.10070811168862714,

            ("PI", "NDS", 0): 0.1119472197450172,
            ("EI", "NDS", 0): 0.12220607591910027,
            ("TS", "NDS", 0): 0.15968650725088385,

            ("PI", "ParEGO", 10): 0.09422696928304863,
            ("EI", "ParEGO", 10): 0.09468429709321047,
            ("TS", "ParEGO", 10): 0.09480830402031681,

            ("PI", "NDS", 10): 0.10152195377567941,
            ("EI", "NDS", 10): 0.09534518826538585,
            ("TS", "NDS", 10): 0.09482336791327406,
        }


    @pytest.mark.parametrize("score", ["PI", "EI", "TS"])
    @pytest.mark.parametrize("unify_method", ["ParEGO", "NDS"])
    @pytest.mark.parametrize("num_rand_basis", [0, 10])
    def test_multi_objective(self, score, unify_method, num_rand_basis):
        vid_ref = self.vid_ref[(score, unify_method, num_rand_basis)]
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        res = self.policy.bayes_search(
            max_num_probes=self.nsearch,
            simulator=self.sim,
            score=score,
            unify_method=self.unify_method[unify_method],
            num_rand_basis=num_rand_basis,
        )
        vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
        assert vid == pytest.approx(vid_ref, rel=1e-3)
        self.policy.get_score(score, xs=np.array([[0.0, 0.0]]))

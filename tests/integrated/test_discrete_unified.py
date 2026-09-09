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
        X = self.X[action]
        t = vlmop2_minus(X)
        return t


class TestDiscreteUnify:
    def setup_method(self):
        self.sim = simulator()
        self.nrand = 10
        self.nsearch = 5
        self.interval = 2
        self.policy = physbo.search.discrete_unified.Policy(
            test_X=self.sim.X, num_objectives=2
        )
        self.policy.set_seed(12345)
        self.unify_method = {
            "ParEGO": physbo.search.unify.ParEGO(
                num_objectives=2, weight_sum=0.5, weight_min=0.5, weights=None
            ),
            "NDS": physbo.search.unify.NDS(num_objectives=2, rank_max=10),
        }

        self.vid_ref = {
            ("PI", "ParEGO", 0): 0.18464354551889506,
            ("EI", "ParEGO", 0): 0.11076136561520267,
            ("TS", "ParEGO", 0): 0.10003702093436417,
            ("PI", "NDS", 0): 0.18343541849063327,
            ("EI", "NDS", 0): 0.18343541849063327,
            ("TS", "NDS", 0): 0.2163315871396495,
            ("PI", "ParEGO", 10): 0.14710521168216184,
            ("EI", "ParEGO", 10): 0.13288175153331716,
            ("TS", "ParEGO", 10): 0.13374086517165518,
            ("PI", "NDS", 10): 0.1954734380647939,
            ("EI", "NDS", 10): 0.16112209109885733,
            ("TS", "NDS", 10): 0.09917790729602605,
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
            interval=self.interval,
            num_rand_basis=num_rand_basis,
            is_disp=False,
        )
        vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
        assert vid == pytest.approx(vid_ref, rel=1e-3)
        # test to run without error
        self.policy.get_score(score, xs=self.sim.X)

    # Seeded reference hypervolumes for score="UCB" (ucb_beta=2.0), matching
    # the deterministic-reference style used in test_multi_objective. These
    # differ from the PI/EI/TS references above, confirming UCB drives a
    # distinct search path through the unified policy.
    vid_ref_ucb = {
        "ParEGO": 0.08592726518218119,
        "NDS": 0.16368641865433553,
    }

    @pytest.mark.parametrize("unify_method", ["ParEGO", "NDS"])
    def test_ucb(self, unify_method):
        # Verify score="UCB" is correctly threaded through the unified
        # single-objective policy by pinning the seeded hypervolume, then check
        # the ucb_beta monotonicity invariant on get_score as a secondary check.
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        res = self.policy.bayes_search(
            max_num_probes=self.nsearch,
            simulator=self.sim,
            score="UCB",
            ucb_beta=2.0,
            unify_method=self.unify_method[unify_method],
            interval=self.interval,
            is_disp=False,
        )
        vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
        assert vid == pytest.approx(self.vid_ref_ucb[unify_method], rel=1e-3)
        # ucb_beta adds the (non-negative) std to the mean, so a larger beta
        # never decreases the score of any candidate.
        f_low = self.policy.get_score("UCB", xs=self.sim.X, ucb_beta=0.0)
        f_high = self.policy.get_score("UCB", xs=self.sim.X, ucb_beta=5.0)
        assert np.all(np.isfinite(f_low)) and np.all(np.isfinite(f_high))
        assert np.all(f_high >= f_low - 1e-8)
        assert not np.allclose(f_low, f_high)

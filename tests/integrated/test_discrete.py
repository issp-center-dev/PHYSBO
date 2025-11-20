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
    return -np.sum((x - 0.5) ** 2, axis=1)


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

    def __call__(self, action):
        return f(self.X[action, :])


class TestDiscrete:
    def setup_method(self):
        self.sim = simulator()
        self.policy = physbo.search.discrete.Policy(test_X=self.sim.X)
        self.policy.set_seed(12345)
        self.nrand = 10
        self.nsearch = 5
        self.num_rand_basis = 100

    def test_random_search(self):
        nsearch = 20
        res = self.policy.random_search(max_num_probes=nsearch, simulator=self.sim)
        best_fx, best_action = res.export_all_sequence_best_fx()
        print(best_fx[-1])
        print(best_action[-1])
        assert best_fx[-1] == pytest.approx(-0.01, abs=0.001)
        assert best_action[-1] == 61

    @pytest.mark.parametrize(
        "score, best_fx_ref, best_action_ref",
        [
            ("EI", 0.0, 60),
            ("PI", 0.0, 60),
            ("TS", 0.0, 60),
        ],
    )
    def test_bayes_search(self, score, best_fx_ref, best_action_ref):
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        res = self.policy.bayes_search(
            max_num_probes=self.nsearch, simulator=self.sim, score=score
        )
        best_fx, best_action = res.export_all_sequence_best_fx()
        print(best_fx[-1])
        print(best_action[-1])
        assert best_fx[-1] == pytest.approx(best_fx_ref, abs=0.001)
        assert best_action[-1] == best_action_ref

    @pytest.mark.parametrize(
        "score, best_fx_ref, best_action_ref",
        [
            ("EI", 0.0, 60),
            ("PI", 0.0, 60),
            ("TS", 0.0, 60),
        ],
    )
    def test_bayes_search_rand(self, score, best_fx_ref, best_action_ref):
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        res = self.policy.bayes_search(
            max_num_probes=self.nsearch,
            simulator=self.sim,
            score=score,
            num_rand_basis=self.num_rand_basis,
        )
        best_fx, best_action = res.export_all_sequence_best_fx()
        print(best_fx[-1])
        print(best_action[-1])
        assert best_fx[-1] == pytest.approx(best_fx_ref, abs=0.001)
        assert best_action[-1] == best_action_ref

    def test_interactive(self):
        actions = self.policy.random_search(
            max_num_probes=1, num_search_each_probe=self.nrand, simulator=None
        )
        targets = self.sim(actions)
        self.policy.write(actions, targets)
        physbo.search.utility.show_search_results(self.policy.history, self.nrand)

        actions = self.policy.bayes_search(
            max_num_probes=1,
            num_search_each_probe=self.nsearch,
            simulator=None,
            score="TS",
        )
        targets = self.sim(actions)
        self.policy.write(actions, targets)
        physbo.search.utility.show_search_results(self.policy.history, self.nsearch)

        res = self.policy.history
        best_fx, best_action = res.export_all_sequence_best_fx()
        print(best_fx)
        print(best_action)
        assert best_fx[-1] == pytest.approx(0.0, abs=0.001)
        assert best_action[-1] == 60

    def test_policy_with_initial_data(self):
        actions = self.policy.random_search(
            max_num_probes=1, num_search_each_probe=self.nrand, simulator=None
        )
        targets = self.sim(actions)

        policy_2 = physbo.search.discrete.Policy(
            test_X=self.sim.X, initial_data=(actions, targets)
        )
        best_fx, best_action = policy_2.history.export_all_sequence_best_fx()
        ref_best_fx = np.max(targets)
        ref_best_action = actions[np.argmax(targets)]
        assert best_fx[-1] == pytest.approx(ref_best_fx, abs=0.001)
        assert np.allclose(best_action[-1], ref_best_action, atol=0.1)

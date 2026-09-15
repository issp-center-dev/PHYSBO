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


class TestRange:
    def setup_method(self):
        self.sim = Simulator()
        self.min_X = np.array([0.0, 0.0])
        self.max_X = np.array([1.0, 1.0])
        self.policy = physbo.search.range.Policy(min_X=self.min_X, max_X=self.max_X)
        self.policy.set_seed(12345)
        self.nsearch_random = 20
        self.nrand = 10
        self.nsearch = 5
        self.num_rand_basis = 100

    def test_random_search(self):
        res = self.policy.random_search(
            max_num_probes=self.nsearch_random, simulator=self.sim
        )
        best_fx, best_X = res.export_all_sequence_best_fx()
        print(best_fx[-1])
        print(best_X[-1])
        assert best_fx[-1] == pytest.approx(0.0, abs=0.01)
        assert np.allclose(best_X[-1], [0.5, 0.5], atol=0.1)

    def test_bayes_search(self):
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        res = self.policy.bayes_search(
            max_num_probes=self.nsearch, simulator=self.sim, score="EI"
        )
        best_fx, best_X = res.export_all_sequence_best_fx()
        print(best_fx)
        print(best_X)
        assert best_fx[-1] == pytest.approx(0.0, abs=0.01)
        assert np.allclose(best_X[-1], [0.5, 0.5], atol=0.1)
        self.policy.get_score("EI", xs=np.array([[0.5, 0.5]]))

    def test_bayes_search_rand(self):
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        res = self.policy.bayes_search(
            max_num_probes=self.nsearch,
            simulator=self.sim,
            score="EI",
            num_rand_basis=self.num_rand_basis,
        )
        best_fx, best_X = res.export_all_sequence_best_fx()
        print(best_fx)
        print(best_X)
        assert best_fx[-1] == pytest.approx(0.0, abs=0.02)
        assert np.allclose(best_X[-1], [0.5, 0.5], atol=0.1)
        self.policy.get_score("EI", xs=np.array([[0.5, 0.5]]))

    def test_bayes_search_ucb(self):
        self.policy.random_search(max_num_probes=self.nrand, simulator=self.sim)
        # Best objective reached by the random phase alone, used to confirm the
        # UCB search step actually contributes beyond random sampling.
        best_fx_random, _ = self.policy.history.export_all_sequence_best_fx()
        best_after_random = best_fx_random[-1]

        res = self.policy.bayes_search(
            max_num_probes=self.nsearch,
            simulator=self.sim,
            score="UCB",
            ucb_beta=2.0,
        )
        best_fx, best_X = res.export_all_sequence_best_fx()
        print(best_fx)
        print(best_X)
        # The UCB search must improve on the random-only best (this exact
        # comparison is the primary regression check). The convergence bounds
        # below match the neighboring EI test (abs=0.01, atol=0.1) to avoid
        # brittleness; note the random phase's best (~ -0.0137 for this seed)
        # still fails abs=0.01, so reaching it requires the UCB search to work.
        assert best_fx[-1] > best_after_random
        assert best_fx[-1] == pytest.approx(0.0, abs=0.01)
        assert np.allclose(best_X[-1], [0.5, 0.5], atol=0.1)
        # ucb_beta adds the (non-negative) std to the mean, so a larger beta
        # never decreases the score of any candidate.
        xs = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.1]])
        f_low = self.policy.get_score("UCB", xs=xs, ucb_beta=0.0)
        f_high = self.policy.get_score("UCB", xs=xs, ucb_beta=5.0)
        assert np.all(np.isfinite(f_low)) and np.all(np.isfinite(f_high))
        assert np.all(f_high >= f_low - 1e-8)
        # ucb_beta must actually be threaded through the range get_score path;
        # if it were ignored (defaulted) the two score arrays would be identical.
        assert not np.allclose(f_low, f_high)

    def test_interactive(self):
        actions = self.policy.random_search(
            max_num_probes=1, num_search_each_probe=self.nrand, simulator=None
        )
        targets = self.sim(actions)
        print(actions)
        print(targets)
        self.policy.write(actions, targets)
        physbo.search.utility.show_search_results(self.policy.history, self.nrand)

        actions = self.policy.bayes_search(max_num_probes=1, simulator=None, score="TS")
        targets = self.sim(actions)
        self.policy.write(actions, targets)
        physbo.search.utility.show_search_results(self.policy.history, self.nsearch)

        res = self.policy.history
        best_fx, best_action = res.export_all_sequence_best_fx()
        ref_best_fx = -0.013715469831508703
        ref_best_action = np.array([0.5677250290816866, 0.5955447029792516])
        assert best_fx[-1] == pytest.approx(ref_best_fx, abs=0.001)
        assert np.allclose(best_action[-1], ref_best_action, atol=0.1)

    def test_policy_with_initial_data(self):
        solution = np.array([[0.5, 0.5]])
        solution_target = self.sim(solution)
        action = np.array([self.min_X, solution[0, :], self.max_X])
        target = self.sim(action)
        self.policy = physbo.search.range.Policy(
            min_X=self.min_X, max_X=self.max_X, initial_data=(action, target)
        )
        best_fx, best_action = self.policy.history.export_all_sequence_best_fx()
        assert best_fx[-1] == pytest.approx(solution_target, abs=0.001)
        assert np.allclose(best_action[-1], solution, atol=0.1)

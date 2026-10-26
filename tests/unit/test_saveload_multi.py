# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Save/load round-trip tests for the multi-objective and unified policies."""

import os
from itertools import product

import numpy as np
import numpy.testing
import pytest

physbo = pytest.importorskip("physbo")


def vlmop2_minus(x):
    n = x.shape[1]
    y1 = 1 - np.exp(-1 * np.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
    y2 = 1 - np.exp(-1 * np.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))
    return np.c_[-y1, -y2]


@pytest.fixture
def test_X():
    a = np.linspace(-2, 2, 11)
    return np.array(list(product(a, a)))


class DiscreteSimulator:
    def __init__(self, X):
        self.t = vlmop2_minus(X)

    def __call__(self, action):
        return self.t[action]


class RangeSimulator:
    def __call__(self, X):
        return vlmop2_minus(np.atleast_2d(X))


min_X = np.array([-2.0, -2.0])
max_X = np.array([2.0, 2.0])


def assert_same_pareto(pareto1, pareto2):
    front1, step1 = pareto1.export_front()
    front2, step2 = pareto2.export_front()
    numpy.testing.assert_array_equal(front1, front2)
    numpy.testing.assert_array_equal(step1, step2)


def test_saveload_discrete_multi(test_X, tmp_path):
    sim = DiscreteSimulator(test_X)
    policy = physbo.search.discrete_multi.Policy(test_X=test_X, num_objectives=2)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=5, simulator=sim)
    policy.bayes_search(max_num_probes=1, simulator=sim, score="EHVI", interval=0)

    file_history = os.path.join(tmp_path, "history.npz")
    file_training = os.path.join(tmp_path, "training.pickle")
    file_predictor = os.path.join(tmp_path, "predictor.pickle")
    policy.save(file_history, file_training, file_predictor)

    policy2 = physbo.search.discrete_multi.Policy(test_X=test_X, num_objectives=2)
    policy2.load(file_history, file_training, file_predictor)

    N = policy.history.total_num_search
    assert policy2.history.total_num_search == N
    assert policy2.history.num_runs == policy.history.num_runs
    numpy.testing.assert_array_equal(
        policy2.history.fx[:N], policy.history.fx[:N]
    )
    numpy.testing.assert_array_equal(
        policy2.history.chosen_actions[:N], policy.history.chosen_actions[:N]
    )
    assert_same_pareto(policy.history.pareto, policy2.history.pareto)
    numpy.testing.assert_array_equal(policy2.actions, policy.actions)
    numpy.testing.assert_array_equal(policy2.training.X, policy.training.X)
    numpy.testing.assert_array_equal(policy2.training.t, policy.training.t)

    # the loaded policy is usable: prediction works without re-training
    fmean = policy2.get_post_fmean(test_X)
    assert fmean.shape == (len(test_X), 2)


def test_saveload_discrete_multi_without_training_file(test_X, tmp_path):
    # training data must be reconstructed from the history if not given
    sim = DiscreteSimulator(test_X)
    policy = physbo.search.discrete_multi.Policy(test_X=test_X, num_objectives=2)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=5, simulator=sim)

    file_history = os.path.join(tmp_path, "history.npz")
    policy.save(file_history)

    policy2 = physbo.search.discrete_multi.Policy(test_X=test_X, num_objectives=2)
    policy2.load(file_history)

    numpy.testing.assert_array_equal(policy2.training.X, policy.training.X)
    numpy.testing.assert_array_equal(
        policy2.training.t.reshape(policy.training.t.shape), policy.training.t
    )
    numpy.testing.assert_array_equal(policy2.actions, policy.actions)


def test_saveload_range_multi(tmp_path):
    sim = RangeSimulator()
    policy = physbo.search.range_multi.Policy(
        min_X=min_X, max_X=max_X, num_objectives=2
    )
    policy.set_seed(12345)
    policy.random_search(max_num_probes=5, simulator=sim)
    policy.bayes_search(max_num_probes=1, simulator=sim, score="EHVI", interval=0)

    file_history = os.path.join(tmp_path, "history.npz")
    file_training = os.path.join(tmp_path, "training.pickle")
    file_predictor = os.path.join(tmp_path, "predictor.pickle")
    policy.save(file_history, file_training, file_predictor)

    policy2 = physbo.search.range_multi.Policy(
        min_X=min_X, max_X=max_X, num_objectives=2
    )
    policy2.load(file_history, file_training, file_predictor)

    N = policy.history.total_num_search
    assert policy2.history.total_num_search == N
    assert policy2.history.num_runs == policy.history.num_runs
    numpy.testing.assert_array_equal(
        policy2.history.fx[:N], policy.history.fx[:N]
    )
    numpy.testing.assert_array_equal(
        policy2.history.action_X[:N], policy.history.action_X[:N]
    )
    assert_same_pareto(policy.history.pareto, policy2.history.pareto)
    numpy.testing.assert_array_equal(policy2.training.X, policy.training.X)
    numpy.testing.assert_array_equal(policy2.training.t, policy.training.t)


def test_saveload_discrete_unified(test_X, tmp_path):
    sim = DiscreteSimulator(test_X)
    unify = physbo.search.unify.ParEGO(num_objectives=2)
    policy = physbo.search.discrete_unified.Policy(test_X=test_X, num_objectives=2)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=5, simulator=sim)
    policy.bayes_search(
        max_num_probes=1, simulator=sim, score="EI", unify_method=unify, interval=0
    )

    file_history = os.path.join(tmp_path, "history.npz")
    # Variable.save appends ".npz" via numpy.savez, so the name passed to
    # load must carry the .npz extension explicitly
    file_training = os.path.join(tmp_path, "training.npz")
    file_predictor = os.path.join(tmp_path, "predictor.dump")
    policy.save(file_history, file_training, file_predictor)

    policy2 = physbo.search.discrete_unified.Policy(test_X=test_X, num_objectives=2)
    policy2.load(file_history, file_training, file_predictor)

    N = policy.history.total_num_search
    assert policy2.history.total_num_search == N
    assert policy2.history.num_runs == policy.history.num_runs
    numpy.testing.assert_array_equal(
        policy2.history.fx[:N], policy.history.fx[:N]
    )
    numpy.testing.assert_array_equal(
        policy2.history.chosen_actions[:N], policy.history.chosen_actions[:N]
    )
    assert_same_pareto(policy.history.pareto, policy2.history.pareto)
    numpy.testing.assert_array_equal(policy2.actions, policy.actions)


def test_saveload_range_unified(tmp_path):
    sim = RangeSimulator()
    unify = physbo.search.unify.ParEGO(num_objectives=2)
    policy = physbo.search.range_unified.Policy(
        min_X=min_X, max_X=max_X, num_objectives=2
    )
    policy.set_seed(12345)
    policy.random_search(max_num_probes=5, simulator=sim)
    policy.bayes_search(
        max_num_probes=1, simulator=sim, score="EI", unify_method=unify, interval=0
    )

    file_history = os.path.join(tmp_path, "history.npz")
    file_training = os.path.join(tmp_path, "training.npz")
    file_predictor = os.path.join(tmp_path, "predictor.dump")
    policy.save(file_history, file_training, file_predictor)

    policy2 = physbo.search.range_unified.Policy(
        min_X=min_X, max_X=max_X, num_objectives=2
    )
    policy2.load(file_history, file_training, file_predictor)

    N = policy.history.total_num_search
    assert policy2.history.total_num_search == N
    assert policy2.history.num_runs == policy.history.num_runs
    numpy.testing.assert_array_equal(
        policy2.history.fx[:N], policy.history.fx[:N]
    )
    numpy.testing.assert_array_equal(
        policy2.history.action_X[:N], policy.history.action_X[:N]
    )
    assert_same_pareto(policy.history.pareto, policy2.history.pareto)

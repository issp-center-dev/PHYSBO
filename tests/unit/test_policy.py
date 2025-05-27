# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function

import os
import pickle
import shutil
import tempfile

import pytest
import numpy as np
import numpy.testing


physbo = pytest.importorskip("physbo")


@pytest.fixture
def X():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
        ]
    )


@pytest.fixture
def policy():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
        ]
    )
    return physbo.search.discrete.policy(test_X=X)


def test_write(policy, X):
    simulator = lambda x: 1.0
    ACTIONS = np.array([0, 1], np.int32)

    policy.write(ACTIONS, np.apply_along_axis(simulator, 1, X[ACTIONS]))
    numpy.testing.assert_array_equal(
        ACTIONS, policy.history.chosen_actions[: len(ACTIONS)]
    )
    assert len(X) - len(ACTIONS) == len(policy.actions)


def test_randomsearch(policy, mocker):
    simulator = mocker.MagicMock(return_value=1.0)
    write_spy = mocker.spy(physbo.search.discrete.policy, "write")

    N = 2

    # No simulator is passed: only returns candidates
    res = policy.random_search(1, num_search_each_probe=N)
    assert len(res) == N
    assert policy.history.num_runs == 0
    write_spy.assert_not_called()
    assert simulator.call_count == 0

    # A simulator is passed: N pairs of action and score are registered
    history = policy.random_search(N, simulator=simulator)
    assert history.num_runs == N
    write_spy.assert_called()
    assert simulator.call_count == N


def test_bayes_search(policy, mocker):
    simulator = mocker.MagicMock(side_effect=lambda x: x)
    write_spy = mocker.spy(physbo.search.discrete.policy, "write")
    get_actions_spy = mocker.spy(physbo.search.discrete.policy, "_get_actions")

    N = 2

    # initial training
    policy.random_search(N, simulator=simulator)

    res = policy.bayes_search(max_num_probes=N, simulator=simulator, score="TS")
    assert res.num_runs == N + N
    assert write_spy.call_count == N + N
    assert simulator.call_count == N + N
    assert get_actions_spy.call_count == N


def test_saveload(policy, X):
    simulator = lambda x: x

    N = 2
    policy.random_search(N, simulator=simulator)
    policy.bayes_search(max_num_probes=N, simulator=simulator, score="TS")

    with tempfile.TemporaryDirectory() as tempdir:
        policy.save(
            file_history=os.path.join(tempdir, "history.npz"),
            file_training=os.path.join(tempdir, "training.npz"),
            file_predictor=os.path.join(tempdir, "predictor.dump"),
        )

        policy2 = physbo.search.discrete.policy(test_X=X)
        policy2.load(
            file_history=os.path.join(tempdir, "history.npz"),
            file_training=os.path.join(tempdir, "training.npz"),
            file_predictor=os.path.join(tempdir, "predictor.dump"),
        )
        numpy.testing.assert_array_equal(policy.actions, policy2.actions)
        assert policy.history.num_runs == policy2.history.num_runs


def test_get_score(policy, mocker, X):
    simulator = mocker.MagicMock(return_value=1.0)
    policy.random_search(2, simulator=simulator)
    policy.set_seed(137)

    res = policy.get_score("EI", xs=X)
    ref = np.array(
        [
            3.98940120e-07,
            3.98934542e-07,
            3.98924610e-07,
            3.98914969e-07,
            3.98911183e-07,
            3.98914969e-07,
        ]
    )
    numpy.testing.assert_allclose(res, ref, rtol=1e-4)

    res = policy.get_score("PI", xs=X)
    print(res)
    ref = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    numpy.testing.assert_allclose(res, ref, rtol=1e-4)

    res = policy.get_score("TS", xs=X)
    ref = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    numpy.testing.assert_allclose(res, ref, rtol=1e-4)

    with pytest.raises(NotImplementedError):
        policy.get_score("XX")


def test_get_post_fmean(policy, mocker, X):
    simulator = mocker.MagicMock(return_value=1.0)
    policy.random_search(2, simulator=simulator)
    policy.set_seed(137)

    res = policy.get_post_fmean(X)
    ref = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    numpy.testing.assert_allclose(res, ref)

def test_get_post_fcov(policy, mocker, X):
    simulator = mocker.MagicMock(return_value=1.0)
    policy.random_search(2, simulator=simulator)
    policy.set_seed(137)

    res = policy.get_post_fcov(X)
    ref = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    numpy.testing.assert_allclose(res, ref, atol=np.inf, rtol=1e-10)


def test_get_permutation_importance(policy, mocker, X):
    simulator = mocker.MagicMock(return_value=1.0)
    policy.random_search(2, simulator=simulator)
    policy.set_seed(137)

    res_mean, res_std = policy.get_permutation_importance(n_perm=10)
    ref_mean = np.array([0.0, 0.0, 0.0])
    ref_std = np.array([0.0, 0.0, 0.0])

    numpy.testing.assert_allclose(res_mean, ref_mean)
    numpy.testing.assert_allclose(res_std, ref_std)

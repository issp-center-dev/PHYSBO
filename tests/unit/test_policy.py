from __future__ import print_function

import os
import pickle
import shutil
import tempfile

import pytest
import numpy as np


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
        assert policy.history.num_runs == policy2.history.num_runs


def test_get_score(policy, mocker):
    EI = mocker.patch("physbo.search.score.EI")
    PI = mocker.patch("physbo.search.score.PI")
    TS = mocker.patch("physbo.search.score.TS")

    policy.get_score("EI")
    EI.assert_called_once()

    policy.get_score("PI")
    PI.assert_called_once()

    policy.get_score("TS")
    TS.assert_called_once()

    with pytest.raises(NotImplementedError):
        policy.get_score("XX")

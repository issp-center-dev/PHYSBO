import pytest
import numpy as np
import tempfile
import os

physbo = pytest.importorskip("physbo")

@pytest.fixture
def min_X():
    return np.array([0.0, 0.0])

@pytest.fixture
def max_X():
    return np.array([1.0, 1.0])

@pytest.fixture
def policy(min_X, max_X):
    from physbo.search.range import Policy
    return Policy(min_X=min_X, max_X=max_X)

def test_write(policy):
    # Generate two points arbitrarily
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    t = np.array([1.0, 2.0])
    policy.write(X, t)
    # Check if they are recorded in history
    np.testing.assert_array_equal(policy.history.fx[:2], t)
    np.testing.assert_array_equal(policy.history.action_X[:2], X)
    assert policy.history.total_num_search == 2

def test_random_search(policy, mocker):
    simulator = mocker.MagicMock(return_value=np.array([1.0, 2.0]))
    write_spy = mocker.spy(policy, "write")
    N = 2
    # Without simulator: candidate points are returned
    actions = policy.random_search(1, num_search_each_probe=N)
    assert actions.shape == (N, 2)
    # With simulator: history is updated
    policy.random_search(1, num_search_each_probe=N, simulator=simulator)
    assert policy.history.total_num_search >= N
    assert write_spy.called
    assert simulator.called

def test_bayes_search(policy, mocker):
    # First, add initial data
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    t = np.array([1.0, 2.0])
    policy.write(X, t)
    simulator = mocker.MagicMock(return_value=np.array([3.0]))
    write_spy = mocker.spy(policy, "write")
    get_actions_spy = mocker.spy(policy, "_get_actions")
    N = 1
    # Without simulator: candidate points are returned
    actions = policy.bayes_search(max_num_probes=1, num_search_each_probe=N)
    assert actions.shape == (N, 2)
    # With simulator: history is updated
    policy.bayes_search(max_num_probes=1, num_search_each_probe=N, simulator=simulator)
    assert policy.history.total_num_search >= 3
    assert write_spy.called
    assert get_actions_spy.called
    assert simulator.called

def test_get_score(policy, mocker):
    # First, add initial data
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    t = np.array([1.0, 2.0])
    policy.write(X, t)
    policy.set_seed(42)
    # EI
    res = policy.get_score("EI", xs=X)
    assert isinstance(res, np.ndarray)
    # PI
    res = policy.get_score("PI", xs=X)
    assert isinstance(res, np.ndarray)
    # TS
    res = policy.get_score("TS", xs=X)
    assert isinstance(res, np.ndarray)
    # Unimplemented score
    with pytest.raises(Exception):
        policy.get_score("XX", xs=X)

def test_saveload(policy, min_X, max_X):
    # Initial data
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    t = np.array([1.0, 2.0])
    policy.write(X, t)
    # Save to a temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        file_history = os.path.join(tempdir, "history.npz")
        file_training = os.path.join(tempdir, "training.npz")
        file_predictor = os.path.join(tempdir, "predictor.dump")
        policy.save(file_history, file_training, file_predictor)
        # Load with a new Policy
        from physbo.search.range import Policy
        policy2 = Policy(min_X=min_X, max_X=max_X)
        policy2.load(file_history, file_training, file_predictor)
        np.testing.assert_array_equal(policy.history.fx[:2], policy2.history.fx[:2])
        assert policy.history.total_num_search == policy2.history.total_num_search

def test_set_seed(policy):
    policy.set_seed(123)
    a1 = policy._get_random_action(2)
    policy.set_seed(123)
    a2 = policy._get_random_action(2)
    np.testing.assert_allclose(a1, a2)

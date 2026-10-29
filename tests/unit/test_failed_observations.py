# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Failed observations (non-finite objective values).

Contract: an evaluation whose objective value is not finite (NaN or +-Inf;
for multi-objective problems, any objective) is a *failed* observation.
It is recorded in the history as is and the candidate is consumed, but it
is excluded from the training data, the best-value tracking, and the
Pareto front.

This holds for the discrete policies. The range policies raise ValueError
for a non-finite objective value, since a failed point cannot be removed
from a continuous search space. Their histories still accept failed
observations, so that the files written by the older versions can be loaded.
"""

import os
import pickle
from itertools import product

import numpy as np
import numpy.testing
import pytest

physbo = pytest.importorskip("physbo")


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def X5():
    return np.linspace(0.0, 1.0, 5).reshape(-1, 1)


def f1(X):
    return -np.sum((X - 0.5) ** 2, axis=-1)


def f2(X):
    X = np.atleast_2d(X)
    return np.c_[-np.sum((X - 0.3) ** 2, axis=1), -np.sum((X - 0.7) ** 2, axis=1)]


min_X = np.array([0.0, 0.0])
max_X = np.array([1.0, 1.0])


# ------------------------------------------------------------ single: discrete

def test_discrete_write_failed(X5):
    policy = physbo.search.discrete.Policy(test_X=X5)
    actions = np.array([0, 1, 2, 3])
    t = f1(X5[actions])
    t[2] = np.nan  # action 2 failed
    policy.write(actions, t)

    h = policy.history
    assert h.total_num_search == 4
    assert np.isnan(h.fx[2])
    numpy.testing.assert_array_equal(h.valid_mask, [True, True, False, True])
    # the failed action is consumed
    numpy.testing.assert_array_equal(policy.actions, [4])
    # but excluded from the training data
    assert policy.training.X.shape[0] == 3
    assert np.all(np.isfinite(policy.training.t))
    ok_actions, ok_fx = h.export_valid()
    numpy.testing.assert_array_equal(ok_actions, [0, 1, 3])
    numpy.testing.assert_array_equal(ok_fx, t[[0, 1, 3]])


def test_discrete_inf_is_failed(X5):
    policy = physbo.search.discrete.Policy(test_X=X5)
    policy.write(np.array([0, 1]), np.array([np.inf, -np.inf]))
    numpy.testing.assert_array_equal(policy.history.valid_mask, [False, False])
    assert policy.training.X is None or policy.training.X.shape[0] == 0


def test_discrete_best_fx_ignores_failed(X5):
    policy = physbo.search.discrete.Policy(test_X=X5)
    # first observation fails, then a valid one, then a failed "better" one
    policy.write(np.array([0]), np.array([np.nan]))
    policy.write(np.array([1]), np.array([-1.0]))
    policy.write(np.array([2]), np.array([np.nan]))
    policy.write(np.array([3]), np.array([-2.0]))

    best_fx, best_actions = policy.history.export_all_sequence_best_fx()
    assert np.isnan(best_fx[0])
    numpy.testing.assert_array_equal(best_fx[1:], [-1.0, -1.0, -1.0])
    numpy.testing.assert_array_equal(best_actions[1:], [1, 1, 1])

    best_fx, best_actions = policy.history.export_sequence_best_fx()
    assert np.isnan(best_fx[0])
    numpy.testing.assert_array_equal(best_fx[1:], [-1.0, -1.0, -1.0])
    numpy.testing.assert_array_equal(best_actions[1:], [1, 1, 1])

    # display must not crash on NaN
    policy.history.show_search_results(1)
    policy.history.show_search_results(2)


def test_discrete_bayes_search_after_failure(X5):
    def simnan(action):
        action = np.asarray(action)
        return np.where(action == 2, np.nan, f1(X5[action]))

    policy = physbo.search.discrete.Policy(test_X=X5)
    policy.set_seed(1)
    policy.random_search(max_num_probes=4, simulator=simnan, is_disp=False)
    assert np.isnan(policy.history.fx[: policy.history.total_num_search]).any()

    # the GP only sees valid observations: no LinAlgError, no NaN poisoning
    res = policy.bayes_search(
        max_num_probes=1, simulator=simnan, score="EI", is_disp=False
    )
    assert res.total_num_search == 5
    assert np.all(np.isfinite(policy.get_post_fmean(X5)))
    assert np.all(np.isfinite(policy.get_score("EI", xs=X5)))


def test_discrete_saveload_reconstructs_valid_training(X5, tmp_path):
    policy = physbo.search.discrete.Policy(test_X=X5)
    policy.write(np.array([0, 1, 2]), np.array([-1.0, np.nan, -3.0]))
    file_history = os.path.join(tmp_path, "history.npz")
    policy.save(file_history)

    policy2 = physbo.search.discrete.Policy(test_X=X5)
    policy2.load(file_history)
    numpy.testing.assert_array_equal(policy2.history.valid_mask, [True, False, True])
    assert policy2.training.X.shape[0] == 2
    assert np.all(np.isfinite(policy2.training.t))
    # the failed action stays consumed
    numpy.testing.assert_array_equal(policy2.actions, [3, 4])


# --------------------------------------------------------------- single: range

def test_range_history_best_fx_ignores_failed():
    h = physbo.search.range.History(dim=2)
    h.write(np.array([np.nan]), np.array([[0.1, 0.1]]))
    h.write(np.array([-1.0]), np.array([[0.2, 0.2]]))
    h.write(np.array([np.nan]), np.array([[0.3, 0.3]]))
    h.write(np.array([-2.0]), np.array([[0.4, 0.4]]))

    numpy.testing.assert_array_equal(h.valid_mask, [False, True, False, True])
    ok_X, ok_fx = h.export_valid()
    numpy.testing.assert_array_equal(ok_X, [[0.2, 0.2], [0.4, 0.4]])
    numpy.testing.assert_array_equal(ok_fx, [-1.0, -2.0])

    best_fx, best_X = h.export_all_sequence_best_fx()
    assert np.isnan(best_fx[0])
    numpy.testing.assert_array_equal(best_fx[1:], [-1.0, -1.0, -1.0])
    numpy.testing.assert_array_equal(best_X[1:], [[0.2, 0.2]] * 3)

    best_fx, best_X = h.export_sequence_best_fx()
    assert np.isnan(best_fx[0])
    numpy.testing.assert_array_equal(best_fx[1:], [-1.0, -1.0, -1.0])

    h.show_search_results(1)
    h.show_search_results(2)


def test_range_load_history_with_failed(tmp_path):
    h = physbo.search.range.History(dim=2)
    h.write(
        np.array([-1.0, np.nan, -3.0]),
        np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]),
    )
    file_history = os.path.join(tmp_path, "history.npz")
    h.save(file_history)

    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
    policy.load(file_history)
    numpy.testing.assert_array_equal(policy.history.valid_mask, [True, False, True])
    assert policy.training.X.shape[0] == 2
    best_fx, _ = policy.history.export_all_sequence_best_fx()
    numpy.testing.assert_array_equal(best_fx, [-1.0, -1.0, -1.0])


def test_range_load_history_of_older_version(tmp_path):
    # the older versions compared fx with NaN, and hence best_index kept
    # referring to the failed observation
    h = physbo.search.range.History(dim=1)
    h.write(np.array([np.nan]), np.array([[0.1]]))
    h.write(np.array([1.0]), np.array([[0.2]]))
    h.best_index[0:2] = 0
    file_history = os.path.join(tmp_path, "history.npz")
    h.save(file_history)

    h2 = physbo.search.range.History(dim=1)
    h2.load(file_history)
    numpy.testing.assert_array_equal(h2.best_index[0:2], [-1, 1])
    h2.write(np.array([2.0]), np.array([[0.3]]))
    best_fx, best_X = h2.export_all_sequence_best_fx()
    numpy.testing.assert_array_equal(best_fx, [np.nan, 1.0, 2.0])
    numpy.testing.assert_array_equal(best_X[1:, 0], [0.2, 0.3])


# ----------------------------------------------------------- multi / unified

@pytest.fixture
def grid2():
    a = np.linspace(0.0, 1.0, 5)
    return np.array(list(product(a, a)))


DISCRETE_MULTI = ["discrete_multi", "discrete_unified"]
RANGE_MULTI = ["range_multi", "range_unified"]


def make_policy(kind, grid2):
    mod = getattr(physbo.search, kind)
    if kind.startswith("discrete"):
        return mod.Policy(test_X=grid2, num_objectives=2)
    return mod.Policy(min_X=min_X, max_X=max_X, num_objectives=2)


def unify_kwargs(kind):
    if kind.endswith("unified"):
        return {"unify_method": physbo.search.unify.ParEGO(num_objectives=2)}
    return {}


@pytest.mark.parametrize("kind", DISCRETE_MULTI)
def test_discrete_multi_write_failed(kind, grid2):
    policy = make_policy(kind, grid2)
    actions = np.array([0, 6, 12, 18])
    t = f2(grid2[actions])
    t[1, 0] = np.nan  # one objective failed -> the point failed
    t[3, 1] = np.inf
    policy.write(actions, t)

    h = policy.history
    assert h.total_num_search == 4
    numpy.testing.assert_array_equal(h.valid_mask, [True, False, True, False])
    # consumed, but excluded from the training data
    assert 6 not in policy.actions and 18 not in policy.actions
    assert policy.training.X.shape[0] == 2
    assert np.all(np.isfinite(policy.training.t))
    # the Pareto front never contains a failed point
    front, front_num = h.export_pareto_front()
    assert np.all(np.isfinite(front))
    assert set(front_num).issubset({0, 2})
    ok_actions, ok_fx = h.export_valid()
    numpy.testing.assert_array_equal(ok_actions, [0, 12])
    assert ok_fx.shape == (2, 2)


@pytest.mark.parametrize("kind", DISCRETE_MULTI)
def test_discrete_multi_bayes_search_after_failure(kind, grid2):
    def simnan(action):
        action = np.asarray(action)
        t = f2(grid2[action])
        t[action == 12, 0] = np.nan
        return t

    policy = make_policy(kind, grid2)
    policy.set_seed(1)
    policy.write(np.array([0, 12, 24, 6, 18]), simnan(np.array([0, 12, 24, 6, 18])))
    res = policy.bayes_search(
        max_num_probes=1, simulator=simnan, score="EI" if kind.endswith("unified") else "EHVI",
        is_disp=False, **unify_kwargs(kind),
    )
    assert res.total_num_search == 6
    front, _ = res.export_pareto_front()
    assert np.all(np.isfinite(front))


@pytest.mark.parametrize("kind", DISCRETE_MULTI)
def test_discrete_multi_saveload(kind, grid2, tmp_path):
    policy = make_policy(kind, grid2)
    actions = np.array([0, 6, 12])
    t = f2(grid2[actions])
    t[1, 1] = np.nan
    policy.write(actions, t)
    file_history = os.path.join(tmp_path, "history.npz")
    policy.save(file_history)

    policy2 = make_policy(kind, grid2)
    policy2.load(file_history)
    numpy.testing.assert_array_equal(policy2.history.valid_mask, [True, False, True])
    assert policy2.training.X.shape[0] == 2
    assert 6 not in policy2.actions
    front, front_num = policy2.history.export_pareto_front()
    assert np.all(np.isfinite(front))


@pytest.mark.parametrize("kind", RANGE_MULTI)
def test_range_multi_load_history_with_failed(kind, grid2, tmp_path):
    X = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    t = f2(X)
    t[1, 0] = np.nan
    h = getattr(physbo.search, kind).History(num_objectives=2, dim=2)
    h.write(t, X)
    numpy.testing.assert_array_equal(h.valid_mask, [True, False, True])
    front, front_num = h.export_pareto_front()
    assert np.all(np.isfinite(front))
    assert 1 not in front_num
    ok_X, ok_fx = h.export_valid()
    numpy.testing.assert_array_equal(ok_X, X[[0, 2]])

    file_history = os.path.join(tmp_path, "history.npz")
    h.save(file_history)

    policy = make_policy(kind, grid2)
    policy.load(file_history)
    numpy.testing.assert_array_equal(policy.history.valid_mask, [True, False, True])
    assert policy.training.X.shape[0] == 2


def make_history(kind):
    mod = getattr(physbo.search, kind)
    if kind.startswith("discrete"):
        return mod.History(num_objectives=2)
    return mod.History(num_objectives=2, dim=2)


def write_history(h, kind, t, n):
    if kind.startswith("discrete"):
        h.write(t, np.array([n]))
    else:
        h.write(t, np.full((1, 2), 0.1 * (n + 1)))


@pytest.mark.parametrize("kind", DISCRETE_MULTI + RANGE_MULTI)
def test_multi_load_history_of_older_version(kind, tmp_path):
    # the older versions kept failed observations in the Pareto front
    ts = np.array([[1.0, 2.0], [np.nan, 0.0], [2.0, 3.0], [3.0, 1.0]])
    h = make_history(kind)
    for n, t in enumerate(ts):
        write_history(h, kind, t.reshape(1, -1), n)
    front, front_num = h.export_pareto_front()
    h.pareto.front = np.array([[np.nan, 0.0], [2.0, 3.0], [3.0, 1.0]])
    h.pareto.front_num = np.array([1, 2, 3])
    h.pareto.reference_min = np.array([[np.nan, -1.0]])
    file_history = os.path.join(tmp_path, "history.pkl")
    h.save(file_history)

    h2 = make_history(kind)
    h2.load(file_history)
    front2, front_num2 = h2.export_pareto_front()
    numpy.testing.assert_array_equal(front2, front)
    numpy.testing.assert_array_equal(front_num2, front_num)
    numpy.testing.assert_array_equal(front_num2, [2, 3])
    assert h2.pareto.num_compared == 4
    assert h2.pareto.reference_min is None


@pytest.mark.parametrize("kind", DISCRETE_MULTI + RANGE_MULTI)
def test_multi_load_keeps_front(kind, tmp_path):
    # without failed observations, loading reproduces the saved front
    rng = np.random.RandomState(12345)
    h = make_history(kind)
    n = 0
    for batch in (1, 3, 2, 1, 4):
        t = rng.rand(batch, 2)
        if kind.startswith("discrete"):
            h.write(t, np.arange(n, n + batch))
        else:
            h.write(t, rng.rand(batch, 2))
        n += batch
    h.pareto.reference_min = np.array([[-1.0, -1.0]])
    file_history = os.path.join(tmp_path, "history.pkl")
    h.save(file_history)

    h2 = make_history(kind)
    h2.load(file_history)
    numpy.testing.assert_array_equal(h2.pareto.front, h.pareto.front)
    numpy.testing.assert_array_equal(h2.pareto.front_num, h.pareto.front_num)
    assert h2.pareto.num_compared == h.pareto.num_compared
    assert h2.pareto.front_updated == h.pareto.front_updated
    numpy.testing.assert_array_equal(h2.pareto.reference_min, [[-1.0, -1.0]])


# ------------------------------------------- range policies: error on failure

RANGE_ALL = ["range"] + RANGE_MULTI
X3 = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])


def make_range_policy(kind, **kwargs):
    mod = getattr(physbo.search, kind)
    if kind == "range":
        return mod.Policy(min_X=min_X, max_X=max_X, **kwargs)
    return mod.Policy(min_X=min_X, max_X=max_X, num_objectives=2, **kwargs)


def values(kind, X, failed=(), bad=np.nan):
    """Objective values at X, where the rows in ``failed`` have failed."""
    if kind == "range":
        t = f1(X)
        t[list(failed)] = bad
    else:
        t = f2(X)
        t[list(failed), 1] = bad
    return t


def score_of(kind):
    if kind == "range" or kind.endswith("unified"):
        return "EI"
    return "EHVI"


def assert_unchanged(policy, n):
    assert policy.history.total_num_search == n
    assert policy.history.num_runs == (1 if n > 0 else 0)
    if n == 0:
        assert policy.training.X is None or policy.training.X.shape[0] == 0
        assert policy.new_data is None
    else:
        assert policy.training.X.shape[0] == n
        assert np.all(np.isfinite(policy.training.t))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kind", RANGE_ALL)
def test_range_write_raises(kind, bad):
    policy = make_range_policy(kind)
    with pytest.raises(ValueError, match="not supported in continuous search"):
        policy.write(X3, values(kind, X3, failed=[1], bad=bad))
    # nothing is written, even for the valid points of the same call
    assert_unchanged(policy, 0)

    # the user can write again with a substitute value
    t = values(kind, X3, failed=[1], bad=bad)
    t[~np.isfinite(t)] = -10.0
    policy.write(X3, t)
    assert_unchanged(policy, 3)


@pytest.mark.parametrize("kind", RANGE_ALL)
def test_range_write_error_shows_failed_points(kind):
    policy = make_range_policy(kind)
    with pytest.raises(ValueError) as e:
        policy.write(X3, values(kind, X3, failed=[1]))
    msg = str(e.value)
    assert "1 of 3 point(s)" in msg
    assert str(X3[1]) in msg
    assert str(X3[0]) not in msg
    assert "penalty" in msg


@pytest.mark.parametrize("kind", RANGE_ALL)
def test_range_initial_data_raises(kind):
    with pytest.raises(ValueError, match="not supported in continuous search"):
        make_range_policy(kind, initial_data=(X3, values(kind, X3, failed=[0])))


@pytest.mark.parametrize("kind", RANGE_ALL)
def test_range_random_search_raises(kind):
    policy = make_range_policy(kind)
    policy.set_seed(1)
    policy.write(X3, values(kind, X3))
    with pytest.raises(ValueError, match="not supported in continuous search"):
        policy.random_search(
            max_num_probes=2,
            simulator=lambda X: values(kind, np.atleast_2d(X), failed=[0]),
            is_disp=False,
        )
    assert_unchanged(policy, 3)


@pytest.mark.parametrize("kind", RANGE_ALL)
def test_range_bayes_search_raises(kind):
    policy = make_range_policy(kind)
    policy.set_seed(1)
    policy.write(X3, values(kind, X3))
    with pytest.raises(ValueError, match="not supported in continuous search"):
        policy.bayes_search(
            max_num_probes=2,
            simulator=lambda X: values(kind, np.atleast_2d(X), failed=[0]),
            score=score_of(kind),
            is_disp=False,
            **unify_kwargs(kind),
        )
    assert_unchanged(policy, 3)

    # the policy stays usable
    res = policy.bayes_search(
        max_num_probes=1,
        simulator=lambda X: values(kind, np.atleast_2d(X)),
        score=score_of(kind),
        is_disp=False,
        **unify_kwargs(kind),
    )
    assert res.total_num_search == 4


# ------------------------------------------------- no valid observation at all

ALL_KINDS = ["discrete"] + DISCRETE_MULTI + RANGE_ALL


def make_any_policy(kind, grid2):
    if kind == "discrete":
        return physbo.search.discrete.Policy(test_X=grid2)
    if kind.startswith("discrete"):
        return make_policy(kind, grid2)
    return make_range_policy(kind)


@pytest.mark.parametrize("kind", ["discrete"] + DISCRETE_MULTI)
def test_bayes_search_with_failed_observations_only(kind, grid2):
    nobj = 1 if kind == "discrete" else 2

    def simnan(action):
        t = np.full((len(action), nobj), np.nan)
        return t[:, 0] if nobj == 1 else t

    policy = make_any_policy(kind, grid2)
    policy.set_seed(1)
    policy.random_search(max_num_probes=3, simulator=simnan, is_disp=False)
    assert policy.history.total_num_search == 3
    with pytest.raises(RuntimeError, match="No valid observation"):
        policy.bayes_search(
            max_num_probes=1,
            simulator=simnan,
            score=score_of("range" if kind == "discrete" else kind),
            is_disp=False,
            **unify_kwargs(kind),
        )
    assert policy.history.total_num_search == 3


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_bayes_search_without_observation(kind, grid2):
    policy = make_any_policy(kind, grid2)
    kwargs = dict(
        score=score_of("range" if kind == "discrete" else kind),
        is_disp=False,
        **unify_kwargs(kind),
    )
    # both with and without a simulator (interactive mode)
    with pytest.raises(RuntimeError, match="No valid observation"):
        policy.bayes_search(max_num_probes=1, simulator=lambda x: None, **kwargs)
    with pytest.raises(RuntimeError, match="No valid observation"):
        policy.bayes_search(max_num_probes=1, simulator=None, **kwargs)
    # learning the hyperparameters needs observations as well
    with pytest.raises(RuntimeError, match="No valid observation"):
        policy.bayes_search(max_num_probes=0, **kwargs)


# ------------------------------------------------------------------ display

def test_discrete_show_search_results(X5, capsys):
    h = physbo.search.discrete.History()
    h.write(np.array([[np.nan]]), np.array([0]))
    h.show_search_results(1)
    assert "no valid observation yet" in capsys.readouterr().out

    h.write(np.array([[-1.0], [np.nan], [-0.5]]), np.array([1, 2, 3]))
    h.write(np.array([[-0.5], [np.inf]]), np.array([4, 0]))
    h.show_search_results(2)
    out = capsys.readouterr().out
    # the first one of the best valid observations, as in
    # export_all_sequence_best_fx
    assert "current best f(x) = -0.500000 (best action=3)" in out
    best_fx, best_actions = h.export_all_sequence_best_fx()
    assert best_fx[-1] == -0.5
    assert best_actions[-1] == 3


# ------------------------------------------------------------------- pareto

def test_pareto_update_front_skips_non_finite():
    from physbo.search.pareto import Pareto

    pareto = Pareto(num_objectives=2)
    pareto.update_front(np.array([[1.0, 3.0], [np.nan, 5.0], [3.0, 1.0], [2.0, np.inf]]))
    front, front_num = pareto.export_front()
    numpy.testing.assert_array_equal(front, [[1.0, 3.0], [3.0, 1.0]])
    # indices keep referring to the rows passed to update_front
    numpy.testing.assert_array_equal(front_num, [0, 2])
    assert pareto.num_compared == 4
    # only-failed batch does not update the front
    pareto.update_front(np.array([[np.nan, np.nan]]))
    assert not pareto.front_updated
    assert pareto.num_compared == 5

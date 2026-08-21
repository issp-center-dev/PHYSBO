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
"""

import os
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

def test_range_write_failed():
    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
    X = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    t = np.array([-1.0, np.nan, -3.0])
    policy.write(X, t)

    h = policy.history
    assert h.total_num_search == 3
    numpy.testing.assert_array_equal(h.valid_mask, [True, False, True])
    assert policy.training.X.shape[0] == 2
    ok_X, ok_fx = h.export_valid()
    numpy.testing.assert_array_equal(ok_X, X[[0, 2]])
    numpy.testing.assert_array_equal(ok_fx, [-1.0, -3.0])


def test_range_best_fx_ignores_failed():
    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
    policy.write(np.array([[0.1, 0.1]]), np.array([np.nan]))
    policy.write(np.array([[0.2, 0.2]]), np.array([-1.0]))
    policy.write(np.array([[0.3, 0.3]]), np.array([np.nan]))
    policy.write(np.array([[0.4, 0.4]]), np.array([-2.0]))

    best_fx, best_X = policy.history.export_all_sequence_best_fx()
    assert np.isnan(best_fx[0])
    numpy.testing.assert_array_equal(best_fx[1:], [-1.0, -1.0, -1.0])
    numpy.testing.assert_array_equal(best_X[1:], [[0.2, 0.2]] * 3)

    best_fx, best_X = policy.history.export_sequence_best_fx()
    assert np.isnan(best_fx[0])
    numpy.testing.assert_array_equal(best_fx[1:], [-1.0, -1.0, -1.0])

    policy.history.show_search_results(1)
    policy.history.show_search_results(2)


def test_range_bayes_search_after_failure():
    def simnan(X):
        X = np.atleast_2d(X)
        val = f1(X)
        return np.where(X[:, 0] > 0.8, np.nan, val)

    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
    policy.set_seed(1)
    policy.write(np.array([[0.9, 0.5], [0.2, 0.2], [0.5, 0.6]]), simnan(np.array([[0.9, 0.5], [0.2, 0.2], [0.5, 0.6]])))
    assert np.isnan(policy.history.fx[0])
    res = policy.bayes_search(
        max_num_probes=1, simulator=simnan, score="EI", is_disp=False
    )
    assert res.total_num_search == 4
    assert np.all(np.isfinite(policy.get_post_fmean(np.array([[0.5, 0.5]]))))


def test_range_saveload_reconstructs_valid_training(tmp_path):
    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
    policy.write(
        np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]), np.array([-1.0, np.nan, -3.0])
    )
    file_history = os.path.join(tmp_path, "history.npz")
    policy.save(file_history)

    policy2 = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
    policy2.load(file_history)
    numpy.testing.assert_array_equal(policy2.history.valid_mask, [True, False, True])
    assert policy2.training.X.shape[0] == 2
    best_fx, _ = policy2.history.export_all_sequence_best_fx()
    numpy.testing.assert_array_equal(best_fx, [-1.0, -1.0, -1.0])


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
def test_range_multi_write_failed(kind, grid2):
    policy = make_policy(kind, grid2)
    X = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    t = f2(X)
    t[1, 0] = np.nan
    policy.write(X, t)

    h = policy.history
    numpy.testing.assert_array_equal(h.valid_mask, [True, False, True])
    assert policy.training.X.shape[0] == 2
    front, front_num = h.export_pareto_front()
    assert np.all(np.isfinite(front))
    assert 1 not in front_num
    ok_X, ok_fx = h.export_valid()
    numpy.testing.assert_array_equal(ok_X, X[[0, 2]])


@pytest.mark.parametrize("kind", RANGE_MULTI)
def test_range_multi_bayes_search_after_failure(kind, grid2):
    def simnan(X):
        X = np.atleast_2d(X)
        t = f2(X)
        t[X[:, 0] > 0.8, 1] = np.nan
        return t

    policy = make_policy(kind, grid2)
    policy.set_seed(1)
    X0 = np.array([[0.9, 0.5], [0.2, 0.2], [0.5, 0.6], [0.3, 0.8]])
    policy.write(X0, simnan(X0))
    assert not policy.history.valid_mask[0]
    res = policy.bayes_search(
        max_num_probes=1, simulator=simnan, score="EI" if kind.endswith("unified") else "EHVI",
        is_disp=False, **unify_kwargs(kind),
    )
    assert res.total_num_search == 5
    front, _ = res.export_pareto_front()
    assert np.all(np.isfinite(front))


@pytest.mark.parametrize("kind", RANGE_MULTI)
def test_range_multi_saveload(kind, grid2, tmp_path):
    policy = make_policy(kind, grid2)
    X = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    t = f2(X)
    t[1, 0] = np.nan
    policy.write(X, t)
    file_history = os.path.join(tmp_path, "history.npz")
    policy.save(file_history)

    policy2 = make_policy(kind, grid2)
    policy2.load(file_history)
    numpy.testing.assert_array_equal(policy2.history.valid_mask, [True, False, True])
    assert policy2.training.X.shape[0] == 2


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

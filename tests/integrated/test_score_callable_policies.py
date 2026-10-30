# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Integration tests for passing a callable as ``score`` to the policies.

Covers how the six policies (discrete, range, discrete_unified,
range_unified, discrete_multi, range_multi) call a score callable, how its
return value is used, the equivalence of string and built-in class scores,
the TS ``alpha`` consistency, and the multiple-probe header message.
"""

import inspect
import re
from itertools import product

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from physbo.search import score as score_single
from physbo.search import score_multi


SINGLE_KINDS = ["discrete", "range", "discrete_unified", "range_unified"]
MULTI_KINDS = ["discrete_multi", "range_multi"]
ALL_KINDS = SINGLE_KINDS + MULTI_KINDS
DISCRETE_KINDS = ["discrete", "discrete_unified", "discrete_multi"]
RANGE_KINDS = ["range", "range_unified", "range_multi"]
UNIFIED_KINDS = ["discrete_unified", "range_unified"]

MIN_X = np.array([0.0, 0.0])
MAX_X = np.array([1.0, 1.0])
INITIAL_ACTIONS = np.array([0, 7, 13, 24])
INITIAL_X_RANGE = np.array([[0.1, 0.1], [0.9, 0.2], [0.3, 0.8], [0.5, 0.5]])
K_SAMPLING = 3


def grid_X():
    a = np.linspace(0.0, 1.0, 5)
    return np.array(list(product(a, a)))


def f_single(x):
    x = np.atleast_2d(x)
    return -np.sum((x - 0.6) ** 2, axis=-1)


def f_multi(x):
    x = np.atleast_2d(x)
    return np.stack(
        [-np.sum((x - 0.25) ** 2, axis=-1), -np.sum((x - 0.75) ** 2, axis=-1)],
        axis=-1,
    )


def is_multi(kind):
    return kind in MULTI_KINDS


def is_range(kind):
    return kind in RANGE_KINDS


def make_config(alpha=None, K=K_SAMPLING):
    """Search config with K virtual samples and a cheap hyperparameter fit."""
    config = physbo.misc.SetConfig()
    config.search.multi_probe_num_sampling = K
    config.learning.is_disp = False
    config.learning.num_init_params_search = 5
    config.learning.max_epoch = 50
    config.learning.max_epoch_init_params_search = 10
    if alpha is not None:
        config.search.alpha = alpha
    return config


def make_policy(kind, config=None, seed=1):
    """A policy with 4 initial observations, seeded."""
    if config is None:
        config = make_config()
    if kind == "discrete":
        X = grid_X()
        policy = physbo.search.discrete.Policy(
            test_X=X,
            config=config,
            initial_data=(INITIAL_ACTIONS, f_single(X[INITIAL_ACTIONS])),
        )
    elif kind == "range":
        policy = physbo.search.range.Policy(
            min_X=MIN_X,
            max_X=MAX_X,
            config=config,
            initial_data=(INITIAL_X_RANGE, f_single(INITIAL_X_RANGE)),
        )
    elif kind in ("discrete_unified", "discrete_multi"):
        X = grid_X()
        mod = getattr(physbo.search, kind)
        policy = mod.Policy(
            test_X=X,
            num_objectives=2,
            config=config,
            initial_data=(INITIAL_ACTIONS, f_multi(X[INITIAL_ACTIONS])),
        )
    elif kind in ("range_unified", "range_multi"):
        mod = getattr(physbo.search, kind)
        policy = mod.Policy(
            num_objectives=2,
            min_X=MIN_X,
            max_X=MAX_X,
            config=config,
            initial_data=(INITIAL_X_RANGE, f_multi(INITIAL_X_RANGE)),
        )
    else:
        raise ValueError(kind)
    policy.set_seed(seed)
    return policy


def make_simulator(kind):
    if kind == "discrete":
        X = grid_X()
        return lambda action: f_single(X[action])
    if kind == "range":
        return lambda x: f_single(x)
    if kind in ("discrete_unified", "discrete_multi"):
        X = grid_X()
        return lambda action: f_multi(X[action])
    return lambda x: f_multi(x)


class GridOptimizer:
    """Deterministic optimizer: evaluates fixed points in order, returns the argmax.

    Every list of values passed back from ``fn`` is kept in ``values_log``.
    """

    def __init__(self, points):
        self.points = np.array(points, dtype=float)
        self.values_log = []

    def __call__(self, fn, mpicomm=None):
        values = np.array([fn(x.copy()) for x in self.points], dtype=float)
        self.values_log.append(values)
        return self.points[int(np.argmax(values))].copy()


def default_grid_points():
    a = np.linspace(0.05, 0.95, 4)
    return np.array(list(product(a, a)))


def search_kwargs(kind, unify_method=None, optimizer=None):
    """Extra bayes_search keyword arguments needed by each policy kind."""
    kw = {}
    if kind in UNIFIED_KINDS:
        if unify_method is None:
            unify_method = physbo.search.unify.NDS(num_objectives=2)
        kw["unify_method"] = unify_method
    if is_range(kind):
        if optimizer is None:
            optimizer = GridOptimizer(default_grid_points())
        kw["optimizer"] = optimizer
    return kw


def proposals(kind, res, n):
    """Last n proposals (indices or X) and objective values from a history."""
    N = res.total_num_search
    if is_range(kind):
        chosen = np.array(res.action_X[N - n : N])
    else:
        chosen = np.array(res.chosen_actions[N - n : N])
    return chosen, np.array(res.fx[N - n : N])


def mean_score_single(predictor, training, test):
    """Deterministic single-objective score: posterior mean + std."""
    fmean = predictor.get_post_fmean(training, test)
    fcov = predictor.get_post_fcov(training, test)
    return (fmean + np.sqrt(fcov)).flatten()


def mean_score_multi(predictor_list, training, test, pareto):
    """Deterministic multi-objective score: sum of posterior means."""
    return np.sum(
        [
            p.get_post_fmean(training, test, objective_index=i)
            for i, p in enumerate(predictor_list)
        ],
        axis=0,
    ).flatten()


def deterministic_score(kind):
    return mean_score_multi if is_multi(kind) else mean_score_single


class Recorder:
    """Score callable recording the raw arguments of every call.

    ``value_fn(record)`` computes the return value from the record; by default
    the posterior-mean based deterministic score is returned.
    """

    def __init__(self, kind, policy=None, value_fn=None):
        self.kind = kind
        self.policy = policy
        self.value_fn = value_fn
        self.calls = []

    def __call__(self, *args, **kwargs):
        rec = {"args": args, "kwargs": dict(kwargs)}
        if len(args) >= 3:
            training, test = args[1], args[2]
            rec["training"] = training
            rec["test"] = test
            rec["training_X"] = np.array(training.X, copy=True)
            rec["training_t"] = np.array(training.t, copy=True)
            rec["test_X"] = np.array(test.X, copy=True)
        if len(args) >= 4:
            rec["pareto"] = args[3]
            if self.policy is not None:
                rec["pareto_is_history"] = args[3] is self.policy.history.pareto
        self.calls.append(rec)
        if self.value_fn is not None:
            return self.value_fn(rec)
        return deterministic_score(self.kind)(*args)


class RecordingUnify:
    """Wraps a unify_method and records its inputs and outputs."""

    def __init__(self, base):
        self.base = base
        self.inputs = []
        self.outputs = []

    def __call__(self, t):
        out = self.base(t)
        self.inputs.append(np.array(t, copy=True))
        self.outputs.append(np.array(out, copy=True))
        return out


def _observed(policy, kind):
    """Observed training data (X, t) the policy holds."""
    return np.array(policy.training.X), np.array(policy.training.t)


def _matching_unify_output(unify, observed_t, unified_t):
    """True if unified_t equals an output of unify for input observed_t."""
    for inp, out in zip(unify.inputs, unify.outputs):
        if (
            inp.shape == observed_t.shape
            and np.array_equal(inp, observed_t)
            and np.array_equal(np.ravel(out), np.ravel(unified_t))
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# C1 / C2 / C3: calling convention, pareto, first-proposal training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [1, 3])
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_bayes_search_calls_score_positionally(kind, N):
    """bayes_search calls the callable with positional (predictor[, _list], training, test[, pareto]) only."""
    policy = make_policy(kind)
    rec = Recorder(kind, policy=policy)
    policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=N,
        simulator=None,
        score=rec,
        is_disp=False,
        **search_kwargs(kind),
    )
    assert len(rec.calls) > 0
    nargs = 4 if is_multi(kind) else 3
    for c in rec.calls:
        assert c["kwargs"] == {}
        assert len(c["args"]) == nargs
        assert isinstance(c["training"], physbo.Variable)
        assert isinstance(c["test"], physbo.Variable)
        if is_multi(kind):
            assert isinstance(c["args"][0], list)
            assert len(c["args"][0]) == 2
            assert isinstance(c["pareto"], physbo.search.pareto.Pareto)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_get_score_calls_score_positionally(kind):
    """Public get_score calls the callable with positional arguments only."""
    policy = make_policy(kind)
    policy.bayes_search(max_num_probes=0, is_disp=False, **search_kwargs(kind))
    rec = Recorder(kind, policy=policy)
    xs = grid_X()[:5]
    res = policy.get_score(rec, xs=xs)
    assert len(res) == 5
    assert len(rec.calls) == 1
    c = rec.calls[0]
    assert c["kwargs"] == {}
    assert len(c["args"]) == (4 if is_multi(kind) else 3)
    assert isinstance(c["training"], physbo.Variable)
    assert isinstance(c["test"], physbo.Variable)


@pytest.mark.parametrize("N", [1, 3])
@pytest.mark.parametrize("kind", MULTI_KINDS)
def test_multi_bayes_search_passes_history_pareto(kind, N):
    """In bayes_search (incl. marginal evaluation) pareto is policy.history.pareto at call time."""
    policy = make_policy(kind)
    rec = Recorder(kind, policy=policy)
    policy.bayes_search(
        max_num_probes=2,
        num_search_each_probe=N,
        simulator=make_simulator(kind),
        score=rec,
        is_disp=False,
        **search_kwargs(kind),
    )
    assert len(rec.calls) > 0
    assert all(c["pareto_is_history"] for c in rec.calls)


@pytest.mark.parametrize("kind", MULTI_KINDS)
def test_multi_get_score_pareto_default_and_explicit(kind):
    """get_score passes history.pareto when pareto is omitted, and the given object otherwise."""
    policy = make_policy(kind)
    policy.bayes_search(max_num_probes=0, is_disp=False, **search_kwargs(kind))
    xs = grid_X()[:4]

    rec = Recorder(kind, policy=policy)
    policy.get_score(rec, xs=xs)
    assert len(rec.calls) == 1
    assert rec.calls[0]["pareto"] is policy.history.pareto

    p = physbo.search.pareto.Pareto(num_objectives=2)
    rec2 = Recorder(kind, policy=policy)
    policy.get_score(rec2, xs=xs, pareto=p)
    assert len(rec2.calls) == 1
    assert rec2.calls[0]["pareto"] is p


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_first_proposal_training_equals_policy_data(kind):
    """For the first proposal, training has the same points and values as the policy's data."""
    policy = make_policy(kind)
    rec = Recorder(kind, policy=policy)
    policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=1,
        simulator=None,
        score=rec,
        is_disp=False,
        **search_kwargs(kind),
    )
    assert len(rec.calls) > 0
    if kind in UNIFIED_KINDS:
        ref = policy.training_unified
    else:
        ref = policy.training
    for c in rec.calls:
        np.testing.assert_array_equal(c["training_X"], ref.X)
        np.testing.assert_array_equal(c["training_t"], ref.t)


@pytest.mark.parametrize("N", [1, 3])
@pytest.mark.parametrize("kind", UNIFIED_KINDS)
def test_unified_training_t_is_unify_method_output(kind, N):
    """Unified policies pass training.t of shape (n, 1) holding the unify_method output (plus virtual rows in marginal)."""
    policy = make_policy(kind)
    unify = RecordingUnify(physbo.search.unify.ParEGO(num_objectives=2))
    rec = Recorder(kind, policy=policy)
    policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=N,
        simulator=None,
        score=rec,
        is_disp=False,
        **search_kwargs(kind, unify_method=unify),
    )
    obs_X, obs_t = _observed(policy, kind)
    n_obs = obs_X.shape[0]
    assert len(rec.calls) > 0
    first = [c for c in rec.calls if c["training_X"].shape[0] == n_obs]
    assert len(first) > 0
    for c in rec.calls:
        n = c["training_X"].shape[0]
        assert c["training_t"].shape == (n, 1)
        np.testing.assert_array_equal(c["training_X"][:n_obs], obs_X)
        assert _matching_unify_output(unify, obs_t, c["training_t"][:n_obs])
    if N > 1:
        assert any(c["training_X"].shape[0] > n_obs for c in rec.calls)


# ---------------------------------------------------------------------------
# C4 / C5: marginal score and use of the return value (discrete)
# ---------------------------------------------------------------------------


def _row_index(Xs, x):
    hits = np.where(np.all(np.isclose(Xs, x), axis=1))[0]
    return hits


@pytest.mark.parametrize("kind", DISCRETE_KINDS)
def test_discrete_marginal_calls_and_mean(kind):
    """Discrete marginal score: K calls per extra point with n_obs + n_chosen training rows; the mean decides."""
    X = grid_X()
    policy = make_policy(kind)
    n_obs = len(INITIAL_ACTIONS)
    n_unsearched = X.shape[0] - n_obs
    idx_first, idx_a, idx_b = 12, 3, 20
    assert idx_first not in INITIAL_ACTIONS
    assert idx_a not in INITIAL_ACTIONS and idx_b not in INITIAL_ACTIONS
    marginal_count = {"n": 0}

    def value_fn(rec):
        test_X = rec["test_X"]
        v = np.zeros(test_X.shape[0])
        n_rows = rec["training_X"].shape[0]
        if n_rows == n_obs:
            v[_row_index(test_X, X[idx_first])] = 10.0
        else:
            k = marginal_count["n"]
            marginal_count["n"] += 1
            # mean: a=1.0 > b=0.9 ; median / most individual argmax: b
            v[_row_index(test_X, X[idx_a])] = 3.0 if k == 0 else 0.0
            v[_row_index(test_X, X[idx_b])] = 0.9
        return v

    rec = Recorder(kind, policy=policy, value_fn=value_fn)
    actions = policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=2,
        simulator=None,
        score=rec,
        is_disp=False,
        **search_kwargs(kind),
    )
    assert list(actions) == [idx_first, idx_a]

    first = [c for c in rec.calls if c["training_X"].shape[0] == n_obs]
    marginal = [c for c in rec.calls if c["training_X"].shape[0] != n_obs]
    assert len(first) == 1
    assert first[0]["test_X"].shape[0] == n_unsearched
    assert len(marginal) == K_SAMPLING
    for c in marginal:
        assert c["training_X"].shape[0] == n_obs + 1
        assert c["test_X"].shape[0] == n_unsearched - 1
        assert len(_row_index(c["test_X"], X[idx_first])) == 0
        np.testing.assert_array_equal(c["training_X"][:n_obs], X[INITIAL_ACTIONS])


@pytest.mark.parametrize("kind", DISCRETE_KINDS)
def test_discrete_marginal_training_grows_with_chosen_points(kind):
    """With 3 proposals, the n-th marginal stage calls the score K times with n_obs + n training rows."""
    policy = make_policy(kind)
    n_obs = len(INITIAL_ACTIONS)
    n_unsearched = grid_X().shape[0] - n_obs
    rec = Recorder(kind, policy=policy)
    actions = policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=3,
        simulator=None,
        score=rec,
        is_disp=False,
        **search_kwargs(kind),
    )
    assert len(actions) == 3
    sizes = [c["training_X"].shape[0] for c in rec.calls]
    assert sizes == [n_obs] + [n_obs + 1] * K_SAMPLING + [n_obs + 2] * K_SAMPLING
    tests = [c["test_X"].shape[0] for c in rec.calls]
    assert tests == (
        [n_unsearched]
        + [n_unsearched - 1] * K_SAMPLING
        + [n_unsearched - 2] * K_SAMPLING
    )


@pytest.mark.parametrize("kind", DISCRETE_KINDS)
def test_discrete_chooses_candidate_with_max_score(kind):
    """A callable that is large only at one candidate makes bayes_search choose that candidate."""
    X = grid_X()
    target = 17
    assert target not in INITIAL_ACTIONS
    policy = make_policy(kind)

    def value_fn(rec):
        v = -np.arange(rec["test_X"].shape[0], dtype=float)
        v[_row_index(rec["test_X"], X[target])] = 100.0
        return v

    rec = Recorder(kind, policy=policy, value_fn=value_fn)
    actions = policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=1,
        simulator=None,
        score=rec,
        is_disp=False,
        **search_kwargs(kind),
    )
    assert list(actions) == [target]


# ---------------------------------------------------------------------------
# C4 / C5 / C6: range policies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", RANGE_KINDS)
def test_range_optimizer_receives_score_value_and_argmax_is_adopted(kind):
    """Range: fn(x) passed to the optimizer equals score_fn's value and the optimizer's point is proposed."""
    points = np.array([[0.2, 0.2], [0.4, 0.7], [0.8, 0.3], [0.6, 0.6]])
    table = np.array([0.5, 2.0, -1.0, 1.5])

    def value_fn(rec):
        i = _row_index(points, rec["test_X"][0])
        return np.array([table[i[0]]])

    policy = make_policy(kind)
    opt = GridOptimizer(points)
    rec = Recorder(kind, policy=policy, value_fn=value_fn)
    X = policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=1,
        simulator=None,
        score=rec,
        is_disp=False,
        **search_kwargs(kind, optimizer=opt),
    )
    assert len(opt.values_log) == 1
    np.testing.assert_allclose(opt.values_log[0], table)
    np.testing.assert_allclose(np.atleast_2d(X)[0], points[1])


@pytest.mark.parametrize("kind", RANGE_KINDS)
def test_range_marginal_fn_is_mean_over_K_virtual_observations(kind):
    """Range marginal: each x is scored K times with n_obs + 1 training rows; fn(x) is the arithmetic mean."""
    points = np.array([[0.2, 0.2], [0.4, 0.7], [0.8, 0.3], [0.6, 0.6]])
    n_obs = INITIAL_X_RANGE.shape[0]
    first_table = np.array([0.0, 0.0, 0.0, 5.0])
    # rows: point index, columns: k-th call for that point in the marginal stage
    marginal_table = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.9, 0.9, 0.9],
            [0.0, 0.1, 0.2],
            [-1.0, -1.0, -1.0],
        ]
    )
    counter = {}

    def value_fn(rec):
        i = _row_index(points, rec["test_X"][0])[0]
        if rec["training_X"].shape[0] == n_obs:
            return np.array([first_table[i]])
        k = counter.get(i, 0)
        counter[i] = k + 1
        return np.array([marginal_table[i, k % K_SAMPLING]])

    policy = make_policy(kind)
    opt = GridOptimizer(points)
    rec = Recorder(kind, policy=policy, value_fn=value_fn)
    X = policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=2,
        simulator=None,
        score=rec,
        is_disp=False,
        **search_kwargs(kind, optimizer=opt),
    )
    X = np.atleast_2d(X)
    np.testing.assert_allclose(X[0], points[3])
    np.testing.assert_allclose(X[1], points[0])

    assert len(opt.values_log) == 2
    np.testing.assert_allclose(opt.values_log[0], first_table)
    np.testing.assert_allclose(
        opt.values_log[1], marginal_table.mean(axis=1), rtol=1e-12
    )

    marginal = [c for c in rec.calls if c["training_X"].shape[0] != n_obs]
    assert len(marginal) == K_SAMPLING * points.shape[0]
    for c in marginal:
        assert c["training_X"].shape[0] == n_obs + 1
        np.testing.assert_array_equal(c["training_X"][:n_obs], INITIAL_X_RANGE)
    for i in range(points.shape[0]):
        assert counter[i] == K_SAMPLING


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("kind", RANGE_KINDS)
def test_range_bayes_search_test_has_one_point(kind, N):
    """Range bayes_search always passes test.X with shape (1, d)."""
    policy = make_policy(kind)
    rec = Recorder(kind, policy=policy)
    policy.bayes_search(
        max_num_probes=2,
        num_search_each_probe=N,
        simulator=make_simulator(kind),
        score=rec,
        is_disp=False,
        **search_kwargs(kind),
    )
    assert len(rec.calls) > 0
    for c in rec.calls:
        assert c["test_X"].shape == (1, 2)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("npoints", [1, 6])
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_get_score_xs_is_passed_whole_in_serial(kind, npoints, parallel):
    """Serial get_score(xs=...) passes all points as test and returns that many scores."""
    policy = make_policy(kind)
    policy.bayes_search(max_num_probes=0, is_disp=False, **search_kwargs(kind))
    xs = grid_X()[3 : 3 + npoints]
    rec = Recorder(kind, policy=policy)
    res = policy.get_score(rec, xs=xs, parallel=parallel)
    assert len(rec.calls) == 1
    np.testing.assert_array_equal(rec.calls[0]["test_X"], xs)
    assert np.ndim(res) == 1
    assert len(res) == npoints


@pytest.mark.parametrize("kind", RANGE_KINDS)
def test_range_get_score_empty_xs_does_not_call(kind):
    """Range get_score with 0 points returns length 0 without calling the callable."""
    policy = make_policy(kind)
    policy.bayes_search(max_num_probes=0, is_disp=False, **search_kwargs(kind))
    rec = Recorder(kind, policy=policy)
    res = policy.get_score(rec, xs=np.zeros((0, 2)))
    assert len(res) == 0
    assert rec.calls == []


@pytest.mark.parametrize("kind", DISCRETE_KINDS)
def test_discrete_get_score_without_candidates_does_not_call(kind):
    """Discrete get_score with no unsearched candidate returns length 0 without calling the callable."""
    X = grid_X()
    policy = make_policy(kind)
    rest = np.array(sorted(set(range(X.shape[0])) - set(INITIAL_ACTIONS)))
    sim = make_simulator(kind)
    policy.write(rest, sim(rest))
    assert len(policy.actions) == 0
    policy.bayes_search(max_num_probes=0, is_disp=False, **search_kwargs(kind))
    rec = Recorder(kind, policy=policy)
    res = policy.get_score(rec)
    assert len(res) == 0
    assert rec.calls == []


# ---------------------------------------------------------------------------
# C7 / C9: bayes_search completes with functions and callable objects
# ---------------------------------------------------------------------------


class CallableSingle:
    def __call__(self, predictor, training, test):
        return mean_score_single(predictor, training, test)


class CallableMulti:
    def __call__(self, predictor_list, training, test, pareto):
        return mean_score_multi(predictor_list, training, test, pareto)


def make_callable(kind, style):
    if style == "function":
        return deterministic_score(kind)
    return CallableMulti() if is_multi(kind) else CallableSingle()


@pytest.mark.parametrize("style", ["function", "object"])
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_bayes_search_with_callable_completes(kind, N, style):
    """bayes_search(score=<function or __call__ object>) completes for single and multiple proposals."""
    policy = make_policy(kind)
    n0 = policy.history.total_num_search
    res = policy.bayes_search(
        max_num_probes=2,
        num_search_each_probe=N,
        simulator=make_simulator(kind),
        score=make_callable(kind, style),
        is_disp=False,
        **search_kwargs(kind),
    )
    assert res.total_num_search == n0 + 2 * N
    chosen, fx = proposals(kind, res, 2 * N)
    assert np.all(np.isfinite(fx))
    if is_range(kind):
        assert np.all(chosen >= MIN_X) and np.all(chosen <= MAX_X)
    else:
        assert len(set(chosen.tolist())) == 2 * N
        assert not set(chosen.tolist()) & set(INITIAL_ACTIONS.tolist())


# ---------------------------------------------------------------------------
# C8: get_score accepts a callable and returns the same shape as strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["function", "object"])
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_get_score_with_callable_has_string_shape(kind, style):
    """get_score(<callable>) returns a 1-D array shaped like get_score(<string>)."""
    policy = make_policy(kind)
    policy.bayes_search(max_num_probes=0, is_disp=False, **search_kwargs(kind))
    xs = grid_X()[2:9]
    string_mode = "EHVI" if is_multi(kind) else "EI"
    ref = policy.get_score(string_mode, xs=xs)
    res = policy.get_score(make_callable(kind, style), xs=xs)
    assert np.ndim(res) == 1
    assert np.shape(res) == np.shape(ref) == (xs.shape[0],)
    if kind in DISCRETE_KINDS:
        ref_a = policy.get_score(string_mode)
        res_a = policy.get_score(make_callable(kind, style))
        assert np.ndim(res_a) == 1
        assert np.shape(res_a) == np.shape(ref_a) == (len(policy.actions),)


# ---------------------------------------------------------------------------
# D2: string and built-in class give identical searches
# ---------------------------------------------------------------------------


TS_ALPHA = 0.5
DISCRETE_MULTI_TS_CANDIDATES = 5


def _builtin_pairs():
    pairs = []
    for kind in SINGLE_KINDS:
        pairs.append((kind, "EI", {}, lambda p: score_single.ExpectedImprovement()))
        pairs.append((kind, "PI", {}, lambda p: score_single.ProbabilityOfImprovement()))
        pairs.append(
            (
                kind,
                "TS",
                {},
                lambda p: score_single.ThompsonSampling(alpha=p.config.search.alpha),
            )
        )
        pairs.append(
            (
                kind,
                "UCB",
                {"ucb_beta": 2.0},
                lambda p: score_single.UpperConfidenceBound(beta=2.0),
            )
        )
    for kind in MULTI_KINDS:
        pairs.append(
            (kind, "EHVI", {}, lambda p: score_multi.ExpectedHypervolumeImprovement())
        )
        pairs.append(
            (
                kind,
                "HVPI",
                {},
                lambda p: score_multi.HypervolumeProbabilityOfImprovement(),
            )
        )
        pairs.append(
            (
                kind,
                "TS",
                {},
                lambda p: score_multi.ThompsonSampling(
                    alpha=p.config.search.alpha,
                    reduced_candidate_num=p.TS_candidate_num,
                ),
            )
        )
    return pairs


def _run_pair_side(kind, score_arg_factory, extra, N):
    policy = make_policy(kind, config=make_config(alpha=TS_ALPHA), seed=7)
    if kind == "discrete_multi":
        policy.TS_candidate_num = DISCRETE_MULTI_TS_CANDIDATES
    score_arg = score_arg_factory(policy)
    optimizer = None
    if is_range(kind):
        optimizer = physbo.search.optimize.random.Optimizer(
            min_X=MIN_X, max_X=MAX_X, nsamples=20
        )
    res = policy.bayes_search(
        max_num_probes=2,
        num_search_each_probe=N,
        simulator=make_simulator(kind),
        score=score_arg,
        is_disp=False,
        **extra,
        **search_kwargs(kind, optimizer=optimizer),
    )
    return proposals(kind, res, 2 * N)


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize(
    "kind, mode, extra, builtin",
    _builtin_pairs(),
    ids=[f"{k}-{m}" for k, m, _, _ in _builtin_pairs()],
)
def test_string_and_builtin_class_give_same_search(kind, mode, extra, builtin, N):
    """score="<name>" and the corresponding built-in instance give the same proposals and fx."""
    chosen_s, fx_s = _run_pair_side(kind, lambda p: mode, extra, N)
    chosen_b, fx_b = _run_pair_side(kind, builtin, {}, N)
    if is_range(kind):
        np.testing.assert_allclose(chosen_b, chosen_s, rtol=1e-12, atol=0)
    else:
        np.testing.assert_array_equal(chosen_b, chosen_s)
    np.testing.assert_allclose(fx_b, fx_s, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# D3: range_unified accepts ucb_beta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [1, 2])
def test_range_unified_bayes_search_ucb_beta_equals_builtin(N):
    """range_unified bayes_search(score="UCB", ucb_beta=b) equals score=UpperConfidenceBound(beta=b)."""
    # With seed 4 the proposals for beta=5 differ from beta=1 (checked when
    # writing this test), so dropping ucb_beta would be detected.
    b = 5.0

    def run(score_arg, **extra):
        policy = make_policy("range_unified", seed=4)
        res = policy.bayes_search(
            max_num_probes=2,
            num_search_each_probe=N,
            simulator=make_simulator("range_unified"),
            score=score_arg,
            is_disp=False,
            **extra,
            **search_kwargs("range_unified"),
        )
        return proposals("range_unified", res, 2 * N)

    X_s, fx_s = run("UCB", ucb_beta=b)
    X_b, fx_b = run(score_single.UpperConfidenceBound(beta=b))
    np.testing.assert_allclose(X_b, X_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(fx_b, fx_s, rtol=1e-12, atol=0)


@pytest.mark.parametrize("b", [0.0, 3.0])
def test_range_unified_get_score_ucb_beta_equals_builtin(b):
    """range_unified get_score("UCB", xs, ucb_beta=b) equals get_score(UpperConfidenceBound(beta=b), xs)."""
    policy = make_policy("range_unified")
    policy.bayes_search(max_num_probes=0, is_disp=False, **search_kwargs("range_unified"))
    xs = grid_X()
    f_s = policy.get_score("UCB", xs=xs, ucb_beta=b)
    f_b = policy.get_score(score_single.UpperConfidenceBound(beta=b), xs=xs)
    np.testing.assert_allclose(f_s, f_b, rtol=1e-12, atol=0)
    # the beta must actually take effect (UCB with beta=1 differs)
    f_1 = policy.get_score(score_single.UpperConfidenceBound(beta=1.0), xs=xs)
    assert not np.allclose(f_s, f_1)


def test_range_unified_bayes_search_positional_arguments_unchanged():
    """Existing positional arguments of range_unified.bayes_search keep their order."""
    unify = physbo.search.unify.NDS(num_objectives=2)

    policy = make_policy("range_unified", seed=4)
    opt = GridOptimizer(default_grid_points())
    # training_list, max_num_probes, num_search_each_probe, predictor, is_disp,
    # disp_pareto_set, simulator, score, unify_method, interval,
    # num_rand_basis, optimizer, ard
    X_pos = policy.bayes_search(
        None, 1, 2, None, False, False, None, "EI", unify, 0, 0, opt, False
    )

    policy = make_policy("range_unified", seed=4)
    opt = GridOptimizer(default_grid_points())
    X_kw = policy.bayes_search(
        training_list=None,
        max_num_probes=1,
        num_search_each_probe=2,
        predictor=None,
        is_disp=False,
        disp_pareto_set=False,
        simulator=None,
        score="EI",
        unify_method=unify,
        interval=0,
        num_rand_basis=0,
        optimizer=opt,
        ard=False,
    )
    np.testing.assert_array_equal(np.atleast_2d(X_pos), np.atleast_2d(X_kw))
    assert np.atleast_2d(X_pos).shape == (2, 2)


# ---------------------------------------------------------------------------
# D4: parameters for string scores do not affect a callable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_callable_ignores_alpha_ucb_beta_and_TS_candidate_num(kind, N):
    """With a callable score, ucb_beta, config.search.alpha and TS_candidate_num do not change the result."""

    def run(alpha, ucb_beta, ts_candidates):
        policy = make_policy(kind, config=make_config(alpha=alpha), seed=5)
        extra = {}
        if is_multi(kind):
            policy.TS_candidate_num = ts_candidates
        elif ucb_beta is not None:
            extra["ucb_beta"] = ucb_beta
        res = policy.bayes_search(
            max_num_probes=2,
            num_search_each_probe=N,
            simulator=make_simulator(kind),
            score=deterministic_score(kind),
            is_disp=False,
            **extra,
            **search_kwargs(kind),
        )
        return proposals(kind, res, 2 * N)

    chosen_a, fx_a = run(None, None, None)
    chosen_b, fx_b = run(0.3, 7.0, 2)
    np.testing.assert_array_equal(chosen_a, chosen_b)
    np.testing.assert_array_equal(fx_a, fx_b)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_get_score_callable_ignores_alpha_and_ucb_beta(kind):
    """get_score(<callable>, alpha=..., ucb_beta=...) equals get_score(<callable>)."""
    policy = make_policy(kind)
    policy.bayes_search(max_num_probes=0, is_disp=False, **search_kwargs(kind))
    xs = grid_X()[1:8]
    fn = deterministic_score(kind)
    ref = policy.get_score(fn, xs=xs)
    extra = {"alpha": 0.2}
    if not is_multi(kind):
        extra["ucb_beta"] = 9.0
    else:
        policy.TS_candidate_num = 1
    res = policy.get_score(fn, xs=xs, **extra)
    np.testing.assert_array_equal(res, ref)


# ---------------------------------------------------------------------------
# E1: TS alpha comes from config.search.alpha on every path
# ---------------------------------------------------------------------------


def _alphas(spy, original):
    sig = inspect.signature(original)
    out = []
    for c in spy.call_args_list:
        bound = sig.bind(*c.args, **c.kwargs)
        bound.apply_defaults()
        out.append(bound.arguments["alpha"])
    return out


@pytest.mark.parametrize("N", [1, 3])
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_string_TS_uses_config_alpha_everywhere(kind, N, mocker):
    """With score="TS", every call of score.TS / score_multi.TS receives config.search.alpha."""
    alpha = 0.5
    module = score_multi if is_multi(kind) else score_single
    original = module.TS
    spy = mocker.spy(module, "TS")
    policy = make_policy(kind, config=make_config(alpha=alpha, K=2))
    policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=N,
        simulator=None,
        score="TS",
        is_disp=False,
        **search_kwargs(kind),
    )
    alphas = _alphas(spy, original)
    n_points = default_grid_points().shape[0] if is_range(kind) else 1
    assert len(alphas) >= n_points * (1 + (N - 1) * 2)
    assert all(a == alpha for a in alphas), alphas


# ---------------------------------------------------------------------------
# F1 / F2: multiple probe header
# ---------------------------------------------------------------------------


HEADER = re.compile(r"^\d{4}-th multiple probe search \((.*)\)$", re.MULTILINE)


def plain_named_score_single(predictor, training, test):
    return mean_score_single(predictor, training, test)


def plain_named_score_multi(predictor_list, training, test, pareto):
    return mean_score_multi(predictor_list, training, test, pareto)


class AnonymousScoreSingle:
    def __call__(self, predictor, training, test):
        return mean_score_single(predictor, training, test)


class AnonymousScoreMulti:
    def __call__(self, predictor_list, training, test, pareto):
        return mean_score_multi(predictor_list, training, test, pareto)


def _display_case(kind, style):
    """(score argument, expected name) for the header test."""
    multi = is_multi(kind)
    if style == "string":
        name = "EHVI" if multi else "EI"
        return name, name
    if style == "builtin":
        if multi:
            return score_multi.ExpectedHypervolumeImprovement(), "EHVI"
        return score_single.ExpectedImprovement(), "EI"
    if style == "function_with_name":

        def fn(*args):
            return deterministic_score(kind)(*args)

        fn.name = "my_function_name"
        return fn, "my_function_name"
    if style == "object_with_name":
        obj = AnonymousScoreMulti() if multi else AnonymousScoreSingle()
        obj.name = "my_object_name"
        return obj, "my_object_name"
    if style == "function_without_name":
        if multi:
            return plain_named_score_multi, "plain_named_score_multi"
        return plain_named_score_single, "plain_named_score_single"
    if style == "object_without_name":
        if multi:
            return AnonymousScoreMulti(), "AnonymousScoreMulti"
        return AnonymousScoreSingle(), "AnonymousScoreSingle"
    raise ValueError(style)


@pytest.mark.parametrize(
    "style",
    [
        "string",
        "builtin",
        "function_with_name",
        "object_with_name",
        "function_without_name",
        "object_without_name",
    ],
)
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_multiple_probe_header_shows_score_name(kind, style, capsys):
    """The multiple-probe header shows the string, the ``name`` attribute, __name__, or the class name."""
    score_arg, expected = _display_case(kind, style)
    policy = make_policy(kind)
    capsys.readouterr()
    policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=2,
        simulator=None,
        score=score_arg,
        is_disp=False,
        **search_kwargs(kind),
    )
    out = capsys.readouterr().out
    names = HEADER.findall(out)
    assert names == [expected]


@pytest.mark.parametrize("style", ["string", "function_without_name"])
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_multiple_probe_header_only_for_multiple_proposals(kind, style, capsys):
    """The header is printed when num_search_each_probe != 1 even with is_disp=False, and not for 1."""
    score_arg, expected = _display_case(kind, style)

    policy = make_policy(kind)
    capsys.readouterr()
    policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=1,
        simulator=None,
        score=score_arg,
        is_disp=False,
        **search_kwargs(kind),
    )
    assert HEADER.findall(capsys.readouterr().out) == []

    policy = make_policy(kind)
    policy.bayes_search(
        max_num_probes=2,
        num_search_each_probe=2,
        simulator=make_simulator(kind),
        score=score_arg,
        is_disp=False,
        **search_kwargs(kind),
    )
    assert HEADER.findall(capsys.readouterr().out) == [expected, expected]

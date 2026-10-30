# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Integration tests for the follow-up items of the callable score contract.

- B5: an uninstantiated class passed as ``score`` to the six policies'
  ``bayes_search`` / ``get_score`` raises ``TypeError`` before anything
  changes.
- E2: the ``alpha`` argument of the public ``get_score("TS", ...)`` reaches
  ``score.TS`` / ``score_multi.TS``.
- E3: ``policy.TS_candidate_num`` reaches ``score_multi.TS`` as
  ``reduced_candidate_num`` on every path of the multi-objective policies.
"""

import copy
import inspect
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
K_SAMPLING = 2


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
    """Deterministic optimizer: evaluates fixed points in order, returns the argmax."""

    def __init__(self, points):
        self.points = np.array(points, dtype=float)

    def __call__(self, fn, mpicomm=None):
        values = np.array([fn(x.copy()) for x in self.points], dtype=float)
        return self.points[int(np.argmax(values))].copy()


def default_grid_points():
    a = np.linspace(0.05, 0.95, 3)
    return np.array(list(product(a, a)))


def search_kwargs(kind):
    """Extra bayes_search keyword arguments needed by each policy kind."""
    kw = {}
    if kind in UNIFIED_KINDS:
        kw["unify_method"] = physbo.search.unify.NDS(num_objectives=2)
    if is_range(kind):
        kw["optimizer"] = GridOptimizer(default_grid_points())
    return kw


def prepared_policy(kind, config=None):
    """A policy whose predictor is ready for the public get_score."""
    policy = make_policy(kind, config=config)
    policy.bayes_search(max_num_probes=0, is_disp=False, **search_kwargs(kind))
    return policy


def _rng_state_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def _history_snapshot(history):
    """Copies of the plain data attributes of a history object."""
    snap = {}
    for name, value in vars(history).items():
        if isinstance(value, np.ndarray):
            snap[name] = np.array(value, copy=True)
        elif isinstance(value, (list, tuple, int, float, str)):
            snap[name] = copy.deepcopy(value)
    return snap


def _assert_history_equal(before, after):
    assert before.keys() == after.keys()
    for name in before:
        if isinstance(before[name], np.ndarray):
            np.testing.assert_array_equal(before[name], after[name], err_msg=name)
        else:
            assert before[name] == after[name], name


class TSRecorder:
    """Replacement for score.TS / score_multi.TS recording bound arguments.

    Records ``alpha``, ``reduced_candidate_num`` (multi only) and the number
    of training points at call time, then delegates to the original.
    """

    def __init__(self, original):
        self.original = original
        self.signature = inspect.signature(original)
        self.calls = []

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        rec = dict(bound.arguments)
        rec["n_training"] = rec["training"].X.shape[0]
        rec["n_test"] = rec["test"].X.shape[0]
        self.calls.append(rec)
        return self.original(*args, **kwargs)


def install_ts_recorder(kind, monkeypatch):
    module = score_multi if is_multi(kind) else score_single
    rec = TSRecorder(module.TS)
    monkeypatch.setattr(module, "TS", rec)
    return rec


# ---------------------------------------------------------------------------
# B5: an uninstantiated class is rejected by the policies
# ---------------------------------------------------------------------------


class UserScoreSingle:
    """User-defined single-objective score with a configurable weight."""

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, predictor, training, test):
        return self.weight * predictor.get_post_fmean(training, test).flatten()


class UserScoreMulti:
    """User-defined multi-objective score with a configurable weight."""

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, predictor_list, training, test, pareto):
        return self.weight * predictor_list[0].get_post_fmean(
            training, test, objective_index=0
        ).flatten()


def class_cases():
    cases = []
    for kind in SINGLE_KINDS:
        for cls in (
            score_single.ExpectedImprovement,
            score_single.ThompsonSampling,
            UserScoreSingle,
        ):
            cases.append(pytest.param(kind, cls, id=f"{kind}-{cls.__name__}"))
    for kind in MULTI_KINDS:
        for cls in (
            score_multi.ExpectedHypervolumeImprovement,
            score_multi.ThompsonSampling,
            UserScoreMulti,
        ):
            cases.append(pytest.param(kind, cls, id=f"{kind}-{cls.__name__}"))
    return cases


def _resolve_message(kind, cls):
    module = score_multi if is_multi(kind) else score_single
    with pytest.raises(TypeError) as excinfo:
        module.resolve(cls)
    return str(excinfo.value)


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("kind,cls", class_cases())
def test_bayes_search_with_class_raises_type_error_without_side_effects(kind, cls, N):
    """bayes_search(score=<class>) raises the resolve TypeError; np.random and history stay unchanged."""
    policy = make_policy(kind)
    history_before = _history_snapshot(policy.history)
    rng_before = np.random.get_state()
    with pytest.raises(TypeError) as excinfo:
        policy.bayes_search(
            max_num_probes=1,
            num_search_each_probe=N,
            simulator=make_simulator(kind),
            score=cls,
            is_disp=False,
            **search_kwargs(kind),
        )
    message = str(excinfo.value)
    assert f"{cls.__name__}()" in message
    assert message == _resolve_message(kind, cls)
    assert _rng_state_equal(rng_before, np.random.get_state())
    _assert_history_equal(history_before, _history_snapshot(policy.history))


@pytest.mark.parametrize("kind,cls", class_cases())
def test_get_score_with_class_raises_type_error_without_side_effects(kind, cls):
    """get_score(<class>, xs=...) raises the resolve TypeError; np.random and history stay unchanged."""
    policy = prepared_policy(kind)
    xs = grid_X()[2:7]
    history_before = _history_snapshot(policy.history)
    rng_before = np.random.get_state()
    with pytest.raises(TypeError) as excinfo:
        policy.get_score(cls, xs=xs)
    message = str(excinfo.value)
    assert f"{cls.__name__}()" in message
    assert message == _resolve_message(kind, cls)
    assert _rng_state_equal(rng_before, np.random.get_state())
    _assert_history_equal(history_before, _history_snapshot(policy.history))


@pytest.mark.parametrize("kind", DISCRETE_KINDS)
def test_discrete_get_score_with_class_on_default_actions_raises_type_error(kind):
    """Discrete get_score(<class>) over the default unsearched actions raises the TypeError."""
    cls = UserScoreMulti if is_multi(kind) else UserScoreSingle
    policy = prepared_policy(kind)
    assert len(policy.actions) > 0
    rng_before = np.random.get_state()
    with pytest.raises(TypeError) as excinfo:
        policy.get_score(cls)
    assert f"{cls.__name__}()" in str(excinfo.value)
    assert _rng_state_equal(rng_before, np.random.get_state())


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_instance_of_user_class_is_still_accepted(kind):
    """An instance of the same user-defined class is accepted by bayes_search and get_score."""
    obj = UserScoreMulti() if is_multi(kind) else UserScoreSingle()
    policy = make_policy(kind)
    res = policy.bayes_search(
        max_num_probes=1,
        simulator=make_simulator(kind),
        score=obj,
        is_disp=False,
        **search_kwargs(kind),
    )
    assert res.total_num_search == len(INITIAL_ACTIONS) + 1
    xs = grid_X()[2:7]
    assert np.shape(policy.get_score(obj, xs=xs)) == (xs.shape[0],)


# ---------------------------------------------------------------------------
# E2: get_score("TS", alpha=a) passes a to TS
# ---------------------------------------------------------------------------


GET_SCORE_ALPHA = 0.5
CONFIG_ALPHA = 0.3


def e2_cases():
    cases = [pytest.param(kind, "xs", id=f"{kind}-xs") for kind in ALL_KINDS]
    cases += [
        pytest.param(kind, "actions", id=f"{kind}-actions") for kind in DISCRETE_KINDS
    ]
    return cases


@pytest.mark.parametrize("kind,where", e2_cases())
def test_get_score_TS_passes_alpha_argument(kind, where, monkeypatch):
    """get_score("TS", ..., alpha=0.5) calls score.TS / score_multi.TS with alpha=0.5."""
    policy = prepared_policy(kind, config=make_config(alpha=CONFIG_ALPHA))
    rec = install_ts_recorder(kind, monkeypatch)
    np.random.seed(7)
    if where == "xs":
        xs = grid_X()[2:9]
        res = policy.get_score("TS", xs=xs, alpha=GET_SCORE_ALPHA)
        assert len(res) == xs.shape[0]
    else:
        res = policy.get_score("TS", alpha=GET_SCORE_ALPHA)
        assert len(res) == len(policy.actions)
    assert len(rec.calls) >= 1
    alphas = [c["alpha"] for c in rec.calls]
    assert all(a == GET_SCORE_ALPHA for a in alphas), alphas


# ---------------------------------------------------------------------------
# E3: TS_candidate_num reaches score_multi.TS as reduced_candidate_num
# ---------------------------------------------------------------------------


TS_CANDIDATES = 5


@pytest.mark.parametrize("N", [1, 3])
@pytest.mark.parametrize("kind", MULTI_KINDS)
def test_bayes_search_TS_passes_TS_candidate_num(kind, N, monkeypatch):
    """score="TS": every score_multi.TS call in bayes_search gets reduced_candidate_num=TS_candidate_num."""
    policy = make_policy(kind)
    policy.TS_candidate_num = TS_CANDIDATES
    n_observed = policy.training.X.shape[0]
    rec = install_ts_recorder(kind, monkeypatch)
    policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=N,
        simulator=None,
        score="TS",
        is_disp=False,
        **search_kwargs(kind),
    )
    assert len(rec.calls) >= 1
    first = [c for c in rec.calls if c["n_training"] == n_observed]
    later = [c for c in rec.calls if c["n_training"] > n_observed]
    assert len(first) >= 1
    if N > 1:
        assert len(later) >= 1
    values = [c["reduced_candidate_num"] for c in rec.calls]
    assert all(v == TS_CANDIDATES for v in values), values


def e3_get_score_cases():
    cases = [pytest.param(kind, "xs", id=f"{kind}-xs") for kind in MULTI_KINDS]
    cases.append(pytest.param("discrete_multi", "actions", id="discrete_multi-actions"))
    return cases


@pytest.mark.parametrize("kind,where", e3_get_score_cases())
def test_get_score_TS_passes_TS_candidate_num(kind, where, monkeypatch):
    """get_score("TS", ...) calls score_multi.TS with reduced_candidate_num=TS_candidate_num."""
    policy = prepared_policy(kind)
    policy.TS_candidate_num = TS_CANDIDATES
    rec = install_ts_recorder(kind, monkeypatch)
    np.random.seed(7)
    if where == "xs":
        xs = grid_X()[2:9]
        res = policy.get_score("TS", xs=xs)
        assert len(res) == xs.shape[0]
    else:
        res = policy.get_score("TS")
        assert len(res) == len(policy.actions)
    assert len(rec.calls) >= 1
    values = [c["reduced_candidate_num"] for c in rec.calls]
    assert all(v == TS_CANDIDATES for v in values), values

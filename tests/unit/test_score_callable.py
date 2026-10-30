# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for callable acquisition functions (score).

Covers the built-in score classes in physbo.search.score and
physbo.search.score_multi, the ``resolve`` helpers, and the dispatch
functions ``score.score`` / ``score_multi.score`` with string and callable
modes.
"""

import inspect
from itertools import product

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from physbo.search import score
from physbo.search import score_multi


def _grid():
    a = np.linspace(0.0, 1.0, 5)
    return np.array(list(product(a, a)))


def _f(x):
    return -np.sum((x - 0.6) ** 2, axis=-1)


def _f_multi(x):
    return np.stack(
        [-np.sum((x - 0.25) ** 2, axis=-1), -np.sum((x - 0.75) ** 2, axis=-1)],
        axis=-1,
    )


INITIAL_ACTIONS = np.array([0, 7, 13, 24])


@pytest.fixture
def single():
    """A trained single-objective GP predictor with its training/test data."""
    X = _grid()
    policy = physbo.search.discrete.Policy(
        test_X=X, initial_data=(INITIAL_ACTIONS, _f(X[INITIAL_ACTIONS]))
    )
    policy.set_seed(1)
    policy.bayes_search(max_num_probes=0, is_disp=False)
    test = policy.test.get_subset(policy.actions)
    return policy.predictor, policy.training, test


@pytest.fixture
def multi():
    """Trained multi-objective GP predictors with training/test data and Pareto."""
    X = _grid()
    policy = physbo.search.discrete_multi.Policy(
        test_X=X,
        num_objectives=2,
        initial_data=(INITIAL_ACTIONS, _f_multi(X[INITIAL_ACTIONS])),
    )
    policy.set_seed(1)
    policy.bayes_search(max_num_probes=0, is_disp=False)
    test = policy.test.get_subset(policy.actions)
    return policy.predictor_list, policy.training, test, policy.history.pareto


def _empty_test(d=2):
    return physbo.Variable(X=np.zeros((0, d)))


def _rng_state():
    state = np.random.get_state()
    return state[1].copy(), state[2], state[3], state[4]


def _assert_rng_unchanged(before):
    after = _rng_state()
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1:] == after[1:]


class _Recorder:
    """Callable that records every call and returns a fixed value."""

    def __init__(self, ret):
        self.ret = ret
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.ret


# ---------------------------------------------------------------------------
# A1: single-objective built-in classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmax", [None, 1.0, -0.3])
def test_expected_improvement_matches_EI(single, fmax):
    """ExpectedImprovement(fmax)(p, tr, te) equals score.EI(p, tr, te, fmax)."""
    predictor, training, test = single
    if fmax is None:
        obj = score.ExpectedImprovement()
    else:
        obj = score.ExpectedImprovement(fmax=fmax)
    expected = score.EI(predictor, training, test, fmax)
    actual = obj(predictor, training, test)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
    assert obj.fmax == fmax


@pytest.mark.parametrize("fmax", [None, 1.0, -0.3])
def test_probability_of_improvement_matches_PI(single, fmax):
    """ProbabilityOfImprovement(fmax)(p, tr, te) equals score.PI(p, tr, te, fmax)."""
    predictor, training, test = single
    if fmax is None:
        obj = score.ProbabilityOfImprovement()
    else:
        obj = score.ProbabilityOfImprovement(fmax=fmax)
    expected = score.PI(predictor, training, test, fmax)
    actual = obj(predictor, training, test)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
    assert obj.fmax == fmax


@pytest.mark.parametrize("alpha", [None, 0.5, 2.0])
def test_thompson_sampling_matches_TS_under_same_seed(single, alpha):
    """ThompsonSampling(alpha)(p, tr, te) equals score.TS(p, tr, te, alpha) for the same seed."""
    predictor, training, test = single
    if alpha is None:
        obj = score.ThompsonSampling()
        alpha_value = 1.0
    else:
        obj = score.ThompsonSampling(alpha=alpha)
        alpha_value = alpha
    np.random.seed(123)
    expected = score.TS(predictor, training, test, alpha_value)
    np.random.seed(123)
    actual = obj(predictor, training, test)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
    assert obj.alpha == alpha_value


@pytest.mark.parametrize("beta", [None, 0.0, 2.0])
def test_upper_confidence_bound_matches_UCB(single, beta):
    """UpperConfidenceBound(beta)(p, tr, te) equals score.UCB(p, tr, te, beta)."""
    predictor, training, test = single
    if beta is None:
        obj = score.UpperConfidenceBound()
        beta_value = 1.0
    else:
        obj = score.UpperConfidenceBound(beta=beta)
        beta_value = beta
    expected = score.UCB(predictor, training, test, beta_value)
    actual = obj(predictor, training, test)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
    assert obj.beta == beta_value


# ---------------------------------------------------------------------------
# A2: multi-objective built-in classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls_name, mode",
    [("ExpectedHypervolumeImprovement", "EHVI"), ("HypervolumeProbabilityOfImprovement", "HVPI")],
)
def test_multi_hypervolume_classes_match_dispatch(multi, cls_name, mode):
    """EHVI/HVPI classes called with (pl, tr, te, pareto) equal score_multi.score(mode, ...)."""
    predictor_list, training, test, pareto = multi
    obj = getattr(score_multi, cls_name)()
    expected = score_multi.score(
        mode,
        predictor_list=predictor_list,
        test=test,
        training=training,
        pareto=pareto,
        alpha=1.0,
        reduced_candidate_num=None,
    )
    actual = obj(predictor_list, training, test, pareto)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "kwargs, alpha, reduced",
    [
        ({}, 1.0, None),
        ({"alpha": 0.5}, 0.5, None),
        ({"alpha": 0.5, "reduced_candidate_num": 5}, 0.5, 5),
        ({"reduced_candidate_num": 5}, 1.0, 5),
    ],
)
def test_multi_thompson_sampling_matches_dispatch(multi, kwargs, alpha, reduced):
    """score_multi.ThompsonSampling equals score_multi.score("TS", ...) with the same alpha/reduced_candidate_num and seed."""
    predictor_list, training, test, pareto = multi
    obj = score_multi.ThompsonSampling(**kwargs)
    assert obj.alpha == alpha
    assert obj.reduced_candidate_num == reduced
    np.random.seed(7)
    expected = score_multi.score(
        "TS",
        predictor_list=predictor_list,
        test=test,
        training=training,
        pareto=pareto,
        alpha=alpha,
        reduced_candidate_num=reduced,
    )
    np.random.seed(7)
    actual = obj(predictor_list, training, test, pareto)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# A3: name attribute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory, name",
    [
        (lambda: score.ExpectedImprovement(), "EI"),
        (lambda: score.ProbabilityOfImprovement(), "PI"),
        (lambda: score.ThompsonSampling(), "TS"),
        (lambda: score.UpperConfidenceBound(), "UCB"),
        (lambda: score_multi.ExpectedHypervolumeImprovement(), "EHVI"),
        (lambda: score_multi.HypervolumeProbabilityOfImprovement(), "HVPI"),
        (lambda: score_multi.ThompsonSampling(), "TS"),
        (lambda: score.ExpectedImprovement(fmax=1.0), "EI"),
        (lambda: score.ThompsonSampling(alpha=0.5), "TS"),
        (lambda: score.UpperConfidenceBound(beta=2.0), "UCB"),
        (lambda: score_multi.ThompsonSampling(alpha=0.5, reduced_candidate_num=3), "TS"),
    ],
)
def test_builtin_instances_have_name(factory, name):
    """Each built-in score instance exposes the documented ``name``."""
    assert factory().name == name


# ---------------------------------------------------------------------------
# A4: constructors do not consume random numbers
# ---------------------------------------------------------------------------


def test_builtin_constructors_do_not_consume_rng():
    """Constructing any built-in score object leaves np.random state untouched."""
    np.random.seed(2024)
    before = _rng_state()
    score.ExpectedImprovement()
    score.ExpectedImprovement(fmax=1.0)
    score.ProbabilityOfImprovement()
    score.ProbabilityOfImprovement(fmax=1.0)
    score.ThompsonSampling()
    score.ThompsonSampling(alpha=0.5)
    score.UpperConfidenceBound()
    score.UpperConfidenceBound(beta=2.0)
    score_multi.ExpectedHypervolumeImprovement()
    score_multi.HypervolumeProbabilityOfImprovement()
    score_multi.ThompsonSampling()
    score_multi.ThompsonSampling(alpha=0.5, reduced_candidate_num=3)
    _assert_rng_unchanged(before)


# ---------------------------------------------------------------------------
# A5: signatures of existing functions are unchanged
# ---------------------------------------------------------------------------


def _params(func):
    return [
        (p.name, p.default)
        for p in inspect.signature(func).parameters.values()
    ]


_EMPTY = inspect.Parameter.empty


@pytest.mark.parametrize(
    "func, expected",
    [
        (lambda: score.EI, [("predictor", _EMPTY), ("training", _EMPTY), ("test", _EMPTY), ("fmax", None)]),
        (lambda: score.PI, [("predictor", _EMPTY), ("training", _EMPTY), ("test", _EMPTY), ("fmax", None)]),
        (lambda: score.TS, [("predictor", _EMPTY), ("training", _EMPTY), ("test", _EMPTY), ("alpha", 1)]),
        (lambda: score.UCB, [("predictor", _EMPTY), ("training", _EMPTY), ("test", _EMPTY), ("beta", 1.0)]),
        (lambda: score_multi.EHVI, [("fmean", _EMPTY), ("fstd", _EMPTY), ("pareto", _EMPTY)]),
        (lambda: score_multi.HVPI, [("fmean", _EMPTY), ("fstd", _EMPTY), ("pareto", _EMPTY)]),
        (
            lambda: score_multi.TS,
            [
                ("predictor_list", _EMPTY),
                ("training", _EMPTY),
                ("test", _EMPTY),
                ("alpha", 1),
                ("reduced_candidate_num", None),
            ],
        ),
    ],
)
def test_existing_score_function_signatures_unchanged(func, expected):
    """The existing score functions keep their parameter names and defaults."""
    assert _params(func()) == expected


# ---------------------------------------------------------------------------
# B1: score.resolve
# ---------------------------------------------------------------------------


def test_resolve_strings_to_builtin_instances():
    """score.resolve maps "EI"/"PI"/"TS"/"UCB" to the built-in classes."""
    assert isinstance(score.resolve("EI"), score.ExpectedImprovement)
    assert isinstance(score.resolve("PI"), score.ProbabilityOfImprovement)
    ts = score.resolve("TS")
    assert isinstance(ts, score.ThompsonSampling)
    assert ts.alpha == 1.0
    ucb = score.resolve("UCB")
    assert isinstance(ucb, score.UpperConfidenceBound)
    assert ucb.beta == 1.0


def test_resolve_passes_alpha_and_ucb_beta():
    """score.resolve sets TS.alpha from ``alpha`` and UCB.beta from ``ucb_beta``."""
    ts = score.resolve("TS", alpha=0.5, ucb_beta=3.0)
    assert isinstance(ts, score.ThompsonSampling)
    assert ts.alpha == 0.5
    ucb = score.resolve("UCB", alpha=0.5, ucb_beta=3.0)
    assert isinstance(ucb, score.UpperConfidenceBound)
    assert ucb.beta == 3.0


def _plain_score_function(predictor, training, test):
    return np.zeros(test.X.shape[0])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _plain_score_function,
        lambda: _Recorder(np.zeros(0)),
        lambda: score.ExpectedImprovement(),
        lambda: score.ThompsonSampling(alpha=0.3),
    ],
)
def test_resolve_returns_callable_itself(factory):
    """score.resolve returns a callable argument unchanged (identity)."""
    fn = factory()
    assert score.resolve(fn) is fn
    assert score.resolve(fn, alpha=0.5, ucb_beta=2.0) is fn


def test_resolve_unknown_string_raises_not_implemented():
    """score.resolve raises NotImplementedError for an unknown string."""
    with pytest.raises(NotImplementedError):
        score.resolve("XYZ")


@pytest.mark.parametrize("bad", [123, None, 1.5, ["EI"]])
def test_resolve_non_string_non_callable_raises_type_error(bad):
    """score.resolve raises TypeError for neither str nor callable."""
    with pytest.raises(TypeError):
        score.resolve(bad)


def test_resolve_does_not_consume_rng():
    """score.resolve leaves np.random state untouched."""
    np.random.seed(99)
    before = _rng_state()
    for mode in ["EI", "PI", "TS", "UCB"]:
        score.resolve(mode, alpha=0.5, ucb_beta=2.0)
    score.resolve(_plain_score_function)
    _assert_rng_unchanged(before)


# ---------------------------------------------------------------------------
# B2: score_multi.resolve
# ---------------------------------------------------------------------------


def test_multi_resolve_strings_to_builtin_instances():
    """score_multi.resolve maps "EHVI"/"HVPI"/"TS" to the built-in classes."""
    assert isinstance(
        score_multi.resolve("EHVI"), score_multi.ExpectedHypervolumeImprovement
    )
    assert isinstance(
        score_multi.resolve("HVPI"), score_multi.HypervolumeProbabilityOfImprovement
    )
    ts = score_multi.resolve("TS")
    assert isinstance(ts, score_multi.ThompsonSampling)
    assert ts.alpha == 1.0
    assert ts.reduced_candidate_num is None


def test_multi_resolve_passes_alpha_and_reduced_candidate_num():
    """score_multi.resolve("TS") copies alpha and reduced_candidate_num."""
    ts = score_multi.resolve("TS", alpha=0.25, reduced_candidate_num=4)
    assert isinstance(ts, score_multi.ThompsonSampling)
    assert ts.alpha == 0.25
    assert ts.reduced_candidate_num == 4


def _plain_multi_score_function(predictor_list, training, test, pareto):
    return np.zeros(test.X.shape[0])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _plain_multi_score_function,
        lambda: _Recorder(np.zeros(0)),
        lambda: score_multi.ExpectedHypervolumeImprovement(),
        lambda: score_multi.ThompsonSampling(alpha=0.3, reduced_candidate_num=2),
    ],
)
def test_multi_resolve_returns_callable_itself(factory):
    """score_multi.resolve returns a callable argument unchanged (identity)."""
    fn = factory()
    assert score_multi.resolve(fn) is fn
    assert score_multi.resolve(fn, alpha=0.5, reduced_candidate_num=3) is fn


def test_multi_resolve_unknown_string_raises_not_implemented():
    """score_multi.resolve raises NotImplementedError for an unknown string."""
    with pytest.raises(NotImplementedError):
        score_multi.resolve("XYZ")


@pytest.mark.parametrize("bad", [123, None, 1.5, ["EHVI"]])
def test_multi_resolve_non_string_non_callable_raises_type_error(bad):
    """score_multi.resolve raises TypeError for neither str nor callable."""
    with pytest.raises(TypeError):
        score_multi.resolve(bad)


def test_multi_resolve_does_not_consume_rng():
    """score_multi.resolve leaves np.random state untouched."""
    np.random.seed(99)
    before = _rng_state()
    for mode in ["EHVI", "HVPI", "TS"]:
        score_multi.resolve(mode, alpha=0.5, reduced_candidate_num=3)
    score_multi.resolve(_plain_multi_score_function)
    _assert_rng_unchanged(before)


# ---------------------------------------------------------------------------
# B3: score.score dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["EI", "PI", "TS", "UCB", "XYZ"])
def test_score_empty_test_returns_empty_for_any_string(mode):
    """score.score returns a length-0 array for 0 test points, even for unknown strings."""
    res = score.score(mode, predictor=None, test=_empty_test(), training=None)
    assert len(res) == 0


def test_score_empty_test_does_not_call_callable():
    """score.score with 0 test points returns length 0 and never calls the callable."""
    fn = _Recorder(np.array([1.0]))
    res = score.score(fn, predictor=None, test=_empty_test(), training=None)
    assert len(res) == 0
    assert fn.calls == []


@pytest.mark.parametrize("fmax", [None, 0.1])
@pytest.mark.parametrize("mode, func", [("EI", "EI"), ("PI", "PI")])
def test_score_string_EI_PI_use_fmax(single, mode, func, fmax):
    """score.score("EI"/"PI") uses kwargs fmax, defaulting to None."""
    predictor, training, test = single
    expected = getattr(score, func)(predictor, training, test, fmax)
    if fmax is None:
        actual = score.score(mode, predictor, test, training=training)
    else:
        actual = score.score(mode, predictor, test, training=training, fmax=fmax)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("alpha", [None, 0.5])
def test_score_string_TS_uses_alpha(single, alpha):
    """score.score("TS") uses kwargs alpha, defaulting to 1.0."""
    predictor, training, test = single
    np.random.seed(5)
    expected = score.TS(predictor, training, test, 1.0 if alpha is None else alpha)
    np.random.seed(5)
    if alpha is None:
        actual = score.score("TS", predictor, test, training=training)
    else:
        actual = score.score("TS", predictor, test, training=training, alpha=alpha)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("ucb_beta", [None, 2.0])
def test_score_string_UCB_uses_ucb_beta(single, ucb_beta):
    """score.score("UCB") uses kwargs ucb_beta, defaulting to 1.0."""
    predictor, training, test = single
    expected = score.UCB(predictor, training, test, 1.0 if ucb_beta is None else ucb_beta)
    if ucb_beta is None:
        actual = score.score("UCB", predictor, test, training=training)
    else:
        actual = score.score(
            "UCB", predictor, test, training=training, ucb_beta=ucb_beta
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_score_unknown_string_message(single):
    """score.score raises NotImplementedError with the documented message for unknown strings."""
    predictor, training, test = single
    with pytest.raises(NotImplementedError) as excinfo:
        score.score("XYZ", predictor, test, training=training)
    assert str(excinfo.value) == "ERROR: mode must be EI, PI, TS or UCB."


def test_score_callable_called_once_positionally(single):
    """score.score(callable) calls it once as fn(predictor, training, test) and returns its value."""
    predictor, training, test = single
    ret = np.arange(test.X.shape[0], dtype=float)
    fn = _Recorder(ret)
    res = score.score(fn, predictor, test, training=training)
    assert res is ret
    assert len(fn.calls) == 1
    args, kwargs = fn.calls[0]
    assert kwargs == {}
    assert len(args) == 3
    assert args[0] is predictor
    assert args[1] is training
    assert args[2] is test


def test_score_callable_ignores_kwargs(single):
    """score.score(callable, fmax=..., alpha=..., ucb_beta=...) forwards none of the kwargs."""
    predictor, training, test = single
    ret = np.ones(test.X.shape[0])
    fn = _Recorder(ret)
    res = score.score(
        fn, predictor, test, training=training, fmax=0.1, alpha=0.5, ucb_beta=2.0
    )
    assert res is ret
    assert len(fn.calls) == 1
    args, kwargs = fn.calls[0]
    assert kwargs == {}
    assert len(args) == 3
    assert args[0] is predictor and args[1] is training and args[2] is test


def test_score_accepts_plain_function(single):
    """score.score accepts a plain function and returns its result."""
    predictor, training, test = single

    def mean_score(p, tr, te):
        return p.get_post_fmean(tr, te)

    res = score.score(mean_score, predictor, test, training=training)
    np.testing.assert_allclose(
        res, predictor.get_post_fmean(training, test), rtol=1e-12, atol=0
    )


# ---------------------------------------------------------------------------
# B4: score_multi.score dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["EHVI", "HVPI", "TS", "XYZ"])
def test_multi_score_empty_test_returns_empty_for_any_string(mode):
    """score_multi.score returns length 0 for 0 test points, even for unknown strings."""
    res = score_multi.score(
        mode, predictor_list=None, test=_empty_test(), training=None
    )
    assert len(res) == 0


def test_multi_score_empty_test_does_not_call_callable(multi):
    """score_multi.score with 0 test points never calls the callable."""
    predictor_list, training, _, pareto = multi
    fn = _Recorder(np.array([1.0]))
    res = score_multi.score(
        fn,
        predictor_list=predictor_list,
        test=_empty_test(),
        training=training,
        pareto=pareto,
    )
    assert len(res) == 0
    assert fn.calls == []


def _fmean_fstd(predictor_list, training, test):
    fmean = np.array(
        [
            p.get_post_fmean(training, test, objective_index=i)
            for i, p in enumerate(predictor_list)
        ]
    ).T
    fcov = np.array(
        [
            p.get_post_fcov(training, test, objective_index=i)
            for i, p in enumerate(predictor_list)
        ]
    ).T
    return fmean, np.sqrt(fcov)


@pytest.mark.parametrize("mode", ["EHVI", "HVPI"])
def test_multi_score_string_hypervolume_uses_pareto(multi, mode):
    """score_multi.score("EHVI"/"HVPI") evaluates the function with kwargs pareto."""
    predictor_list, training, test, pareto = multi
    fmean, fstd = _fmean_fstd(predictor_list, training, test)
    expected = getattr(score_multi, mode)(fmean, fstd, pareto)
    actual = score_multi.score(
        mode,
        predictor_list=predictor_list,
        test=test,
        training=training,
        pareto=pareto,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("mode", ["EHVI", "HVPI"])
def test_multi_score_string_hypervolume_requires_pareto(multi, mode):
    """score_multi.score("EHVI"/"HVPI") without pareto fails."""
    predictor_list, training, test, _ = multi
    with pytest.raises(Exception):
        score_multi.score(
            mode, predictor_list=predictor_list, test=test, training=training
        )


@pytest.mark.parametrize("alpha", [None, 0.5])
@pytest.mark.parametrize("reduced", [None, 5])
def test_multi_score_string_TS_uses_alpha_and_reduced(multi, alpha, reduced):
    """score_multi.score("TS") uses alpha (default 1.0) and reduced_candidate_num."""
    predictor_list, training, test, pareto = multi
    np.random.seed(11)
    expected = score_multi.TS(
        predictor_list,
        training,
        test,
        1.0 if alpha is None else alpha,
        reduced_candidate_num=reduced,
    )
    kwargs = {"reduced_candidate_num": reduced, "pareto": pareto}
    if alpha is not None:
        kwargs["alpha"] = alpha
    np.random.seed(11)
    actual = score_multi.score(
        "TS", predictor_list=predictor_list, test=test, training=training, **kwargs
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_multi_score_string_TS_requires_reduced_candidate_num(multi):
    """score_multi.score("TS") without reduced_candidate_num raises KeyError."""
    predictor_list, training, test, pareto = multi
    with pytest.raises(KeyError):
        score_multi.score(
            "TS",
            predictor_list=predictor_list,
            test=test,
            training=training,
            pareto=pareto,
        )


def test_multi_score_unknown_string_message(multi):
    """score_multi.score raises NotImplementedError with the documented message."""
    predictor_list, training, test, pareto = multi
    with pytest.raises(NotImplementedError) as excinfo:
        score_multi.score(
            "XYZ",
            predictor_list=predictor_list,
            test=test,
            training=training,
            pareto=pareto,
            reduced_candidate_num=None,
        )
    assert str(excinfo.value) == "mode must be EHVI, HVPI or TS."


def test_multi_score_callable_called_once_positionally(multi):
    """score_multi.score(callable) calls fn(pl, training, test, pareto) once and returns its value."""
    predictor_list, training, test, pareto = multi
    ret = np.arange(test.X.shape[0], dtype=float)
    fn = _Recorder(ret)
    res = score_multi.score(
        fn,
        predictor_list=predictor_list,
        test=test,
        training=training,
        pareto=pareto,
        alpha=0.5,
        reduced_candidate_num=3,
    )
    assert res is ret
    assert len(fn.calls) == 1
    args, kwargs = fn.calls[0]
    assert kwargs == {}
    assert len(args) == 4
    assert args[0] is predictor_list
    assert args[1] is training
    assert args[2] is test
    assert args[3] is pareto


def test_multi_score_callable_without_pareto_receives_none(multi):
    """score_multi.score(callable) without pareto calls fn with pareto=None."""
    predictor_list, training, test, _ = multi
    fn = _Recorder(np.zeros(test.X.shape[0]))
    score_multi.score(fn, predictor_list=predictor_list, test=test, training=training)
    args, _ = fn.calls[0]
    assert args[3] is None


def test_multi_score_TS_instance_without_pareto_matches_string(multi):
    """A ThompsonSampling instance needs no pareto, as the string "TS" does not."""
    predictor_list, training, test, _ = multi
    np.random.seed(3)
    expected = score_multi.score(
        "TS",
        predictor_list=predictor_list,
        test=test,
        training=training,
        reduced_candidate_num=None,
    )
    np.random.seed(3)
    actual = score_multi.score(
        score_multi.ThompsonSampling(),
        predictor_list=predictor_list,
        test=test,
        training=training,
    )
    np.testing.assert_array_equal(actual, expected)

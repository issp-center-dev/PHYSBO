# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The reference points of EHVI and HVPI are prepared alike for the names
and the built-in instances, without candidates and before the optimizer of
range_multi; other scores leave them alone."""

import warnings
from types import SimpleNamespace

import numpy as np
import numpy.testing
import pytest

import physbo
from physbo.search import score_multi
from physbo.search.pareto import Pareto


def estimated_reference_points(front):
    """The documented estimates: front_min - 2 w / n and front_max + 100 w,
    where an objective without extent takes the largest width of the others
    (1 for a front without any extent)."""
    front_min = front.min(axis=0)
    front_max = front.max(axis=0)
    w = front_max - front_min
    w = np.where(w > 0, w, w.max() if np.any(w > 0) else 1.0)
    return front_min - 2 * w / front.shape[0], front_max + 100 * w


FIRST_FRONT = [[1.0, 1.0]]
# beyond both estimates of FIRST_FRONT, [-1, -1] and [101, 101]
GROWN_FRONT = [[150.0, -5.0], [-5.0, 150.0]]


def grown_pareto():
    """A front that has grown beyond both estimated reference points."""
    p = Pareto(2)
    p.update_front(np.array(FIRST_FRONT))
    p.prepare_reference_points()
    numpy.testing.assert_allclose(p.reference_min, [[-1.0, -1.0]])
    numpy.testing.assert_allclose(p.reference_max, [[101.0, 101.0]])
    p.update_front(np.array(GROWN_FRONT))
    return p


def fresh_pareto():
    """A front of three points without reference points."""
    p = Pareto(2)
    p.update_front(np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]))
    return p


PARETOS = {"grown": grown_pareto, "fresh": fresh_pareto}


def no_candidates():
    return SimpleNamespace(X=np.zeros((0, 2)))


def assert_same_reference(actual, expected):
    if expected is None:
        assert actual is None
    else:
        numpy.testing.assert_allclose(actual, expected)


def score_without_candidates(mode, p):
    with warnings.catch_warnings():
        # the estimates are replaced silently
        warnings.simplefilter("error")
        out = score_multi.score(mode, None, no_candidates(), None, pareto=p)
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)
    return out


EHVI_MODES = [
    pytest.param("EHVI", id="EHVI-string"),
    pytest.param(score_multi.ExpectedHypervolumeImprovement(), id="EHVI-instance"),
]
HVPI_MODES = [
    pytest.param("HVPI", id="HVPI-string"),
    pytest.param(
        score_multi.HypervolumeProbabilityOfImprovement(), id="HVPI-instance"
    ),
]


# I1a


@pytest.mark.parametrize("mode", EHVI_MODES)
@pytest.mark.parametrize("which", list(PARETOS))
def test_ehvi_without_candidates_prepares_as_upper_true(mode, which):
    p = PARETOS[which]()
    expected = PARETOS[which]()
    expected.prepare_reference_points(upper=True)

    score_without_candidates(mode, p)

    assert_same_reference(p.reference_min, expected.reference_min)
    assert_same_reference(p.reference_max, expected.reference_max)


@pytest.mark.parametrize("mode", EHVI_MODES)
def test_ehvi_without_candidates_estimates_both_again(mode):
    p = grown_pareto()

    score_without_candidates(mode, p)

    lower, upper = estimated_reference_points(p.front)
    numpy.testing.assert_allclose(np.ravel(p.reference_min), lower)
    numpy.testing.assert_allclose(np.ravel(p.reference_max), upper)


# I1b


@pytest.mark.parametrize("mode", HVPI_MODES)
@pytest.mark.parametrize("which", list(PARETOS))
def test_hvpi_without_candidates_prepares_as_upper_false(mode, which):
    p = PARETOS[which]()
    expected = PARETOS[which]()
    expected.prepare_reference_points(upper=False)

    score_without_candidates(mode, p)

    assert_same_reference(p.reference_min, expected.reference_min)
    assert_same_reference(p.reference_max, expected.reference_max)


@pytest.mark.parametrize("mode", HVPI_MODES)
def test_hvpi_without_candidates_estimates_only_reference_min_again(mode):
    p = grown_pareto()
    reference_max = p.reference_max

    score_without_candidates(mode, p)

    lower, _ = estimated_reference_points(p.front)
    numpy.testing.assert_allclose(np.ravel(p.reference_min), lower)
    assert p.reference_max is reference_max
    numpy.testing.assert_allclose(p.reference_max, [[101.0, 101.0]])


@pytest.mark.parametrize("mode", HVPI_MODES)
def test_hvpi_without_candidates_leaves_a_missing_reference_max(mode):
    p = fresh_pareto()

    score_without_candidates(mode, p)

    lower, _ = estimated_reference_points(p.front)
    numpy.testing.assert_allclose(np.ravel(p.reference_min), lower)
    assert p.reference_max is None


# I1c


class Recorder:
    """A user score: records its calls."""

    def __init__(self, name=None):
        self.calls = []
        if name is not None:
            self.name = name

    def __call__(self, predictor_list, training, test, pareto):
        self.calls.append(test)
        return np.zeros(test.X.shape[0])


def EHVI(predictor_list, training, test, pareto):  # noqa: N802
    """A user function that happens to be named EHVI."""
    EHVI.calls.append(test)
    return np.zeros(test.X.shape[0])


EHVI.calls = []


def other_modes():
    return {
        "TS-string": ("TS", None),
        "TS-instance": (score_multi.ThompsonSampling(), None),
        "user-function-named-EHVI": (EHVI, EHVI.calls),
        "user-object-named-EHVI": (lambda r: (r, r.calls))(Recorder(name="EHVI")),
        "user-object-named-HVPI": (lambda r: (r, r.calls))(Recorder(name="HVPI")),
    }


@pytest.mark.parametrize("key", list(other_modes()))
@pytest.mark.parametrize("which", list(PARETOS))
def test_other_modes_without_candidates_leave_the_reference_points(key, which):
    mode, calls = other_modes()[key]
    n_calls = None if calls is None else len(calls)
    p = PARETOS[which]()
    reference_min, reference_max = p.reference_min, p.reference_max

    score_without_candidates(mode, p)

    assert p.reference_min is reference_min
    assert p.reference_max is reference_max
    if calls is not None:
        assert len(calls) == n_calls  # the callable is not called


# I1d


ALL_MODES = [
    pytest.param("EHVI", id="EHVI-string"),
    pytest.param("HVPI", id="HVPI-string"),
    pytest.param("TS", id="TS-string"),
    pytest.param("XYZ", id="unknown-string"),
    pytest.param(score_multi.ExpectedHypervolumeImprovement(), id="EHVI-instance"),
    pytest.param(
        score_multi.HypervolumeProbabilityOfImprovement(), id="HVPI-instance"
    ),
    pytest.param(score_multi.ThompsonSampling(), id="TS-instance"),
]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_without_pareto_and_candidates_returns_an_empty_array(mode):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = score_multi.score(mode, None, no_candidates(), None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)


def test_without_pareto_and_candidates_a_user_callable_is_not_called():
    r = Recorder(name="EHVI")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = score_multi.score(r, None, no_candidates(), None)
    assert out.shape == (0,)
    assert r.calls == []


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("given", [False, True])
def test_an_empty_front_without_candidates_is_left_alone(mode, given):
    p = Pareto(2)
    if given:
        p.set_reference_min(np.array([[-1.0, -1.0]]))
        p.set_reference_max(np.array([[3.0, 3.0]]))
    reference_min, reference_max = p.reference_min, p.reference_max

    score_without_candidates(mode, p)

    assert p.reference_min is reference_min
    assert p.reference_max is reference_max


# I2


def grown_policy():
    """A range_multi policy whose front has grown beyond both estimates."""
    policy = physbo.search.range_multi.Policy(
        min_X=np.zeros(2), max_X=np.ones(2), num_objectives=2
    )
    policy.set_seed(0)
    policy.write(np.array([[0.2, 0.3]]), np.array(FIRST_FRONT))
    policy.history.pareto.prepare_reference_points()
    policy.write(np.array([[0.5, 0.1], [0.9, 0.7]]), np.array(GROWN_FRONT))
    return policy


def front_inside(pareto, upper):
    if np.any(pareto.front < np.ravel(pareto.reference_min)):
        return False
    return not upper or not np.any(pareto.front > np.ravel(pareto.reference_max))


# an observation beyond the reference points estimated at the first
# step: below reference_min in the first objective, above reference_max in
# the second one
SECOND_OBSERVATION = np.array([[-1000.0, 1.0e6]])


@pytest.mark.parametrize(
    "mode, upper",
    [
        pytest.param(score_multi.ExpectedHypervolumeImprovement(), True, id="EHVI"),
        pytest.param(
            score_multi.HypervolumeProbabilityOfImprovement(), False, id="HVPI"
        ),
    ],
)
def test_range_policy_prepares_for_built_in_instances(mode, upper):
    policy = grown_policy()
    pareto = policy.history.pareto
    old_max = pareto.reference_max
    calls = []

    def never_evaluates(fn, mpicomm=None):
        if not calls:
            lower, upper_estimate = estimated_reference_points(pareto.front)
            numpy.testing.assert_allclose(np.ravel(pareto.reference_min), lower)
            if upper:
                numpy.testing.assert_allclose(
                    np.ravel(pareto.reference_max), upper_estimate
                )
        assert front_inside(pareto, upper)
        if not upper:
            # HVPI does not use reference_max
            assert pareto.reference_max is old_max
        calls.append(pareto.front.copy())
        return np.array([0.4, 0.6])

    # observed at the end of the first step, so in the front at the second
    observations = iter([SECOND_OBSERVATION, np.array([[0.0, 0.0]])])
    policy.bayes_search(
        max_num_probes=2,
        simulator=lambda x: next(observations),
        score=mode,
        optimizer=never_evaluates,
        is_disp=False,
    )

    assert len(calls) == 2
    # the second step indeed saw a front beyond the first estimates
    assert not np.any(np.all(calls[0] == SECOND_OBSERVATION, axis=1))
    assert np.any(np.all(calls[1] == SECOND_OBSERVATION, axis=1))


@pytest.mark.parametrize("name", [None, "EHVI", "HVPI"])
def test_range_policy_does_not_prepare_for_a_user_callable(name):
    policy = grown_policy()
    pareto = policy.history.pareto
    reference_min, reference_max = pareto.reference_min, pareto.reference_max
    user_score = Recorder(name=name)
    calls = []

    def never_evaluates(fn, mpicomm=None):
        assert pareto.reference_min is reference_min
        assert pareto.reference_max is reference_max
        calls.append(True)
        return np.array([0.4, 0.6])

    policy.bayes_search(
        max_num_probes=1,
        simulator=lambda x: np.array([[0.0, 0.0]]),
        score=user_score,
        optimizer=never_evaluates,
        is_disp=False,
    )

    assert calls
    assert user_score.calls == []

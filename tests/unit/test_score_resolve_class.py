# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for ``resolve`` rejecting an uninstantiated class (contract B5).

A class object passed as ``score`` raises ``TypeError`` whose message
contains the class name followed by ``()``, while instances of the same
classes and plain functions are still accepted unchanged.
"""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from physbo.search import score
from physbo.search import score_multi


class UserScoreSingle:
    """User-defined single-objective score with a configurable weight."""

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, predictor, training, test):
        return self.weight * predictor.get_post_fmean(training, test)


class UserScoreMulti:
    """User-defined multi-objective score with a configurable weight."""

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, predictor_list, training, test, pareto):
        return self.weight * np.zeros(test.X.shape[0])


def _plain_single(predictor, training, test):
    return predictor.get_post_fmean(training, test)


def _plain_multi(predictor_list, training, test, pareto):
    return np.zeros(test.X.shape[0])


SINGLE_CLASSES = [
    score.ExpectedImprovement,
    score.ProbabilityOfImprovement,
    score.ThompsonSampling,
    score.UpperConfidenceBound,
    UserScoreSingle,
]
MULTI_CLASSES = [
    score_multi.ExpectedHypervolumeImprovement,
    score_multi.HypervolumeProbabilityOfImprovement,
    score_multi.ThompsonSampling,
    UserScoreMulti,
]


def _ids(classes):
    return [c.__name__ for c in classes]


def _rng_state_equal(a, b):
    return (
        a[0] == b[0]
        and np.array_equal(a[1], b[1])
        and a[2:] == b[2:]
    )


# ---------------------------------------------------------------------------
# B5: score.resolve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", SINGLE_CLASSES, ids=_ids(SINGLE_CLASSES))
def test_resolve_class_raises_type_error_suggesting_instance(cls):
    """score.resolve(<class>) raises TypeError whose message contains "<ClassName>()"."""
    with pytest.raises(TypeError) as excinfo:
        score.resolve(cls)
    assert f"{cls.__name__}()" in str(excinfo.value)


@pytest.mark.parametrize("cls", SINGLE_CLASSES, ids=_ids(SINGLE_CLASSES))
def test_resolve_class_with_parameters_raises_type_error(cls):
    """score.resolve(<class>, alpha=..., ucb_beta=...) also raises the TypeError."""
    with pytest.raises(TypeError) as excinfo:
        score.resolve(cls, alpha=0.5, ucb_beta=2.0)
    assert f"{cls.__name__}()" in str(excinfo.value)


def test_resolve_multi_class_given_to_single_resolve_raises_type_error():
    """A multi-objective built-in class passed to score.resolve is still rejected as a class."""
    cls = score_multi.ThompsonSampling
    with pytest.raises(TypeError) as excinfo:
        score.resolve(cls)
    assert f"{cls.__name__}()" in str(excinfo.value)


@pytest.mark.parametrize("cls", SINGLE_CLASSES, ids=_ids(SINGLE_CLASSES))
def test_resolve_instance_of_class_is_returned_unchanged(cls):
    """score.resolve(<instance>) still returns the very same instance."""
    obj = cls()
    assert score.resolve(obj) is obj


def test_resolve_function_is_returned_unchanged():
    """score.resolve(<function>) still returns the very same function."""
    assert score.resolve(_plain_single) is _plain_single


def test_resolve_class_does_not_consume_rng():
    """score.resolve(<class>) leaves np.random state untouched."""
    np.random.seed(123)
    before = np.random.get_state()
    for cls in SINGLE_CLASSES:
        with pytest.raises(TypeError):
            score.resolve(cls)
    assert _rng_state_equal(before, np.random.get_state())


# ---------------------------------------------------------------------------
# B5: score_multi.resolve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", MULTI_CLASSES, ids=_ids(MULTI_CLASSES))
def test_multi_resolve_class_raises_type_error_suggesting_instance(cls):
    """score_multi.resolve(<class>) raises TypeError whose message contains "<ClassName>()"."""
    with pytest.raises(TypeError) as excinfo:
        score_multi.resolve(cls)
    assert f"{cls.__name__}()" in str(excinfo.value)


@pytest.mark.parametrize("cls", MULTI_CLASSES, ids=_ids(MULTI_CLASSES))
def test_multi_resolve_class_with_parameters_raises_type_error(cls):
    """score_multi.resolve(<class>, alpha=..., reduced_candidate_num=...) also raises the TypeError."""
    with pytest.raises(TypeError) as excinfo:
        score_multi.resolve(cls, alpha=0.5, reduced_candidate_num=3)
    assert f"{cls.__name__}()" in str(excinfo.value)


def test_multi_resolve_single_class_given_to_multi_resolve_raises_type_error():
    """A single-objective built-in class passed to score_multi.resolve is still rejected as a class."""
    cls = score.ExpectedImprovement
    with pytest.raises(TypeError) as excinfo:
        score_multi.resolve(cls)
    assert f"{cls.__name__}()" in str(excinfo.value)


@pytest.mark.parametrize("cls", MULTI_CLASSES, ids=_ids(MULTI_CLASSES))
def test_multi_resolve_instance_of_class_is_returned_unchanged(cls):
    """score_multi.resolve(<instance>) still returns the very same instance."""
    obj = cls()
    assert score_multi.resolve(obj) is obj


def test_multi_resolve_function_is_returned_unchanged():
    """score_multi.resolve(<function>) still returns the very same function."""
    assert score_multi.resolve(_plain_multi) is _plain_multi


def test_multi_resolve_class_does_not_consume_rng():
    """score_multi.resolve(<class>) leaves np.random state untouched."""
    np.random.seed(123)
    before = np.random.get_state()
    for cls in MULTI_CLASSES:
        with pytest.raises(TypeError):
            score_multi.resolve(cls)
    assert _rng_state_equal(before, np.random.get_state())

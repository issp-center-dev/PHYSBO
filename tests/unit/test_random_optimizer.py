# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import warnings

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from physbo.search.optimize.random import Optimizer


def run(scores, nsamples=None):
    """Run the optimizer on the given sequence of scores.

    Returns the returned point, the samples in the order of evaluation and the
    warnings issued.
    """
    scores = list(scores)
    samples = []

    def fn(x):
        samples.append(x.copy())
        return scores[len(samples) - 1]

    opt = Optimizer(
        min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=nsamples or len(scores)
    )
    np.random.seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = opt(fn)
    return x, samples, [w.category for w in caught]


@pytest.mark.parametrize(
    "scores",
    [
        [np.nan],
        [np.nan, np.nan, np.nan],
        [-np.inf, -np.inf],
        [np.nan, -np.inf, np.nan],
        [-np.inf, np.nan],
        [np.float32(np.nan), np.float16(-np.inf)],
        [np.array([np.nan]), np.array([-np.inf])],
    ],
)
def test_no_score_larger_than_minus_inf_returns_first_sample(scores):
    x, samples, caught = run(scores)
    assert x is not None
    assert np.array_equal(x, samples[0])
    assert caught == [RuntimeWarning]


@pytest.mark.parametrize(
    "scores, best",
    [
        ([np.nan, 1.0, np.nan], 1),
        ([-np.inf, -np.inf, -5.0], 2),
        ([np.nan, 2.0, 3.0, 1.0], 2),
        ([1.0, 1.0], 0),
        ([np.inf, np.nan, np.inf], 0),
        ([np.float32(np.nan), np.float32(1.0), np.float16(2.0)], 2),
        ([np.array([np.nan]), np.array([1.0]), np.array([0.5])], 1),
        ([np.longdouble(-np.inf), 3, np.int64(4)], 2),
    ],
)
def test_best_sample_without_warning(scores, best):
    x, samples, caught = run(scores)
    assert np.array_equal(x, samples[best])
    assert caught == []


def test_warning_can_be_turned_into_error():
    opt = Optimizer(min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="larger than -inf"):
            opt(lambda x: np.nan)


def test_range_policy_gives_simulator_a_point_within_bounds():
    """With objective values so large that every EI score is nan, the range
    policy passes a point within the bounds to the simulator, not nan."""
    np.random.seed(0)
    calls = []

    def sim(x):
        x = np.atleast_2d(x)
        calls.append(x.copy())
        return 1e300 * np.sum(x, axis=1)

    policy = physbo.search.range.Policy(min_X=np.zeros(2), max_X=np.ones(2))
    policy.random_search(max_num_probes=3, simulator=sim, is_disp=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        policy.bayes_search(max_num_probes=2, simulator=sim, score="EI", is_disp=False)
    assert len(calls) == 5
    for x in calls[3:]:
        assert np.all(np.isfinite(x))
        assert np.all((0 <= x) & (x <= 1))
    assert any(
        w.category is RuntimeWarning and "larger than -inf" in str(w.message)
        for w in caught
    )

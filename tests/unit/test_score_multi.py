# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function

from types import SimpleNamespace

import pytest
import numpy as np

physbo = pytest.importorskip("physbo")


def make_predictor(mocker):
    """A predictor whose posterior samples are random per objective."""

    def get_post_samples(training, test, alpha, objective_index, rng=None):
        return np.random.randn(test.X.shape[0])

    p = mocker.MagicMock()
    p.get_post_samples = mocker.MagicMock(side_effect=get_post_samples)
    return p


@pytest.fixture
def predictor_list(mocker):
    return [make_predictor(mocker), make_predictor(mocker)]


@pytest.fixture
def test_points():
    # 10 candidate points in 2D
    return SimpleNamespace(X=np.arange(20, dtype=float).reshape(10, 2))


def test_TS_returns_onehot_over_all_candidates(predictor_list, test_points):
    """TS must return a one-hot score with length == number of candidates."""
    np.random.seed(0)
    score = physbo.search.score_multi.TS(predictor_list, None, test_points)
    N = test_points.X.shape[0]
    assert score.shape == (N,)
    assert np.count_nonzero(score) == 1
    assert score.sum() == pytest.approx(1.0)


def test_TS_reduced_candidate_samples_full_set(predictor_list, test_points):
    """Regression test for the reduced-candidate sampling bug.

    With ``reduced_candidate_num`` smaller than the number of candidates, the
    subset of candidates considered for the Pareto front must be drawn from the
    *entire* candidate set, not only from the first ``reduced_candidate_num``
    indices.  The previous implementation did
    ``np.random.choice(np.arange(reduced_candidate_num), ...)`` which could only
    ever select indices ``[0, reduced_candidate_num)``.
    """
    N = test_points.X.shape[0]  # 10
    reduced = 3
    np.random.seed(0)

    chosen = set()
    for _ in range(300):
        score = physbo.search.score_multi.TS(
            predictor_list, None, test_points, reduced_candidate_num=reduced
        )
        chosen.add(int(np.argmax(score)))

    # With the bug, every chosen index is < reduced. After the fix, candidates
    # from across the full range [0, N) can be selected.
    assert max(chosen) >= reduced
    # Sanity: indices stay within range.
    assert min(chosen) >= 0 and max(chosen) < N


def _reduction_calls(choice_spy):
    """Calls to np.random.choice that perform candidate reduction.

    The reduction call has the form ``np.random.choice(N, k, replace=False)``;
    the unrelated Pareto-front pick is ``np.random.choice(front_num)`` (a single
    positional arg, no ``replace``).
    """
    return [
        c
        for c in choice_spy.call_args_list
        if c.kwargs.get("replace") is False or len(c.args) >= 2
    ]


def test_TS_reduced_branch_samples_from_full_set(predictor_list, test_points, mocker):
    """The reduction call must draw from the FULL candidate set, not [0, k).

    This is the core of the bug fix and would fail both for the original
    ``np.random.choice(np.arange(k), ...)`` (first arg an array, not N) and for a
    ``<=`` -> ``<`` mutation.
    """
    N = test_points.X.shape[0]  # 10
    k = 3
    choice_spy = mocker.spy(np.random, "choice")
    np.random.seed(0)
    physbo.search.score_multi.TS(
        predictor_list, None, test_points, reduced_candidate_num=k
    )

    reductions = _reduction_calls(choice_spy)
    assert len(reductions) == 1
    first = reductions[0]
    # Population is the full candidate count N (scalar), sample size is k.
    assert np.ndim(first.args[0]) == 0 and int(first.args[0]) == N
    assert int(first.args[1]) == k
    # Reduction must sample without replacement (no duplicate candidates).
    # np.random.choice accepts ``replace`` positionally or by keyword.
    replace = (
        first.kwargs["replace"]
        if "replace" in first.kwargs
        else (first.args[2] if len(first.args) > 2 else None)
    )
    assert replace is False


@pytest.mark.parametrize("reduced_candidate_num", [None, 10, 20])
def test_TS_full_range_branch_skips_reduction(
    predictor_list, test_points, mocker, reduced_candidate_num
):
    """For ``None``, ``== N`` and ``> N`` no reduction sampling happens.

    ``score.shape[0] <= reduced_candidate_num`` (and the ``None`` case) take the
    full-range branch, so np.random.choice must not be called to reduce
    candidates. A ``<=`` -> ``<`` mutation would make the ``== N`` case reduce
    and fail this test. The output must remain one-hot over all candidates.
    """
    N = test_points.X.shape[0]  # 10
    choice_spy = mocker.spy(np.random, "choice")
    np.random.seed(0)
    score = physbo.search.score_multi.TS(
        predictor_list,
        None,
        test_points,
        reduced_candidate_num=reduced_candidate_num,
    )
    assert _reduction_calls(choice_spy) == []
    assert score.shape == (N,)
    assert np.count_nonzero(score) == 1
    assert score.sum() == pytest.approx(1.0)

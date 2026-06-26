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

    def get_post_samples(training, test, alpha, objective_index):
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

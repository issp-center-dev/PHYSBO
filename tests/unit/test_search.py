# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function

import pytest
import numpy as np
import numpy.random as random

physbo = pytest.importorskip("physbo")


def get_post_fmean(training, test):
    print(test)
    N = test.shape[0]
    return random.randn(N, 1)


def get_post_fcov(training, test):
    print(test)
    N = test.shape[0]
    return np.ones((N, 1))


def get_post_samples(training, test, alpha=1.0):
    print(test)
    N = test.shape[0]
    return random.randn(N, 1)


@pytest.fixture
def predictor(mocker):
    p = mocker.MagicMock()
    p.get_post_fmean = mocker.MagicMock(side_effect=get_post_fmean)
    p.get_post_fcov = mocker.MagicMock(side_effect=get_post_fcov)
    p.get_post_samples = mocker.MagicMock(side_effect=get_post_samples)
    return p


@pytest.fixture
def X():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
        ]
    )


@pytest.fixture
def Y():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
    )


def test_EI(predictor, X, Y):
    N = Y.shape[0]
    score = physbo.search.score.EI(predictor, X, Y)
    assert score.shape[0] == N
    predictor.get_post_fmean.assert_any_call(X, Y)
    predictor.get_post_fcov.assert_called_once_with(X, Y)
    predictor.get_post_samples.assert_not_called()

    predictor.get_post_fmean.reset_mock()
    predictor.get_post_fcov.reset_mock()
    predictor.get_post_samples.reset_mock()

    score = physbo.search.score.EI(predictor, X, Y, fmax=1.0)
    assert score.shape[0] == N
    predictor.get_post_fmean.assert_called_once_with(X, Y)
    predictor.get_post_fcov.assert_called_once_with(X, Y)
    predictor.get_post_samples.assert_not_called()


def test_PI(predictor, X, Y):
    N = Y.shape[0]
    score = physbo.search.score.PI(predictor, X, Y)
    assert score.shape[0] == N
    predictor.get_post_fmean.assert_any_call(X, Y)
    predictor.get_post_fcov.assert_called_once_with(X, Y)
    predictor.get_post_samples.assert_not_called()

    predictor.get_post_fmean.reset_mock()
    predictor.get_post_fcov.reset_mock()
    predictor.get_post_samples.reset_mock()

    score = physbo.search.score.EI(predictor, X, Y, fmax=1.0)
    assert score.shape[0] == N
    predictor.get_post_fmean.assert_called_once_with(X, Y)
    predictor.get_post_fcov.assert_called_once_with(X, Y)
    predictor.get_post_samples.assert_not_called()


def test_TS(predictor, X, Y):
    N = Y.shape[0]
    alpha = 1.0
    score = physbo.search.score.TS(predictor, X, Y, alpha=alpha)
    assert score.shape[0] == N
    predictor.get_post_fmean.assert_not_called()
    predictor.get_post_fcov.assert_not_called()
    predictor.get_post_samples.assert_called_once_with(X, Y, alpha=alpha)


def test_UCB(predictor, X, Y):
    N = Y.shape[0]
    score = physbo.search.score.UCB(predictor, X, Y, beta=1.0)
    assert score.shape[0] == N
    predictor.get_post_fmean.assert_called_once_with(X, Y)
    predictor.get_post_fcov.assert_called_once_with(X, Y)
    predictor.get_post_samples.assert_not_called()


def test_UCB_beta_controls_exploration(mocker, X, Y):
    # Deterministic predictor: fixed mean and variance per candidate.
    fmean = np.array([1.0, 2.0, 0.0, 0.0])
    fcov = np.array([0.0, 0.0, 4.0, 1.0])
    p = mocker.MagicMock()
    p.get_post_fmean = mocker.MagicMock(return_value=fmean)
    p.get_post_fcov = mocker.MagicMock(return_value=fcov)

    # beta == 0 reduces to greedy exploitation of the posterior mean.
    greedy = physbo.search.score.UCB(p, X, Y, beta=0.0)
    np.testing.assert_allclose(greedy, fmean)
    assert np.argmax(greedy) == 1

    # Large beta favors the high-variance candidate (index 2, std == 2).
    explore = physbo.search.score.UCB(p, X, Y, beta=10.0)
    np.testing.assert_allclose(explore, fmean + 10.0 * np.sqrt(fcov))
    assert np.argmax(explore) == 2


def test_score_dispatch_UCB(X, Y, mocker):
    N = X.shape[0]
    test = mocker.MagicMock()
    test.X = X
    p = mocker.MagicMock()
    p.get_post_fmean = mocker.MagicMock(return_value=np.arange(N, dtype=float))
    p.get_post_fcov = mocker.MagicMock(return_value=np.ones(N))

    score = physbo.search.score.score("UCB", p, test, training=Y, ucb_beta=2.0)
    assert score.shape[0] == N
    # score = fmean + 2 * sqrt(fcov) = arange(N) + 2
    np.testing.assert_allclose(score, np.arange(N, dtype=float) + 2.0)


def test_score_dispatch_unknown_mode(X, Y, mocker):
    test = mocker.MagicMock()
    test.X = X
    with pytest.raises(NotImplementedError):
        physbo.search.score.score("NOPE", mocker.MagicMock(), test, training=Y)

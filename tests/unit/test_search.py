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


def get_post_samples(training, test, alpha=1.0, rng=None):
    print(test)
    N = test.shape[0]
    return random.randn(N, 1)


@pytest.fixture
def predictor(mocker):
    p = mocker.MagicMock()
    p.get_post_fmean = mocker.MagicMock(side_effect=get_post_fmean)
    p.get_post_fcov = mocker.MagicMock(side_effect=get_post_fcov)
    p.get_post_samples = mocker.MagicMock(side_effect=get_post_samples)
    # a plain MagicMock auto-creates any attribute, which would make
    # score.TS take the draw/evaluate path meant for the BLM predictor
    del p.draw_post_sample_params
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
    predictor.get_post_samples.assert_called_once_with(X, Y, alpha=alpha, rng=None)

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Mathematical correctness tests for the acquisition functions.

EI and PI are compared against their definitions evaluated by numerical
integration over the Gaussian posterior (independent of the closed-form
implementation in physbo.search.score):

    EI(x) = E[max(f(x) - fmax, 0)]
    PI(x) = P[f(x) > fmax]
"""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from physbo.search import score


class FakePredictor:
    """Predictor stub returning prescribed posterior moments.

    The training set is recognized by identity so that the default-fmax
    branch (max of the posterior mean over the training data) can be
    exercised as well.
    """

    def __init__(self, fmean, fcov, training=None, train_fmean=None):
        self.fmean = np.asarray(fmean, dtype=float)
        self.fcov = np.asarray(fcov, dtype=float)
        # use a private sentinel so that a test set of None never matches
        self.training = training if training is not None else object()
        self.train_fmean = train_fmean

    def get_post_fmean(self, training, test):
        if test is self.training:
            return np.asarray(self.train_fmean, dtype=float)
        return self.fmean

    def get_post_fcov(self, training, test):
        return self.fcov

    def get_post_samples(self, training, test, alpha=1.0):
        return self.fmean + alpha * np.sqrt(self.fcov)


def ei_by_integration(fmean, fstd, fmax):
    x = np.linspace(fmean - 12 * fstd, fmean + 12 * fstd, 200001)
    density = np.exp(-0.5 * ((x - fmean) / fstd) ** 2) / (
        fstd * np.sqrt(2 * np.pi)
    )
    return np.trapezoid(np.maximum(x - fmax, 0.0) * density, x)


def pi_by_integration(fmean, fstd, fmax):
    # integrate the density over [fmax, fmean + 12 sigma]; starting the
    # grid exactly at fmax keeps the integrand smooth
    x = np.linspace(fmax, fmean + 12 * fstd, 200001)
    density = np.exp(-0.5 * ((x - fmean) / fstd) ** 2) / (
        fstd * np.sqrt(2 * np.pi)
    )
    return np.trapezoid(density, x)


@pytest.mark.parametrize(
    "fmean, fstd, fmax",
    [
        (0.0, 1.0, 0.0),
        (1.0, 2.0, 0.0),
        (0.0, 1.0, 1.5),
        (-0.5, 0.1, 0.0),
        (3.0, 0.5, 1.0),
    ],
)
def test_EI_against_integration(fmean, fstd, fmax):
    predictor = FakePredictor(fmean=[fmean], fcov=[fstd**2])
    res = score.EI(predictor, None, None, fmax=fmax)
    ref = ei_by_integration(fmean, fstd, fmax)
    assert res[0] == pytest.approx(ref, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize(
    "fmean, fstd, fmax",
    [
        (0.0, 1.0, 0.0),
        (1.0, 2.0, 0.0),
        (0.0, 1.0, 1.5),
        (-0.5, 0.1, 0.0),
        (3.0, 0.5, 1.0),
    ],
)
def test_PI_against_integration(fmean, fstd, fmax):
    predictor = FakePredictor(fmean=[fmean], fcov=[fstd**2])
    res = score.PI(predictor, None, None, fmax=fmax)
    ref = pi_by_integration(fmean, fstd, fmax)
    assert res[0] == pytest.approx(ref, rel=1e-6, abs=1e-12)


def test_EI_vectorized():
    fmean = np.array([0.0, 1.0, 2.0])
    fstd = np.array([1.0, 0.5, 2.0])
    predictor = FakePredictor(fmean=fmean, fcov=fstd**2)
    res = score.EI(predictor, None, None, fmax=1.0)
    assert res.shape == (3,)
    for i in range(3):
        assert res[i] == pytest.approx(
            ei_by_integration(fmean[i], fstd[i], 1.0), rel=1e-6, abs=1e-12
        )


def test_EI_properties():
    # EI is non-negative and increasing in the posterior mean
    fmean = np.array([-1.0, 0.0, 1.0, 2.0])
    predictor = FakePredictor(fmean=fmean, fcov=np.ones(4))
    res = score.EI(predictor, None, None, fmax=0.5)
    assert np.all(res >= 0.0)
    assert np.all(np.diff(res) > 0.0)
    # EI is bounded below by the expected improvement of the mean
    assert np.all(res >= fmean - 0.5)


def test_PI_properties():
    fmean = np.array([-1.0, 0.0, 1.0, 2.0])
    predictor = FakePredictor(fmean=fmean, fcov=np.ones(4))
    res = score.PI(predictor, None, None, fmax=0.5)
    assert np.all((res >= 0.0) & (res <= 1.0))
    assert np.all(np.diff(res) > 0.0)


def test_default_fmax_is_max_training_mean():
    # when fmax is omitted, the maximum posterior mean over the training
    # data must be used
    training = object()
    predictor = FakePredictor(
        fmean=[1.0],
        fcov=[1.0],
        training=training,
        train_fmean=[0.3, 0.7, -0.2],
    )
    res_default = score.EI(predictor, training, None)
    res_explicit = score.EI(predictor, training, None, fmax=0.7)
    assert res_default[0] == pytest.approx(res_explicit[0])

    res_default = score.PI(predictor, training, None)
    res_explicit = score.PI(predictor, training, None, fmax=0.7)
    assert res_default[0] == pytest.approx(res_explicit[0])


def test_TS_returns_flattened_samples():
    predictor = FakePredictor(fmean=np.array([[1.0], [2.0]]), fcov=np.zeros((2, 1)))
    res = score.TS(predictor, None, None, alpha=1.0)
    assert res.shape == (2,)
    np.testing.assert_allclose(res, [1.0, 2.0])


def test_score_unknown_mode():
    class Test:
        X = np.zeros((2, 2))

    predictor = FakePredictor(fmean=[0.0], fcov=[1.0])
    with pytest.raises(NotImplementedError):
        score.score("XX", predictor, Test())


def test_score_empty_test():
    class Test:
        X = np.zeros((0, 2))

    predictor = FakePredictor(fmean=[0.0], fcov=[1.0])
    res = score.score("EI", predictor, Test())
    assert res.shape == (0,)

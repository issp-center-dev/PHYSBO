# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Gradient checks for the GP hyperparameter learning.

The analytic gradient of the marginal likelihood (chaining the gradients
of the covariance, likelihood, and mean modules) is compared with central
finite differences.
"""

import numpy as np
import numpy.testing
import pytest

physbo = pytest.importorskip("physbo")


def make_model(dim, ard):
    cov = physbo.gp.cov.Gauss(dim, ard=ard)
    mean = physbo.gp.mean.Const()
    lik = physbo.gp.lik.Gauss()
    return physbo.gp.core.Model(lik=lik, mean=mean, cov=cov)


@pytest.fixture
def data():
    rng = np.random.RandomState(12345)
    X = rng.randn(10, 3)
    t = np.sin(X[:, 0]) + 0.1 * rng.randn(10)
    return X, t


@pytest.mark.parametrize("ard", [False, True])
def test_grad_marlik_finite_difference(data, ard):
    X, t = data
    model = make_model(X.shape[1], ard)
    params = model.cat_params(model.lik.params, model.prior.params)

    grad = model.get_grad_marlik(params, X, t)
    assert grad.shape == (len(params),)

    h = 1e-6
    for i in range(len(params)):
        dp = np.zeros_like(params)
        dp[i] = h
        fd = (
            model.eval_marlik(params + dp, X, t)
            - model.eval_marlik(params - dp, X, t)
        ) / (2 * h)
        assert grad[i] == pytest.approx(fd, rel=1e-4, abs=1e-8), f"param {i}"


@pytest.mark.parametrize("ard", [False, True])
def test_cov_grad_finite_difference(data, ard):
    X, _ = data
    cov = physbo.gp.cov.Gauss(X.shape[1], ard=ard)
    params = cov.params

    grad = cov.get_grad(X, params)
    n_params = len(params)
    N = X.shape[0]
    assert grad.shape == (n_params, N, N)

    h = 1e-6
    for i in range(n_params):
        dp = np.zeros_like(params)
        dp[i] = h
        fd = (cov.get_cov(X, params=params + dp) - cov.get_cov(X, params=params - dp)) / (
            2 * h
        )
        numpy.testing.assert_allclose(
            grad[i], fd, rtol=1e-5, atol=1e-8, err_msg=f"param {i}"
        )


def test_lik_grad_finite_difference():
    lik = physbo.gp.lik.Gauss()
    num_data = 5
    params = np.array([0.3])

    grad = lik.get_grad(num_data, params)
    h = 1e-6
    fd = (
        lik.get_cov(num_data, params + h) - lik.get_cov(num_data, params - h)
    ) / (2 * h)
    numpy.testing.assert_allclose(grad.toarray() if hasattr(grad, "toarray") else grad,
                                  fd.toarray() if hasattr(fd, "toarray") else fd,
                                  rtol=1e-5, atol=1e-8)


def test_mean_grad():
    mean = physbo.gp.mean.Const()
    num_data = 5
    params = np.array([0.7])

    grad = mean.get_grad(num_data, params)
    h = 1e-6
    fd = (
        mean.get_mean(num_data, params + h) - mean.get_mean(num_data, params - h)
    ) / (2 * h)
    numpy.testing.assert_allclose(np.asarray(grad).reshape(-1), fd, rtol=1e-5)


@pytest.mark.parametrize("method", ["adam", "bfgs"])
def test_fit_improves_marlik(data, method):
    from physbo.misc import _set_config

    X, t = data
    model = make_model(X.shape[1], ard=False)
    params0 = model.cat_params(model.lik.params, model.prior.params)
    marlik0 = model.eval_marlik(params0, X, t)

    if method == "adam":
        learning_config = _set_config.Adam()
        learning_config.max_epoch = 200
    else:
        learning_config = _set_config.Batch()
    config = physbo.misc.SetConfig(learning_config=learning_config)
    model.fit(X, t, config)

    params1 = model.cat_params(model.lik.params, model.prior.params)
    marlik1 = model.eval_marlik(params1, X, t)
    assert np.all(np.isfinite(params1))
    # eval_marlik returns the negative log marginal likelihood: smaller is better
    assert marlik1 < marlik0

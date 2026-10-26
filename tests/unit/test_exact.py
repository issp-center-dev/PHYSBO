# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests for two GP speedups.

* ``exact.get_grad_marlik`` inverts the covariance from its Cholesky factor
  with ``dpotri`` instead of ``scipy.linalg.inv``; the gradient is checked
  against the one obtained with the LU-based inverse.  The comparison with a
  finite-difference gradient is in ``test_gp_gradient.py``.
* ``cov._gauss.Gauss.get_cand_params`` samples the random pairwise distances
  in one vectorized ``np.random.randint((M, 2))`` call; the vectorized pair
  sampling is checked to be identical to the previous per-iteration loop for a
  fixed seed.
"""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")


def _get_grad_marlik_lu_reference(gp, X, t, params):
    """Reimplementation of get_grad_marlik with the previous LU-based inverse.

    Mirrors the previous implementation (scipy.linalg.inv of A) so the
    production code, which inverts A from its Cholesky factor, can be checked
    against it.
    """
    import scipy.linalg

    from physbo import misc

    ndata = X.shape[0]
    lik_params, prior_params = gp.decomp_params(params)
    fmu = gp.prior.get_mean(ndata, prior_params)
    G = gp.prior.get_cov(X, params=prior_params)
    B = gp.lik.get_cov(ndata, lik_params)
    A = G + B + 1e-8 * np.identity(ndata)
    U = scipy.linalg.cholesky(A, check_finite=False)
    alpha = misc.gauss_elim(U, t - fmu)
    invA = scipy.linalg.inv(A, check_finite=False)

    grad = np.zeros(gp.num_params)
    lik_grad = gp.lik.get_grad(ndata, lik_params)
    grad[0] = -0.5 * lik_grad.dot(alpha).dot(alpha) + 0.5 * misc.traceAB2(invA, lik_grad)
    n = gp.lik.num_params
    if gp.prior.mean.num_params != 0:
        mean_grad = gp.prior.get_grad_mean(ndata, prior_params)
        grad[n : n + gp.prior.mean.num_params] = -np.inner(alpha, mean_grad)
        n += gp.prior.mean.num_params
    cov_grad = gp.prior.get_grad_cov(X, prior_params)
    grad[n:] = -0.5 * cov_grad.dot(alpha).dot(alpha) + 0.5 * misc.traceAB3(invA, cov_grad)
    return grad


@pytest.mark.parametrize("ard", [False, True])
@pytest.mark.parametrize("case", ["generic", "single", "close"])
def test_get_grad_marlik_matches_lu_inverse(ard, case):
    from physbo.gp.inf import exact

    rng = np.random.RandomState(0)
    N, d = {"generic": 20, "single": 1, "close": 6}[case], 2
    X = rng.randn(N, d)
    if case == "close":
        # pairs of nearby points
        X[1::2] = X[0::2] + 1e-4 * rng.randn(N // 2, d)
    t = rng.randn(N)
    gp = physbo.gp.Model(
        lik=physbo.gp.lik.Gauss(),
        mean=physbo.gp.mean.Const(),
        cov=physbo.gp.cov.Gauss(num_dim=d, ard=ard),
    )
    # A generic, non-degenerate parameter vector.
    params = 0.3 * rng.randn(gp.num_params)
    rtol = 1e-10
    if case == "close":
        # small noise: an ill-conditioned covariance (cond ~ 1e6)
        params[0] = np.log(1e-3)
        rtol = 1e-6
    grad = exact.get_grad_marlik(gp, X, t, params.copy())
    grad_lu = _get_grad_marlik_lu_reference(gp, X, t, params.copy())
    assert np.all(np.isfinite(grad))
    np.testing.assert_allclose(grad, grad_lu, rtol=rtol, atol=rtol * np.max(np.abs(grad_lu)))


def _get_cand_params_loop_reference(X, t):
    """Reimplementation of the pre-vectorization non-ARD get_cand_params.

    Mirrors the previous implementation (per-iteration randint loop) so the
    vectorized production code can be checked against it under the same seed.
    """
    num_data = X.shape[0]
    M = max(2000, int(np.floor(num_data / 5)))
    dist = np.zeros(M)
    for m in range(M):
        a = np.random.randint(0, X.shape[0], 2)
        dist[m] = np.linalg.norm(X[a[0], :] - X[a[1], :])
    dist = np.sort(dist)
    tmp = int(np.floor(M / 10))
    n = np.random.randint(0, 5)
    width = dist[(2 * n + 1) * tmp]
    scale = np.std(t)
    return np.append(np.log(width + 1e-8), np.log(scale))


def test_get_cand_params_matches_old_loop_implementation():
    """Production get_cand_params must equal the old loop algorithm per-seed.

    This exercises the real ``cov.get_cand_params`` (not just an in-test
    snippet), covering the full code path including the post-sort
    ``np.random.randint(0, 5)`` draw whose RNG state depends on the pair
    sampling.
    """
    X = np.random.RandomState(1).randn(50, 3)
    t = np.random.RandomState(4).randn(50)
    cov = physbo.gp.cov.Gauss(num_dim=X.shape[1], ard=False)

    state = np.random.get_state()
    try:
        np.random.seed(42)
        ref = _get_cand_params_loop_reference(X, t)
        np.random.seed(42)
        got = cov.get_cand_params(X, t)
    finally:
        np.random.set_state(state)

    np.testing.assert_array_equal(got, ref)


def test_get_cand_params_is_reproducible_for_fixed_seed():
    X = np.random.RandomState(2).randn(40, 2)
    t = np.random.RandomState(3).randn(40)
    cov = physbo.gp.cov.Gauss(num_dim=X.shape[1], ard=False)

    state = np.random.get_state()
    try:
        np.random.seed(7)
        p1 = cov.get_cand_params(X, t)
        np.random.seed(7)
        p2 = cov.get_cand_params(X, t)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(p1, p2)

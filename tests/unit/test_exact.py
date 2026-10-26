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
  in one vectorized ``np.random.randint((M, 2))`` call; it is checked to leave
  the global RNG in exactly the same state as the previous per-iteration loop
  and to give the same result up to rounding (the vectorized norms can differ
  from the per-pair norms in the last bit).
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
@pytest.mark.parametrize("case", ["generic", "single", "generic_trace", "close_trace"])
def test_get_grad_marlik_matches_lu_inverse(ard, case):
    from physbo.gp.inf import exact

    rng = np.random.RandomState(0)
    N = {"generic": 20, "single": 1, "close": 6}[case.replace("_trace", "")]
    d = 2
    X = rng.randn(N, d)
    if case.startswith("close"):
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
    if case.startswith("close"):
        # small noise: an ill-conditioned covariance (cond ~ 1e6)
        params[0] = np.log(1e-3)
        rtol = 1e-6
    if case.endswith("_trace"):
        # targets equal to the constant mean: alpha = 0, so every component
        # is a trace term of the inverse; compared component by component.
        # (With random targets and small noise, the alpha terms would
        # dominate and hide errors of the inverse.)
        t = np.full(N, params[gp.lik.num_params])
    grad = exact.get_grad_marlik(gp, X, t, params.copy())
    grad_lu = _get_grad_marlik_lu_reference(gp, X, t, params.copy())
    assert np.all(np.isfinite(grad))
    if case.endswith("_trace"):
        np.testing.assert_allclose(grad, grad_lu, rtol=rtol, atol=0)
    else:
        np.testing.assert_allclose(
            grad, grad_lu, rtol=rtol, atol=rtol * np.max(np.abs(grad_lu))
        )


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


@pytest.mark.parametrize(
    "seed, num_data, dim",
    [(42, 50, 3), (7, 1, 2), (3, 2, 4), (11, 1000, 200), (5, 10010, 2)],
)
def test_get_cand_params_matches_old_loop_implementation(seed, num_data, dim):
    """Production get_cand_params must match the old loop algorithm per seed.

    The global RNG must end in exactly the same state (the same pairs are
    drawn, and the post-sort ``np.random.randint(0, 5)`` draw is unchanged);
    the result agrees up to rounding, since the vectorized norms can differ
    from the per-pair norms in the last bit.  num_data = 10010 takes the
    M = num_data / 5 > 2000 branch; num_data = 1 gives zero distances and a
    zero standard deviation.
    """
    X = np.random.RandomState(1).randn(num_data, dim)
    t = np.random.RandomState(4).randn(num_data)
    cov = physbo.gp.cov.Gauss(num_dim=dim, ard=False)

    state = np.random.get_state()
    try:
        with np.errstate(divide="ignore"):  # log(std) of a single target
            np.random.seed(seed)
            ref = _get_cand_params_loop_reference(X, t)
            state_ref = np.random.get_state()
            np.random.seed(seed)
            got = cov.get_cand_params(X, t)
            state_got = np.random.get_state()
    finally:
        np.random.set_state(state)

    assert state_got[0] == state_ref[0]
    np.testing.assert_array_equal(state_got[1], state_ref[1])
    assert state_got[2:] == state_ref[2:]
    # the result is in log space: atol covers a width close to 1
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_get_cand_params_honors_global_seed():
    """The same global seed gives the same result, different seeds differ."""
    X = np.random.RandomState(2).randn(40, 2)
    t = np.random.RandomState(3).randn(40)
    cov = physbo.gp.cov.Gauss(num_dim=X.shape[1], ard=False)

    state = np.random.get_state()
    try:
        results = []
        for seed in (7, 7, 8, 9, 10):
            np.random.seed(seed)
            results.append(cov.get_cand_params(X, t))
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(results[0], results[1])
    assert any(not np.array_equal(results[0], r) for r in results[2:])

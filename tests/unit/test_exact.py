# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests locking down the GP speedups in perf/gp-grad-and-init-hyperparam.

* ``exact.get_grad_marlik`` now reuses the Cholesky factor via ``cho_solve``
  instead of ``scipy.linalg.inv``; we check it against a finite-difference
  gradient of ``eval_marlik``.
* ``cov._gauss.Gauss.get_cand_params`` now samples the random pairwise
  distances in one vectorized ``np.random.randint((M, 2))`` call; we check the
  vectorized pair sampling is identical to the previous per-iteration loop for a
  fixed seed.
"""

from __future__ import print_function

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")


def _build_gp(d):
    cov = physbo.gp.cov.Gauss(num_dim=d, ard=False)
    mean = physbo.gp.mean.Const()
    lik = physbo.gp.lik.Gauss()
    return physbo.gp.Model(lik=lik, mean=mean, cov=cov)


def test_get_grad_marlik_matches_finite_difference():
    from physbo.gp.inf import exact

    rng = np.random.RandomState(0)
    N, d = 8, 2
    X = rng.randn(N, d)
    t = rng.randn(N)

    gp = _build_gp(d)
    # A generic, non-degenerate parameter vector.
    params = 0.3 * rng.randn(gp.num_params)

    grad = exact.get_grad_marlik(gp, X, t, params)

    eps = 1e-6
    grad_fd = np.zeros_like(grad)
    for i in range(gp.num_params):
        pp = params.copy()
        pp[i] += eps
        pm = params.copy()
        pm[i] -= eps
        grad_fd[i] = (
            exact.eval_marlik(gp, X, t, pp) - exact.eval_marlik(gp, X, t, pm)
        ) / (2 * eps)

    # Observed central-difference error for these points is ~1e-9 (abs) /
    # ~1e-8 (rel) across seeds, so this tolerance is tight enough to catch a
    # real gradient regression while leaving a comfortable margin.
    np.testing.assert_allclose(grad, grad_fd, rtol=1e-6, atol=1e-7)


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

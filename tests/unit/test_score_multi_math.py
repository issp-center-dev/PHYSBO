# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Numerical tests for the multi-objective acquisition functions EHVI and HVPI."""

import numpy as np
import numpy.testing
import pytest
import scipy.stats

physbo = pytest.importorskip("physbo")

from physbo.search import score_multi
from physbo.search.pareto import Pareto


def ehvi_pairwise(fmean, fstd, pareto):
    """EHVI summed over all pairs of cells (Couckuyt et al., 2014).

    This is the O(N * n_cell**2 * n_obj) formulation that physbo used before
    the factorized one, kept here as a reference.
    """
    n_obj = pareto.num_objectives
    N = fmean.shape[0]
    front = np.r_[
        np.array(pareto.reference_min).reshape((1, n_obj)),
        pareto.front,
        np.array(pareto.reference_max).reshape((1, n_obj)),
    ]
    ax = np.arange(n_obj)
    l = front[pareto.cells.ub, ax] * -1
    u = front[pareto.cells.lb, ax] * -1
    n_cell = pareto.cells.lb.shape[0]
    l = np.tile(l, (n_cell, 1, 1, 1))
    u = np.tile(u, (n_cell, 1, 1, 1))
    a = l.transpose((2, 1, 0, 3))
    b = u.transpose((2, 1, 0, 3))
    fmean = fmean.reshape((1, N, 1, n_obj)) * -1
    fstd = fstd.reshape((1, N, 1, n_obj))
    norm = scipy.stats.norm
    phi_min_bu = norm.pdf((np.minimum(b, u) - fmean) / fstd)
    phi_max_al = norm.pdf((np.maximum(a, l) - fmean) / fstd)
    Phi_l = norm.cdf((l - fmean) / fstd)
    Phi_u = norm.cdf((u - fmean) / fstd)
    Phi_a = norm.cdf((a - fmean) / fstd)
    Phi_b = norm.cdf((b - fmean) / fstd)
    is_type_A = np.logical_and(a < u, l < b)
    is_type_B = u <= a
    EI_A = (
        (b - a) * (np.maximum(Phi_a, Phi_l) - Phi_l)
        + (b - fmean) * (np.minimum(Phi_b, Phi_u) - np.maximum(Phi_a, Phi_l))
        + fstd * (phi_min_bu - phi_max_al)
    )
    EI_B = (b - a) * (Phi_u - Phi_l)
    G = EI_A * is_type_A + EI_B * is_type_B
    return np.sum(np.sum(np.prod(G, axis=3), axis=0), axis=1)


def make_problem(rng, n_obj, n_points, N):
    pareto = Pareto(n_obj)
    pareto.update_front(rng.normal(size=(n_points, n_obj)))
    fmean = rng.normal(size=(N, n_obj))
    fstd = rng.uniform(0.05, 1.0, size=(N, n_obj))
    # candidates on the front and almost deterministic ones
    k = min(3, pareto.front.shape[0])
    fmean[:k] = pareto.front[:k]
    fstd[k : 2 * k] = 1e-8
    return fmean, fstd, pareto


@pytest.mark.parametrize(
    "n_obj, n_points", [(2, 1), (2, 5), (2, 40), (3, 4), (3, 15), (4, 6)]
)
def test_EHVI_matches_pairwise_formulation(n_obj, n_points):
    rng = np.random.default_rng(n_obj * 100 + n_points)
    fmean, fstd, pareto = make_problem(rng, n_obj, n_points, N=50)
    res = score_multi.EHVI(fmean, fstd, pareto)
    ref = ehvi_pairwise(fmean, fstd, pareto)
    numpy.testing.assert_allclose(res, ref, rtol=1e-10, atol=1e-12 * np.max(ref))


def test_EHVI_blocks(monkeypatch):
    rng = np.random.default_rng(1)
    fmean, fstd, pareto = make_problem(rng, 3, 10, N=101)
    ref = score_multi.EHVI(fmean, fstd, pareto)
    # a few candidates per block, with a last block of a different size
    monkeypatch.setattr(score_multi, "_EHVI_BLOCK_ELEMENTS", 7 * pareto.cells.lb.size)
    res = score_multi.EHVI(fmean, fstd, pareto)
    numpy.testing.assert_array_equal(res, ref)


def test_EHVI_against_monte_carlo():
    # 2 objectives (maximized): front (0, 2), (1, 1), (2, 0)
    pareto = Pareto(2)
    pareto.update_front(np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]))
    pareto.set_reference_min([-1.0, -1.0])
    pareto.set_reference_max([3.0, 3.0])
    fmean = np.array([[1.2, 1.3], [0.5, 0.5], [2.5, 0.2]])
    fstd = np.array([[0.4, 0.3], [0.2, 0.2], [0.3, 0.5]])
    res = score_multi.EHVI(fmean, fstd, pareto)

    # hypervolume improvement of y inside the box: the area dominated by y
    # (and above reference_min) minus the part already dominated by the front
    rng = np.random.default_rng(0)
    xs = np.linspace(-1.0, 3.0, 401)
    xc = 0.5 * (xs[1:] + xs[:-1])
    dx = xs[1] - xs[0]
    front_height = np.full(xc.shape, -1.0)
    for f in pareto.front:
        front_height = np.where(xc <= f[0], np.maximum(front_height, f[1]), front_height)
    for i in range(len(fmean)):
        y = fmean[i] + fstd[i] * rng.standard_normal((20000, 2))
        inside = np.all(y <= 3.0, axis=1)
        height = np.clip(y[:, 1:2] - front_height[None, :], 0.0, None)
        hvi = np.sum(height * (xc[None, :] <= y[:, 0:1]), axis=1) * dx
        mc = np.mean(hvi * inside)
        assert res[i] == pytest.approx(mc, rel=0.05, abs=2e-3)


def test_HVPI_matches_scipy_stats():
    rng = np.random.default_rng(2)
    fmean, fstd, pareto = make_problem(rng, 3, 10, N=50)
    res = score_multi.HVPI(fmean, fstd, pareto)

    n_obj = 3
    front = np.r_[
        np.array(pareto.reference_min).reshape((1, n_obj)),
        pareto.front,
        np.full((1, n_obj), np.inf),
    ]
    ax = np.arange(n_obj)
    l = -front[pareto.cells.ub, ax][None]
    u = -front[pareto.cells.lb, ax][None]
    m = -fmean[:, None, :]
    s = fstd[:, None, :]
    norm = scipy.stats.norm
    poi = np.sum(np.prod(norm.cdf((u - m) / s) - norm.cdf((l - m) / s), axis=2), axis=1)
    hv = np.sum(np.prod(u - np.maximum(l, m), axis=2) * np.all(m < u, axis=2), axis=1)
    numpy.testing.assert_array_equal(res, hv * poi)

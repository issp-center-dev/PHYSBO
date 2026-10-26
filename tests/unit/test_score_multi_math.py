# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Numerical tests for the multi-objective acquisition functions EHVI and HVPI."""

import itertools

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
    # fewer target elements than one candidate needs: one candidate per block
    monkeypatch.setattr(score_multi, "_EHVI_BLOCK_ELEMENTS", 1)
    numpy.testing.assert_array_equal(score_multi.EHVI(fmean, fstd, pareto), ref)


def hvi_exact(y, front, reference_min, reference_max):
    """Hypervolume improvement of each row of y (objectives maximized).

    The volume of the box [reference_min, y] not dominated by the front,
    by inclusion-exclusion over the front points; as in physbo, points
    beyond reference_max do not contribute.  It does not use the cells of
    the non-dominated region.
    """
    y = np.atleast_2d(y)
    hvi = np.zeros(y.shape[0])
    for k in range(front.shape[0] + 1):
        for subset in itertools.combinations(range(front.shape[0]), k):
            top = np.minimum(y, np.min(front[list(subset)], axis=0, initial=np.inf))
            hvi += (-1) ** k * np.prod(np.clip(top - reference_min, 0.0, None), axis=1)
    hvi[np.any(y > reference_max, axis=1)] = 0.0
    return hvi


EHVI_ORACLE_CASES = {
    # 2 objectives: front (0, 2), (1, 1), (2, 0) in the box [-1, 3]^2
    "2d": (
        [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]],
        [-1.0, -1.0],
        [3.0, 3.0],
        [[1.2, 1.3], [0.8, 0.9], [2.5, 0.2], [2.9, 2.8]],
        [[0.4, 0.3], [0.3, 0.3], [0.3, 0.5], [0.2, 0.1]],
    ),
    # 3 objectives: front (0, 1, 2), (1, 2, 0), (2, 0, 1) in the box [-1, 3]^3
    "3d": (
        [[0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [2.0, 0.0, 1.0]],
        [-1.0, -1.0, -1.0],
        [3.0, 3.0, 3.0],
        [[1.0, 1.0, 1.0], [1.5, 1.5, 0.5], [0.5, 0.5, 0.5], [2.8, 0.5, 2.5]],
        [[0.4, 0.3, 0.5], [0.3, 0.3, 0.3], [0.2, 0.2, 0.2], [0.2, 0.4, 0.3]],
    ),
}


def make_oracle_problem(name):
    front, rmin, rmax, fmean, fstd = (np.array(a, dtype=float) for a in EHVI_ORACLE_CASES[name])
    pareto = Pareto(front.shape[1])
    pareto.update_front(front)
    pareto.set_reference_min(rmin)
    pareto.set_reference_max(rmax)
    return pareto, rmin, rmax, fmean, fstd


@pytest.mark.parametrize("name", sorted(EHVI_ORACLE_CASES))
def test_EHVI_against_monte_carlo(name):
    pareto, rmin, rmax, fmean, fstd = make_oracle_problem(name)
    res = score_multi.EHVI(fmean, fstd, pareto)
    rng = np.random.default_rng(0)
    n = 200000
    for i in range(fmean.shape[0]):
        y = fmean[i] + fstd[i] * rng.standard_normal((n, fmean.shape[1]))
        hvi = hvi_exact(y, pareto.front, rmin, rmax)
        stderr = np.std(hvi) / np.sqrt(n)
        assert np.mean(hvi) > 10 * stderr  # the case is not trivially zero
        assert abs(res[i] - np.mean(hvi)) < 4 * stderr


@pytest.mark.parametrize("name", sorted(EHVI_ORACLE_CASES))
def test_EHVI_deterministic_limit(name):
    # for fstd -> 0 the EHVI tends to the hypervolume improvement of fmean
    pareto, rmin, rmax, fmean, _ = make_oracle_problem(name)
    fmean = np.r_[fmean, pareto.front[:1] - 0.1]  # add a dominated candidate
    res = score_multi.EHVI(fmean, np.full(fmean.shape, 1e-9), pareto)
    numpy.testing.assert_allclose(
        res, hvi_exact(fmean, pareto.front, rmin, rmax), rtol=1e-6, atol=1e-12
    )


def test_EHVI_objective_permutation():
    rng = np.random.default_rng(4)
    fmean, fstd, pareto = make_problem(rng, 3, 12, N=30)
    res = score_multi.EHVI(fmean, fstd, pareto)
    perm = [2, 0, 1]
    permuted = Pareto(3)
    permuted.update_front(pareto.front[:, perm])
    permuted.set_reference_min(np.ravel(pareto.reference_min)[perm])
    permuted.set_reference_max(np.ravel(pareto.reference_max)[perm])
    numpy.testing.assert_allclose(
        score_multi.EHVI(fmean[:, perm], fstd[:, perm], permuted), res, rtol=1e-12
    )


def test_HVPI_matches_scipy_stats():
    rng = np.random.default_rng(2)
    fmean, fstd, pareto = make_problem(rng, 3, 10, N=50)
    res = score_multi.HVPI(fmean, fstd, pareto)

    # the HVPI expression is unchanged; this pins that scipy.special.ndtr
    # gives exactly the values of scipy.stats.norm.cdf used before
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


def test_EHVI_accepts_flattened_inputs():
    rng = np.random.default_rng(3)
    fmean, fstd, pareto = make_problem(rng, 2, 6, N=20)
    res = score_multi.EHVI(fmean, fstd, pareto)  # sets the reference points
    ref = ehvi_pairwise(fmean, fstd, pareto)
    numpy.testing.assert_allclose(res, ref, rtol=1e-10)
    numpy.testing.assert_allclose(
        score_multi.EHVI(fmean, fstd.ravel(), pareto), ref, rtol=1e-10
    )
    numpy.testing.assert_allclose(
        score_multi.EHVI(fmean[:, None, :], fstd[None], pareto), ref, rtol=1e-10
    )
    assert score_multi.EHVI(fmean[:0], fstd[:0], pareto).shape == (0,)
    hvpi = score_multi.HVPI(fmean, fstd, pareto)
    numpy.testing.assert_array_equal(score_multi.HVPI(fmean, fstd.ravel(), pareto), hvpi)
    assert score_multi.HVPI(fmean[:0], fstd[:0], pareto).shape == (0,)


def test_EHVI_one_objective():
    pareto = Pareto(1)
    pareto.update_front(np.array([[0.5], [1.0]]))
    pareto.set_reference_min([-1.0])
    pareto.set_reference_max([3.0])
    fmean = np.array([0.0, 1.0, 2.0, 2.9])
    fstd = np.array([0.5, 0.3, 0.2, 0.4])
    res = score_multi.EHVI(fmean, fstd, pareto)
    numpy.testing.assert_allclose(
        res, ehvi_pairwise(fmean[:, None], fstd[:, None], pareto), rtol=1e-12
    )
    # E[(Y - 1)^+ ; Y <= 3] for Y ~ N(m, s^2), by quadrature: as in the
    # pairwise formulation, points beyond reference_max do not contribute
    y = np.linspace(1.0, 3.0, 200001)
    for m, s, r in zip(fmean, fstd, res):
        w = scipy.stats.norm.pdf(y, m, s)
        f = (y - 1.0) * w
        # trapezoidal rule (np.trapezoid is not available in NumPy 1.x)
        integral = np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(y))
        assert r == pytest.approx(integral, rel=1e-8)


def test_EHVI_near_reference_points():
    # candidates around reference_min/reference_max in each objective
    pareto = Pareto(3)
    pareto.update_front(np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [2.0, 0.0, 1.0]]))
    pareto.set_reference_min([-1.0, -1.0, -1.0])
    pareto.set_reference_max([3.0, 3.0, 3.0])
    fmean = np.array(
        [
            [3.0, 1.0, 1.0],
            [3.5, 2.5, 1.0],
            [-1.0, 3.0, 2.9],
            [2.9, 2.9, 2.9],
            [-1.5, -1.5, -1.5],
            [1.0, 2.0, 0.0],
        ]
    )
    for scale in (1e-3, 0.3, 2.0):
        fstd = np.full(fmean.shape, scale)
        numpy.testing.assert_allclose(
            score_multi.EHVI(fmean, fstd, pareto),
            ehvi_pairwise(fmean, fstd, pareto),
            rtol=1e-10,
            atol=1e-14,
        )


def test_normal_cdf_pdf_promote_like_scipy_stats():
    from physbo.search.score import _norm_cdf, _norm_pdf

    x = np.r_[np.linspace(-40.0, 40.0, 1001), np.nan, np.inf, -np.inf]
    numpy.testing.assert_array_equal(_norm_cdf(x), scipy.stats.norm.cdf(x))
    numpy.testing.assert_array_equal(_norm_pdf(x), scipy.stats.norm.pdf(x))
    x32 = x.astype(np.float32)
    assert _norm_cdf(x32).dtype == scipy.stats.norm.cdf(x32).dtype
    assert _norm_pdf(x32).dtype == scipy.stats.norm.pdf(x32).dtype
    numpy.testing.assert_array_equal(_norm_cdf(x32), scipy.stats.norm.cdf(x32))
    numpy.testing.assert_array_equal(_norm_pdf(x32), scipy.stats.norm.pdf(x32))

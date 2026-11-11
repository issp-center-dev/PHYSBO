# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The box decomposition of the non-dominated region (Lacour et al. 2017)
covers the same region as the binary search, with fewer cells."""

import copy

import numpy as np
import pytest

from physbo.search import pareto as pareto_module
from physbo.search import score_multi
from physbo.search.pareto import Pareto


def make_front(rng, kind, n, m):
    if kind == "sphere":
        a = np.abs(rng.standard_normal((n, m)))
        return a / np.linalg.norm(a, axis=1, keepdims=True)
    if kind == "random":
        return rng.random((n, m))
    return rng.integers(0, 5, (n, m)).astype(float)  # many ties


CASES = [
    (m, kind, n)
    for m in (2, 3, 4, 5)
    for kind in ("sphere", "random", "grid")
    for n in (1, 6, 25)
]


@pytest.mark.parametrize("m,kind,n", CASES)
def test_same_region_as_binary_search(m, kind, n):
    rng = np.random.default_rng(1000 * m + 10 * n + len(kind))
    p = Pareto(m)
    p.update_front(make_front(rng, kind, n, m))
    # a tight box keeps the cell sums free of cancellation
    lo = p.front.min(axis=0) - 0.1
    hi = p.front.max(axis=0) + 0.1
    p.set_reference_min(lo)
    p.set_reference_max(hi)
    q = copy.deepcopy(p)
    p.divide_non_dominated_region(method="box")
    q.divide_non_dominated_region(method="binary_search")

    padded = np.r_[[lo], p.front, [hi]]
    ax = np.arange(m)
    assert np.all(padded[p.cells.lb, ax] < padded[p.cells.ub, ax])
    if m >= 3 and p.front.shape[0] >= 6:
        assert p.cells.lb.shape[0] < q.cells.lb.shape[0]

    np.testing.assert_allclose(
        p.volume_in_dominance(lo, hi), q.volume_in_dominance(lo, hi), rtol=1e-12
    )
    fmean = rng.random((40, m)) * 1.4 - 0.2
    fstd = rng.random((40, m)) * 0.3 + 0.02
    np.testing.assert_allclose(
        score_multi.EHVI(fmean, fstd, p), score_multi.EHVI(fmean, fstd, q), rtol=1e-10
    )
    np.testing.assert_allclose(
        score_multi.HVPI(fmean, fstd, p), score_multi.HVPI(fmean, fstd, q), rtol=1e-10
    )


def test_cells_do_not_depend_on_reference_points():
    rng = np.random.default_rng(7)
    p = Pareto(4)
    p.update_front(make_front(rng, "sphere", 30, 4))
    p.divide_non_dominated_region()
    lb, ub = p.cells.lb.copy(), p.cells.ub.copy()
    p.set_reference_min(np.full(4, -5.0))
    p.divide_non_dominated_region()
    assert np.array_equal(p.cells.lb, lb) and np.array_equal(p.cells.ub, ub)


def test_large_magnitude_front():
    # front.min - 1 rounds back to front.min beyond about 2**53, which would put
    # the lowest front coordinates on the row of reference_min
    rng = np.random.default_rng(0)
    a = np.abs(rng.standard_normal((10, 3)))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    scale = 1e17
    p = Pareto(3)
    p.update_front(3 * scale + scale * a)
    p.set_reference_min(p.front.min(axis=0) - 0.1 * scale)
    p.set_reference_max(p.front.max(axis=0) + 0.1 * scale)
    q = copy.deepcopy(p)
    p.divide_non_dominated_region(method="box")
    q.divide_non_dominated_region(method="binary_search")
    fmean = 3 * scale + scale * rng.random((20, 3))
    fstd = 0.1 * scale * np.ones((20, 3))
    np.testing.assert_allclose(
        score_multi.EHVI(fmean, fstd, p), score_multi.EHVI(fmean, fstd, q), rtol=1e-10
    )


def test_defaults_and_options():
    rng = np.random.default_rng(3)
    p2 = Pareto(2)
    p2.update_front(rng.random((20, 2)))
    p2.divide_non_dominated_region()
    assert p2.cells.lb.shape[0] == p2.front.shape[0] + 1  # strips
    with pytest.raises(ValueError):
        p2.divide_non_dominated_region(method="foo")
    # force_binary_search keeps selecting the binary search
    p3 = Pareto(3)
    p3.update_front(make_front(rng, "sphere", 30, 3))
    q3 = copy.deepcopy(p3)
    p3.divide_non_dominated_region(force_binary_search=True)
    q3.divide_non_dominated_region(method="binary_search")
    assert np.array_equal(p3.cells.lb, q3.cells.lb)
    # an empty front gives the whole box
    p0 = Pareto(3)
    p0.divide_non_dominated_region(method="box")
    assert p0.cells.lb.tolist() == [[0, 0, 0]] and p0.cells.ub.tolist() == [[1, 1, 1]]


def _local_upper_bounds_rebuilding(points, ref, fill_value):
    """Algorithm 1 of Lacour et al. with the arrays rebuilt for every point:
    the reference for the rows of Pareto._local_upper_bounds and their order."""
    M = len(ref)
    Z0 = np.repeat(np.asarray(fill_value, dtype=float)[:, None], M, axis=1)
    Z0[np.arange(M), np.arange(M)] = ref
    U = np.asarray(ref, dtype=float)[None, :].copy()
    Z = Z0[None, :, :]
    off = ~np.eye(M, dtype=bool)
    for z in points:
        hit = np.all(U > z, axis=1)
        if not hit.any():
            continue
        Uh, Zh = U[hit], Z[hit]
        newU, newZ = [U[~hit]], [Z[~hit]]
        for j in range(M):
            zmax = np.max(Zh[:, j, off[j]], axis=1, initial=-np.inf)
            ok = z[j] >= zmax
            if ok.any():
                u2 = Uh[ok].copy()
                u2[:, j] = z[j]
                z2 = Zh[ok].copy()
                z2[:, :, j] = z
                newU.append(u2)
                newZ.append(z2)
        U = np.concatenate(newU)
        Z = np.concatenate(newZ)
    return U, Z


@pytest.mark.parametrize("rows", [1, 1024])
@pytest.mark.parametrize(
    "m,kind,n",
    [(2, "random", 40), (3, "sphere", 80), (4, "sphere", 60), (4, "grid", 300),
     (5, "random", 200)],
)
def test_local_upper_bounds_rows_and_order(monkeypatch, m, kind, n, rows):
    # Both passes of the box decomposition give the rows of the reference in
    # the same order, so that the cells and the order of the sums over them
    # do not depend on how the rows are stored.  `rows` is the initial size
    # of the arrays: starting from 1, they grow and are compacted from the
    # first points on, whatever the size of the front.
    monkeypatch.setattr(pareto_module, "_LOCAL_UPPER_BOUNDS_ROWS", rows)
    rng = np.random.default_rng(100 * m + n)
    p = Pareto(m)
    p.update_front(make_front(rng, kind, n, m))
    rmin = np.min(p.front, axis=0) - 1.0
    first = (-p.front, -rmin, np.full(m, -np.inf))
    U1, Z1 = Pareto._local_upper_bounds(*first)
    U1_ref, Z1_ref = _local_upper_bounds_rebuilding(*first)
    assert np.array_equal(U1, U1_ref) and np.array_equal(Z1, Z1_ref)
    second = (-U1, np.full(m, np.inf), rmin)
    U2, Z2 = Pareto._local_upper_bounds(*second)
    U2_ref, Z2_ref = _local_upper_bounds_rebuilding(*second)
    assert np.array_equal(U2, U2_ref) and np.array_equal(Z2, Z2_ref)


def _coverage_error(p):
    """On the grid cut by the coordinates of the front, the cells must cover
    every cell that is not dominated exactly once and no dominated one.
    This does not use the binary search."""
    m = p.num_objectives
    padded = np.r_[np.full((1, m), -np.inf), p.front, np.full((1, m), np.inf)]
    ax = np.arange(m)
    lo, hi = padded[p.cells.lb, ax], padded[p.cells.ub, ax]
    if not np.all(lo < hi):
        return "improper cell"
    ticks = [np.unique(padded[:, d]) for d in range(m)]
    cover = np.zeros([len(t) - 1 for t in ticks], dtype=int)
    for l, h in zip(lo, hi):
        cover[
            tuple(
                slice(np.searchsorted(ticks[d], l[d]), np.searchsorted(ticks[d], h[d]))
                for d in range(m)
            )
        ] += 1
    dominated = np.zeros(cover.shape, dtype=bool)
    for y in p.front:
        dominated[
            tuple(slice(0, np.searchsorted(ticks[d], y[d])) for d in range(m))
        ] = True
    if np.any(cover[dominated] != 0):
        return "a dominated cell is covered"
    if np.any(cover[~dominated] != 1):
        return "a cell is not covered exactly once"
    return None


@pytest.mark.parametrize("m,n", [(6, 6), (7, 5)])
@pytest.mark.parametrize("last", ["any", "grid", 1, 2, 3])
def test_slabs_tile_the_region_exactly(m, n, last):
    # 6 or more objectives: slabs along the last objective.  `last` is the
    # number of distinct values of the last objective (1: no cut; 2: nothing
    # of the front lies in the upper slab), "grid" has ties in all objectives.
    rng = np.random.default_rng(10 * m + n)
    for _ in range(5):
        front = make_front(rng, "grid" if last == "grid" else "sphere", n, m)
        if isinstance(last, int):
            front[:, -1] = rng.permutation(np.arange(n) % last)
        p = Pareto(m)
        p.update_front(front)
        p.divide_non_dominated_region(method="box")
        assert _coverage_error(p) is None


def _box(front):
    p = Pareto(front.shape[1])
    p.update_front(front)
    p.divide_non_dominated_region(method="box")
    return p


def test_slabs_from_six_objectives(monkeypatch):
    rng = np.random.default_rng(5)
    fronts = {m: make_front(rng, "sphere", 25, m) for m in (4, 5, 6)}
    with_slabs = {m: _box(f) for m, f in fronts.items()}
    monkeypatch.setattr(pareto_module, "_BOX_SLABS", 1)
    for m, p in with_slabs.items():
        q = _box(fronts[m])
        same = np.array_equal(p.cells.lb, q.cells.lb) and np.array_equal(
            p.cells.ub, q.cells.ub
        )
        assert same == (m < 6)
        lo, hi = p.front.min(axis=0) - 0.1, p.front.max(axis=0) + 0.1
        np.testing.assert_allclose(
            p.volume_in_dominance(lo, hi), q.volume_in_dominance(lo, hi), rtol=1e-12
        )
        for r in (p, q):
            r.set_reference_min(lo)
            r.set_reference_max(hi)
        fmean = rng.random((20, m)) * 1.4 - 0.2
        fstd = rng.random((20, m)) * 0.3 + 0.02
        for score in (score_multi.EHVI, score_multi.HVPI):
            np.testing.assert_allclose(
                score(fmean, fstd, p), score(fmean, fstd, q), rtol=1e-10
            )


def _cells_and_bounds(monkeypatch, front, slabs):
    """Number of cells, and of local upper bounds of the largest pass (the
    time of a pass grows faster than its bounds), with this number of slabs."""
    bounds = []
    original = Pareto._local_upper_bounds

    def counting(points, ref, fill_value):
        U, Z = original(points, ref, fill_value)
        bounds.append(U.shape[0])
        return U, Z

    with monkeypatch.context() as mp:
        mp.setattr(pareto_module, "_BOX_SLABS", slabs)
        mp.setattr(Pareto, "_local_upper_bounds", staticmethod(counting))
        n_cells = _box(front).cells.lb.shape[0]
    return n_cells, max(bounds)


def test_slabs_save_cells_and_bounds(monkeypatch):
    # what the slabs are for, on a front without structure
    front = make_front(np.random.default_rng(6), "sphere", 30, 6)
    cells, bounds = _cells_and_bounds(monkeypatch, front, 3)
    cells_1, bounds_1 = _cells_and_bounds(monkeypatch, front, 1)
    assert cells < cells_1
    # without lowering the points above a slab to its upper edge, the lowest
    # slab would need as many bounds as the whole front
    assert bounds < 0.8 * bounds_1


def test_slabs_with_ties_in_the_last_objective(monkeypatch):
    # Most points share the value of the last objective.  The slabs are cut
    # at quantiles of the points; at quantiles of the distinct values, the
    # tied points would be decomposed in every slab and the cells would be
    # about 1.5 times as many as without slabs.
    rng = np.random.default_rng(10)
    front = make_front(rng, "sphere", 40, 6)
    front[rng.random(40) < 0.8, -1] = 0.5
    cells, _ = _cells_and_bounds(monkeypatch, front, 3)
    cells_1, _ = _cells_and_bounds(monkeypatch, front, 1)
    assert cells < 1.1 * cells_1

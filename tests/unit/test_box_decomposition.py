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

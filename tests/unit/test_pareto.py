# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Numerical tests for the Pareto front and hypervolume calculations.

All objectives are maximized. Reference volumes are computed by hand:
``volume_in_dominance(ref_min, ref_max)`` is the volume of the part of the
box [ref_min, ref_max] dominated by the Pareto front.
"""

import numpy as np
import numpy.testing
import pytest

physbo = pytest.importorskip("physbo")

from physbo.search.pareto import Pareto, dominate
from physbo.search.unify.nds import nds_impl, nds_impl_naive


def test_dominate():
    assert dominate([2.0, 2.0], [1.0, 1.0])
    assert dominate([2.0, 1.0], [1.0, 1.0])
    assert not dominate([1.0, 1.0], [1.0, 1.0])  # equal points do not dominate
    assert not dominate([2.0, 0.0], [1.0, 1.0])  # incomparable
    assert not dominate([1.0, 1.0], [2.0, 2.0])


def test_update_front():
    pareto = Pareto(num_objectives=2)
    pareto.update_front(np.array([[1.0, 3.0], [3.0, 1.0]]))
    assert pareto.front_updated

    # front is sorted in ascending order of the first objective
    front, front_num = pareto.export_front()
    numpy.testing.assert_array_equal(front, [[1.0, 3.0], [3.0, 1.0]])
    numpy.testing.assert_array_equal(front_num, [0, 1])

    # a dominated point does not change the front
    pareto.update_front(np.array([[0.5, 0.5]]))
    assert not pareto.front_updated
    front, front_num = pareto.export_front()
    numpy.testing.assert_array_equal(front, [[1.0, 3.0], [3.0, 1.0]])

    # an incomparable point joins the front
    pareto.update_front(np.array([[2.0, 2.0]]))
    assert pareto.front_updated
    front, front_num = pareto.export_front()
    numpy.testing.assert_array_equal(
        front, [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
    )
    numpy.testing.assert_array_equal(front_num, [0, 3, 1])

    # a dominating point removes dominated members from the front
    pareto.update_front(np.array([[3.0, 3.0]]))
    assert pareto.front_updated
    front, front_num = pareto.export_front()
    numpy.testing.assert_array_equal(front, [[3.0, 3.0]])
    numpy.testing.assert_array_equal(front_num, [4])


def test_update_front_equal_point():
    # current behavior: a point identical to a front member is kept as a
    # duplicate entry (equal points do not dominate each other)
    pareto = Pareto(num_objectives=2)
    pareto.update_front(np.array([[1.0, 2.0]]))
    pareto.update_front(np.array([[1.0, 2.0]]))
    front, front_num = pareto.export_front()
    numpy.testing.assert_array_equal(front, [[1.0, 2.0], [1.0, 2.0]])
    numpy.testing.assert_array_equal(sorted(front_num), [0, 1])
    # duplicates must not break the volume calculation
    v = pareto.volume_in_dominance([0.0, 0.0], [2.0, 3.0])
    assert v == pytest.approx(2.0)


@pytest.mark.parametrize("force_binary_search", [False, True])
def test_volume_in_dominance_2d(force_binary_search):
    pareto = Pareto(num_objectives=2)
    pareto.update_front(
        np.array([[1.0, 3.0], [3.0, 1.0], [2.0, 2.0], [0.5, 0.5]])
    )
    pareto.divide_non_dominated_region(force_binary_search=force_binary_search)

    # dominated area in [0,4]^2:
    #   x in [0,1]: height 3, x in [1,2]: height 2, x in [2,3]: height 1
    v = pareto.volume_in_dominance([0.0, 0.0], [4.0, 4.0])
    assert v == pytest.approx(6.0)

    ratio = pareto.volume_in_dominance([0.0, 0.0], [4.0, 4.0], dominance_ratio=True)
    assert ratio == pytest.approx(6.0 / 16.0)


def test_volume_in_dominance_3d_single_point():
    pareto = Pareto(num_objectives=3)
    pareto.update_front(np.array([[1.0, 1.0, 1.0]]))
    # dominated volume in [0,2]^3 is the unit cube [0,1]^3
    v = pareto.volume_in_dominance([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    assert v == pytest.approx(1.0)


def test_volume_in_dominance_3d_two_points():
    pareto = Pareto(num_objectives=3)
    pareto.update_front(np.array([[2.0, 1.0, 1.0], [1.0, 1.0, 2.0]]))
    # union of [0,(2,1,1)] (volume 2) and [0,(1,1,2)] (volume 2),
    # intersection [0,(1,1,1)] (volume 1) -> union volume 3
    v = pareto.volume_in_dominance([0.0, 0.0, 0.0], [3.0, 3.0, 3.0])
    assert v == pytest.approx(3.0)


def test_volume_random_2d_binary_search_consistency():
    # the generic binary-search division must agree with the 2D algorithm
    rng = np.random.RandomState(12345)
    t = rng.rand(30, 2)

    pareto1 = Pareto(num_objectives=2)
    pareto1.update_front(t)
    pareto1.divide_non_dominated_region(force_binary_search=False)
    v1 = pareto1.volume_in_dominance([0.0, 0.0], [1.0, 1.0])

    pareto2 = Pareto(num_objectives=2)
    pareto2.update_front(t)
    pareto2.divide_non_dominated_region(force_binary_search=True)
    v2 = pareto2.volume_in_dominance([0.0, 0.0], [1.0, 1.0])

    assert v1 == pytest.approx(v2)

    # Monte-Carlo estimate of the dominated volume as an independent check
    p = rng.rand(200000, 2)
    front, _ = pareto1.export_front()
    dominated = np.zeros(len(p), dtype=bool)
    for f in front:
        dominated |= np.all(p <= f, axis=1)
    assert v1 == pytest.approx(dominated.mean(), abs=0.01)


def front_brute_force(t, num_compared=0):
    """Non-dominated rows of t, sorted stably on the first objective."""
    keep = [
        i
        for i in range(len(t))
        if not any(dominate(t[j], t[i]) for j in range(len(t)) if j != i)
    ]
    keep = np.array(keep, dtype=int)
    order = np.argsort(t[keep, 0], kind="stable")
    return t[keep[order]], num_compared + keep[order]


@pytest.mark.parametrize("seed", range(5))
def test_update_front_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    num_objectives = 2 + seed % 3
    pareto = Pareto(num_objectives)
    seen = np.zeros((0, num_objectives))
    for n in (1, 7, 0, 30, 12):
        # small integers: many ties and duplicated points
        t = rng.integers(0, 4, size=(n, num_objectives)).astype(float)
        before = pareto.front_num.copy()
        pareto.update_front(t)
        seen = np.r_[seen, t]
        front, front_num = front_brute_force(seen)
        numpy.testing.assert_array_equal(pareto.front, front)
        numpy.testing.assert_array_equal(pareto.front_num, front_num)
        assert pareto.num_compared == len(seen)
        assert pareto.front_updated == (
            not np.array_equal(np.sort(before), np.sort(front_num))
        )


@pytest.mark.parametrize("num_objectives", [3, 4, 5, 8])
def test_update_front_matches_brute_force_large_front(num_objectives):
    # points near a simplex: large fronts with many ties on the first
    # objective, so that the tie order is checked beyond tiny fronts
    rng = np.random.default_rng(num_objectives)
    pareto = Pareto(num_objectives)
    seen = np.zeros((0, num_objectives))
    sizes = []
    for n in (5, 40, 0, 60, 25):
        t = rng.integers(0, 5, size=(n, num_objectives)).astype(float)
        t[:, -1] = 4 * (num_objectives - 1) - t[:, :-1].sum(axis=1) - rng.integers(0, 2, n)
        before = pareto.front_num.copy()
        pareto.update_front(t)
        seen = np.r_[seen, t]
        front, front_num = front_brute_force(seen)
        numpy.testing.assert_array_equal(pareto.front, front)
        numpy.testing.assert_array_equal(pareto.front_num, front_num)
        assert pareto.front_updated == (
            not np.array_equal(np.sort(before), np.sort(front_num))
        )
        sizes.append(len(front_num))
    assert max(sizes) > 30  # 66-130 points with this seed


def test_update_front_antichain_order():
    # one batch of mutually incomparable points in shuffled order: all of
    # them stay, sorted stably on the first objective
    # (x, y, -y) with distinct y are incomparable; x has ties
    rng = np.random.default_rng(0)
    x = rng.permutation(np.repeat(np.arange(150.0), 2))
    y = rng.permutation(len(x)).astype(float)
    t = np.c_[x, y, -y]
    pareto = Pareto(3)
    pareto.update_front(t)
    numpy.testing.assert_array_equal(pareto.front_num, np.argsort(x, kind="stable"))


def test_update_front_signed_zero_lexicographic():
    # -0.0 and 0.0 are equal in the first objective, so only the point with
    # the larger second objective survives, whatever the row order
    for t in ([[0.0, 1.0], [-0.0, 2.0]], [[-0.0, 1.0], [0.0, 2.0]]):
        for rows in (t, t[::-1]):
            pareto = Pareto(2)
            pareto.update_front(np.array(rows))
            numpy.testing.assert_array_equal(pareto.front[:, 1], [2.0])


def test_update_front_one_objective_many_ties():
    pareto = Pareto(1)
    pareto.update_front(np.r_[np.full((10, 1), 2.0), [[1.0]], np.full((10, 1), 2.0)])
    pareto.update_front(np.r_[[[0.0]], np.full((10, 1), 2.0)])
    numpy.testing.assert_array_equal(
        pareto.front_num, np.r_[np.arange(10), np.arange(11, 21), np.arange(22, 32)]
    )


def test_update_front_empty_on_fresh_front():
    for t in (np.zeros((0, 2)), np.full((3, 2), np.nan)):
        pareto = Pareto(2)
        pareto.update_front(t)
        assert not pareto.front_updated
        assert pareto.cells is None
        assert pareto.front.shape == (0, 2)
        assert pareto.num_compared == t.shape[0]


def test_rebuild_front():
    t = np.array([[1.0, 3.0], [3.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.5, 0.5]])
    # an object with a front, a consumed failed row, cells and reference points
    pareto = Pareto(2)
    pareto.update_front(np.array([[9.0, 9.0], [np.nan, 0.0]]))
    pareto.divide_non_dominated_region()
    pareto.set_reference_min([np.nan, 0.0])
    pareto.set_reference_max([10.0, 10.0])
    pareto.rebuild_front(t, [2, 4, 5])
    numpy.testing.assert_array_equal(pareto.front_num, [0, 2, 1])
    assert pareto.num_compared == 5
    assert pareto.cells is None
    assert not pareto.front_updated  # the last run added nothing
    # non-finite reference points are cleared, finite ones are kept
    assert pareto.reference_min is None
    numpy.testing.assert_array_equal(pareto.reference_max, [10.0, 10.0])
    fresh = Pareto(2)
    for start, stop in ((0, 2), (2, 4), (4, 5)):
        fresh.update_front(t[start:stop])
    numpy.testing.assert_array_equal(pareto.front, fresh.front)
    numpy.testing.assert_array_equal(pareto.front_num, fresh.front_num)


def test_rebuild_front_without_observations():
    # nothing is replayed, so only rebuild_front itself resets the state
    pareto = Pareto(2)
    pareto.update_front(np.array([[1.0, 3.0], [3.0, 1.0]]))
    pareto.divide_non_dominated_region()
    pareto.set_reference_min([0.0, 0.0])
    pareto.set_reference_max([np.inf, 10.0])
    pareto.rebuild_front(np.zeros((0, 2)), [])
    assert pareto.cells is None
    assert pareto.front.shape == (0, 2) and pareto.front_num.size == 0
    assert pareto.num_compared == 0
    assert not pareto.front_updated
    numpy.testing.assert_array_equal(pareto.reference_min, [0.0, 0.0])
    assert pareto.reference_max is None


def test_update_front_without_new_front_points():
    pareto = Pareto(2)
    pareto.update_front(np.array([[1.0, 3.0], [3.0, 1.0]]))
    pareto.divide_non_dominated_region()
    cells = pareto.cells
    front = pareto.front.copy()
    # an empty batch, failed observations only, and a dominated point
    for t in (np.zeros((0, 2)), np.array([[np.nan, 1.0], [np.inf, -np.inf]]), [0.5, 0.5]):
        num_compared = pareto.num_compared
        pareto.update_front(t)
        assert not pareto.front_updated
        assert pareto.cells is cells
        numpy.testing.assert_array_equal(pareto.front, front)
        numpy.testing.assert_array_equal(pareto.front_num, [0, 1])
        assert pareto.num_compared == num_compared + np.atleast_2d(t).shape[0]
    # a new point joining the front clears the cells
    pareto.update_front([2.0, 2.0])
    assert pareto.front_updated
    assert pareto.cells is None
    numpy.testing.assert_array_equal(pareto.front_num, [0, 5, 1])


def test_update_front_one_objective():
    pareto = Pareto(1)
    pareto.update_front(np.array([[1.0], [3.0], [2.0], [3.0]]))
    numpy.testing.assert_array_equal(pareto.front, [[3.0], [3.0]])
    numpy.testing.assert_array_equal(pareto.front_num, [1, 3])
    pareto.update_front(np.array([[4.0]]))
    numpy.testing.assert_array_equal(pareto.front_num, [4])


def test_update_front_signed_zero_ties():
    pareto = Pareto(2)
    pareto.update_front(np.array([[0.0, 1.0], [-0.0, 1.0], [1.0, 0.0]]))
    # equal points do not dominate each other; ties keep the row order
    numpy.testing.assert_array_equal(pareto.front_num, [0, 1, 2])
    assert not np.signbit(pareto.front[0, 0]) and np.signbit(pareto.front[1, 0])


def test_update_front_large_values():
    pareto = Pareto(2)
    pareto.update_front(np.array([[-1e308, 0.0]]))
    with np.errstate(over="raise"):
        pareto.update_front(np.array([[1e308, 0.0]]))
    assert pareto.front_updated
    numpy.testing.assert_array_equal(pareto.front, [[1e308, 0.0]])
    numpy.testing.assert_array_equal(pareto.front_num, [1])
    assert pareto.num_compared == 2


def test_update_front_signed_zero_ties_across_updates():
    pareto = Pareto(2)
    pareto.update_front(np.array([[-0.0, 1.0], [1.0, 0.0]]))
    pareto.update_front(np.array([[0.0, 1.0]]))
    # the equal point joins; on ties the current front comes first
    assert pareto.front_updated
    numpy.testing.assert_array_equal(pareto.front_num, [0, 2, 1])
    assert np.signbit(pareto.front[0, 0]) and not np.signbit(pareto.front[1, 0])


def test_cross_dominance_blocks(monkeypatch):
    import physbo.search.pareto as pareto_module

    rng = np.random.default_rng(7)
    front_t = rng.integers(0, 5, size=(40, 3)).astype(float)
    new_t = rng.integers(0, 5, size=(23, 3)).astype(float)
    ref = pareto_module._cross_dominance(front_t, new_t)
    # blocks of 2 new points (last one partial), and a budget below front.size
    for budget in (2 * front_t.size, 1):
        monkeypatch.setattr(pareto_module, "_CROSS_DOMINANCE_BLOCK_ELEMENTS", budget)
        res = pareto_module._cross_dominance(front_t, new_t)
        numpy.testing.assert_array_equal(res[0], ref[0])
        numpy.testing.assert_array_equal(res[1], ref[1])
    keep_front = [not any(dominate(n, f) for n in new_t) for f in front_t]
    keep_new = [not any(dominate(f, n) for f in front_t) for n in new_t]
    numpy.testing.assert_array_equal(ref[0], keep_front)
    numpy.testing.assert_array_equal(ref[1], keep_new)


def test_nds_ranking():
    # A=(2,2) and C=(0,3) are rank 1; B=(1,1) is rank 2; D=(0,0) is rank 3
    t = np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 3.0], [0.0, 0.0]])
    res = nds_impl(t, rank_max=10)
    numpy.testing.assert_allclose(
        res, np.array([[1.0], [1.0 / 2.0], [1.0], [1.0 / 3.0]])
    )


def test_nds_impl_matches_naive():
    rng = np.random.RandomState(12345)
    t = rng.rand(50, 3)
    res = nds_impl(t, rank_max=10)
    ref = nds_impl_naive(t, rank_max=10)
    numpy.testing.assert_allclose(res, ref)

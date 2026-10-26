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

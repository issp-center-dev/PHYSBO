# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import copy

import numpy as np
import time


class Rectangles(object):
    def __init__(self, n_dim, dtype):
        """
        Initilize a set of hyper-rectangle.

        :param n_dim: dimension of rectangles
        """
        self.n_dim = n_dim
        self.lb = np.zeros((0, self.n_dim), dtype=dtype)
        self.ub = np.zeros((0, self.n_dim), dtype=dtype)

    def add(self, lb, ub):
        """
        Add new rectangles.

        :param lb: lower bounds of rectangles
        :param ub: upper bounds of rectangles
        """
        self.lb = np.r_[self.lb, lb]
        self.ub = np.r_[self.ub, ub]


def dominate(t1, t2):
    """domination rule for maximization problem"""
    return np.all(t1 >= t2) and np.any(t1 > t2)


def _merge_fronts(
    front1: np.ndarray,
    front2: np.ndarray,
    indices1: np.ndarray,
    indices2: np.ndarray,
    maximize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    assert front1.ndim == 2
    assert front2.ndim == 2
    N1, D = front1.shape
    N2, D2 = front2.shape
    assert D == D2
    if N1 == 0:
        return front2, indices2
    if N2 == 0:
        return front1, indices1

    diff = np.zeros((N2, N1, D))
    for i2 in range(N2):
        diff[i2, :, :] = front2[i2, :] - front1
    # dominated_1[i2, i1] = True if front1[i1] is dominated by front2[i2]
    # dominated_2[i2, j1] = True if front2[i2] is dominated by front1[i1]
    if maximize:
        dominated_1 = np.all(diff >= 0, axis=2) & np.any(diff > 0, axis=2)
        dominated_2 = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)
    else:
        dominated_1 = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)
        dominated_2 = np.all(diff >= 0, axis=2) & np.any(diff > 0, axis=2)


    undominated_1 = ~(dominated_1.any(axis=0))
    undominated_2 = ~(dominated_2.any(axis=1))

    new_front1 = front1[undominated_1]
    new_indices1 = indices1[undominated_1]
    new_front2 = front2[undominated_2]
    new_indices2 = indices2[undominated_2]
    new_N1 = len(new_front1)
    new_N2 = len(new_front2)

    i1 = 0
    i2 = 0
    new_front = []
    new_indices = []
    while i1 < new_N1 or i2 < new_N2:
        if i1 == new_N1:
            new_front.append(new_front2[i2])
            new_indices.append(new_indices2[i2])
            i2 += 1
        elif i2 == new_N2:
            new_front.append(new_front1[i1])
            new_indices.append(new_indices1[i1])
            i1 += 1
        elif new_front1[i1, 0] > new_front2[i2, 0]:
            new_front.append(new_front2[i2])
            new_indices.append(new_indices2[i2])
            i2 += 1
        else:
            new_front.append(new_front1[i1])
            new_indices.append(new_indices1[i1])
            i1 += 1
    res_front = np.array(new_front).reshape(-1, D)
    res_indices = np.array(new_indices).reshape(-1)
    return res_front, res_indices


def _extract_non_dominated_points(t: np.ndarray, indices: np.ndarray, maximize: bool = True):
    assert t.ndim == 2
    N = t.shape[0]
    if N < 2:
        return t, indices

    Nleft = N // 2

    tleft = copy.deepcopy(t[:Nleft, :])
    ileft = copy.deepcopy(indices[:Nleft])

    tleft, ileft = _extract_non_dominated_points(tleft, ileft, maximize)

    tright = copy.deepcopy(t[Nleft:, :])
    iright = copy.deepcopy(indices[Nleft:])

    tright, iright = _extract_non_dominated_points(tright, iright, maximize)

    merged_front, merged_indices = _merge_fronts(tleft, tright, ileft, iright, maximize)
    return merged_front, merged_indices


class Pareto(object):
    def __init__(self, num_objectives, dom_rule=None):
        self.num_objectives = num_objectives
        self.front = np.zeros((0, self.num_objectives))
        self.front_num = np.zeros(0, dtype=int)
        self.num_compared = 0
        self.dom_rule = dom_rule
        self.front_updated = False

        if self.dom_rule is None:
            self.dom_rule = dominate

        # self.cells = Rectangles(num_objectives, int)
        self.cells = None
        self.reference_min = None
        self.reference_max = None

    def update_front(self, t):
        return self.__update_front_new(t)

    def __update_front_new(self, t):
        """
        Update the non-dominated set of points.

        Pareto set is sorted on the first objective in ascending order.
        """
        t = np.array(t)
        if t.ndim == 1:
            t = t.reshape((1, -1))
        assert t.shape[1] == self.num_objectives
        N = t.shape[0]

        indices = self.num_compared + np.arange(N)
        right_front, right_indices = _extract_non_dominated_points(t, indices, maximize=True)
        new_front, new_indices = _merge_fronts(self.front, right_front, self.front_num, right_indices, maximize=True)

        updated = False
        for i in new_indices:
            if i in right_indices:
                updated = True
                break
        self.front_updated = updated
        self.num_compared += N
        if self.front_updated:
            self.cells = None
            self.front = new_front
            self.front_num = new_indices


    def __update_front_old(self, t):
        """
        Update the non-dominated set of points.

        Pareto set is sorted on the first objective in ascending order.
        """
        t = np.array(t)
        if t.ndim == 1:
            tt = t.reshape((1, -1))
        else:
            tt = t

        assert tt.shape[1] == self.num_objectives

        front_updated = False

        for k in range(len(tt)):
            point = tt[k]
            is_front = True
            for i in range(len(self.front)):
                if self.dom_rule(self.front[i], point):
                    is_front = False
                    break

            if is_front:
                front_updated = True
                dom_filter = np.full(len(self.front), True, dtype=bool)
                for i in range(len(self.front)):
                    if self.dom_rule(point, self.front[i]):
                        dom_filter[i] = False

                self.front = np.r_[self.front[dom_filter], point[np.newaxis, :]]
                self.front_num = np.r_[self.front_num[dom_filter], self.num_compared]

            self.num_compared += 1

        if front_updated:
            sorted_idx = self.front[:, 0].argsort()
            self.front = self.front[sorted_idx, :]
            self.front_num = self.front_num[sorted_idx]
            self.cells = None # clear cells -- need to update cells before using them
            # self.divide_non_dominated_region()

        self.front_updated = front_updated

    def export_front(self):
        return self.front, self.front_num

    def set_reference_min(self, reference_min=None):
        if reference_min is None:
            # estimate reference min point
            front_min = np.min(self.front, axis=0, keepdims=True)
            w = np.max(self.front, axis=0, keepdims=True) - front_min
            reference_min = front_min - w * 2 / self.front.shape[0]

        self.reference_min = reference_min

    def set_reference_max(self, reference_max=None):
        if reference_max is None:
            # estimate reference max point
            front_max = np.max(self.front, axis=0, keepdims=True)
            w = front_max - np.min(self.front, axis=0, keepdims=True)
            reference_max = front_max + w * 100

        self.reference_max = reference_max

    def volume_in_dominance(self, ref_min, ref_max, dominance_ratio=False):
        if self.cells is None:
            self.divide_non_dominated_region()
        assert isinstance(self.cells, Rectangles) # for type checking

        ref_min = np.array(ref_min)
        ref_max = np.array(ref_max)
        v_all = np.prod(ref_max - ref_min)

        front = np.r_[[ref_min], self.front, [ref_max]]
        ax = np.arange(self.num_objectives)
        lb = front[self.cells.lb, ax]
        ub = front[self.cells.ub, ax]

        v_non_dom = np.sum(np.prod(ub - lb, axis=1))

        if dominance_ratio:
            return (v_all - v_non_dom) / v_all
        else:
            return v_all - v_non_dom

    def divide_non_dominated_region(self, force_binary_search=False, debug=False):
        # clear rectangles
        self.cells = Rectangles(self.num_objectives, int)

        if self.num_objectives == 2 and not force_binary_search:
            self.__divide_2d()
        else:
            self.__divide_using_binary_search(debug=debug)

    def __divide_2d(self):
        """
        Divide non-dominated region into vertical rectangles for the case of 2-objectives.

        Assumes that Pareto set has been sorted on the first objective in ascending order.

        Notes:
            In 2-dimensional cases, the second objective has be sorted in decending order.
        """
        n_cells = self.front.shape[0] + 1
        lb_idx = [[i, (i + 1) % n_cells] for i in range(n_cells)]
        ub_idx = [[i + 1, n_cells] for i in range(n_cells)]

        self.cells.add(lb_idx, ub_idx)

    def __included_in_non_dom_region(self, p):
        return np.all(
            np.any(self.front <= p, axis=1)
        )  # revised on 2025/03/25, vectorized with np.any

    def __divide_using_binary_search(self, debug=False):
        front = np.r_[
            np.full((1, self.num_objectives), -np.inf),
            self.front,
            np.full((1, self.num_objectives), np.inf),
        ]

        # Pareto front indices sorted in each dimension
        front_idx = np.r_[
            np.zeros((1, self.num_objectives), dtype=int),
            np.argsort(self.front, axis=0) + 1,
            np.full((1, self.num_objectives), self.front.shape[0] + 1, dtype=int),
        ]

        if debug:
            print("front_idx shape", front_idx.shape)

        rect_candidates = [[np.copy(front_idx[0]), np.copy(front_idx[-1])]]

        call_time = 0

        start_time = time.time()

        # modified on 2025/03/25, optimized checking of non-dominated region

        while rect_candidates:
            rect = rect_candidates.pop()

            lb_idx = [front_idx[rect[0][d], d] for d in range(self.num_objectives)]
            ub_idx = [front_idx[rect[1][d], d] for d in range(self.num_objectives)]
            lb = [front[lb_idx[d], d] for d in range(self.num_objectives)]
            ub = [front[ub_idx[d], d] for d in range(self.num_objectives)]

            # first check if lb is in the non-dominated region, if so, add it to the cells
            if self.__included_in_non_dom_region(lb):
                self.cells.add([lb_idx], [ub_idx])
                call_time += 1
                continue

            # when lb is not in the non-dominated region, check if ub is in the non-dominated region
            # if ub is in the non-dominated region, it means that the whole rectangle is in the dominated region
            if not self.__included_in_non_dom_region(ub):
                call_time += 1
                continue

            # only when lb is not in the non-dominated region and ub is in the non-dominated region,
            # we need to divide the rectangle into two sub-rectangles

            rect_sizes = rect[1] - rect[0]
            if np.any(rect_sizes > 1):
                div_dim = np.argmax(rect_sizes)
                div_point = rect[0][div_dim] + int(round(rect_sizes[div_dim] / 2.0))

                # Left sub-rectangle
                left_ub_idx = np.copy(rect[1])
                left_ub_idx[div_dim] = div_point
                rect_candidates.append([np.copy(rect[0]), left_ub_idx])

                # Right sub-rectangle
                right_lb_idx = np.copy(rect[0])
                right_lb_idx[div_dim] = div_point
                rect_candidates.append([right_lb_idx, np.copy(rect[1])])

        end_time = time.time()

        if debug:
            print("Execution time:", end_time - start_time, "seconds")
            print(f"call times: {call_time} times")

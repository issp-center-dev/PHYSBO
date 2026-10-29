# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import warnings

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
    # coerce to ndarray: plain lists would be compared lexicographically
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    return np.all(t1 >= t2) and np.any(t1 > t2)


def _non_dominated_mask(t: np.ndarray) -> np.ndarray:
    """
    Mask of the points of t not dominated by any other point (maximization).

    The points are visited in descending lexicographic order.  The first
    remaining point cannot be dominated by any later point, so it is
    non-dominated; the points it dominates are removed at once.  This costs
    O(N * front_size * n_obj) vectorized operations.
    """
    N = t.shape[0]
    mask = np.zeros(N, dtype=bool)
    # np.lexsort uses the last key as the primary one
    order = np.lexsort(t.T[::-1])[::-1]
    rest = t[order]
    rest_idx = order
    while rest_idx.size > 0:
        p = rest[0]
        mask[rest_idx[0]] = True
        others = rest[1:]
        dominated = np.all(others <= p, axis=1) & np.any(others < p, axis=1)
        rest = others[~dominated]
        rest_idx = rest_idx[1:][~dominated]
    return mask


# target number of elements of the (block, front_size, n_obj) temporaries;
# a block holds at least one new point
_CROSS_DOMINANCE_BLOCK_ELEMENTS = 1 << 20


def _cross_dominance(front: np.ndarray, t: np.ndarray):
    """
    Masks of the points of front not dominated by any point of t, and of the
    points of t not dominated by any point of front (maximization).
    """
    keep_front = np.ones(front.shape[0], dtype=bool)
    keep_new = np.ones(t.shape[0], dtype=bool)
    if front.shape[0] == 0 or t.shape[0] == 0:
        return keep_front, keep_new
    block = max(1, _CROSS_DOMINANCE_BLOCK_ELEMENTS // front.size)
    for start in range(0, t.shape[0], block):
        stop = min(start + block, t.shape[0])
        # compared directly: the difference of the values may overflow
        ge = t[start:stop, None, :] >= front[None, :, :]
        le = t[start:stop, None, :] <= front[None, :, :]
        # new point dominates front point: all(ge) and any(t > front)
        dominates = np.all(ge, axis=2) & ~np.all(le, axis=2)
        dominated = np.all(le, axis=2) & ~np.all(ge, axis=2)
        keep_front &= ~np.any(dominates, axis=0)
        keep_new[start:stop] = ~np.any(dominated, axis=1)
    return keep_front, keep_new


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
        # the reference points last estimated from the front; a reference
        # point is treated as estimated while it is this very object
        self._estimated_reference_min = None
        self._estimated_reference_max = None

    def update_front(self, t):
        return self.__update_front_new(t)

    def __update_front_new(self, t):
        """
        Update the non-dominated set of points.

        Pareto set is sorted on the first objective in ascending order; on
        ties, the points of the current front come first and then the new
        points in the order of the rows of t.  front_updated tells whether
        any of the new points joined the front.
        """
        t = np.array(t, dtype=float)
        if t.ndim == 1:
            t = t.reshape((1, -1))
        assert t.shape[1] == self.num_objectives
        N = t.shape[0]

        indices = self.num_compared + np.arange(N)

        # failed observations (any non-finite objective) never join the
        # front; they are skipped here while their indices are still
        # consumed so that front_num keeps referring to the rows of t
        valid = np.all(np.isfinite(t), axis=1)
        if not np.all(valid):
            t = t[valid]
            indices = indices[valid]

        # non-dominated points among the new points
        mask = _non_dominated_mask(t)
        t = t[mask]
        indices = indices[mask]

        # remove the points of the current front dominated by a new point and
        # the new points dominated by a point of the current front
        keep_front, keep_new = _cross_dominance(self.front, t)
        # the front is updated if any of the new points joins it; a point of
        # the current front can be removed only by a new point that joins
        self.front_updated = bool(np.any(keep_new))
        self.num_compared += N
        if not self.front_updated:
            return

        # sorted on the first objective in ascending order; ties keep the
        # current front first and then the order of the rows of t
        all_t = np.r_[self.front[keep_front], t[keep_new]]
        all_indices = np.r_[self.front_num[keep_front], indices[keep_new]]
        sorted_idx = np.argsort(all_t[:, 0], kind="stable")
        self.front = all_t[sorted_idx]
        self.front_num = all_indices[sorted_idx]
        self.cells = None

    def rebuild_front(self, t, terminal_num_run):
        """
        Recompute the non-dominated set of points from the observations.

        The front is updated in the same order as the observations were
        written. The reference points are kept unless they are not finite.

        Parameters
        ----------
        t: numpy.ndarray
            N x num_objectives dimensional array. All the observations.
        terminal_num_run: numpy.ndarray
            The number of the observations at the end of each run.
        """
        self.front = np.zeros((0, self.num_objectives))
        self.front_num = np.zeros(0, dtype=int)
        self.num_compared = 0
        self.front_updated = False
        self.cells = None

        st = 0
        for en in terminal_num_run:
            self.update_front(t[st:en])
            st = en

        # reference points estimated from a front with failed observations
        if self.reference_min is not None and not np.all(
            np.isfinite(self.reference_min)
        ):
            self.reference_min = None
        if self.reference_max is not None and not np.all(
            np.isfinite(self.reference_max)
        ):
            self.reference_max = None

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
        """
        Set the lower reference point.

        Without an argument, it is estimated from the current front, and
        prepare_reference_points estimates it again when the front no
        longer lies above it.  A given reference point is kept as it is.
        """
        if reference_min is None:
            # estimate reference min point
            front_min = np.min(self.front, axis=0, keepdims=True)
            w = np.max(self.front, axis=0, keepdims=True) - front_min
            reference_min = front_min - w * 2 / self.front.shape[0]
            self._estimated_reference_min = reference_min
        else:
            self._estimated_reference_min = None

        self.reference_min = reference_min

    def set_reference_max(self, reference_max=None):
        """
        Set the upper reference point.

        Without an argument, it is estimated from the current front, and
        prepare_reference_points estimates it again when the front no
        longer lies below it.  A given reference point is kept as it is.
        """
        if reference_max is None:
            # estimate reference max point
            front_max = np.max(self.front, axis=0, keepdims=True)
            w = front_max - np.min(self.front, axis=0, keepdims=True)
            reference_max = front_max + w * 100
            self._estimated_reference_max = reference_max
        else:
            self._estimated_reference_max = None

        self.reference_max = reference_max

    def prepare_reference_points(self, upper=True):
        """
        Make the reference points valid for the current front.

        EHVI needs the front between reference_min and reference_max, and
        HVPI needs it above reference_min.  A reference point that is not
        set is estimated from the front.  An estimated reference point that
        the front has grown beyond is estimated again; a reference point set
        by the user is kept, with a warning.  The warning is always issued
        from this module, whichever function calls this method, so that
        warning filters treat all calls alike (and all processes of a
        parallel run alike).

        Parameters
        ----------
        upper: bool
            Whether reference_max is needed as well.
        """
        if self.reference_min is None:
            self.set_reference_min()
        elif self._front_beyond(self.reference_min, lower=True):
            # getattr: objects pickled by older versions lack the attribute
            if self.reference_min is getattr(self, "_estimated_reference_min", None):
                self.set_reference_min()
            else:
                warnings.warn(
                    "the Pareto front is not above reference_min; "
                    "the hypervolume-based scores are not reliable",
                    RuntimeWarning,
                )
        if not upper:
            return
        if self.reference_max is None:
            self.set_reference_max()
        elif self._front_beyond(self.reference_max, lower=False):
            if self.reference_max is getattr(self, "_estimated_reference_max", None):
                self.set_reference_max()
            else:
                warnings.warn(
                    "the Pareto front is not below reference_max; "
                    "EHVI is not reliable",
                    RuntimeWarning,
                )

    def _front_beyond(self, reference, lower):
        """Whether a point of the front lies beyond the reference point."""
        reference = np.asarray(reference, dtype=float).reshape(-1)
        if lower:
            return bool(np.any(self.front < reference))
        return bool(np.any(self.front > reference))

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

        # the cells are collected in lists and added at once at the end;
        # appending them one by one to the arrays costs O(n_cells**2)
        cells_lb = []
        cells_ub = []

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
                cells_lb.append(lb_idx)
                cells_ub.append(ub_idx)
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

        if cells_lb:
            self.cells.add(cells_lb, cells_ub)

        end_time = time.time()

        if debug:
            print("Execution time:", end_time - start_time, "seconds")
            print(f"call times: {call_time} times")

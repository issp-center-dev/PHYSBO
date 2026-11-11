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

# initial number of rows of the arrays of Pareto._local_upper_bounds; the
# arrays double when they are full
_LOCAL_UPPER_BOUNDS_ROWS = 1024


# The box decomposition of a front with _BOX_SLAB_MIN_OBJECTIVES or more
# objectives is made of _BOX_SLABS slabs along the last objective (see
# Pareto.__divide_box_decomposition).  The slabs are a heuristic: they take
# less time and, for most fronts, give fewer cells, but how much depends on
# the front.  With 3 or 4 objectives they save neither.  With 5 they save a
# few percent of the cells at best and cost about 20% when the last
# objective follows another one, so they start at 6; such a last objective
# still costs about 15% of the cells there.  The number of slabs is not
# critical: 2 to 4 give similar numbers of cells, many give more.
_BOX_SLABS = 3
_BOX_SLAB_MIN_OBJECTIVES = 6


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

        Without an argument, it is estimated from the current front as
        front_min - 2 w / n, with w the width of the front in each objective
        (an objective in which the front has no extent takes the largest
        width of the others, or 1) and n the number of points of the front;
        prepare_reference_points estimates it again when the front no longer
        lies above it.  A given reference point is kept as it is.
        """
        if reference_min is None:
            # estimate reference min point
            front_min = np.min(self.front, axis=0, keepdims=True)
            w = self._front_width()
            reference_min = front_min - w * 2 / self.front.shape[0]
            self._estimated_reference_min = reference_min
        else:
            self._estimated_reference_min = None

        self.reference_min = reference_min

    def set_reference_max(self, reference_max=None):
        """
        Set the upper reference point.

        Without an argument, it is estimated from the current front as
        front_max + 100 w, with w as for set_reference_min;
        prepare_reference_points estimates it again when the front no longer
        lies below it.  A given reference point is kept as it is.
        """
        if reference_max is None:
            # estimate reference max point
            front_max = np.max(self.front, axis=0, keepdims=True)
            w = self._front_width()
            reference_max = front_max + w * 100
            self._estimated_reference_max = reference_max
        else:
            self._estimated_reference_max = None

        self.reference_max = reference_max

    def _front_width(self):
        """
        Width of the front in each objective, for estimating the reference
        points.

        An objective in which the front has no extent (all its points share
        the value, e.g. a front of a single point) would give reference
        points of zero width there, and EHVI would vanish for every
        candidate; it gets the largest width of the other objectives, or 1
        when the front has no extent at all.  (For values so large that the
        resulting offsets are below their resolution, a reference point may
        still round to the front; such objectives need given reference
        points.)
        """
        w = np.max(self.front, axis=0, keepdims=True) - np.min(
            self.front, axis=0, keepdims=True
        )
        if np.all(w > 0):
            return w
        fill = np.max(w) if np.any(w > 0) else 1.0
        return np.where(w > 0, w, fill)

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

    def divide_non_dominated_region(
        self, force_binary_search=False, debug=False, method="auto"
    ):
        """
        Partition the region not dominated by the front into disjoint
        hyper-rectangles, stored in ``self.cells`` as index bounds into
        ``[reference_min; front; reference_max]``.

        Parameters
        ----------
        force_binary_search: bool
            Same as ``method="binary_search"``.
        debug: bool
            Print timings of the binary search.
        method: str
            ``"auto"`` (default): vertical strips for 2 objectives, ``"box"``
            otherwise.
            ``"box"``: the box decomposition of Lacour, Klamroth and Fonseca
            (2017), built from local upper bounds as in BoTorch's
            ``FastNondominatedPartitioning``.  Far fewer cells than the binary
            search for 3 or more objectives (4 objectives, 150 front points:
            about 1,600 instead of 2.7 million).  For 6 or more objectives
            the region is cut into slabs along the last objective, and each
            slab is decomposed in this way.
            ``"binary_search"``: recursive bisection of the bounding box.

        All methods cover the same region, so EHVI, HVPI and
        ``volume_in_dominance`` agree up to rounding.
        """
        if force_binary_search:
            method = "binary_search"
        if method not in ("auto", "box", "binary_search"):
            raise ValueError(
                f"unknown method {method!r} (use 'auto', 'box' or 'binary_search')"
            )
        # clear rectangles
        self.cells = Rectangles(self.num_objectives, int)

        if method == "binary_search":
            self.__divide_using_binary_search(debug=debug)
        elif self.num_objectives == 2 and method == "auto":
            self.__divide_2d()
        else:
            self.__divide_box_decomposition()

    # --- Box decomposition from local upper bounds ---------------------------
    #
    # Lacour, Klamroth & Fonseca, "A box decomposition algorithm to compute the
    # hypervolume indicator", Computers & Operations Research 79 (2017)
    # 347-360.  Algorithm 1 maintains, for a minimization problem, the local
    # upper bounds U of the search region and for each u in U its defining
    # points z^j(u) (one per objective).  Following BoTorch's
    # FastNondominatedPartitioning (MIT licence), the region not dominated by
    # the front of a maximization problem is decomposed in two passes:
    #   1. Algorithm 1 on the negated front gives local upper bounds U; -U are
    #      the local lower bounds of the maximization problem.
    #   2. Algorithm 1 again, with -U as the points of a minimization problem
    #      whose reference point is +inf; Eq. (2) of the paper turns its local
    #      upper bounds and defining points into disjoint boxes that tile the
    #      region not dominated by the front above reference_min.
    # Every box coordinate is a coordinate of the front, of reference_min or
    # +inf (mapped to reference_max), so the boxes are stored as index bounds.
    #
    # For _BOX_SLAB_MIN_OBJECTIVES or more objectives, the region is first cut
    # into _BOX_SLABS slabs a <= y < b along the last objective, at values of
    # the front.  In a slab, the points with that objective <= a dominate
    # nothing, and the others dominate the same part of the slab with that
    # objective lowered to b.  Each slab is thus the decomposition of a
    # smaller front, cut to the slab.  The cost of Algorithm 1 grows faster
    # than the number of its bounds, so the slabs take less time in total;
    # for most fronts they also give fewer boxes (see _BOX_SLABS).
    #
    # _local_upper_bounds and the conversion by Eq. (2) in _box_cells are
    # adapted from BoTorch
    # (botorch/utils/multi_objective/box_decompositions/utils.py:
    # compute_local_upper_bounds and get_partition_bounds), which is
    # distributed under the following licence:
    #
    #   MIT License
    #
    #   Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    #   Permission is hereby granted, free of charge, to any person obtaining
    #   a copy of this software and associated documentation files (the
    #   "Software"), to deal in the Software without restriction, including
    #   without limitation the rights to use, copy, modify, merge, publish,
    #   distribute, sublicense, and/or sell copies of the Software, and to
    #   permit persons to whom the Software is furnished to do so, subject to
    #   the following conditions:
    #
    #   The above copyright notice and this permission notice shall be
    #   included in all copies or substantial portions of the Software.
    #
    #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    #   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    #   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    #   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    #   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    #   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    #   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    @staticmethod
    def _local_upper_bounds(points, ref, fill_value):
        """
        Algorithm 1 of Lacour et al. (minimization): local upper bounds of the
        points (rows) inside the box bounded above by ``ref``.

        Returns ``U`` (k x M) and the defining points ``Z`` (k x M x M) with
        ``Z[k, :, j]`` the defining point z^j of ``U[k]``.  The dummy defining
        points have ``ref[j]`` on the diagonal and ``fill_value`` elsewhere.
        """
        M = len(ref)
        Z0 = np.repeat(np.asarray(fill_value, dtype=float)[:, None], M, axis=1)
        Z0[np.arange(M), np.arange(M)] = ref
        # A point replaces only a few of the bounds.  They are marked as dead
        # and the new ones are appended, so that the other rows are not copied
        # for every point; dead rows are dropped, keeping the order of the
        # others, once they are the majority.  The rows come out in the order
        # in which rebuilding the arrays for every point would leave them.
        cap = _LOCAL_UPPER_BOUNDS_ROWS
        U = np.empty((cap, M))
        Z = np.empty((cap, M, M))
        alive = np.zeros(cap, dtype=bool)
        U[0], Z[0], alive[0] = ref, Z0, True
        k = 1  # rows in use, alive or dead
        n_dead = 0
        off = ~np.eye(M, dtype=bool)
        for z in points:
            # the bounds with U > z in every objective, selected one objective
            # at a time, so that the later ones are compared on fewer rows (in
            # the second pass an objective keeps about half of the rows left)
            hit = np.flatnonzero(alive[:k] & (U[:k, 0] > z[0]))
            for d in range(1, M):
                hit = hit[U[hit, d] > z[d]]
            if hit.size == 0:
                continue
            Uh, Zh = U[hit], Z[hit]
            alive[hit] = False
            n_dead += hit.size
            for j in range(M):
                # max over the other defining points of their j-th coordinate
                zmax = np.max(Zh[:, j, off[j]], axis=1, initial=-np.inf)
                ok = z[j] >= zmax
                c = int(np.count_nonzero(ok))
                if c == 0:
                    continue
                if k + c > cap:
                    cap = max(2 * cap, k + c)
                    U = np.concatenate([U[:k], np.empty((cap - k, M))])
                    Z = np.concatenate([Z[:k], np.empty((cap - k, M, M))])
                    alive = np.concatenate([alive[:k], np.zeros(cap - k, dtype=bool)])
                U[k : k + c] = Uh[ok]
                U[k : k + c, j] = z[j]
                Z[k : k + c] = Zh[ok]
                Z[k : k + c, :, j] = z
                alive[k : k + c] = True
                k += c
            if 2 * n_dead > k:
                keep = alive[:k]
                U[: k - n_dead] = U[:k][keep]
                Z[: k - n_dead] = Z[:k][keep]
                alive[: k - n_dead] = True
                alive[k - n_dead : k] = False
                k -= n_dead
                n_dead = 0
        keep = alive[:k]
        return U[:k][keep], Z[:k][keep]

    def __divide_box_decomposition(self):
        M = self.num_objectives
        nf = self.front.shape[0]
        if nf == 0:  # the whole box, as the binary search gives
            self.cells.add(np.zeros((1, M), dtype=int), np.ones((1, M), dtype=int))
            return
        ax = M - 1
        edges = [-np.inf, np.inf]
        if M >= _BOX_SLAB_MIN_OBJECTIVES:
            # Cut at values of the front, with about as many points per slab.
            # (Not as many distinct values: points that share a value would
            # then lie in most of the slabs and be decomposed in each.)
            s = np.sort(self.front[:, ax])
            cut = {s[(nf * k) // _BOX_SLABS] for k in range(1, _BOX_SLABS)} - {s[0]}
            edges = [-np.inf] + sorted(cut) + [np.inf]
        los, his = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            if len(edges) == 2:
                sub = self.front
            else:
                sub = self.front[self.front[:, ax] > a]  # a copy
                # Lowered to b, the points above the slab differ in the other
                # objectives only, which leaves far fewer bounds.  The region
                # is the same without this step; the time is not.
                sub[:, ax] = np.minimum(sub[:, ax], b)
                sub = sub[_non_dominated_mask(sub)]
            lo, hi = self._box_cells(sub)
            lo[:, ax] = np.maximum(lo[:, ax], a)
            hi[:, ax] = np.minimum(hi[:, ax], b)
            keep = lo[:, ax] < hi[:, ax]
            los.append(lo[keep])
            his.append(hi[keep])
        lo = np.concatenate(los)
        hi = np.concatenate(his)
        if lo.shape[0] == 0:
            return
        # value -> row of [reference_min; front; reference_max], per objective
        # (+ 0.0 turns -0.0 into 0.0 so that both map to the same key)
        lb = np.empty(lo.shape, dtype=int)
        ub = np.empty(hi.shape, dtype=int)
        for d in range(M):
            index = {-np.inf: 0}
            for i in range(nf):
                index.setdefault(self.front[i, d] + 0.0, i + 1)
            index[np.inf] = nf + 1
            lb[:, d] = [index[x + 0.0] for x in lo[:, d]]
            ub[:, d] = [index[x + 0.0] for x in hi[:, d]]
        self.cells.add(lb, ub)

    @staticmethod
    def _box_cells(front):
        """
        Boxes [lo, hi) that tile the region not dominated by ``front`` (the
        two passes described before ``_local_upper_bounds``), as values:
        -inf stands for reference_min and +inf for reference_max.
        """
        M = front.shape[1]
        if front.shape[0] == 0:  # nothing is dominated
            return np.full((1, M), -np.inf), np.full((1, M), np.inf)
        # Any point strictly below the front gives the same boxes, so they do
        # not depend on reference_min.  The next double below the front stays
        # strictly below it at any magnitude, unlike front.min - 1, which
        # rounds back to front.min beyond about 2**53; lo == rmin below would
        # then also turn coordinates of the front into reference_min.
        rmin = np.nextafter(np.min(front, axis=0), -np.inf)
        # pass 1: minimization of -front with reference point -rmin
        U1, _ = Pareto._local_upper_bounds(-front, -rmin, np.full(M, -np.inf))
        L = -U1  # local lower bounds of the maximization problem
        # pass 2: points L (minimization), reference point +inf, dummies at rmin
        U2, Z2 = Pareto._local_upper_bounds(L, np.full(M, np.inf), rmin)
        # Eq. (2): l_1 = z^1_1(u), u_1 = inf; l_j = max_{i<j} z^i_j(u), u_j = u_j
        lo = np.empty_like(U2)
        hi = np.empty_like(U2)
        lo[:, 0] = Z2[:, 0, 0]
        hi[:, 0] = np.inf
        for j in range(1, M):
            lo[:, j] = np.max(Z2[:, j, :j], axis=1)
            hi[:, j] = U2[:, j]
        keep = np.all(hi > lo, axis=1)
        lo, hi = lo[keep], hi[keep]
        lo[lo == rmin] = -np.inf  # compared objective by objective
        return lo, hi

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

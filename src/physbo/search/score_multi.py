# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import scipy.special

from .pareto import Pareto
from .score import _norm_pdf

# maximum number of elements of each temporary array in EHVI
_EHVI_BLOCK_ELEMENTS = 1 << 18


def score(mode, predictor_list, test, training, **kwargs):
    if test.X.shape[0] == 0:
        return np.zeros(0)

    if mode == "EHVI":
        pareto = kwargs["pareto"]
        fmean, fstd = _get_fmean_fstd(predictor_list, training, test)
        f = EHVI(fmean, fstd, pareto)
    elif mode == "HVPI":
        pareto = kwargs["pareto"]
        fmean, fstd = _get_fmean_fstd(predictor_list, training, test)
        f = HVPI(fmean, fstd, pareto)
    elif mode == "TS":
        alpha = kwargs.get("alpha", 1.0)
        reduced_candidate_num = kwargs["reduced_candidate_num"]
        f = TS(
            predictor_list,
            training,
            test,
            alpha,
            reduced_candidate_num=reduced_candidate_num,
        )
    else:
        raise NotImplementedError("mode must be EHVI, HVPI or TS.")
    return f


def HVPI(fmean, fstd, pareto):
    """
    Calculate Hypervolume-based Probability of Improvement (HVPI).

    Reference: (Couckuyt et al., 2014) Fast calculation of multiobjective probability of improvement and expected improvement criteria for Pareto optimization

    Parameters
    ----------
    fmean: numpy.ndarray
        mean of the predictive distribution
    fstd: numpy.ndarray
        standard deviation of the predictive distribution
    pareto: Pareto object
        Pareto object

    Returns
    -------
    score: numpy.ndarray
        score of the candidate
    """

    if pareto.cells is None:
        pareto.divide_non_dominated_region()

    N = fmean.shape[0]
    n_obj = pareto.num_objectives

    if pareto.reference_min is None:
        pareto.set_reference_min()
    reference_min = pareto.reference_min

    # Pareto front with reference points
    # shape: (front_size, n_obj)
    front = np.r_[
        np.array(reference_min).reshape((1, n_obj)),
        pareto.front,
        np.full((1, n_obj), np.inf),
    ]

    ax = np.arange(n_obj)
    n_cell = pareto.cells.lb.shape[0]

    # convert to minimization problem
    l = front[pareto.cells.ub, ax].reshape((1, n_cell, n_obj)) * -1
    u = front[pareto.cells.lb, ax].reshape((1, n_cell, n_obj)) * -1

    # convert to minimization problem
    fmean = fmean.reshape((N, 1, n_obj)) * -1
    fstd = fstd.reshape((N, 1, n_obj))

    # calculate cdf
    Phi_l = scipy.special.ndtr((l - fmean) / fstd)
    Phi_u = scipy.special.ndtr((u - fmean) / fstd)

    #  calculate PoI
    poi = np.sum(np.prod(Phi_u - Phi_l, axis=2), axis=1)  # shape: (N, 1)

    # calculate hypervolume contribution of fmean point
    hv_valid = np.all(fmean < u, axis=2)  # shape: (N, n_cell)
    hv = np.prod(u - np.maximum(l, fmean), axis=2)  # shape: (N, n_cell)
    hv = np.sum(hv * hv_valid, axis=1)  # shape: (N, 1)

    # HVPoI
    score = hv * poi
    return score


def EHVI(fmean, fstd, pareto):
    """
    Calculate Expected Hyper-Volume Improvement (EHVI).

    Reference: (Couckuyt et al., 2014) Fast calculation of multiobjective probability of improvement and expected improvement criteria for Pareto optimization

    Parameters
    ----------
    fmean: numpy.ndarray
        mean of the predictive distribution
    fstd: numpy.ndarray
        standard deviation of the predictive distribution
    pareto: Pareto object
        Pareto object

    Returns
    -------
    score: numpy.ndarray
        score of the candidate
    """

    if pareto.cells is None:
        pareto.divide_non_dominated_region()

    N = fmean.shape[0]
    n_obj = pareto.num_objectives

    if pareto.reference_min is None:
        pareto.set_reference_min()
    if pareto.reference_max is None:
        pareto.set_reference_max()
    reference_min = pareto.reference_min
    reference_max = pareto.reference_max

    # Pareto front with reference points
    # shape: (front_size, n_obj)
    front = np.r_[
        np.array(reference_min).reshape((1, n_obj)),
        pareto.front,
        np.array(reference_max).reshape((1, n_obj)),
    ]

    ax = np.arange(n_obj)

    # cell bounds in the minimization problem, shape: (n_cell, n_obj)
    l = front[pareto.cells.ub, ax] * -1
    u = front[pareto.cells.lb, ax] * -1

    # The hypervolume improvement of a point y is the sum over the cells
    # [l, u] of the non-dominated region of vol([l, u] & {z >= y}), which
    # vanishes for a dominated y.  Its expectation can therefore be taken
    # over the whole box y >= lo spanned by the cells, where it factorizes
    # over the objectives: each cell costs n_obj closed-form terms, i.e.,
    # O(N * n_cell * n_obj) instead of the O(N * n_cell**2 * n_obj) of
    # summing over all pairs of cells.
    lo = np.min(l, axis=0)  # shape: (n_obj,)
    width = u - l

    # convert to minimization problem
    fmean = fmean * -1
    score = np.empty(N)

    # bound the size of the (block, n_cell, n_obj) temporaries
    block = max(1, _EHVI_BLOCK_ELEMENTS // max(1, l.size))
    for start in range(0, N, block):
        stop = min(start + block, N)
        m = fmean[start:stop, None, :]  # shape: (n, 1, n_obj)
        s = fstd[start:stop, None, :]
        zl = (l - m) / s  # shape: (n, n_cell, n_obj)
        zu = (u - m) / s
        Phi_l = scipy.special.ndtr(zl)
        Phi_lo = scipy.special.ndtr((lo - m) / s)
        # E[(u - max(l, Y))^+ ; Y >= lo] for Y ~ N(m, s^2)
        G = (
            width * (Phi_l - Phi_lo)
            + (u - m) * (scipy.special.ndtr(zu) - Phi_l)
            + s * (_norm_pdf(zu) - _norm_pdf(zl))
        )
        score[start:stop] = np.sum(np.prod(G, axis=2), axis=1)
    return score


def TS(predictor_list, training, test, alpha=1, reduced_candidate_num=None):
    """Thompson Sampling for multi-objective optimization.

    Parameters
    ----------
    predictor_list : list of Predictor
        List of predictors for each objective.
    training : Variable
        Training data containing all objectives.
    test : Variable
        Test points.
    alpha : float, optional
        Scaling parameter for sampling from posterior distribution.
        Default is 1.
    reduced_candidate_num : int, optional
        Number of candidates to randomly select for Pareto front calculation.
        If None or less than number of test points, all points are used.
        Default is None.

    Returns
    -------
    score_res: numpy.ndarray
        Score array with 1 at the chosen index and 0 elsewhere.
    """

    score = [
        predictor.get_post_samples(training, test, alpha=alpha, objective_index=i)
        for i, predictor in enumerate(predictor_list)
    ]
    score = np.array(score).reshape((len(predictor_list), test.X.shape[0])).T
    pareto = Pareto(num_objectives=len(predictor_list))

    if reduced_candidate_num is None or score.shape[0] <= reduced_candidate_num:
        use_idx = np.arange(score.shape[0])
    else:
        use_idx = np.random.choice(
            score.shape[0], reduced_candidate_num, replace=False
        )

    # pareto.update_front(score)
    pareto.update_front(score[use_idx, :])

    # randomly choose candidate from pareto frontiers
    chosen_idx = np.random.choice(pareto.front_num)
    score_res = np.zeros(score.shape[0])
    score_res[use_idx[chosen_idx]] = 1  # only chosen_idx th value is one.

    return score_res


def _get_fmean_fstd(predictor_list, training, test):
    fmean = [
        predictor.get_post_fmean(training, test, objective_index=i)
        for i, predictor in enumerate(predictor_list)
    ]
    fcov = [
        predictor.get_post_fcov(training, test, objective_index=i)
        for i, predictor in enumerate(predictor_list)
    ]

    # shape: (N, n_obj)
    fmean = np.array(fmean).T
    fstd = np.sqrt(np.array(fcov)).T
    return fmean, fstd

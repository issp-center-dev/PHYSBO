# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import scipy.stats

from .pareto import Pareto


def score(mode, predictor_list, test, training, **kwargs):
    """Calculate multi-objective acquisition scores for test data.

    Parameters
    ----------
    mode : str or callable
        "EHVI", "HVPI", "TS", or a callable called as
        ``mode(predictor_list, training, test, pareto)``.
    predictor_list : list of predictor objects
        Predictor for each objective.
    test : physbo.Variable
        Test inputs.
    training : physbo.Variable
        Training dataset containing all objectives.

    Other Parameters
    ----------------
    pareto : Pareto
        Required for EHVI, HVPI and callable modes.
    alpha : float
        Posterior sampling scale for string "TS". Default is 1.0.
    reduced_candidate_num : int or None
        Required candidate limit for string "TS"; None uses all candidates.
        Callable modes ignore this parameter and alpha.

    Returns
    -------
    numpy.ndarray
        Acquisition scores of shape ``(test.X.shape[0],)``.
    """
    if test.X.shape[0] == 0:
        return np.zeros(0)

    if isinstance(mode, str) and mode == "TS":
        score_fn = resolve(
            mode,
            alpha=kwargs.get("alpha", 1.0),
            reduced_candidate_num=kwargs["reduced_candidate_num"],
        )
        pareto = kwargs.get("pareto")
    else:
        score_fn = resolve(mode)
        pareto = kwargs["pareto"]
    return score_fn(predictor_list, training, test, pareto)


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
    Phi_l = scipy.stats.norm.cdf((l - fmean) / fstd)
    Phi_u = scipy.stats.norm.cdf((u - fmean) / fstd)

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

    # convert to minimization problem
    l = front[pareto.cells.ub, ax] * -1
    u = front[pareto.cells.lb, ax] * -1

    n_cell = pareto.cells.lb.shape[0]

    # shape: (n_cell, 1, n_cell, n_obj)
    l = np.tile(l, (n_cell, 1, 1, 1))
    u = np.tile(u, (n_cell, 1, 1, 1))
    a = l.transpose((2, 1, 0, 3))
    b = u.transpose((2, 1, 0, 3))

    # convert to minimization problem
    fmean = fmean.reshape((1, N, 1, n_obj)) * -1
    fstd = fstd.reshape((1, N, 1, n_obj))

    # calculate pdf, cdf
    phi_min_bu = scipy.stats.norm.pdf((np.minimum(b, u) - fmean) / fstd)
    phi_max_al = scipy.stats.norm.pdf((np.maximum(a, l) - fmean) / fstd)
    Phi_l = scipy.stats.norm.cdf((l - fmean) / fstd)
    Phi_u = scipy.stats.norm.cdf((u - fmean) / fstd)
    Phi_a = scipy.stats.norm.cdf((a - fmean) / fstd)
    Phi_b = scipy.stats.norm.cdf((b - fmean) / fstd)

    ## calculate G
    is_type_A = np.logical_and(a < u, l < b)
    is_type_B = u <= a

    # note: Phi[max_or_min(x,y)] = max_or_min(Phi[x], Phi[y])
    EI_A = (
        (b - a) * (np.maximum(Phi_a, Phi_l) - Phi_l)
        + (b - fmean) * (np.minimum(Phi_b, Phi_u) - np.maximum(Phi_a, Phi_l))
        + fstd * (phi_min_bu - phi_max_al)
    )
    EI_B = (b - a) * (Phi_u - Phi_l)

    G = EI_A * is_type_A + EI_B * is_type_B
    score = np.sum(np.sum(np.prod(G, axis=3), axis=0), axis=1)  # shape: (N, 1)
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


class ExpectedHypervolumeImprovement:
    """Callable hypervolume acquisition function wrapping :func:`EHVI`.

    Notes
    -----
    Called with predictor_list, training, test and pareto as positional
    arguments. Returns scores of shape ``(test.X.shape[0],)``.

    See Also
    --------
    EHVI : Acquisition function using posterior means and standard deviations.
    """

    name = "EHVI"

    def __call__(self, predictor_list, training, test, pareto):
        """Return acquisition scores for test points."""
        fmean, fstd = _get_fmean_fstd(predictor_list, training, test)
        return EHVI(fmean, fstd, pareto)


class HypervolumeProbabilityOfImprovement:
    """Callable hypervolume acquisition function wrapping :func:`HVPI`.

    Notes
    -----
    Called with predictor_list, training, test and pareto as positional
    arguments. Returns scores of shape ``(test.X.shape[0],)``.

    See Also
    --------
    HVPI : Acquisition function using posterior means and standard deviations.
    """

    name = "HVPI"

    def __call__(self, predictor_list, training, test, pareto):
        """Return acquisition scores for test points."""
        fmean, fstd = _get_fmean_fstd(predictor_list, training, test)
        return HVPI(fmean, fstd, pareto)


class ThompsonSampling:
    """Callable multi-objective Thompson sampling wrapping :func:`TS`.

    Parameters
    ----------
    alpha : float, optional
        Posterior sampling scale. Default is 1.0.
    reduced_candidate_num : int, optional
        Maximum number of candidates used for Pareto front calculation.
        Default is None, which uses all candidates.

    See Also
    --------
    TS : Multi-objective Thompson sampling acquisition function.
    """

    name = "TS"

    def __init__(self, alpha=1.0, reduced_candidate_num=None):
        self.alpha = alpha
        self.reduced_candidate_num = reduced_candidate_num

    def __call__(self, predictor_list, training, test, pareto):
        """Return scores of shape ``(test.X.shape[0],)``; pareto is unused."""
        return TS(
            predictor_list,
            training,
            test,
            self.alpha,
            reduced_candidate_num=self.reduced_candidate_num,
        )


def resolve(score, *, alpha=1.0, reduced_candidate_num=None):
    """Resolve a multi-objective score name or return a callable unchanged.

    Parameters
    ----------
    score : str or callable
        "EHVI", "HVPI", "TS", or a callable accepting predictor_list,
        training, test and pareto as positional arguments.
    alpha : float, optional
        Posterior sampling scale for "TS". Default is 1.0.
    reduced_candidate_num : int, optional
        Candidate limit for "TS". Default is None, using all candidates.

    Returns
    -------
    callable
        Acquisition function. Existing callables are returned unchanged.

    Raises
    ------
    NotImplementedError
        If the score name is unknown.
    TypeError
        If score is neither a string nor a callable.
    """
    if isinstance(score, str):
        if score == "EHVI":
            return ExpectedHypervolumeImprovement()
        elif score == "HVPI":
            return HypervolumeProbabilityOfImprovement()
        elif score == "TS":
            return ThompsonSampling(
                alpha=alpha, reduced_candidate_num=reduced_candidate_num
            )
        raise NotImplementedError("mode must be EHVI, HVPI or TS.")
    if callable(score):
        return score
    raise TypeError("score must be a string or callable.")

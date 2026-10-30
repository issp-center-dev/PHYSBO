# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

from .pareto import Pareto
from .._rng import attach_rng, get_rng
from .score import _norm_cdf, _norm_pdf

# target number of elements of each (block, n_cell, n_obj) temporary array in
# EHVI; a block holds at least one candidate, so the temporaries never get
# smaller than (1, n_cell, n_obj)
_EHVI_BLOCK_ELEMENTS = 1 << 18


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
        Required for EHVI and HVPI. Passed to callable modes, as None when
        omitted.
    alpha : float
        Posterior sampling scale for string "TS". Default is 1.0.
    reduced_candidate_num : int or None
        Required candidate limit for string "TS"; None uses all candidates.
        Callable modes ignore this parameter and alpha.
    rng : rng object
        Random number generator for string "TS" (default: the global
        ``numpy.random`` state); a callable mode draws from its own.

    Returns
    -------
    numpy.ndarray
        Acquisition scores of shape ``(test.X.shape[0],)``.
    """
    if test.X.shape[0] == 0:
        upper = _reference_points_upper(mode)
        pareto = kwargs.get("pareto")
        if upper is not None and pareto is not None and pareto.front.shape[0] > 0:
            # as EHVI/HVPI do on the other processes of a parallel run, so
            # that they all use the same reference points and warn (or raise)
            # together
            pareto.prepare_reference_points(upper=upper)
        return np.zeros(0)

    if isinstance(mode, str) and mode == "TS":
        score_fn = resolve(
            mode,
            alpha=kwargs.get("alpha", 1.0),
            reduced_candidate_num=kwargs["reduced_candidate_num"],
            rng=kwargs.get("rng", None),
        )
    else:
        score_fn = resolve(mode, rng=kwargs.get("rng", None))
    return score_fn(predictor_list, training, test, kwargs.get("pareto"))


def _reference_points_upper(mode):
    """Whether ``mode`` needs both reference points (EHVI: True), only the
    lower one (HVPI: False), or none (None), as a name or a built-in instance.

    A user callable gets None even if its name is "EHVI"; it prepares the
    reference points itself if it needs them.
    """
    if isinstance(mode, str):
        return {"EHVI": True, "HVPI": False}.get(mode)
    if isinstance(mode, ExpectedHypervolumeImprovement):
        return True
    if isinstance(mode, HypervolumeProbabilityOfImprovement):
        return False
    return None


def HVPI(fmean, fstd, pareto):
    """
    Calculate Hypervolume-based Probability of Improvement (HVPI).

    Reference: (Couckuyt et al., 2014) Fast calculation of multiobjective probability of improvement and expected improvement criteria for Pareto optimization

    Parameters
    ----------
    fmean: numpy.ndarray
        mean of the predictive distribution, shape (N, n_obj); the first
        axis must be the N candidates, the rest is reshaped to (N, n_obj)
    fstd: numpy.ndarray
        standard deviation of the predictive distribution, any shape with
        N * n_obj elements (reshaped to (N, n_obj))
    pareto: Pareto object
        Pareto object.  The non-dominated region is divided into cells and
        reference_min is set by default if not done yet.

    The result holds only while the front lies above reference_min.

    Returns
    -------
    score: numpy.ndarray
        score of the candidates, shape (N,)
    """

    if pareto.cells is None:
        pareto.divide_non_dominated_region()

    N = fmean.shape[0]
    n_obj = pareto.num_objectives

    pareto.prepare_reference_points(upper=False)
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
    Phi_l = _norm_cdf((l - fmean) / fstd)
    Phi_u = _norm_cdf((u - fmean) / fstd)

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
        mean of the predictive distribution, shape (N, n_obj); the first
        axis must be the N candidates, the rest is reshaped to (N, n_obj)
    fstd: numpy.ndarray
        standard deviation of the predictive distribution, any shape with
        N * n_obj elements (reshaped to (N, n_obj))
    pareto: Pareto object
        Pareto object.  The non-dominated region is divided into cells and
        the reference points are set by default if not done yet.

    The result is the expected hypervolume improvement when the objectives
    are independent normal variables with mean fmean and standard deviation
    fstd, where predictions beyond reference_max in any objective contribute
    zero (they are not clipped).  It holds only while the front lies between
    reference_min and reference_max.

    Returns
    -------
    score: numpy.ndarray
        score of the candidates, shape (N,)
    """

    if pareto.cells is None:
        pareto.divide_non_dominated_region()

    N = fmean.shape[0]
    n_obj = pareto.num_objectives

    pareto.prepare_reference_points()
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
    # lo is -reference_max (when the front lies between the reference
    # points): points beyond reference_max do not contribute
    lo = np.min(l, axis=0)  # shape: (n_obj,)
    width = u - l

    # convert to minimization problem
    fmean = fmean.reshape((N, n_obj)) * -1
    fstd = fstd.reshape((N, n_obj))
    score = np.empty(N)

    # process the candidates in blocks to bound the temporary memory
    block = max(1, _EHVI_BLOCK_ELEMENTS // max(1, l.size))
    for start in range(0, N, block):
        stop = min(start + block, N)
        m = fmean[start:stop, None, :]  # shape: (n, 1, n_obj)
        s = fstd[start:stop, None, :]
        zl = (l - m) / s  # shape: (n, n_cell, n_obj)
        zu = (u - m) / s
        Phi_l = _norm_cdf(zl)
        Phi_lo = _norm_cdf((lo - m) / s)
        # E[(u - max(l, Y))^+ ; Y >= lo] for Y ~ N(m, s^2)
        G = (
            width * (Phi_l - Phi_lo)
            + (u - m) * (_norm_cdf(zu) - Phi_l)
            + s * (_norm_pdf(zu) - _norm_pdf(zl))
        )
        score[start:stop] = np.sum(np.prod(G, axis=2), axis=1)
    return score


def TS(
    predictor_list, training, test, alpha=1, reduced_candidate_num=None, rng=None
):
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

    rng = get_rng(rng)
    score = [
        predictor.get_post_samples(
            training, test, alpha=alpha, objective_index=i, rng=rng
        )
        for i, predictor in enumerate(predictor_list)
    ]
    score = np.array(score).reshape((len(predictor_list), test.X.shape[0])).T
    pareto = Pareto(num_objectives=len(predictor_list))

    if reduced_candidate_num is None or score.shape[0] <= reduced_candidate_num:
        use_idx = np.arange(score.shape[0])
    else:
        use_idx = rng.choice(score.shape[0], reduced_candidate_num, replace=False)

    # pareto.update_front(score)
    pareto.update_front(score[use_idx, :])

    # randomly choose candidate from pareto frontiers
    chosen_idx = rng.choice(pareto.front_num)
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
    rng : rng object, optional
        Random number generator for the posterior samples and the choice
        among the Pareto front; None (default) draws from the global
        ``numpy.random`` state. An instance given to a policy without a
        generator of its own borrows the policy's (see
        :func:`physbo._rng.attach_rng`).

    See Also
    --------
    TS : Multi-objective Thompson sampling acquisition function.
    """

    name = "TS"

    def __init__(self, alpha=1.0, reduced_candidate_num=None, rng=None):
        self.alpha = alpha
        self.reduced_candidate_num = reduced_candidate_num
        self.rng = get_rng(rng)

    def __call__(self, predictor_list, training, test, pareto):
        """Return scores of shape ``(test.X.shape[0],)``; pareto is unused."""
        return TS(
            predictor_list,
            training,
            test,
            self.alpha,
            reduced_candidate_num=self.reduced_candidate_num,
            rng=self.rng,
        )


def resolve(score, *, alpha=1.0, reduced_candidate_num=None, rng=None):
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
    rng : rng object, optional
        Random number generator for "TS". A callable that holds a
        generator in an ``rng`` attribute and was not given one of its
        own (the built-in :class:`ThompsonSampling`) is attached to it
        (see :func:`physbo._rng.attach_rng`). Default is None, the global
        ``numpy.random`` state.

    Returns
    -------
    callable
        Acquisition function. Existing callables are returned unchanged.

    Raises
    ------
    NotImplementedError
        If the score name is unknown.
    TypeError
        If score is a class rather than an instance, or is neither a string
        nor a callable.
    """
    if isinstance(score, type):
        raise TypeError(
            f"score must be an instance, not a class. Pass {score.__name__}() instead."
        )
    if isinstance(score, str):
        if score == "EHVI":
            return ExpectedHypervolumeImprovement()
        elif score == "HVPI":
            return HypervolumeProbabilityOfImprovement()
        elif score == "TS":
            return ThompsonSampling(
                alpha=alpha, reduced_candidate_num=reduced_candidate_num, rng=rng
            )
        raise NotImplementedError("mode must be EHVI, HVPI or TS.")
    if callable(score):
        if rng is not None:
            attach_rng(score, rng)
        return score
    raise TypeError("score must be a string or callable.")

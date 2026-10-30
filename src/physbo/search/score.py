# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import scipy.special

from .. import _mpi
from .._rng import attach_rng, get_rng

_SQRT_2PI = np.sqrt(2 * np.pi)


def _norm_cdf(x):
    """Standard normal CDF, as scipy.stats.norm.cdf without its overhead."""
    # scipy.stats promotes the input to float64; so does this function
    return scipy.special.ndtr(np.asarray(x, dtype=float))


def _norm_pdf(x):
    """Standard normal PDF, as scipy.stats.norm.pdf without its overhead."""
    x = np.asarray(x, dtype=float)
    return np.exp(-(x**2) / 2.0) / _SQRT_2PI


def score(mode, predictor, test, training=None, **kwargs):
    """
    Calculate scores (acquisition function) for test data.

    Parameters
    ----------
    mode: str or callable
        Kind of score.

        "EI", "PI", "TS", and "UCB" are available. A callable is called as
        ``mode(predictor, training, test)`` and ignores score parameters.

    predictor: predictor object
        Base class is defined in physbo.predictor.

    training: physbo.variable
        Training dataset.
        If the predictor is not trained, use this for training.

    test: physbo.variable
        Inputs

    Other Parameters
    ----------------
    fmax: float
        Max value of mean of posterior probability distribution.
        If not set, the maximum value of posterior mean for training is used.
        Used only for mode == "EI" and "PI"

    alpha: float
        noise for sampling source (default: 1.0)
        Used only for mode == "TS"

    rng: rng object
        random number generator (default: global numpy.random state)
        Used only for mode == "TS"; a callable mode draws from its own.

    ucb_beta: float
        Trade-off parameter between exploitation and exploration (default: 1.0).
        The score is ``fmean + ucb_beta * fstd``; larger values favor
        exploration. Used only for mode == "UCB".

    Returns
    -------
    score: numpy.ndarray

    Raises
    ------
    NotImplementedError
        If unknown mode is given
    """

    if test.X.shape[0] == 0:
        return np.zeros(0)

    score_fn = resolve(
        mode,
        alpha=kwargs.get("alpha", 1.0),
        ucb_beta=kwargs.get("ucb_beta", 1.0),
        rng=kwargs.get("rng", None),
    )
    if isinstance(mode, str) and mode in ("EI", "PI"):
        score_fn.fmax = kwargs.get("fmax", None)
    return score_fn(predictor, training, test)


def EI(predictor, training, test, fmax=None):
    """
    Maximum expected improvement.

    Parameters
    ----------
    predictor: predictor object
        Base class is defined in physbo.predictor.
    training: physbo.variable
        Training dataset.
        If the predictor is not trained, use this for training.
    test: physbo.variable
        Inputs
    fmax: float
        Max value of posterior probability distribution.
        If not set, the maximum value of posterior mean for training is used.
    Returns
    -------
    score: numpy.ndarray
    """
    fmean = predictor.get_post_fmean(training, test)
    fcov = predictor.get_post_fcov(training, test)
    fstd = np.sqrt(fcov)

    if fmax is None:
        fmax = np.max(predictor.get_post_fmean(training, training))

    temp1 = fmean - fmax
    temp2 = temp1 / fstd
    score = temp1 * _norm_cdf(temp2) + fstd * _norm_pdf(temp2)
    return score


def PI(predictor, training, test, fmax=None):
    """
    Maximum probability of improvement.

    Parameters
    ----------
    predictor: predictor object
        Base class is defined in physbo.predictor.
    training: physbo.variable
        Training dataset.
        If the predictor is not trained, use this for training.
    test: physbo.variable
        Inputs
    fmax: float
        Max value of posterior probability distribution.
        If not set, the maximum value of posterior mean for training is used.
    Returns
    -------
    score: numpy.ndarray
    """
    fmean = predictor.get_post_fmean(training, test)
    fcov = predictor.get_post_fcov(training, test)
    fstd = np.sqrt(fcov)

    if fmax is None:
        fmax = np.max(predictor.get_post_fmean(training, training))

    temp = (fmean - fmax) / fstd
    score = _norm_cdf(temp)
    return score


def TS(predictor, training, test, alpha=1, rng=None):
    """
    Thompson sampling (See Sec. 2.1 in Materials Discovery Volume 4, June 2016, Pages 18-21)

    Parameters
    ----------
    predictor: predictor object
        Base class is defined in physbo.predictor.
    training: physbo.variable
        Training dataset.
        If the predictor is not trained, use this for training.
    test: physbo.variable
        Inputs
    alpha: float
        noise for sampling source
        (default: 1.0)
    rng: rng object, optional
        random number generator (default: global numpy.random state)
    Returns
    -------
    score: numpy.ndarray

    Notes
    -----
    For predictors with a finite-dimensional posterior sample
    representation (the BLM predictor, i.e. ``num_rand_basis > 0``), the
    sample is a weight vector, drawn once and evaluated at every test
    point. In a parallel run the policies draw it on rank 0 and broadcast
    it (see :class:`ThompsonSampling`), so that every rank scores its
    candidates with the same function sample. The GP predictor
    (``num_rand_basis == 0``) has no finite sample representation; in a
    parallel run each rank then samples the marginal distribution of its
    own candidates only, which is an approximation: cross-rank
    correlations are ignored, and the result depends on the number of
    ranks. For TS in parallel runs, the BLM predictor is recommended.
    """
    if hasattr(predictor, "draw_post_sample_params"):
        w_hat = predictor.draw_post_sample_params(training, alpha=alpha, rng=rng)
        return predictor.evaluate_post_sample(w_hat, test).flatten()

    # no finite sample representation (e.g. GP): sample locally
    test_X = getattr(test, "X", None)
    if test_X is not None and test_X.shape[0] == 0:
        return np.zeros(0)
    return (
        predictor.get_post_samples(training, test, alpha=alpha, rng=rng)
    ).flatten()


def UCB(predictor, training, test, beta=1.0):
    """
    Upper Confidence Bound (UCB).

    The score is ``fmean + beta * fstd``, where ``fmean`` and ``fstd`` are the
    mean and the standard deviation of the posterior predictive distribution.
    Since PHYSBO maximizes the objective, larger ``beta`` puts more weight on
    exploration (high-variance candidates), while ``beta == 0`` reduces to
    greedy exploitation of the posterior mean.

    Parameters
    ----------
    predictor: predictor object
        Base class is defined in physbo.predictor.
    training: physbo.variable
        Training dataset.
        If the predictor is not trained, use this for training.
    test: physbo.variable
        Inputs
    beta: float
        Trade-off parameter between exploitation and exploration.
        (default: 1.0)
    Returns
    -------
    score: numpy.ndarray
    """
    fmean = predictor.get_post_fmean(training, test)
    fcov = predictor.get_post_fcov(training, test)
    fstd = np.sqrt(fcov)
    return (fmean + beta * fstd).flatten()


class ExpectedImprovement:
    """Callable acquisition function wrapping :func:`EI`.

    Parameters
    ----------
    fmax : float, optional
        Maximum posterior mean; None uses the training data.
        Default is None.

    See Also
    --------
    EI : Acquisition function with predictor, training and test arguments.
    """

    name = "EI"

    def __init__(self, fmax=None):
        self.fmax = fmax

    def __call__(self, predictor, training, test):
        """Return acquisition scores of shape ``(test.X.shape[0],)``."""
        return EI(predictor, training, test, self.fmax)


class ProbabilityOfImprovement:
    """Callable acquisition function wrapping :func:`PI`.

    Parameters
    ----------
    fmax : float, optional
        Maximum posterior mean; None uses the training data.
        Default is None.

    See Also
    --------
    PI : Acquisition function with predictor, training and test arguments.
    """

    name = "PI"

    def __init__(self, fmax=None):
        self.fmax = fmax

    def __call__(self, predictor, training, test):
        """Return acquisition scores of shape ``(test.X.shape[0],)``."""
        return PI(predictor, training, test, self.fmax)


class ThompsonSampling:
    """Callable acquisition function wrapping :func:`TS`.

    Parameters
    ----------
    alpha : float, optional
        Scaling parameter for posterior sampling.
        Default is 1.0.
    rng : rng object, optional
        Random number generator for the posterior sample; None (default)
        draws from the global ``numpy.random`` state. An instance given
        to a policy without a generator of its own borrows the policy's
        (see :func:`physbo._rng.attach_rng`).

    Notes
    -----
    Each call draws a new posterior sample. In a parallel run the
    policies share one sample between the processes: :meth:`draw` draws
    it on rank 0, it is broadcast, and :meth:`with_sample` gives the score
    function evaluating it (see :func:`share_sample`).

    See Also
    --------
    TS : Acquisition function with predictor, training and test arguments.
    """

    name = "TS"

    def __init__(self, alpha=1.0, rng=None):
        self.alpha = alpha
        self.rng = get_rng(rng)

    def __call__(self, predictor, training, test):
        """Return acquisition scores of shape ``(test.X.shape[0],)``."""
        return TS(predictor, training, test, self.alpha, rng=self.rng)

    def draw(self, predictor, training):
        """Draw the posterior sample to be shared between the processes.

        Returns the sample of a predictor with a finite-dimensional
        representation of the posterior (the BLM predictor), or None for
        a predictor without one (the GP predictor), which then samples
        in each call.
        """
        if not hasattr(predictor, "draw_post_sample_params"):
            return None
        return predictor.draw_post_sample_params(
            training, alpha=self.alpha, rng=self.rng
        )

    def with_sample(self, sample):
        """The score function evaluating ``sample`` from :meth:`draw`, or
        this object if ``sample`` is None."""
        if sample is None:
            return self
        return _SampledThompsonSampling(sample)


class _SampledThompsonSampling:
    """Thompson sampling evaluating a given posterior sample (deterministic)."""

    name = "TS"

    def __init__(self, sample):
        self.sample = sample

    def __call__(self, predictor, training, test):
        return predictor.evaluate_post_sample(self.sample, test).flatten()


def share_sample(mode, predictor, training, comm):
    """The score evaluating one posterior sample on every process.

    For a :class:`ThompsonSampling` instance in a parallel run (``comm``
    given), the sample is drawn on rank 0 from its generator, broadcast,
    and the returned score evaluates it on every process. Any other
    ``mode``, or ``comm`` None, is returned unchanged. A collective
    operation when ``comm`` is given; an exception raised while drawing
    is raised on every process (see :mod:`physbo._mpi`).
    """
    if comm is None or not isinstance(mode, ThompsonSampling):
        return mode
    if comm.Get_rank() == 0:
        sample, error = _mpi.call(comm, mode.draw, predictor, training)
    else:
        sample, error = None, None
    return mode.with_sample(_mpi.bcast(comm, sample, error, root=0))


def share_samples(mode, predictor, models, K, comm):
    """The posterior samples of ``K`` models, drawn on rank 0 and broadcast.

    ``models(k)`` returns the predictor and the training data of the
    ``k``-th model (the same on every process). For a
    :class:`ThompsonSampling` instance in a parallel run whose
    ``predictor`` has a finite-dimensional posterior representation, the
    returned list holds one sample per model, for
    :meth:`ThompsonSampling.with_sample`; otherwise None is returned, and
    the score draws in each call. A collective operation when it draws.
    """
    if (
        comm is None
        or not isinstance(mode, ThompsonSampling)
        or not hasattr(predictor, "draw_post_sample_params")
    ):
        return None
    if comm.Get_rank() == 0:
        samples, error = _mpi.call(
            comm, lambda: [mode.draw(*models(k)) for k in range(K)]
        )
    else:
        samples, error = None, None
    return _mpi.bcast(comm, samples, error, root=0)


class UpperConfidenceBound:
    """Callable acquisition function wrapping :func:`UCB`.

    Parameters
    ----------
    beta : float, optional
        Weight of the posterior standard deviation.
        Default is 1.0.

    See Also
    --------
    UCB : Acquisition function with predictor, training and test arguments.
    """

    name = "UCB"

    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, predictor, training, test):
        """Return acquisition scores of shape ``(test.X.shape[0],)``."""
        return UCB(predictor, training, test, self.beta)


def resolve(score, *, alpha=1.0, ucb_beta=1.0, rng=None):
    """Resolve a score name or return a callable unchanged.

    Parameters
    ----------
    score : str or callable
        "EI", "PI", "TS", "UCB", or a callable accepting predictor, training
        and test as positional arguments.
    alpha : float, optional
        Posterior sampling scale for "TS". Default is 1.0.
    ucb_beta : float, optional
        Exploration weight for "UCB". Default is 1.0.
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
        if score == "EI":
            return ExpectedImprovement()
        elif score == "PI":
            return ProbabilityOfImprovement()
        elif score == "TS":
            return ThompsonSampling(alpha=alpha, rng=rng)
        elif score == "UCB":
            return UpperConfidenceBound(beta=ucb_beta)
        raise NotImplementedError("ERROR: mode must be EI, PI, TS or UCB.")
    if callable(score):
        if rng is not None:
            attach_rng(score, rng)
        return score
    raise TypeError("score must be a string or callable.")

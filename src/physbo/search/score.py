# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import scipy.stats


def score(mode, predictor, test, training=None, **kwargs):
    """
    Calculate scores (acquisition function) for test data.

    Parameters
    ----------
    mode: str
        Kind of score.

        "EI", "PI", and "TS" are available.

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
        Used only for mode == "TS"

    comm: MPI.Comm
        MPI communicator (default: None).
        Used only for mode == "TS". If given, the posterior sample is
        drawn on rank 0 and broadcast so that all ranks score their
        candidates with the same sample. All ranks must call this
        function (collective operation).

    Returns
    -------
    score: numpy.ndarray

    Raises
    ------
    NotImplementedError
        If unknown mode is given
    """

    comm = kwargs.get("comm", None)

    # NOTE: when comm is given, TS is a collective operation; a rank with
    # an empty candidate set must still participate in the broadcast, so
    # the early return is skipped in that case (TS handles empty input).
    if test.X.shape[0] == 0 and (comm is None or mode != "TS"):
        return np.zeros(0)

    if mode == "EI":
        fmax = kwargs.get("fmax", None)
        return EI(predictor, training, test, fmax)
    elif mode == "PI":
        fmax = kwargs.get("fmax", None)
        return PI(predictor, training, test, fmax)
    elif mode == "TS":
        alpha = kwargs.get("alpha", 1.0)
        rng = kwargs.get("rng", None)
        return TS(predictor, training, test, alpha, rng=rng, comm=comm)
    else:
        raise NotImplementedError("ERROR: mode must be EI, PI or TS.")


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
    score = temp1 * scipy.stats.norm.cdf(temp2) + fstd * scipy.stats.norm.pdf(temp2)
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
    score = scipy.stats.norm.cdf(temp)
    return score


def TS(predictor, training, test, alpha=1, rng=None, comm=None):
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
    comm: MPI.Comm, optional
        MPI communicator (default: None).
        If given, the posterior sample is drawn on rank 0 and broadcast
        so that all ranks evaluate the same sample (collective operation;
        all ranks must call this function).
    Returns
    -------
    score: numpy.ndarray

    Notes
    -----
    For predictors with a finite-dimensional posterior sample
    representation (the BLM predictor, i.e. ``num_rand_basis > 0``), the
    sample is a weight vector; under MPI it is drawn on rank 0 and
    broadcast, so that the score corresponds to a single function sample
    regardless of the number of ranks. The GP predictor
    (``num_rand_basis == 0``) has no finite sample representation; under
    MPI each rank then samples the marginal distribution of its own
    candidates only, which is an approximation: cross-rank correlations
    are ignored, and the result depends on the number of ranks. For TS
    with MPI, the BLM predictor is recommended.
    """
    if hasattr(predictor, "draw_post_sample_params"):
        # draw on rank 0, evaluate everywhere (deterministic given the sample)
        if comm is None or comm.rank == 0:
            w_hat = predictor.draw_post_sample_params(training, alpha=alpha, rng=rng)
        else:
            w_hat = None
        if comm is not None:
            w_hat = comm.bcast(w_hat, root=0)
        return predictor.evaluate_post_sample(w_hat, test).flatten()

    # no finite sample representation (e.g. GP): sample locally
    test_X = getattr(test, "X", None)
    if test_X is not None and test_X.shape[0] == 0:
        return np.zeros(0)
    return (
        predictor.get_post_samples(training, test, alpha=alpha, rng=rng)
    ).flatten()

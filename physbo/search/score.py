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

    if mode == "EI":
        fmax = kwargs.get("fmax", None)
        return EI(predictor, training, test, fmax)
    elif mode == "PI":
        fmax = kwargs.get("fmax", None)
        return PI(predictor, training, test, fmax)
    elif mode == "TS":
        alpha = kwargs.get("alpha", 1.0)
        return TS(predictor, training, test, alpha)
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


def TS(predictor, training, test, alpha=1):
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
    Returns
    -------
    score: numpy.ndarray
    """
    return (predictor.get_post_samples(training, test, alpha=alpha)).flatten()

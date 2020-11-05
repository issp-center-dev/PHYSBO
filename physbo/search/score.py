import numpy as np
import scipy.stats

def EI(predictor, training, test, fmax=None):
    """
    Maximum expected improvement.

    Parameters
    ----------
    predictor: predictor object
            Base class is defined in physbo.predictor.
    training: physbo.variable
            Training dataset. If already trained, the model does not use this.
    test: physbo.variable
            Inputs
    fmax: float
            Max value of posterior probability distribution.
            If not set fmax, the max value of posterior mean of weights is set.
    Returns
    -------
    score: numpy.ndarray
    """
    fmean = predictor.get_post_fmean(training, test)
    fcov = predictor.get_post_fcov(training, test)
    fstd = np.sqrt(fcov)

    if fmax is None:
        fmax = np.max(predictor.get_post_fmean(training, training))

    temp1 = (fmean - fmax)
    temp2 = temp1 / fstd
    score = temp1 * scipy.stats.norm.cdf(temp2) \
        + fstd * scipy.stats.norm.pdf(temp2)
    return score


def PI(predictor, training, test, fmax=None):
    """
    Maximum probability of improvement.

    Parameters
    ----------
    predictor: predictor object
            Base class is defined in physbo.predictor.
    training: physbo.variable
            Training dataset. If already trained, the model does not use this.
    test: physbo.variable
            Inputs
    fmax: float
            Max value of posterior probability distribution.
            If not set fmax, the max value of posterior mean of weights is set.
    Returns
    -------
    score: numpy.ndarray
    """
    fmean = predictor.get_post_fmean(training, test)
    fcov = predictor.get_post_fcov(training, test)
    fstd = np.sqrt(fcov)

    if fmax is None:
        fmax = np.max(predictor.get_post_fmean(training, training))

    temp = (fmean - fmax)/fstd
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
            Training dataset. If already trained, the model does not use this.
    test: physbo.variable
            Inputs
    alpha: float
            noise for sampling source
            (default: 1.0)
    Returns
    -------
    score: numpy.ndarray
    """
    score = predictor.get_post_samples(training, test, alpha=alpha)

    try:
        score.shape[1]
        score[0, :]
    except:
        pass

    return score

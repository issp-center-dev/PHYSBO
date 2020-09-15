import numpy as np
import scipy
from ... import misc


def prepare(blm, X, t, Psi=None):
    """
    initializes auxiaialy parameters for quick sampling

    ``blm.stats`` will be updated.

    Parameters
    ==========
    blm: physbo.blm.core.model
        model
    X: numpy.ndarray
        inputs
    t: numpy.ndarray
        target (label)
    Psi:
        feature maps (default: blm.lik.get_basis(X))
    """
    if Psi is None:
        Psi = blm.lik.get_basis(X)
    PsiT = Psi.transpose()
    G = np.dot(PsiT, Psi) * blm.lik.cov.prec
    A = G + blm.prior.get_prec()
    L = scipy.linalg.cholesky(A, check_finite=False)
    b = PsiT.dot(t - blm.lik.linear.bias)
    alpha = misc.gauss_elim(L, b)
    blm.stats = (L, b, alpha)


def update_stats(blm, x, t, psi=None):
    """
    calculates new auxiaialy parameters for quick sampling by fast-update

    Parameters
    ==========
    blm: physbo.blm.core.model
        model
    x: numpy.ndarray
        input
    t: numpy.ndarray
        target (label)
    psi:
        feature map (default: blm.lik.get_basis(X))

    Returns
    =======
    (L, b, alpha): Tuple
        new auxially parameters

    Notes
    =====
    ``blm.stats[0]`` (L) will be mutated while the others not.
    """
    if psi is None:
        psi = blm.lik.get_basis(x)
    L = blm.stats[0]
    b = blm.stats[1] + (t - blm.lik.linear.bias) * psi
    misc.cholupdate(L, psi * np.sqrt(blm.lik.cov.prec))
    alpha = misc.gauss_elim(L, b)
    return (L, b, alpha)


def sampling(blm, w_mu=None, N=1, alpha=1.0):
    """
    draws samples of weights

    Parameters
    ==========
    blm: physbo.blm.core.model
        model
    w_mu: numpy.ndarray
        mean of weight
    N: int
        the number of samples
        (default: 1)
    alpha: float
        noise for sampling source
        (default: 1.0)

    Returns
    =======
    numpy.ndarray
        samples of weights
    """
    if w_mu is None:
        w_mu = get_post_params_mean(blm)
    if N == 1:
        z = np.random.randn(blm.nbasis) * alpha
    else:
        z = np.random.randn(blm.nbasis, N) * alpha

    L = blm.stats[0]
    invLz = scipy.linalg.solve_triangular(
        L, z, lower=False, overwrite_b=False, check_finite=False
    )
    return (invLz.transpose() + w_mu).transpose()


def get_post_params_mean(blm):
    """
    calculates mean of weight

    Parameters
    ==========
    blm: physbo.blm.core.model

    Returns
    =======
    numpy.ndarray
    """
    return blm.stats[2] * blm.lik.cov.prec


def get_post_fmean(blm, X, Psi=None, w=None):
    """
    calculates posterior mean of model

    Parameters
    ==========
    blm: physbo.blm.core.model
    X: numpy.ndarray
        inputs
    Psi: numpy.ndarray
        feature maps
        (default: blm.lik.linear.basis.get_basis(X))
    w: numpy.ndarray
        weights
        (default: get_post_params_mean(blm))

    Returns
    =======
    numpy.ndarray
    """
    if Psi is None:
        Psi = blm.lik.linear.basis.get_basis(X)

    if w is None:
        w = get_post_params_mean(blm)
    return Psi.dot(w) + blm.lik.linear.bias


def get_post_fcov(blm, X, Psi=None, diag=True):
    """
    calculates posterior covariance of model

    Parameters
    ==========
    blm: physbo.blm.core.model
    X: numpy.ndarray
        inputs
    Psi: numpy.ndarray
        feature maps
        (default: blm.lik.linear.basis.get_basis(X))
    diag: bool
        if True, returns only variances as a diagonal matrix
        (default: True)

    Returns
    =======
    numpy.ndarray
    """
    if Psi is None:
        Psi = blm.lik.linear.basis.get_basis(X)

    L = blm.stats[0]
    R = scipy.linalg.solve_triangular(
        L.transpose(),
        Psi.transpose(),
        lower=True,
        overwrite_b=False,
        check_finite=False,
    )
    RT = R.transpose()

    if diag is True:
        fcov = misc.diagAB(RT, R)
    else:
        fcov = np.dot(RT, R)

    return fcov

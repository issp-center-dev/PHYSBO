# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import scipy
from scipy.linalg import lapack
from scipy.spatial import distance

from ... import misc
from .. import cov, lik, mean

# largest noise variance handled by the fast path
_MAX_NOISE_VARIANCE = np.finfo(float).max / 4


def _uses_gauss_fast_path(gp, params=None):
    """
    True for the default model: Gaussian likelihood, constant or zero mean and
    Gaussian kernel (with or without ARD).  Subclasses use the generic path,
    and so does a noise variance close to overflow (or not finite), whose
    floating-point exceptions the generic path defines.

    The fast path (_gauss_factorize, _eval_marlik_gauss and
    _get_grad_marlik_gauss) inlines the noise covariance and the derivatives
    of lik.Gauss, mean.Const, mean.Zero and cov.Gauss, and the 1e-8 jitter of
    the generic path; keep them consistent when any of these change.
    tests/unit/test_exact_fast_path.py compares both paths.

    Under NumPy's default floating-point error handling, both paths give the
    same values up to roundoff and raise the same exceptions.  When
    underflow or overflow is turned into an exception with np.errstate, the
    two paths may raise for different inputs, since they do not perform
    exactly the same elementwise operations.
    """
    if not (
        type(gp.lik) is lik.Gauss
        and type(gp.prior.cov) is cov.Gauss
        and type(gp.prior.mean) in (mean.Const, mean.Zero)
    ):
        return False
    lik_params, _ = gp.decomp_params(params)
    # this only selects the path: the selected path then warns or raises on
    # overflow or underflow as the generic path always did
    with np.errstate(all="ignore"):
        var = np.asarray(gp.lik.trans_params(lik_params)) ** 2
    # compared, not multiplied: the generic path computes 2 * var
    return bool(np.all(var < _MAX_NOISE_VARIANCE))


def _inputs_allow_fast_path(gp, X, t):
    """
    True when the data are what the fast path assumes: plain arrays (not
    subclasses such as np.matrix or masked arrays), X of shape (N, num_dim)
    with N > 0 (for ARD; the isotropic kernel accepts any number of columns)
    and t of shape (N,) with a real numeric dtype.  Other inputs, which the
    generic path broadcasts, returns or rejects in its own way, use the
    generic path.
    """
    if type(X) is not np.ndarray or type(t) is not np.ndarray:
        return False
    if X.ndim != 2 or X.shape[0] == 0 or t.shape != (X.shape[0],):
        return False
    if t.dtype.kind not in "biuf":
        return False
    c = gp.prior.cov
    return not getattr(c, "ard", False) or X.shape[1] == getattr(c, "num_dim", None)


def _gauss_factorize(gp, X, t, params):
    """
    Cholesky factor of the covariance of the default model.

    The kernel matrix and the parameter bounds are those of the generic path
    (gp.prior.get_cov), but the identity matrices of the generic path are
    avoided.

    Returns
    -------
    A: numpy.ndarray
        A = G + (noise variance + 1e-8) I
    U: numpy.ndarray
        upper Cholesky factor of A
    res: numpy.ndarray
        t minus the prior mean
    G: numpy.ndarray
        kernel matrix
    width: numpy.ndarray
        kernel width(s) after applying the parameter bounds
    var: numpy.float64
        noise variance
    """
    ndata = X.shape[0]
    lik_params, prior_params = gp.decomp_params(params)
    fmu = gp.prior.get_mean(ndata, params=prior_params)
    G = gp.prior.get_cov(X, params=prior_params)
    # the bounds were applied in place by get_cov; prepare is idempotent
    _, width, _ = gp.prior.cov.prepare(prior_params[gp.prior.mean.num_params :])
    # squared in the dtype of the parameters, as gp.lik.get_cov does, then
    # promoted as by the product with the float64 identity of the generic path
    var = np.float64((np.asarray(gp.lik.trans_params(lik_params)) ** 2).reshape(-1)[0])

    A = G.copy()
    diag = np.diag_indices(ndata)
    A[diag] += var
    A[diag] += 1e-8
    # the same factorization as the generic path, also when it fails
    U = scipy.linalg.cholesky(A, check_finite=False)
    return A, U, t - fmu, G, width, var


def _check_lapack_info(name, info):
    if info != 0:
        raise np.linalg.LinAlgError("%s failed with info = %d" % (name, info))


def _eval_marlik_gauss(gp, X, t, params):
    """Marginal likelihood of the default model (see _gauss_factorize)."""
    ndata = X.shape[0]
    _, U, res, _, _, _ = _gauss_factorize(gp, X, t, params)
    # solve U^T alpha = res
    alpha, info = lapack.dtrtrs(U, res, lower=0, trans=1)
    _check_lapack_info("dtrtrs", info)
    return (
        0.5 * ndata * np.log(2 * np.pi)
        + np.sum(np.log(np.diag(U)))
        + 0.5 * np.inner(alpha, alpha)
    )


def _get_grad_marlik_gauss(gp, X, t, params):
    """
    Gradient of the marginal likelihood of the default model.

    The same contractions as in the generic path, without forming its
    (num_params, N, N) tensor of derivatives.
    """
    A, U, res, G, width, var = _gauss_factorize(gp, X, t, params)
    alpha, info = lapack.dpotrs(U, res, lower=0)
    _check_lapack_info("dpotrs", info)
    # inverted as in the generic path, which may also fail for a nearly
    # singular A that has a Cholesky factor
    invA = scipy.linalg.inv(A, overwrite_a=True, check_finite=False)

    # each derivative is 0.5 * (tr(A^-1 dA) - alpha^T dA alpha), with the two
    # terms evaluated separately as in the generic path (forming
    # A^-1 - alpha alpha^T first loses accuracy for nearly singular A)
    grad = np.zeros(gp.num_params)
    # likelihood: dA/dlog(std) = 2 var I, applied to alpha first as in the
    # generic path (alpha^T alpha alone may overflow)
    with np.errstate(over="ignore"):  # like the generic matrix product
        dA_alpha = (2 * var) * alpha
    grad[0] = 0.5 * np.einsum("i,->", np.diag(invA), 2 * var) - 0.5 * np.dot(dA_alpha, alpha)
    n = gp.lik.num_params
    # mean: d(t - mean)/dmean = -1
    if gp.prior.mean.num_params != 0:
        # an inner product with ones, as mean.Const.get_grad in the generic path
        grad[n] = -np.inner(alpha, np.ones(alpha.shape[0]))
        n += gp.prior.mean.num_params
    # kernel: dG/dlog(width_d) = G * ((x_i,d - x_j,d) / width_d)^2 and
    # dG/dlog(scale) = 2 G.  The squared distances and the products are
    # computed as in cov.Gauss.get_grad and cov._gauss.grad_width, and the
    # traces with misc.traceAB2 (einsum) as in the generic path, so that
    # the floating-point exceptions are the same
    if gp.prior.cov.ard:
        for d in range(X.shape[1]):
            # divided by an array, not a scalar, so that the dtype is promoted
            # as in grad_width also under the NumPy 1 casting rules
            sq = ((X[:, d, None] - X[None, :, d]) / width[d : d + 1]) ** 2
            dG = np.einsum("ij,ij->ij", sq, G)
            grad[n + d] = 0.5 * misc.traceAB2(invA, dG) - 0.5 * np.dot(dG.dot(alpha), alpha)
    else:
        dist = distance.pdist(X / width, "euclidean")
        dG = G * distance.squareform(dist**2)
        grad[n] = 0.5 * misc.traceAB2(invA, dG) - 0.5 * np.dot(dG.dot(alpha), alpha)
    dG = 2 * G
    grad[-1] = 0.5 * misc.traceAB2(invA, dG) - 0.5 * np.dot(dG.dot(alpha), alpha)
    return grad


def eval_marlik(gp, X, t, params=None):
    """
    Evaluating marginal likelihood.

    Parameters
    ----------
    gp: physbo.gp.core.Model
    X: numpy.ndarray
        N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.

    t:  numpy.ndarray
        N dimensional array.
        The negative energy of each search candidate (value of the objective function to be optimized).
    params: numpy.ndarray
        Parameters.

    Returns
    -------
    marlik: float
        Marginal likelihood.
    """
    # the default model takes a faster path with the same result
    if _inputs_allow_fast_path(gp, X, t) and _uses_gauss_fast_path(gp, params):
        return _eval_marlik_gauss(gp, X, t, params)

    ndata, ndims = X.shape
    lik_params, prior_params = gp.decomp_params(params)

    fmu = gp.prior.get_mean(ndata, params=prior_params)
    G = gp.prior.get_cov(X, params=prior_params)
    B = gp.lik.get_cov(ndata, params=lik_params)

    A = G + B + 1e-8 * np.identity(ndata)
    res = t - fmu
    U = scipy.linalg.cholesky(A, check_finite=False)
    alpha = scipy.linalg.solve_triangular(
        U.transpose(), res, lower=True, overwrite_b=False, check_finite=False
    )
    marlik = (
        0.5 * ndata * np.log(2 * np.pi)
        + np.sum(np.log(np.diag(U)))
        + 0.5 * np.inner(alpha, alpha)
    )
    return marlik


def get_grad_marlik(gp, X, t, params=None):
    """
    Evaluating gradiant of marginal likelihood.

    Parameters
    ----------
    gp: physbo.gp.core.Model
    X: numpy.ndarray
        N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.

    t:  numpy.ndarray
        N dimensional array.
        The negative energy of each search candidate (value of the objective function to be optimized).
    params: numpy.ndarray
        Parameters.

    Returns
    -------
    grad_marlik: numpy.ndarray
        Gradiant of marginal likelihood.
    """
    # the default model takes a faster path with the same result
    if _inputs_allow_fast_path(gp, X, t) and _uses_gauss_fast_path(gp, params):
        return _get_grad_marlik_gauss(gp, X, t, params)

    ndata, ndims = X.shape
    lik_params, prior_params = gp.decomp_params(params)

    fmu = gp.prior.get_mean(ndata, prior_params)
    G = gp.prior.get_cov(X, params=prior_params)
    B = gp.lik.get_cov(ndata, lik_params)

    A = G + B + 1e-8 * np.identity(ndata)
    U = scipy.linalg.cholesky(A, check_finite=False)
    res = t - fmu
    alpha = misc.gauss_elim(U, res)
    invA = scipy.linalg.inv(A, check_finite=False)

    grad_marlik = np.zeros(gp.num_params)

    """ lik """
    if gp.lik.num_params != 0:
        lik_grad = gp.lik.get_grad(ndata, lik_params)
        temp = lik_grad.dot(alpha)
        grad_marlik[0 : gp.lik.num_params] = -0.5 * temp.dot(
            alpha
        ) + 0.5 * misc.traceAB2(invA, lik_grad)

    ntemp = gp.lik.num_params
    """ prior """
    if gp.prior.mean.num_params != 0:
        mean_grad = gp.prior.get_grad_mean(ndata, prior_params)
        grad_marlik[ntemp : ntemp + gp.prior.mean.num_params] = -np.inner(
            alpha, mean_grad
        )

    ntemp += gp.prior.mean.num_params

    if gp.prior.cov.num_params != 0:
        cov_grad = gp.prior.get_grad_cov(X, prior_params)
        temp = cov_grad.dot(alpha)
        grad_marlik[ntemp:] = -0.5 * temp.dot(alpha) + 0.5 * misc.traceAB3(
            invA, cov_grad
        )

    return grad_marlik


def prepare(gp, X, t, params=None):
    """

    Parameters
    ----------
    gp: physbo.gp.core.Model
    X: numpy.ndarray
        N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.

    t:  numpy.ndarray
        N dimensional array.
        The negative energy of each search candidate (value of the objective function to be optimized).
    params: numpy.ndarray
        Parameters.

    Returns
    -------
    stats: tupple
    """
    ndata = X.shape[0]
    # ndims = X.shape[1]

    if params is None:
        params = np.copy(gp.params)

    lik_params, prior_params = gp.decomp_params(params)

    G = gp.prior.get_cov(X, params=prior_params)
    fmu = gp.prior.get_mean(ndata, params=prior_params)
    B = gp.lik.get_cov(ndata, params=lik_params)
    A = G + B + 1e-8 * np.identity(ndata)
    U = scipy.linalg.cholesky(A, check_finite=False)
    residual = t - fmu
    alpha = misc.gauss_elim(U, residual)
    stats = (U, alpha)

    return stats


def get_post_fmean(gp, X, Z, params=None):
    """
    Calculating the mean of posterior

    Parameters
    ----------
    gp: physbo.gp.core.Model
    X: numpy.ndarray
        N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
    Z: numpy.ndarray
        N x d dimensional matrix. Each row of Z denotes the d-dimensional feature vector of tests.
    params: numpy.ndarray
        Parameters.

    Returns
    -------
    numpy.ndarray
    """

    ntest = Z.shape[0]

    lik_params, prior_params = gp.decomp_params(params)

    alpha = gp.stats[1]

    fmu = gp.prior.get_mean(ntest)
    G = gp.prior.get_cov(X=Z, Z=X, params=prior_params)

    return G.dot(alpha) + fmu


def get_post_fcov(gp, X, Z, params=None, diag=True):
    """
    Calculating the covariance of posterior

    Parameters
    ----------
    gp: physbo.gp.core.Model
    X: numpy.ndarray
        N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
    Z: numpy.ndarray
        N x d dimensional matrix. Each row of Z denotes the d-dimensional feature vector of tests.
    params: numpy.ndarray
        Parameters.
    diag: bool
        If true, only variances (diagonal elements) are returned.

    Returns
    -------
        numpy.ndarray
            Returned shape is (num_points) if diag=true, (num_points, num_points) if diag=false,
            where num_points is the number of points in X.
    """

    lik_params, prior_params = gp.decomp_params(params)

    U = gp.stats[0]
    # alpha = gp.stats[1]

    G = gp.prior.get_cov(X=X, Z=Z, params=prior_params)

    invUG = scipy.linalg.solve_triangular(
        U.transpose(), G, lower=True, overwrite_b=False, check_finite=False
    )

    if diag:
        diagK = gp.prior.get_cov(X=Z, params=prior_params, diag=True)
        diag_invUG2 = misc.diagAB(invUG.transpose(), invUG)
        post_cov = diagK - diag_invUG2
    else:
        K = gp.prior.get_cov(X=Z, params=prior_params)
        post_cov = K - np.dot(invUG.transpose(), invUG)

    return post_cov

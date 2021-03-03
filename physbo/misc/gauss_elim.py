import numpy as np
import scipy


def gauss_elim(U, t):
    """
    Calculate alpha using scipy.linalg.solve_triangular.
    alpha = (U^T U)^-1 t = U^-1 [(U^T)-1 t]

    Parameters
    ----------
    U: (M, M) array_like
    A triangular matrix
    t: (M,) or (M, N) array_like

    Returns
    -------
    alpha: numpy.ndarray
    Solution to the system L^T alpha = t. Shape of return matches t.
    """
    alpha = scipy.linalg.solve_triangular(
        U.transpose(), t, lower=True, overwrite_b=False, check_finite=False
    )

    alpha = scipy.linalg.solve_triangular(
        U, alpha, lower=False, overwrite_b=False, check_finite=False
    )
    return alpha

import numpy as np
import scipy

def gauss_elim( L, t ):
    """
    Calculate alpha using scipy.linalg.solve_triangular.
    alpha = (L^T L)^-1 t = L^-1 [(L^T)-1 t]

    Parameters
    ----------
    L: (M, M) array_like
    A triangular matrix
    t: (M,) or (M, N) array_like
    
    Returns
    -------
    alpha: numpy.ndarray
    Solution to the system L^T alpha = t. Shape of return matches t.
    """
    alpha = scipy.linalg.solve_triangular( L.transpose(), t, \
                        lower=True, overwrite_b = False, check_finite=False )

    alpha = scipy.linalg.solve_triangular( L, alpha, \
                        lower=False, overwrite_b = False, check_finite=False )
    return alpha

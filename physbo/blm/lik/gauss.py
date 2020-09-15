import numpy as np

class gauss:
    """ 
    Gaussian

    Attributes
    ==========
    linear
    cov: blm.lik.cov
        covariance
    stats
    """
    def __init__( self, linear, cov ):
        self.linear = linear
        self.cov = cov
        self.stats = ()

    def get_cov( self, N, params = None ):
        """
        Returns covariance matrix

        Parameters
        ==========
        N: int
            dimension
        params: float
            half of log of variance
            (default: self.cov.params)

        Returns
        =======
        numpy.ndarray
            NxN covariance matrix
        """
        if params is None:
            params = np.copy( self.cov.params )

        return self.cov.get_cov( N, params )

    def get_prec( self, N, params = None ):
        """
        Returns precision matrix

        Parameters
        ==========
        N: int
            dimension
        params: float
            half of log of variance
            (default: self.cov.params)

        Returns
        =======
        numpy.ndarray
            NxN precision matrix
        """

        if params is None:
            params = np.copy( self.cov.params )

        return self.cov.get_cov( N, params )

    def get_basis( self, X ):
        """
        calculates value of basis function at input

        Parameter
        =========
        X: numpy.ndarray
            input

        See also
        ========
        blm.basis.fourier.get_basis
        """
        return self.linear.basis.get_basis( X )

    def get_mean( self, X, Psi = None, params = None, bias = None ):
        """
        calculates mean value

        Parameters
        ==========
        X: numpy.ndarray
            raw input
        Psi: numpy.ndarray
            value of feature maps
        params: numpy.ndarray
            weight
        bias: float
            bias

        See also
        ========
        blm.basis.fourier.get_mean
        """
        return self.linear.get_mean( X, Psi, params, bias )

    def set_params( self, params ):
        """
        sets parameters
        """
        self.linear.set_params( params )

    def set_bias( self, bias ):
        """
        sets bias
        """
        self.linear.set_bias( bias )

    def sampling( self, fmean ):
        """
        draws samples

        Parameters
        ==========
        fmean: numpy.ndarray
            means of samples

        Returns
        =======
        samples: numpy.ndarray
        """
        num_data = fmean.shape[0]
        eps = np.sqrt(self.cov.sigma2) * np.random.randn( num_data )
        return fmean + eps

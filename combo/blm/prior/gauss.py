import numpy as np

class cov_const:
    """
    isotropic variance-covariance

    All elements have the same variance and are independent with each other

    Attributes
    ==========
    params: float
        half of log of covariance
    sigma2: float
        covariance
    prec: float
        precision (= inv. of covariance)
    """
    def __init__( self, params = None ):
        """
        Parameters
        ==========
        params: float
            half of log of covariance
            (default: numpy.log(1))
        """
        if params is None:
            self.params = np.log(1)
        self.sigma2, self.prec = self._trans_params( params )

    def get_cov( self, nbasis, params = None ):
        """
        computes the covariance

        Parameters
        ==========
        nbasis: int
            the number of components
        params: float
            half of log of variance
            (default: self.params)

        Returns
        =======
        numpy.ndarray
            nbasis-by-n-basis covariance matrix
        """
        if params is None:
            params = self.params
        sigma2, prec = self._trans_params( params )
        return np.identity( nbasis ) * sigma2

    def get_prec( self, nbasis, params = None ):
        """
        computes the precision

        Parameters
        ==========
        nbasis: int
            the number of components
        params: float
            half of log of variance
            (default: self.params)

        Returns
        =======
        numpy.ndarray
            nbasis-by-n-basis precision matrix
        """
        if params is None:
            params = self.params
        sigma2, prec = self._trans_params( params )
        return np.identity( nbasis ) * prec

    def set_params( self, params ):
        """
        sets params

        Parameters
        ==========
        params: float
            half of log of variance
        """
        self.params = params
        self.sigma2, self.prec = self._trans_params( params )

    def _trans_params( self, params = None ):
        """
        calculates variance and precise from params

        Parameters
        ==========
        params: float
            half of log of variance
            (default: self.params)

        Returns
        =======
        sigma2: float
            variance
        prec: float
            precise (= inv. of variance)
        """
        if params is None:
            params = self.params

        sigma2 = np.exp( 2 * params )
        prec = 1/sigma2

        return sigma2, prec

class gauss:
    """
    Gaussian prior

    Attributes
    ==========
    nbasis: int
        number of components
    cov: cov_const
        covariance
    """
    def __init__( self, nbasis, cov = None ):
        """
        Parameters
        ==========
        nbasis: int
            number of components
        cov: cov_const
            (default: cov_const())
        """
        self._init_cov( cov )
        self.nbasis = nbasis

    def get_mean( self, params = None ):
        """
        calculates the mean value of priors

        Parameter
        =========
        params: float
            half of log of variance
            (not used)

        Returns
        =======
        numpy.ndarray
        """
        return np.zeros( self.nbasis )

    def get_cov( self, params = None ):
        """
        calculates the variance-covariance matrix of priors

        Parameter
        =========
        params: float
            half of log of variance
            (default: self.cov.params)

        Returns
        =======
        numpy.ndarray
        """
        return self.cov.get_cov( self.nbasis, params )

    def get_prec( self, params = None ):
        """
        calculates the precise matrix of priors

        Parameter
        =========
        params: float
            half of log of variance
            (default: self.cov.params)

        Returns
        =======
        numpy.ndarray
        """
        return self.cov.get_prec( self.nbasis, params )

    def set_params( self, params ):
        """
        sets params

        Parameter
        =========
        params: float
            half of log of variance
        """
        self.cov.set_params( params )

    def _init_cov( self, cov ):
        """
        initialize covariance

        Parameters
        ==========
        cov: cov_const
            default: ``cov_const()``
        """
        self.cov = cov
        if cov is None:
            self.cov = cov_const()

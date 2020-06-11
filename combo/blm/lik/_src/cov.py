import numpy as np

class cov:
    """
    Covariance

    Attributes
    ==========
    params: float
        half of log of variance
    nparams: int
        number of parameters
    sigma2: float
        variance
    prec: float
        inv. of variance
    """
    def __init__( self, params = None ):
        self.params = params
        if self.params is None:
            self.params = np.log(1)
        self.nparams = 1
        self.sigma2, self.prec = self._trans_params( params )

    def get_cov( self, N, params = None ):
        '''
        compute the covariance of prior

        Parameters
        ==========
        N: int
            dimension
        params:
            half of log of variance
            (default: self.params)

        Returns
        =======
        numpy.ndarray
            NxN covariance matrix
        '''
        if params is None:
            params = self.params

        sigma2, prec = self._trans_params( params )
        return np.identity( N ) * sigma2

    def get_prec( self, N, params = None ):
        '''
        compute the precision of prior

        Parameters
        ==========
        N: int
            dimension
        params:
            half of log of variance
            (default: self.params)

        Returns
        =======
        numpy.ndarray
            inverse of covariance matrix
        '''
        if params is None:
            params = self.params
        sigma2, prec = self._trans_params( params )
        return np.identity( N ) * prec

    def set_params( self, params ):
        '''
        set the parameter
        
        Parameters
        ==========
        params: float
            half of log of variance
        '''
        self.params = params
        self.sigma2, self.prec = self._trans_params( params )

    def _trans_params( self, params = None ):
        ''' 
        transform the parameter into variance and precision 

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
            precision (inv. of variance)
        '''
        if params is None:
            params = np.copy(self.params)

        sigma2 = np.exp( 2 * params )
        prec = 1/sigma2
        return sigma2, prec

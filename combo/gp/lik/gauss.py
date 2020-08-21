# -*- coding:utf-8 -*-
import numpy as np
import scipy
#from scipy.stats import multivariate_normal

class gauss:
    ''' Gaussian likelihood function '''
    def __init__( self, std = 1, max_params = 1e6, min_params = 1e-6 ):
        """

        Parameters
        ----------
        std: numpy.ndarray or float
            standard deviation.
        max_params: float
            The maximum value of the parameter.
            If the parameter is greater than this value, it will be replaced by this value.
        min_params: float
            The minimum value of the parameter.
            If the parameter is less than this value, it will be replaced by this value.
        """
        self.min_params = np.log( min_params )
        self.max_params = np.log( max_params )
        self.num_params = 1
        self.std = std
        self.params = np.log( std )
        self.set_params( self.params )

    def supp_params( self, params = None ):
        """
        Set maximum (minimum) values for parameters when the parameter is greater(less) than this value.

        Parameters
        ----------
        params: numpy.ndarray
            Parameters for optimization.
            Array of real elements of size (n,), where ‘n’ is the number of independent variables.
        Returns
        -------

        """
        if params is None:
            params = np.copy( params )

        if params > self.max_params :
            params = self.max_params

        if params < self.min_params :
            params =  self.min_params

        return params

    def trans_params( self, params = None ):
        """
        Get exp[params].

        Parameters
        ----------
        params: numpy.ndarray
            Parameters for optimization.
            Array of real elements of size (n,), where ‘n’ is the number of independent variables.

        Returns
        -------
        std: numpy.ndarray
        """
        if params is None:
            params = np.copy( self.params )

        std = np.exp( params )
        return std

    def get_params_bound( self ):
        """
        Get boundary array.

        Returns
        -------
        bound: list
            A num_params-dimensional array with the tuple (min_params, max_params).
        """
        bound = [ ( self.min_params, self.max_params ) for i in range(0, self.num_params) ]
        return bound

    def get_cov( self, num_data, params = None ):
        """
        Get a covariance matrix

        Parameters
        ----------
        num_data: int
        params: numpy.ndarray
            Parameters for optimization.
            Array of real elements of size (n,), where ‘n’ is the number of independent variables.

        Returns
        -------
        numpy.ndarray
            Diagonal element matrix of exp(2.0*params)
        """
        std = self.trans_params( params )
        var = std ** 2
        return var * np.identity( num_data )

    def get_grad( self, num_data, params = None ):
        """
        Get a gradient matrix

        Parameters
        ----------
        num_data: int
        params: numpy.ndarray
            Parameters for optimization.
            Array of real elements of size (n,), where ‘n’ is the number of independent variables.

        Returns
        -------
        numpy.ndarray
            Diagonal element matrix of 2.0 * exp(2.0*params)
        """
        std = self.trans_params( params )
        var = std ** 2
        return var * np.identity( num_data ) * 2

    def set_params( self, params ):
        """
        Set parameters.

        Parameters
        ----------
        params: numpy.ndarray
            Parameters for optimization.
            Array of real elements of size (n,), where ‘n’ is the number of independent variables.

        Returns
        -------

        """
        self.params = self.supp_params( params )
        self.std = self.trans_params( params )

    def get_cand_params( self, t ):
        """
        Getting candidate parameters.

        Parameters
        ----------
        t:  numpy.ndarray
            N dimensional array. The negative energy of each search candidate (value of the objective function to be optimized).

        Returns
        -------
        numpy.ndarray
            log[ standard deviation of t] - log 10.0
        """
        return np.log( np.std(t) / 10 )

    #[TODO] Check: This function seems not to be used.
    def sampling( self, fmean):
        """
        Sampling by adding noise

        Parameters
        ----------
        fmean: numpy.ndarray

        Returns
        -------

        """
        num_data = fmean.shape[0]
        eps = self.std * np.random.randn( num_data )
        return fmean + eps

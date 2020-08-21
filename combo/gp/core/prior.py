import numpy as np
import scipy

class prior:
    ''' prior of gaussian process '''
    def __init__( self, mean, cov ):
        """

        Parameters
        ----------
        mean: numpy.ndarray
            mean values of prior
        cov: numpy.ndarray
            covariance matrix of priors
        """
        self.mean = mean
        self.cov = cov
        self.num_params = self.cov.num_params + self.mean.num_params
        self.params = self.cat_params( self.mean.params, self.cov.params )

    def cat_params( self, mean_params, cov_params ):
        """

        Parameters
        ----------
        mean_params: numpy.ndarray
            Mean values of parameters
        cov_params: numpy.ndarray
            Covariance matrix of parameters
        Returns
        -------
        numpy.ndarray
        """
        return  np.append( mean_params, cov_params )

    def decomp_params( self, params ):
        """
        decomposing the parameters to those of mean values and covariance matrix for priors

        Parameters
        ----------
        params: numpy.ndarray
            parameters

        Returns
        -------
        mean_params: numpy.ndarray
        cov_params: numpy.ndarray
        """
        if params is None:
            params = np.copy( self.params )

        mean_params = params[0:self.mean.num_params ]
        cov_params = params[self.mean.num_params:]
        return mean_params, cov_params

    def get_mean( self, num_data, params = None ):
        """
        Calculating the mean value of priors

        Parameters
        ----------
        num_data: int
            Total number of data
        params: numpy.ndarray
            Parameters
        Returns
        -------
        float
        """
        if params is None:
            params = np.copy( self.params )
        return self.mean.get_mean( num_data, params[0:self.mean.num_params] )

    def get_cov( self, X, Z = None, params = None, diag = False ):
        """
        Calculating the variance-covariance matrix of priors

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        Z: numpy.ndarray
            N x d dimensional matrix. Each row of Z denotes the d-dimensional feature vector of tests.
        params: numpy.ndarray
           Parameters.
        diag: bool
            If X is the diagonalization matrix, true.
        Returns
        -------
        numpy.ndarray
        """
        if params is None:
            params = np.copy( self.params )

        return self.cov.get_cov( X, Z, params = params[self.mean.num_params:], diag = diag  )

    def get_grad_mean( self, num_data, params = None ):
        """
        Calculating the gradiant of mean values of priors

        Parameters
        ----------
        num_data: int
            Total number of data
        params: numpy.ndarray
            Parameters

        Returns
        -------
        numpy.ndarray

        """
        if params is None:
            params = np.copy( self.params )

        mean_params, cov_params = self.decomp_params( params )
        return self.mean.get_grad( num_data, params = mean_params )

    def get_grad_cov( self, X, params = None ):
        """
        Calculating the covariance matrix priors

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        params: numpy.ndarray
           Parameters.

        Returns
        -------
        numpy.ndarray

        """
        if params is None:
            params = np.copy( self.params )
        mean_params, cov_params =self.decomp_params( params )
        return self.cov.get_grad( X, params = cov_params )

    def set_params( self, params ):
        """
        Setting parameters

        Parameters
        ----------
        params: numpy.ndarray
           Parameters.
        """
        mean_params, cov_params = self.decomp_params( params )
        self.set_mean_params( mean_params )
        self.set_cov_params( cov_params )

    def set_mean_params( self, params ):
        """
        Setting parameters for mean values of priors

        Parameters
        ----------
        params: numpy.ndarray
            Parameters
        """
        if self.mean.num_params != 0:
            self.params[0:self.mean.num_params ] = params
            self.mean.set_params( params )

    def set_cov_params( self, params ):
        """
        Setting parameters for covariance matrix of priors

        Parameters
        ----------
        params: numpy.ndarray
            Parameters
        """
        self.params[self.mean.num_params:] = params
        self.cov.set_params( params )

    def sampling( self, X, N = 1 ):
        """
        Sampling from GP prior

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        N: int

        Returns
        -------
        float

        """
        num_data = X.shape[0]
        G = self.get_cov( X ) + 1e-8 * np.identity( num_data )
        L = scipy.linalg.cholesky( G, check_finite = False )
        Z = np.random.randn( N, num_data )
        return np.dot(Z,L) + self.get_mean( num_data )

import numpy as np

class const:
    ''' constant '''
    def __init__( self, params = None,  max_params = 1e12, min_params = -1e12 ):
        """

        Parameters
        ----------
        params: numpy.ndarray
            Parameters
        max_params: float
            Threshold value for specifying the maximum value of the parameter
        min_params: float
            Threshold value for specifying the minimum value of the parameter

        """
        self.max_params = max_params
        self.min_params = min_params
        self.init_params( params )
        self.num_params = 1

    def supp_params( self, params ):
        """
        Setting maximum and minimum value of parameters.

        Parameters
        ----------
        params: numpy.ndarray
            parameters

        Returns
        -------
            numpy.ndarray
        """
        if params > self.max_params:
            params = self.max_params

        if params < self.min_params:
            params = self.min_params

        return params

    def get_params_bound( self ):
        """
        Getting the boundary list for parameters

        Returns
        -------
        bound: list
            num_params array with the tupple (min_param, max_params)

        """
        bound = [( self.min_params, self.max_params ) for i in range(0, self.num_params)]
        return bound

    def get_mean( self, num_data, params = None ):
        """

        Parameters
        ----------
        num_data: int
            total number of data
        params: numpy.ndarray
            parameters

        Returns
        -------
            numpy.ndarray
        """
        if params is None:
            params = np.copy( self.params )
        return params * np.ones( num_data )

    def get_grad( self, num_data, params = None ):
        """
        Returning a new array of (num_data), filled with ones.

        Parameters
        ----------
        num_data: int
            total number of data
        params: object
            not used

        Returns
        -------
            numpy.ndarray
        """
        return np.ones( num_data )

    def set_params( self, params ):
        """
        Setting parameters defined in const class.

        Parameters
        ----------
        params: numpy.ndarray
            parameters

        Returns
        -------
            numpy.ndarray
        """
        self.params = params

    def init_params( self, params ):
        """
        Initializing parameters

        Parameters
        ----------
        params: numpy.ndarray
            parameters

        Returns
        -------
        params: numpy.ndarray
            initialized parameters
        """
        if params is None:
            self.params = 0
        else:
            self.params = self.supp_params( params )

    def get_cand_params( self, t ):
        """
        Getting the median array of candidates.

        Parameters
        ----------
        t: array_like
            Input array or object that can be converted to an array

        Returns
        -------
        median: numpy.ndarray
            A new array holding the result.

        """
        return np.median( t )

import numpy as np

class zero:
    ''' zero '''
    def __init__(self):
        self.num_params = 0
        self.params = np.array([])

    def get_mean(self, num_data, params = None):
        """
        Returning numpy.zeros(num_data)

        Parameters
        ----------
        num_data: int
            dimension of numpy.zeros
        params: object
            not used

        Returns
        -------
            numpy.ndarray

        """
        return np.zeros(num_data)

    def get_grad( self, num_data, params = None):
        """
        Returning empty numpy.ndarray

        Parameters
        ----------
        num_data: int
            not used
        params: object
            not used

        Returns
        -------
            numpy.ndarray

        """
        return np.array([])

    def set_params( self, params ):
        """
        Not defined

        Parameters
        ----------
        params

        Returns
        -------

        """
        pass

# -*- coding:utf-8 -*-
import numpy as np

class fourier:
    '''
    random feature maps
    ``Psi(X; W,b) = cos[X * Wt + b] * alpha``
        where

            - X: input, N-by-d matrix
            - W: weight, l-by-d matrix
            - Wt: transpose of W
            - b: bias, 1-by-l matrix
            - alpha: coefficient
        
        and

            - N: number of data
            - d: dimension of input
            - l: number of basis

    Attributes
    ==========
    params: Tuple
        W, b, alpha
    nbasis: int
        number of basis

    References
    ==========
    A. Rahimi and B. Recht, "Random features for large-scale kernel machines,"
    in "Advances in neural information processing systems," 2007, pp. 1177-1184.
    '''

    def __init__( self, params ):
        """
        Parameters
        ----------
        params: Tuple
            W, b, alpha
        """
        self._check_params( params )
        self._check_len_params( params )
        self.params = params
        self.nbasis = self.params[1].shape[0]

    def get_basis( self, X, params = None ):
        '''
        compute the value of basis

        Parameters
        ==========
        X: numpy.ndarray
            input
        params: Tuple
            W, b, alpha
            (default: self.params)

        Returns
        =======
        Psi(X; W,b): numpy.ndarray
            N-by-l matrix

            ``cos[X * Wt + b] * alpha``

            where ``Wt`` is the transpose of ``W``.
        '''
        if params is None:
            params = self.params

        self._check_params( params )
        self._check_len_params( params )

        return np.cos( np.dot( X, params[0].transpose() ) + params[1] ) * params[2]

    def set_params( self, params ):
        """
        update basis parameters

        Parameters
        ==========
        params: tuple
            W, b, alpha

        """
        self._check_params( params )
        self._check_len_params( params )
        self.params = params

    def show( self ):
        """
        print parameters
        """
        print 'W = ', self.params[0]
        print 'b = ', self.params[1]
        print 'alpha = ', self.params[2]

    def _check_params( self, params ):
        """
        Parameters
        ==========
        params: tuple
            W, b, alpha

        Raises
        ======
        ValueError
            if ``params`` is not a 3-dimensional tuple
        """
        if not isinstance( params, tuple):
            raise ValueError( 'The variable < params > must be a tuple.' )

        if len(params) !=3:
            raise ValueError( 'The variable < params > must be 3-dimensional tuple.' )

    def _check_len_params( self, params ):
        """
        Parameters
        ==========
        params: tuple
            W, b, alpha


        Raises
        ======
        ValueError
            when dim of W and b are mismatch
            or alpha is not a scalar
        """
        if params[0].shape[0] != params[1].shape[0]:
            raise ValueError( 'The length of 0-axis of W must be same as the length of b.' )

        if hasattr(params[2], "__len__"):
            if len(params[2]) !=1:
                raise ValueError('The third entry of <params> must be a scalar.')
            else:
                if isinstance(params[2], str):
                    raise ValueError('The third entry of <params> must be a scalar.')

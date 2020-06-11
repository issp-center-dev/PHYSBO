import numpy as np

class linear:
    """

    Attributes
    ==========
    basis:
        basis for random feature map
    nbasis: int
        number of basis
    bias:
    params:
    _init_params:
        initial value of the parameter
    """
    def __init__( self, basis, params = None, bias = None ):
        self.basis  = basis
        self.nbasis = basis.nbasis
        self._init_params = params
        self.bias = bias
        self.params = params

        if params is None:
            self.params = np.zeros(self.nbasis)
        self.nparams = self.nbasis

    def get_mean( self, X, Psi = None, params = None, bias = None ):
        """
        calculate mean values

        Parameters
        ==========
        X: numpy.ndarray
            input as an N-by-d matrix
        Psi: numpy.ndarray
            feature maps ``Psi(X)`` as an N-by-l matrix
            (default: self.get_basis(X))
        params: numpy.ndarray
            weight as a vector with size l
            (default: self.params)
        bias: float
            (default: self.bias)

        Returns
        =======
        numpy.ndarray
            Psi * params + bias

        """
        if params is None:
            params = np.copy( self.params)

        if bias is None:
            bias = np.copy( self.bias )

        if Psi is None:
            Psi = self.get_basis( X )

        return Psi.dot(params) + bias

    def set_params( self, params ):
        """
        set parameters

        Parameters
        ==========
        params: np.ndarray
        """
        self.params = params

    def set_bias( self, bias ):
        """
        set bias

        Parameters
        ==========
        bias: float
        """
        self.bias = bias

    def _init_params( self, params ):
        """
        initialize parameters

        Parameters
        ==========
        params: np.ndarray
            (default: numpy.zeros(self.nbasis))
        """
        if params is None:
            self.params = np.zeros( self.nbasis )

        self.params = params

    def _init_bias( self, bias ):
        """
        initialize bias

        Parameters
        ==========
        bias: float
            (default: 0)
        """
        if bias is None:
            self.bias = 0

        self.bias = bias

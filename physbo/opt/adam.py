import numpy as np

class adam:
    """
    Optimizer of f(x) with the adam method

    Attributes
    ==========
    params: numpy.ndarray
        current input, x
    nparams: int
        dimension
    grad: function
        gradient function, g(x) = f'(x)
    m: numpy.ndarray
    v: numpy.ndarray
    epoch: int
        the number of update already done
    max_epoch: int
        the maximum number of update
    alpha: float
    beta: float
    gamma: float
    epsilon: float
    """
    def __init__( self, params, grad, options={} ):
        """

        Parameters
        ==========
        params:
        grad:
        options: dict
            Hyperparameters for the adam method

                - "alpha" (default: 0.001)
                - "beta" (default: 0.9)
                - "gamma" (default: 0.9999)
                - "epsilon" (default: 1e-8)
                - "max_epoch" (default: 4000)
        """
        self.grad = grad
        self.params = params
        self.nparams = params.shape[0]
        self._set_options( options )
        self.m = np.zeros( self.nparams )
        self.v = np.zeros( self.nparams )
        self.epoch = 0

    def set_params( self, params ):
        self.params = params

    def update( self, params, *args, **kwargs ):
        """
        calculates the updates of params

        Parameters
        ==========
        params: numpy.ndarray
            input
        args:
            will be passed to self.grad
        kwargs:
            will be passed to self.grad

        Returns
        =======
        numpy.ndarray
            update of params
        """
        g = self.grad( params, *args, **kwargs )
        self.m = self.m * self.beta + g * ( 1 - self.beta )
        self.v = self.v * self.gamma + g**2 * ( 1 - self.gamma )
        hat_m = self.m / ( 1 - self.beta ** ( self.epoch + 1 ) )
        hat_v = self.v / ( 1 - self.gamma ** ( self.epoch + 1 ) )
        self.epoch += 1
        return - self.alpha * hat_m / ( np.sqrt( hat_v ) + self.epsilon )

    def run( self, *args, **kwargs ):
        params = self.params
        for epoch in xrange( self.max_epoch ):
            update  = self.update( params, *args, **kwargs )
            params += update

    def _set_options( self, options ):
        """
        set hyperparameters for the method

        Parameters
        ==========
        options: dict

        """
        self.alpha = options.get( 'alpha', 0.001 )
        self.beta = options.get( 'beta', 0.9 )
        self.gamma = options.get( 'gamma', 0.9999 )
        self.epsilon = options.get( 'epsilon', 1e-8 )
        self.max_epoch = options.get( 'max_epoch', 4000 )

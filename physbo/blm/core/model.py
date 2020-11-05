import numpy as np
from .. import inf


class model:
    """
    Baysean Linear Model

    Attributes
    ==========
    prior: physbo.blm.prior.gauss
        prior distribution of weights
    lik: physbo.blm.lik.gauss
        kernel
    nbasis: int
        number of features in random feature map
    stats: Tuple
        auxially parameters for sampling
    method: str
        sampling method
    """
    def __init__( self, lik, prior, options = {} ):
        self.prior = prior
        self.lik = lik
        self.nbasis = self.lik.linear.basis.nbasis
        self._init_prior( prior )
        self._set_options( options )
        self.stats = ()

    def prepare( self, X, t, Psi = None ):
        """
        initializes model by using the first training dataset 

        Parameters
        ==========
        X: numpy.ndarray
            inputs
        t: numpy.ndarray
            target (label)
        Psi: numpy.ndarray
            feature maps

        See also
        ========
        physbo.blm.inf.exact.prepare
        """
        if self.method == 'exact':
            inf.exact.prepare( blm = self, X = X, t = t, Psi = Psi )
        else:
            pass

    def update_stats( self, x, t, psi = None ):
        """
        updates model by using another training data

        Parameters
        ==========
        x: numpy.ndarray
            input
        t: float
            target (label)
        psi: numpy.ndarray
            feature map

        See also
        ========
        physbo.blm.inf.exact.update_stats
        """
        if self.method == 'exact':
            self.stats = inf.exact.update_stats( self, x, t, psi )
        else:
            pass

    def get_post_params_mean( self ):
        """
        calculates posterior mean of weights

        Returns
        =======
        numpy.ndarray

        See also
        ========
        physbo.blm.inf.exact.get_post_params_mean
        """
        if self.method == 'exact':
            self.lik.linear.params = inf.exact.get_post_params_mean( blm = self )

    def get_post_fmean( self, X, Psi = None, w = None ):
        """
        calculates posterior mean of model (function)

        Parameters
        ==========
        X: numpy.ndarray
            inputs
        Psi: numpy.ndarray
            feature maps
        w: numpy.ndarray
            weight

        See also
        ========
        physbo.blm.inf.exact.get_post_fmean
        """
        if self.method == 'exact':
            fmu = inf.exact.get_post_fmean( self, X, Psi, w )
        else:
            pass
        return fmu

    def sampling( self, w_mu = None, N = 1, alpha = 1.0 ):
        """
        draws samples of weights

        Parameters
        ==========
        blm: physbo.blm.core.model
            model
        w_mu: numpy.ndarray
            mean of weight
        N: int
            the number of samples
            (default: 1)
        alpha: float
            noise for sampling source
            (default: 1.0)

        Returns
        =======
        numpy.ndarray
            samples of weights

        See also
        ========
        physbo.blm.inf.exact.sampling
        """
        if self.method == 'exact':
            w_hat = inf.exact.sampling( self, w_mu, N, alpha= alpha )
        else:
            pass
        return w_hat

    def post_sampling( self, Xtest, Psi = None, N = 1, alpha = 1.0):
        """
        draws samples of mean value of model

        Parameters
        ==========
        Xtest: numpy.ndarray
            inputs
        Psi: numpy.ndarray
            feature maps
            (default: ``blm.lik.get_basis(Xtest)``)
        N: int
            number of samples
            (default: 1)
        alpha: float
            noise for sampling source

        Returns
        =======
        numpy.ndarray
        """
        if Psi is None:
            Psi = blm.lik.get_basis( Xtest )
        w_hat = self.sampling( N = N, alpha = alpha)
        return Psi.dot( w_hat ) + self.lik.linear.bias

    def predict_sampling( self, Xtest, Psi = None, N=1 ):
        """
        draws samples from model

        Parameters
        ==========
        Xtest: numpy.ndarray
            inputs
        Psi: numpy.ndarray
            feature map
            (default: ``blm.lik.get_basis(Xtest)``)
        N: int
            number of samples
            (default: 1)

        Returns
        =======
        numpy.ndarray
        """
        fmean = self.post_sampling(Xtest, Psi, N=N)
        return fmean + np.sqrt( self.lik.cov.sigma2 ) * np.random.randn( Xtest.shape[0], N )

    def get_post_fcov( self, X, Psi = None, diag = True ):
        """
        calculates posterior covariance of model

        Parameters
        ==========
        X: numpy.ndarray
            inputs
        Psi: numpy.ndarray
            feature maps
            (default: blm.lik.linear.basis.get_basis(X))
        diag: bool
            if True, returns only variances as a diagonal matrix
            (default: True)

        Returns
        =======
        numpy.ndarray

        See also
        ========
        physbo.blm.inf.exact.get_post_fcov
        """
        if self.method == 'exact':
            fcov = inf.exact.get_post_fcov( self, X, Psi, diag = True)
        else:
            pass
        return fcov

    def _set_options( self, options ):
        """
        read options

        Parameters
        ==========
        options: dict
            
            - 'method' : sampling method

                - 'exact' (default)
        """
        self.method = options.get('method','exact')

    def _init_prior( self, prior ):
        """
        sets the prior distribution

        Parameters
        ==========
        prior: physbo.blm.prior.gauss
            if None, prior.gauss(self.nbasis)
        """
        if prior is None:
            prior = prior.gauss( self.nbasis )
        self.prior = prior

from ..predictor import base_predictor


class predictor(base_predictor):
    """ Predictor using Baysean linear model

    Attributes
    ==========
    blm: physbo.blm.core.model
    config: physbo.misc.set_config
        configuration
    """

    def __init__(self, config, model=None):
        """

        Parameters
        ==========
        config: physbo.misc.set_config
            configuration
        model: physbo.gp.core.model

        See also
        ========
        physbo.predictor.base_predictor
        """
        super(predictor, self).__init__(config, model)
        self.blm = None

    def fit(self, training, num_basis=None):
        """
        fit model to training dataset

        Parameters
        ==========
        training: physbo.variable
            dataset for training
        num_basis: int
            the number of basis (default: self.config.predict.num_basis)
        """
        if num_basis is None:
            num_basis = self.config.predict.num_basis

        if self.model.prior.cov.num_dim is None:
            self.model.prior.cov.num_dim = training.X.shape[1]
        self.model.fit(training.X, training.t, self.config)
        self.blm = self.model.export_blm(num_basis)
        self.delete_stats()

    def prepare(self, training):
        """
        initializes model by using training data set

        Parameters
        ==========
        training: physbo.variable
            dataset for training
        """
        self.blm.prepare(training.X, training.t, training.Z)

    def delete_stats(self):
        """
        resets model
        """
        self.blm.stats = None

    def get_basis(self, X):
        """
        calculates feature maps Psi(X)

        Parameters
        ==========
        X: numpy.ndarray
            inputs

        Returns
        =======
        Psi: numpy.ndarray
            feature maps
        """
        return self.blm.lik.get_basis(X)

    def get_post_fmean(self, training, test):
        """
        calculates posterior mean value of model

        Parameters
        ==========
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs

        Returns
        =======
        numpy.ndarray
        """
        if self.blm.stats is None:
            self.prepare(training)
        return self.blm.get_post_fmean(test.X, test.Z)

    def get_post_fcov(self, training, test):
        """
        calculates posterior variance-covariance matrix of model

        Parameters
        ==========
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs

        Returns
        =======
        numpy.ndarray
        """
        if self.blm.stats is None:
            self.prepare(training)
        return self.blm.get_post_fcov(test.X, test.Z)

    def get_post_params(self, training, test):
        """
        calculates posterior weights

        Parameters
        ==========
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs (not used)

        Returns
        =======
        numpy.ndarray
        """
        if self.blm.stats is None:
            self.prepare(training)
        return self.blm.get_post_params_mean()

    def get_post_samples(self, training, test, N=1, alpha=1.0):
        """
        draws samples of mean values of model

        Parameters
        ==========
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs
        N: int
            number of samples
            (default: 1)
        alpha: float
            noise for sampling source
            (default: 1.0)

        Returns
        =======
        numpy.ndarray
        """
        if self.blm.stats is None:
            self.prepare(training)
        return self.blm.post_sampling(test.X, Psi=test.Z, N=N, alpha=alpha)

    def get_predict_samples(self, training, test, N=1):
        """
        draws samples of values of model

        Parameters
        ==========
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs
        N: int
            number of samples
            (default: 1)
        alpha: float
            noise for sampling source
            (default: 1.0)

        Returns
        =======
        numpy.ndarray
        """
        if self.blm.stats is None:
            self.prepare(training)
        return self.blm.predict_sampling(test.X, Psi=test.Z, N=N).transpose()

    def update(self, training, test):
        """
        updates the model.

        If not yet initialized (prepared), the model will be prepared by ``training``.
        Otherwise, the model will be updated by ``test``.

        Parameters
        ==========
        training: physbo.variable
            training dataset for initialization (preparation).
            If already prepared, the model ignore this.
        test: physbo.variable
            training data for update.
            If not prepared, the model ignore this.
        """
        if self.model.stats is None:
            self.prepare(training)
            return None

        if hasattr(test.t, '__len__'):
            N = len(test.t)
        else:
            N = 1

        if N == 1:
            if test.Z is None:
                try:
                    test.X.shape[1]
                    self.blm.update_stats(test.X[0, :], test.t)
                except:
                    self.blm.update_stats(test.X, test.t)
            else:
                try:
                    test.Z.shape[1]
                    self.blm.update_stats(test.X[0, :], test.t, psi=test.Z[0, :])
                except:
                    self.blm.update_stats(test.X, test.t, psi=test.Z)
        else:
            for n in xrange(N):
                if test.Z is None:
                    self.blm.update_stats(test.X[n, :], test.t[n])
                else:
                    self.blm.update_stats(test.X[n, :], test.t[n], psi=test.Z[n, :])

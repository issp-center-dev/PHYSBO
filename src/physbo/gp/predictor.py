# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import physbo.predictor


class predictor(physbo.predictor.base_predictor):
    """predictor"""

    def __init__(self, config, model=None):
        """

        Parameters
        ----------
        config: physbo.misc.set_config
            configuration
        model: physbo.gp.core.model
        """
        super(predictor, self).__init__(config, model)

    def fit(self, training, num_basis=None):
        """
        Fitting model to training dataset

        Parameters
        ----------
        training: physbo.variable
            dataset for training
        num_basis: int
            the number of basis (default: self.config.predict.num_basis)
        """
        if self.model.prior.cov.num_dim is None:
            self.model.prior.cov.num_dim = training.X.shape[1]
        self.model.fit(training.X, training.t, self.config)
        self.delete_stats()

    def get_basis(self, *args, **kwds):
        """

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        pass

    def get_post_params(self, *args, **kwds):
        """

        Parameters
        ----------
        args
        kwds

        Returns
        -------

        """
        pass

    def update(self, training, test):
        self.prepare(training)

    def prepare(self, training):
        """
        Initializing model by using training data set

        Parameters
        ----------
        training: physbo.variable
            dataset for training

        """
        self.model.prepare(training.X, training.t)

    def delete_stats(self):
        self.model.stats = None

    def get_post_fmean(self, training, test):
        """
        Calculating posterior mean value of model

        Parameters
        ----------
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs

        Returns
        -------
        numpy.ndarray

        """
        if self.model.stats is None:
            self.prepare(training)
        return self.model.get_post_fmean(training.X, test.X)

    def get_post_fcov(self, training, test, diag=True):
        """
        Calculating posterior variance-covariance matrix of model

        Parameters
        ----------
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs
        diag: bool
            If true, only variances (diagonal elements) are returned.

        Returns
        -------
        numpy.ndarray
            Returned shape is (num_points) if diag=true, (num_points, num_points) if diag=false,
            where num_points is the number of points in test.

        """
        if self.model.stats is None:
            self.prepare(training)
        return self.model.get_post_fcov(training.X, test.X, diag=diag)

    def get_post_samples(self, training, test, alpha=1):
        """
        Drawing samples of mean values of model

        Parameters
        ----------
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs (not used)
        alpha: float
            tuning parameter of the covariance by multiplying alpha**2 for np.random.multivariate_normal.
        Returns
        -------
        numpy.ndarray

        """
        if self.model.stats is None:
            self.prepare(training)
        return self.model.post_sampling(training.X, test.X, alpha=alpha)

    def get_predict_samples(self, training, test, N=1):
        """
        Drawing samples of values of model

        Parameters
        ----------
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs
        N: int
            number of samples
            (default: 1)

        Returns
        -------
        numpy.ndarray (N x len(test))

        """
        if self.model.stats is None:
            self.prepare(training)
        return self.model.predict_sampling(training.X, test.X, N=N)

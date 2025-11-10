# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import physbo.predictor


class Predictor(physbo.predictor.BasePredictor):
    """Predictor using Baysean linear model

    Attributes
    ==========
    blm: physbo.blm.core.Model
    config: physbo.misc.SetConfig
        configuration
    """

    def __init__(self, config, model=None):
        """

        Parameters
        ==========
        config: physbo.misc.SetConfig
            configuration
        model: physbo.gp.core.Model

        See also
        ========
        physbo.predictor.BasePredictor
        """
        super(Predictor, self).__init__(config, model)
        self.blm = None

    def fit(self, training, num_basis=None, comm=None, objective_index=0):
        """
        fit model to training dataset

        Parameters
        ==========
        training: physbo.variable
            dataset for training
        num_basis: int
            the number of basis (default: self.config.predict.num_basis)
        comm: MPI.Comm
            MPI communicator
        objective_index: int
            Index of objective column to use when training.t is 2D (default: 0)
        """
        if num_basis is None:
            num_basis = self.config.predict.num_basis

        if self.model.prior.cov.num_dim is None:
            self.model.prior.cov.num_dim = training.X.shape[1]
        # Extract 1D t for model fitting: if 2D, take specified column; if 1D, use as is
        t_fit = training.t[:, objective_index] if training.t.ndim == 2 else training.t
        self.model.fit(training.X, t_fit, self.config, comm=comm)
        self.blm = self.model.export_blm(num_basis, comm=comm)
        self.delete_stats()

    def prepare(self, training, objective_index=0):
        """
        initializes model by using training data set

        Parameters
        ==========
        training: physbo.variable
            dataset for training
        objective_index: int
            Index of objective column to use when training.t is 2D (default: 0)
        """
        # Extract 1D t for model preparation: if 2D, take specified column; if 1D, use as is
        t_prep = training.t[:, objective_index] if training.t.ndim == 2 else training.t
        # Extract basis for this objective: Z is (k, N, n), get (N, n) for this objective
        Z_prep = training.Z[objective_index, :, :] if training.Z is not None else None
        self.blm.prepare(training.X, t_prep, Z_prep)

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

    def get_post_fmean(self, training, test, objective_index=0):
        """
        calculates posterior mean value of model

        Parameters
        ==========
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs
        objective_index: int
            Index of objective column to use when training.t is 2D (default: 0)

        Returns
        =======
        numpy.ndarray
            Returned shape is (num_points),
            where num_points is the number of points in test.
        """
        if self.blm.stats is None:
            self.prepare(training, objective_index=objective_index)
        # Extract basis for this objective: Z is (k, N, n), get (N, n) for this objective
        Z_test = test.Z[objective_index, :, :] if test.Z is not None else None
        return self.blm.get_post_fmean(test.X, Z_test)

    def get_post_fcov(self, training, test, diag=True, objective_index=0):
        """
        calculates posterior variance-covariance matrix of model

        Parameters
        ==========
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs
        diag: bool
            If true, only variances (diagonal elements) are returned.
        objective_index: int
            Index of objective column to use when training.t is 2D (default: 0)
        Returns
        =======
        numpy.ndarray
            Returned shape is (num_points) if diag=true, (num_points, num_points) if diag=false,
            where num_points is the number of points in test.
        """
        if self.blm.stats is None:
            self.prepare(training, objective_index=objective_index)
        # Extract basis for this objective: Z is (k, N, n), get (N, n) for this objective
        Z_test = test.Z[objective_index, :, :] if test.Z is not None else None
        return self.blm.get_post_fcov(test.X, Z_test, diag)

    def get_post_params(self, training, test, objective_index=0):
        """
        calculates posterior weights

        Parameters
        ==========
        training: physbo.variable
            training dataset. If already trained, the model does not use this.
        test: physbo.variable
            inputs (not used)
        objective_index: int
            Index of objective column to use when training.t is 2D (default: 0)

        Returns
        =======
        numpy.ndarray
        """
        if self.blm.stats is None:
            self.prepare(training, objective_index=objective_index)
        return self.blm.get_post_params_mean()

    def get_post_samples(self, training, test, N=1, alpha=1.0, objective_index=0):
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
        objective_index: int
            Index of objective column to use when training.t is 2D (default: 0)

        Returns
        =======
        numpy.ndarray
        """
        if self.blm.stats is None:
            self.prepare(training, objective_index=objective_index)
        # Extract basis for this objective: Z is (k, N, n), get (N, n) for this objective
        Z_test = test.Z[objective_index, :, :] if test.Z is not None else None
        return self.blm.post_sampling(test.X, Psi=Z_test, N=N, alpha=alpha)

    def get_predict_samples(self, training, test, N=1, objective_index=0):
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
        objective_index: int
            Index of objective column to use when training.t is 2D (default: 0)

        Returns
        =======
        numpy.ndarray (N x len(test))
        """
        if self.blm.stats is None:
            self.prepare(training, objective_index=objective_index)
        # Extract basis for this objective: Z is (k, N, n), get (N, n) for this objective
        Z_test = test.Z[objective_index, :, :] if test.Z is not None else None
        return self.blm.predict_sampling(test.X, Psi=Z_test, N=N).transpose()

    def update(self, training, test, objective_index=0):
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
        objective_index: int
            Index of objective column to use when training.t or test.t is 2D (default: 0)
        """
        if self.model.stats is None:
            self.prepare(training, objective_index=objective_index)
            return None

        N = test.X.shape[0]

        if N == 1:
            if test.t.ndim == 2:
                t_val = test.t[0, objective_index]
            else:
                t_val = test.t[0] if test.t.ndim == 1 else test.t
            if test.Z is None:
                self.blm.update_stats(test.X, t_val)
            else:
                # Extract basis for this objective: Z is (k, N, n), get (n,) for this objective and first sample
                if test.Z.ndim == 3:
                    psi_val = test.Z[objective_index, 0, :]
                elif test.Z.ndim == 1:
                    psi_val = test.Z
                else:
                    psi_val = test.Z[0, :]
                self.blm.update_stats(test.X[0, :], t_val, psi=psi_val)
        else:
            for n in range(N):
                if test.t.ndim == 2:
                    t_val = test.t[n, objective_index]
                else:
                    t_val = test.t[n]
                if test.Z is None:
                    # Extract n-th row value: if 2D, take row; if 1D, take scalar
                    self.blm.update_stats(test.X[n, :], t_val)
                else:
                    # Extract basis for this objective: Z is (k, N, n), get (n,) for this objective and n-th sample
                    if test.Z.ndim == 3:
                        psi_val = test.Z[objective_index, n, :]
                    elif test.Z.ndim == 1:
                        psi_val = test.Z
                    else:
                        psi_val = test.Z[n, :]
                    self.blm.update_stats(test.X[n, :], t_val, psi=psi_val)

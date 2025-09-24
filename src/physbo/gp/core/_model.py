# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

from ... import blm
from .. import inf
from . import learning
from ._prior import Prior

from physbo.misc.permutation_importance import get_permutation_importance


class Model:
    def __init__(self, lik, mean, cov, inf="exact"):
        """

        Parameters
        ----------
        lik
        mean
        cov
        inf
        """
        self.lik = lik
        self.prior = Prior(mean=mean, cov=cov)
        self.inf = inf
        self.num_params = self.lik.num_params + self.prior.num_params
        self.params = self.cat_params(self.lik.params, self.prior.params)
        self.stats = ()

    def cat_params(self, lik_params, prior_params):
        """
        Concatinate the likelihood and prior parameters

        Parameters
        ----------
        lik_params: numpy.ndarray
            Parameters for likelihood
        prior_params: numpy.ndarray
            Parameters for prior
        Returns
        -------
        params: numpy.ndarray
            parameters about likelihood and prior
        """
        params = np.append(lik_params, prior_params)
        return params

    def decomp_params(self, params=None):
        """
        decomposing the parameters to those of likelifood and priors

        Parameters
        ----------
        params: numpy.ndarray
            parameters

        Returns
        -------
        lik_params: numpy.ndarray
        prior_params: numpy.ndarray
        """
        if params is None:
            params = np.copy(self.params)

        lik_params = params[0 : self.lik.num_params]
        prior_params = params[self.lik.num_params :]
        return lik_params, prior_params

    def set_params(self, params):
        """
        Setting parameters

        Parameters
        ----------
        params: numpy.ndarray
           Parameters.
        """
        self.params = params
        lik_params, prior_params = self.decomp_params(params)
        self.lik.set_params(lik_params)
        self.prior.set_params(prior_params)

    def sub_sampling(self, X, t, N):
        """
        Make subset for sampling

        Parameters
        ----------
        X: numpy.ndarray
           Each row of X denotes the d-dimensional feature vector of search candidate.
        t: numpy.ndarray
           The negative energy of each search candidate (value of the objective function to be optimized).
        N: int
           Total number of data in subset
        Returns
        -------
        subX: numpy.ndarray
        subt: numpy.ndarray
        """
        num_data = X.shape[0]

        if N is not None and N < num_data:
            index = np.random.permutation(num_data)
            subX = X[index[0:N], :]
            subt = t[index[0:N]]
        else:
            subX = X
            subt = t
        return subX, subt

    def export_blm(self, num_basis, comm=None):
        """
        Exporting the blm(Baysean linear model) predictor

        Parameters
        ----------
        num_basis: int
            Total number of basis
        comm: MPI.Comm
            MPI communicator
        Returns
        -------
        physbo.blm.core.Model
            Bayesian linear model
        """
        if not hasattr(self.prior.cov, "rand_expans"):
            raise ValueError("The kernel must be.")

        basis_params = self.prior.cov.rand_expans(num_basis)
        if comm is not None:
            basis_params = comm.bcast(basis_params, root=0)
        basis = blm.basis.Fourier(basis_params)
        prior = blm.prior.Gauss(num_basis)
        lik = blm.lik.Gauss(
            blm.lik.Linear(basis, bias=self.prior.get_mean(1)),
            blm.lik.Cov(self.lik.params),
        )
        blr = blm.core.Model(lik, prior)

        return blr

    def eval_marlik(self, params, X, t, N=None):
        """
        Evaluating marginal likelihood.

        Parameters
        ----------
        params: numpy.ndarray
            Parameters.
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        t:  numpy.ndarray
            N dimensional array.
            The negative energy of each search candidate (value of the objective function to be optimized).
        N: int
            Total number of subset data (if not specified, all dataset is used)
        Returns
        -------
            marlik: float
            Marginal likelihood.
        """
        subX, subt = self.sub_sampling(X, t, N)
        if self.inf == "exact":
            marlik = inf.exact.eval_marlik(self, subX, subt, params=params)
        else:
            pass

        return marlik

    def get_grad_marlik(self, params, X, t, N=None):
        """
        Evaluating gradiant of marginal likelihood.

        Parameters
        ----------
        params: numpy.ndarray
            Parameters.
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        t:  numpy.ndarray
            N dimensional array.
            The negative energy of each search candidate (value of the objective function to be optimized).
        N: int
            Total number of subset data (if not specified, all dataset is used)

        Returns
        -------
        grad_marlik: numpy.ndarray
            Gradiant of marginal likelihood.
        """
        subX, subt = self.sub_sampling(X, t, N)
        if self.inf == "exact":
            grad_marlik = inf.exact.get_grad_marlik(self, subX, subt, params=params)
        return grad_marlik

    def get_params_bound(self):
        """
        Getting boundary of the parameters.

        Returns
        -------
        bound: list
            An array with the tuple (min_params, max_params).
        """
        if self.lik.num_params != 0:
            bound = self.lik.get_params_bound()

        if self.prior.mean.num_params != 0:
            bound.extend(self.prior.mean.get_params_bound())

        if self.prior.cov.num_params != 0:
            bound.extend(self.prior.cov.get_params_bound())
        return bound

    def prepare(self, X, t, params=None):
        """

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.

        t:  numpy.ndarray
            N dimensional array.
            The negative energy of each search candidate (value of the objective function to be optimized).
        params: numpy.ndarray
            Parameters.
        """
        if params is None:
            params = np.copy(self.params)
        if self.inf == "exact":
            self.stats = inf.exact.prepare(self, X, t, params)
        else:
            pass

    def get_post_fmean(self, X, Z, params=None):
        """
        Calculating posterior mean of model (function)

        Parameters
        ==========
        X: numpy.ndarray
            inputs
        Z: numpy.ndarray
            feature maps
        params: numpy.ndarray
            Parameters
        See also
        ========
        physbo.gp.inf.exact.get_post_fmean
        """
        if params is None:
            params = np.copy(self.params)

        if self.inf == "exact":
            post_fmu = inf.exact.get_post_fmean(self, X, Z, params)

        return post_fmu

    def get_post_fcov(self, X, Z, params=None, diag=True):
        """
        Calculating posterior covariance matrix of model (function)

        Parameters
        ----------
        X: numpy.ndarray
            inputs
        Z: numpy.ndarray
            feature maps
        params: numpy.ndarray
            Parameters
        diag: bool
            If true, only variances (diagonal elements) are returned.

        Returns
        -------
        numpy.ndarray
            Returned shape is (num_points) if diag=true, (num_points, num_points) if diag=false,
            where num_points is the number of points in X.

        """
        if params is None:
            params = np.copy(self.params)

        if self.inf == "exact":
            post_fcov = inf.exact.get_post_fcov(self, X, Z, params, diag)

        return post_fcov

    def post_sampling(self, X, Z, params=None, N=1, alpha=1):
        """
        draws samples of mean value of model

        Parameters
        ==========
        X: numpy.ndarray
            inputs
        Z: numpy.ndarray
            feature maps
        N: int
            number of samples
            (default: 1)
        alpha: float
            noise for sampling source
        Returns
        =======
        numpy.ndarray
        """
        if params is None:
            params = np.copy(self.params)

        fmean = self.get_post_fmean(X, Z, params=None)
        fcov = self.get_post_fcov(X, Z, params=None, diag=False)
        return np.random.multivariate_normal(fmean, fcov * alpha**2, N)

    def predict_sampling(self, X, Z, params=None, N=1):
        """

        Parameters
        ----------
        X: numpy.ndarray
            training datasets
        Z: numpy.ndarray
            input for sampling objective values
        params: numpy.ndarray
            Parameters
        N: int
            number of samples
            (default: 1)

        Returns
        -------
        numpy.ndarray

        """
        if params is None:
            params = np.copy(self.params)

        ndata = Z.shape[0]
        if ndata == 0:
            return np.zeros((N, 0))
        fmean = self.get_post_fmean(X, Z, params=None)
        fcov = self.get_post_fcov(X, Z, params=None, diag=False) + self.lik.get_cov(
            ndata
        )

        return np.random.multivariate_normal(fmean, fcov, N)

    def print_params(self):
        """
        Printing parameters
        """
        print("\n")
        if self.lik.num_params != 0:
            print("likelihood parameter =  ", self.lik.params)

        if self.prior.mean.num_params != 0:
            print("mean parameter in GP prior: ", self.prior.mean.params)

        print("covariance parameter in GP prior: ", self.prior.cov.params)
        print("\n")

    def get_cand_params(self, X, t):
        """
        Getting candidate for parameters

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.

        t:  numpy.ndarray
            N dimensional array.
            The negative energy of each search candidate (value of the objective function to be optimized).
        Returns
        -------
        params: numpy.ndarray
            Parameters
        """
        params = np.zeros(self.num_params)
        if self.lik.num_params != 0:
            params[0 : self.lik.num_params] = self.lik.get_cand_params(t)

        temp = self.lik.num_params

        if self.prior.mean.num_params != 0:
            params[temp : temp + self.prior.mean.num_params] = (
                self.prior.mean.get_cand_params(t)
            )

        temp += self.prior.mean.num_params

        if self.prior.cov.num_params != 0:
            params[temp:] = self.prior.cov.get_cand_params(X, t)

        return params

    def fit(self, X, t, config, comm=None):
        """
        Fitting function (update parameters)

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.

        t:  numpy.ndarray
            N dimensional array.
            The negative energy of each search candidate (value of the objective function to be optimized).
        config: physbo.misc.set_config object
        comm: MPI.Comm
            MPI communicator

        """
        method = config.learning.method

        if method == "adam":
            adam = learning.Adam(self, config)
            params = adam.run(X, t)

        if method in ("bfgs", "batch"):
            bfgs = learning.Batch(self, config)
            params = bfgs.run(X, t)

        if comm is not None:
            params = comm.bcast(params, root=0)

        self.set_params(params)

    def get_permutation_importance(
        self, X, t, n_perm: int, comm=None, split_features_parallel=False
    ):
        """
        Calculating permutation importance of model

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        t:  numpy.ndarray
            N dimensional array.
            The negative energy of each search candidate (value of the objective function to be optimized).
        n_perm: int
            Number of permutations
        comm: MPI.Comm
            MPI communicator

        Returns
        =======
        numpy.ndarray
            importance_mean
        numpy.ndarray
            importance_std
        """
        return get_permutation_importance(
            self,
            X,
            t,
            n_perm,
            comm=comm,
            split_features_parallel=split_features_parallel,
        )


class SFS(Model):
    def __init__(self, lik, mean, cov, inf="exact", config=None):
        super().__init__(lik, mean, cov, inf)

        self.config = config

    def prepare(self, X, t, params=None):
        return super().prepare(X, t, params)

    def fit(self, X, t):
        """
        Fitting function (update parameters)

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.

        t:  numpy.ndarray
            N dimensional array.
            The negative energy of each search candidate (value of the objective function to be optimized).
        config: physbo.misc.SetConfig object

        """
        # config = SetConfig()
        method = self.config.learning.method

        if method == "adam":
            adam = learning.Adam(self, self.config)
            params = adam.run(X, t)

        if method in ("bfgs", "batch"):
            bfgs = learning.Batch(self, self.config)
            params = bfgs.run(X, t)

        self.set_params(params)

        self.prepare(X, t, params=None)

        self.xtrain = X

    def get_post_fmean(self, X, Z, params=None):
        return super().get_post_fmean(X, Z, params)

    def predict(self, Z, params=None):
        """
        Calculating posterior mean of model (function)

        Parameters
        ==========
        X: numpy.ndarray
            inputs
        Z: numpy.ndarray
            feature maps
        params: numpy.ndarray
            Parameters
        See also
        ========
        physbo.gp.inf.exact.get_post_fmean
        """

        if params is None:
            params = np.copy(self.params)

        if self.inf == "exact":
            post_fmu = inf.exact.get_post_fmean(self, self.xtrain, Z, params)

        return post_fmu

    def get_params(self, deep=True):
        mean = self.prior.mean
        cov = self.prior.cov
        config = self.config

        return {"lik": self.lik, "mean": mean, "cov": cov, "config": config}

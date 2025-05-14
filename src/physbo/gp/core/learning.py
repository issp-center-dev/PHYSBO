# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# coding=utf-8
import numpy as np
import scipy.optimize


class batch(object):
    """
    basis class for batch learning
    """

    def __init__(self, gp, config):
        """

        Parameters
        ----------
        gp : physbo.gp.core.model object
        config: physbo.misc.set_config object
        """

        self.gp = gp
        self.config = config

    def run(self, X, t):
        """
        Performing optimization using the L-BFGS-B algorithm

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        t: numpy.ndarray
            N-dimensional vector that represents the corresponding negative energy of search candidates.
        Returns
        -------
        numpy.ndarray
            The solution of the optimization.
        """
        batch_size = self.config.learning.batch_size
        sub_X, sub_t = self.gp.sub_sampling(X, t, batch_size)

        if self.config.learning.num_init_params_search != 0:
            is_init_params_search = True
        else:
            is_init_params_search = False

        if is_init_params_search:
            params = self.init_params_search(sub_X, sub_t)
        else:
            params = np.copy(self.gp.params)

        params = self.one_run(params, sub_X, sub_t)
        return params

    def one_run(self, params, X, t, max_iter=None):
        """

        Parameters
        ----------
        params: numpy.ndarray
            Initial guess for optimization.
            Array of real elements of size (n,), where ‘n’ is the number of independent variables.

        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        t: numpy.ndarray
            N-dimensional vector that represents the corresponding negative energy of search candidates.
        max_iter: int
            Maximum number of iterations to perform.
        Returns
        -------
        numpy.ndarray
            The solution of the optimization.
        """

        # is_disp: Set to True to print convergence messages.
        is_disp = True

        if max_iter is None:
            is_disp = self.config.learning.is_disp
            max_iter = int(self.config.learning.max_iter)

        args = (X, t)
        bound = self.gp.get_params_bound()
        res = scipy.optimize.minimize(
            fun=self.gp.eval_marlik,
            args=args,
            x0=params,
            method="L-BFGS-B",
            jac=self.gp.get_grad_marlik,
            bounds=bound,
            options={"disp": is_disp, "maxiter": max_iter},
        )

        return res.x

    def init_params_search(self, X, t):
        """

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        t: numpy.ndarray
            N-dimensional vector that represents the corresponding negative energy of search candidates.

        Returns
        -------
        numpy.ndarray
            The parameters which give the minimum marginal likelihood.
        """
        num_init_params_search = self.config.learning.num_init_params_search
        max_iter = int(self.config.learning.max_iter_init_params_search)
        min_params = np.zeros(self.gp.num_params)
        min_marlik = np.inf

        for i in range(num_init_params_search):
            params = self.gp.get_cand_params(X, t)
            params = self.one_run(params, X, t, max_iter)
            marlik = self.gp.eval_marlik(params, X, t)

            if min_marlik > marlik:
                min_marlik = marlik
                min_params = params

        # print 'minimum marginal likelihood = ', min_marlik
        return min_params


class online(object):
    """
    base class for online learning
    """

    def __init__(self, gp, config):
        """

        Parameters
        ----------
        gp : model  (gp.core.model)
        config: set_config  (misc.set_config)
        """
        self.gp = gp
        self.config = config
        self.num_iter = 0

    def run(self, X, t):
        """
        Run initial search and hyper parameter running.

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        t: numpy.ndarray
            N-dimensional vector that represents the corresponding negative energy of search candidates.

        Returns
        -------
        numpy.ndarray
            The solution of the optimization.

        """
        if self.config.learning.num_init_params_search != 0:
            is_init_params_search = True
        else:
            is_init_params_search = False

        is_disp = self.config.learning.is_disp
        if is_init_params_search:
            if is_disp:
                print("Start the initial hyper parameter searching ...")
            params = self.init_params_search(X, t)
            if is_disp:
                print("Done\n")
        else:
            params = np.copy(self.params)

        if is_disp:
            print("Start the hyper parameter learning ...")
        params = self.one_run(params, X, t)
        if is_disp:
            print("Done\n")

        return params

    def one_run(self, params, X, t, max_epoch=None, is_disp=False):
        """

        Parameters
        ----------
        params: numpy.ndarray
            Parameters for optimization.
            Array of real elements of size (n,), where ‘n’ is the number of independent variables.
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        t: numpy.ndarray
            N-dimensional vector that represents the corresponding negative energy of search candidates.
        max_epoch: int
            Maximum candidate epochs
        Returns
        -------
        numpy.ndarray
            The solution of the optimization.

        """
        num_data = X.shape[0]
        batch_size = self.config.learning.batch_size

        if batch_size > num_data:
            batch_size = num_data

        if max_epoch is None:
            max_epoch = self.config.learning.max_epoch
            is_disp = self.config.learning.is_disp

        num_disp = self.config.learning.num_disp
        eval_size = self.config.learning.eval_size
        eval_X, eval_t = self.gp.sub_sampling(X, t, eval_size)
        timing = range(0, max_epoch, int(np.floor(max_epoch / num_disp)))
        temp = 0

        for num_epoch in range(0, max_epoch):
            perm = np.random.permutation(num_data)

            if is_disp and temp < num_disp and num_epoch == timing[temp]:
                self.disp_marlik(params, eval_X, eval_t, num_epoch)
                temp += 1

            for n in range(0, num_data, batch_size):
                tmp_index = perm[n : n + batch_size]
                if len(tmp_index) == batch_size:
                    self.num_iter += 1
                    subX = X[tmp_index, :]
                    subt = t[tmp_index]
                    params += self.get_one_update(params, subX, subt)

        if is_disp:
            self.disp_marlik(params, eval_X, eval_t, num_epoch + 1)

        self.reset()
        return params

    def disp_marlik(self, params, eval_X, eval_t, num_epoch=None):
        """
        Displaying marginal likelihood

        Parameters
        ----------
        params: numpy.ndarray
            Parameters for optimization.
            Array of real elements of size (n,), where ‘n’ is the number of independent variables.
        eval_X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        eval_t: numpy.ndarray
            N-dimensional vector that represents the corresponding negative energy of search candidates.
        num_epoch: int
            Number of epochs

        Returns
        -------

        """
        marlik = self.gp.eval_marlik(params, eval_X, eval_t)
        if num_epoch is not None:
            print(num_epoch, end=" ")
            print("-th epoch", end=" ")

        print("marginal likelihood", marlik)

    def init_params_search(self, X, t):
        """
        Initial parameter searchs

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        t: numpy.ndarray
            N-dimensional vector that represents the corresponding negative energy of search candidates.

        Returns
        -------
        numpy.ndarray
            The parameter which gives the minimum likelihood.
        """
        num_init_params_search = self.config.learning.num_init_params_search
        is_disp = self.config.learning.is_disp
        max_epoch = self.config.learning.max_epoch_init_params_search
        eval_size = self.config.learning.eval_size
        eval_X, eval_t = self.gp.sub_sampling(X, t, eval_size)
        min_params = np.zeros(self.gp.num_params)
        min_marlik = np.inf

        for i in range(num_init_params_search):
            params = self.gp.get_cand_params(X, t)

            params = self.one_run(params, X, t, max_epoch)
            marlik = self.gp.eval_marlik(params, eval_X, eval_t)

            if min_marlik > marlik:
                min_marlik = marlik
                min_params = params

        # print 'minimum marginal likelihood = ', min_marlik
        return min_params

    def get_one_update(self, params, X, t):
        raise NotImplementedError


class adam(online):
    """default"""

    def __init__(self, gp, config):
        """

        Parameters
        ----------
        gp : physbo.gp.core.model object
        config: physbo.misc.set_config object
        """
        super(adam, self).__init__(gp, config)

        self.alpha = self.config.learning.alpha
        self.beta = self.config.learning.beta
        self.gamma = self.config.learning.gamma
        self.epsilon = self.config.learning.epsilon
        self.m = np.zeros(self.gp.num_params)
        self.v = np.zeros(self.gp.num_params)

    def reset(self):
        self.m = np.zeros(self.gp.num_params)
        self.v = np.zeros(self.gp.num_params)
        self.num_iter = 0

    def get_one_update(self, params, X, t):
        """

        Parameters
        ----------
        params: numpy.ndarray
            Parameters for optimization.
            Array of real elements of size (n,), where ‘n’ is the number of independent variables.
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
        t: numpy.ndarray
            N-dimensional vector that represents the corresponding negative energy of search candidates.
        Returns
        -------

        """
        grad = self.gp.get_grad_marlik(params, X, t)
        self.m = self.m * self.beta + grad * (1 - self.beta)
        self.v = self.v * self.gamma + grad**2 * (1 - self.gamma)
        hat_m = self.m / (1 - self.beta ** (self.num_iter))
        hat_v = self.v / (1 - self.gamma ** (self.num_iter))
        return -self.alpha * hat_m / (np.sqrt(hat_v) + self.epsilon)

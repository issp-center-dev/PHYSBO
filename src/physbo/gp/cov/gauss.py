# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# -*- coding:utf-8 -*-
import numpy as np
from scipy import spatial
from ._src.enhance_gauss import grad_width64


class gauss:
    """gaussian kernel"""

    def __init__(
        self,
        num_dim,
        width=3,
        scale=1,
        ard=False,
        max_width=1e6,
        min_width=1e-6,
        max_scale=1e6,
        min_scale=1e-6,
    ):
        """

        Parameters
        ----------
        num_dim: int
        width: float
        scale: float
        ard: bool
            flag to use Automatic Relevance Determination (ARD).
        max_width: float
            Maximum value of width
        min_width: float
            Minimum value of width
        max_scale: float
            Maximum value of scale
        min_scale: float
            Minimum value of scale
        """
        self.ard = ard
        self.num_dim = num_dim
        self.scale = scale
        self.max_ln_width = np.log(max_width)
        self.min_ln_width = np.log(min_width)
        self.max_ln_scale = np.log(max_scale)
        self.min_ln_scale = np.log(min_scale)

        if self.ard:
            # with ARD
            self.num_params = num_dim + 1
            if isinstance(width, np.ndarray) and len(width) == self.num_dim:
                self.width = width
            else:
                self.width = width * np.ones(self.num_dim)
        else:
            # without ARD
            self.width = width
            self.num_params = 2

        params = self.cat_params(self.width, self.scale)
        self.set_params(params)

    def print_params(self):
        """
        show the current kernel parameters

        """

        print(" Parameters of Gaussian kernel \n ")
        print(" width  = ", +self.width)
        print(" scale  = ", +self.scale)
        print(" scale2 = ", +self.scale**2)
        print(" \n")

    def prepare(self, params=None):
        """
        Setting parameters

        Parameters
        ----------
        params: numpy.ndarray
            parameters

        Returns
        -------
        params: numpy.ndarray
        width: int
        scale: int

        """
        if params is None:
            params = self.params
            width = self.width
            scale = self.scale
        else:
            params = self.supp_params(params)
            width, scale = self.decomp_params(params)

        return params, width, scale

    def get_grad(self, X, params=None):
        """
        Getting gradiant values of X

        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.

        params: numpy.ndarray

        Returns
        -------
        grad: numpy.ndarray

        """
        num_data = X.shape[0]
        params, width, scale = self.prepare(params)
        G = self.get_cov(X, params=params)

        grad = np.zeros((self.num_params, num_data, num_data))
        if self.ard:
            grad[0 : self.num_params - 1, :, :] = grad_width64(X, width, G)
        else:
            pairwise_dists = spatial.distance.pdist(X / width, "euclidean")
            grad[0, :, :] = G * spatial.distance.squareform(pairwise_dists**2)

        grad[-1, :, :] = 2 * G
        return grad

    def get_cov(self, X, Z=None, params=None, diag=False):
        """
        compute the covariant matrix
        Parameters
        ----------
        X: numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.

        Z: numpy.ndarray
            N x d dimensional matrix. Each row of Z denotes the d-dimensional feature vector of search candidate.

        params: numpy.ndarray
            Parameters

        diag: bool
            If true, only variances (diagonal elements) are returned.

        Returns
        -------
        G: numpy.ndarray
            covariant matrix
            Returned shape is (num_points) if diag=true, (num_points, num_points) if diag=false,
            where num_points is the number of points in X.
        """
        params, width, scale = self.prepare(params)
        scale2 = scale**2

        if Z is None:
            if diag:
                G = scale2 * np.ones(X.shape[0])
            else:
                pairwise_dists = spatial.distance.squareform(
                    spatial.distance.pdist(X / width, "euclidean") ** 2
                )
                G = np.exp(-0.5 * pairwise_dists) * scale2
        else:
            pairwise_dists = (
                spatial.distance.cdist(X / width, Z / width, "euclidean") ** 2
            )
            G = np.exp(-0.5 * pairwise_dists) * scale2

        return G

    def set_params(self, params):
        """
        set kernel parameters

        Parameters
        ----------
        params: numpy.ndarray
            Parameters for optimization.

        """
        params = self.supp_params(params)
        self.params = params
        self.width, self.scale = self.decomp_params(params)

    def supp_params(self, params):
        """
        Set maximum (minimum) values for parameters when the parameter is greater(less) than this value.

        Parameters
        ----------
        params: numpy.ndarray
            Parameters for optimization.
            Array of real elements of size (n,), where ‘n’ is the number of independent variables.

        Returns
        -------
        params: numpy.ndarray

        """
        index = np.where(params[0:-1] > self.max_ln_width)
        params[index[0]] = self.max_ln_width

        index = np.where(params[0:-1] < self.min_ln_width)
        params[index[0]] = self.min_ln_width

        if params[-1] > self.max_ln_scale:
            params[-1] = self.max_ln_scale

        if params[-1] < self.min_ln_scale:
            params[-1] = self.min_ln_scale

        return params

    def decomp_params(self, params):
        """
        decompose the parameters defined on the log region
        into width and scale parameters

        Parameters
        ----------
        params: numpy.ndarray
            parameters

        Returns
        -------
        width: float
        scale: float
        """

        width = np.exp(params[0:-1])
        scale = np.exp(params[-1])
        return width, scale

    def save(self, file_name):
        """
        save the gaussian kernel

        Parameters
        ----------
        file_name: str
            file name to save the information of the kernel

        """
        kwarg = {
            "name": "gauss",
            "params": self.params,
            "ard": self.ard,
            "num_dim": self.num_dim,
            "max_ln_scale": self.max_ln_scale,
            "min_ln_scale": self.min_ln_scale,
            "max_ln_width": self.max_ln_width,
            "min_ln_width": self.min_ln_width,
            "num_params": self.num_params,
        }
        with open(file_name, "wb") as f:
            np.savez(f, **kwarg)

    def load(self, file_name):
        """
        Recovering the Gaussian kernel from file
        Parameters
        ----------
         file_name: str
            file name to load the information of the kernel

        """
        temp = np.load(file_name)

        self.num_dim = temp["num_dim"]
        self.ard = temp["ard"]
        self.max_ln_scale = temp["max_ln_scale"]
        self.min_ln_scale = temp["min_ln_scale"]
        self.max_ln_width = temp["max_ln_width"]
        self.min_ln_width = temp["min_ln_width"]
        params = temp["params"]
        self.set_params(params)

    def get_params_bound(self):
        """
        Getting boundary array.

        Returns
        -------
        bound: list
            A num_params-dimensional array with the tuple (min_params, max_params).

        """

        if self.ard:
            bound = [
                (self.min_ln_width, self.max_ln_width) for i in range(0, self.num_dim)
            ]
        else:
            bound = [(self.min_ln_width, self.max_ln_width)]

        bound.append((self.min_ln_scale, self.max_ln_scale))
        return bound

    def cat_params(self, width, scale):
        """
        Taking the logarithm of width and scale parameters
        and concatinate them into one ndarray

        Parameters
        ----------
        width: int
        scale: int

        Returns
        -------
        params: numpy.ndarray
            Parameters
        """
        params = np.zeros(self.num_params)
        params[0:-1] = np.log(width)
        params[-1] = np.log(scale)
        return params

    def rand_expans(self, num_basis, params=None):
        """
        Kernel Expansion

        Parameters
        ----------
        num_basis: int
            total number of basis
        params: numpy.ndarray
            Parameters

        Returns
        -------
        tupple (W, b, amp)
        """
        params, width, scale = self.prepare(params)
        scale2 = scale**2
        amp = np.sqrt((2 * scale2) / num_basis)
        W = np.random.randn(num_basis, self.num_dim) / width
        b = np.random.rand(num_basis) * 2 * np.pi
        return (W, b, amp)

    def get_cand_params(self, X, t):
        """
        Getting candidate parameters.

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

        """
        if self.ard:
            # with ARD
            width = np.zeros(self.num_dim)
            scale = np.std(t)
            u = np.random.uniform(0.4, 0.8)
            width = u * (np.max(X, 0) - np.min(X, 0)) * np.sqrt(self.num_dim)

            index = np.where(np.abs(width) < 1e-6)
            width[index[0]] = 1e-6
            params = np.append(np.log(width), np.log(scale))
        else:
            # without ARD
            num_data = X.shape[0]
            M = max(2000, int(np.floor(num_data / 5)))

            dist = np.zeros(M)

            for m in range(M):
                a = np.random.randint(0, X.shape[0], 2)
                dist[m] = np.linalg.norm(X[a[0], :] - X[a[1], :])

            dist = np.sort(dist)
            tmp = int(np.floor(M / 10))
            n = np.random.randint(0, 5)
            width = dist[(2 * n + 1) * tmp]
            scale = np.std(t)
            params = np.append(np.log(width + 1e-8), np.log(scale))
        return params

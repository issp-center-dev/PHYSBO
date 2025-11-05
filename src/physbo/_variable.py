# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np


class Variable(object):
    """ Variable class

    Variable class represents a set of pairs of input (X) and output (t).

    """
    def __init__(self, X=None, t=None, Z=None):
        """

        Parameters
        ----------
        X:  numpy array
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of each search candidate.
        t:  numpy array
            N x k dimensional array, where k is the number of objectives.
            The negative energy of each search candidate (value of the objective function to be optimized).
        Z: numpy array

        """
        if X is not None:
            self.X = np.array(X)
        else:
            self.X = None
        if Z is not None:
            self.Z = np.array(Z)
        else:
            self.Z = None
        if t is not None:
            self.t = np.array(t)
        else:
            self.t = None
        self.check_shape()

    def check_shape(self):
        if self.X is not None:
            assert self.X.ndim == 2, "X must be a 2D array"
            nX = self.X.shape[0]
        else:
            nX = None
        if self.t is not None:
            assert self.t.ndim == 1 or self.t.ndim == 2, "t must be a 1D or 2D array"
            if self.t.ndim == 1:
                nt = len(self.t)
            else:
                nt = self.t.shape[0]
        else:
            nt = None
        if self.Z is not None:
            assert self.Z.ndim == 2, "Z must be a 2D array"
            nZ = self.Z.shape[0]
        else:
            nZ = None

        assert nX is None or nt is None or nX == nt, "The number of X and t must be the same"
        assert nX is None or nZ is None or nX == nZ, "The number of X and Z must be the same"
        assert nt is None or nZ is None or nt == nZ, "The number of t and Z must be the same"

    def __len__(self):
        if self.X is not None:
            return self.X.shape[0]
        else:
            return 0

    def get_subset(self, index):
        """
        Getting subset of variables.

        Parameters
        ----------
        index: int or array of int
            Index of selected action.
        Returns
        -------
        variable: physbo.variable
        """
        if isinstance(index, int):
            index = [index]
        temp_X = self.X[index, :] if self.X is not None else None
        if self.t is not None:
            if self.t.ndim == 1:
                temp_t = self.t[index]
            else:
                temp_t = self.t[index, :] if self.t is not None else None
        else:
            temp_t = None
        temp_Z = self.Z[index, :] if self.Z is not None else None

        return Variable(X=temp_X, t=temp_t, Z=temp_Z)

    def delete(self, num_row):
        """
        Deleting variables of X, t, Z whose indexes are specified by num_row.

        Parameters
        ----------
        num_row: numpy array
            Index array to be deleted.

        Returns
        -------

        """
        self.delete_X(num_row)
        self.delete_t(num_row)
        self.delete_Z(num_row)
        self.check_shape()

    def add(self, X=None, t=None, Z=None):
        """
        Adding variables of X, t, Z.

        Parameters
        ----------
        X:  numpy array
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of each search candidate.
        t:  numpy array
            N dimensional array (single-objective) or N x k dimensional matrix (multi-objective).
            The negative energy of each search candidate (value of the objective function to be optimized).
        Z

        Returns
        -------

        """
        self.add_X(X)
        self.add_t(t)
        self.add_Z(Z)
        self.check_shape()

    def delete_X(self, num_row):
        """
        Deleting variables of X whose indexes are specified by num_row.


        Parameters
        ----------
        num_row: numpy array
            Index array to be deleted.

        Returns
        -------

        """
        if self.X is not None:
            self.X = np.delete(self.X, num_row, 0)

    def delete_t(self, num_row):
        """
        Deleting variables of t whose indexes are specified by num_row.

        Parameters
        ----------
        num_row: numpy array
            Index array to be deleted.

        Returns
        -------

        """
        if self.t is not None:
            self.t = np.delete(self.t, num_row, axis=0)

    def delete_Z(self, num_row):
        """
        Deleting variables of Z whose indexes are specified by num_row.

        Parameters
        ----------
        num_row: numpy array
            Index array to be deleted.

        Returns
        -------

        """
        if self.Z is not None:
            self.Z = np.delete(self.Z, num_row, 0)

    def add_X(self, X=None):
        """
        Adding variable X. If self.X is None, self.X is set as X.

        Parameters
        ----------
        X:  numpy array
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of each search candidate.

        Returns
        -------

        """
        if X is not None:
            if self.X is not None:
                self.X = np.vstack((self.X, X))
            else:
                self.X = X

    def add_t(self, t=None):
        """
        Adding variable t. If self.t is None, self.t is set as t.

        Parameters
        ----------
        t:  numpy array
            N dimensional array (single-objective) or N x k dimensional matrix (multi-objective).
            The negative energy of each search candidate (value of the objective function to be optimized).

        Returns
        -------

        """
        if t is None:
            return

        if not isinstance(t, np.ndarray):
            t = np.array([t])

        # Ensure consistent shape for concatenation
        if self.t is not None:
            # Normalize shapes
            t_ndim = t.ndim
            self_t_ndim = self.t.ndim

            # Convert 1D to 2D if needed for consistency
            if self_t_ndim == 1 and t_ndim == 2:
                # self.t is 1D, t is 2D -> convert self.t to 2D
                # Match the number of columns from t
                self.t = self.t[:, np.newaxis]
                self_t_ndim = 2
            elif self_t_ndim == 2 and t_ndim == 1:
                # self.t is 2D, t is 1D -> convert t to 2D
                # Match the number of columns from self.t
                n_cols = self.t.shape[1]
                if n_cols == 1:
                    t = t[:, np.newaxis]
                else:
                    # For multi-objective, broadcast 1D to match number of columns
                    # This represents repeating the same value for all objectives
                    t = np.tile(t[:, np.newaxis], (1, n_cols))
                t_ndim = 2

            if self_t_ndim == 1:
                # Both 1D: use hstack for backward compatibility
                self.t = np.hstack((self.t, t))
            else:
                # Both 2D: use vstack
                self.t = np.vstack((self.t, t))
        else:
            self.t = t

    def add_Z(self, Z=None):
        """
        Adding variable Z. If self.Z is None, self.Z is set as Z.

        Parameters
        ----------
        Z

        Returns
        -------

        """
        if Z is not None:
            if self.Z is None:
                self.Z = Z
            else:
                self.Z = np.vstack((self.Z, Z))

    def save(self, file_name):
        """
        Saving variables X, t, Z to the file.

        Parameters
        ----------
        file_name: str
            A file name for saving variables X, t, Z using numpy.savez_compressed.

        Returns
        -------

        """
        np.savez_compressed(file_name, X=self.X, t=self.t, Z=self.Z, version=2)

    def load(self, file_name):
        """
        Loading variables X, t, Z from the file.

        Parameters
        ----------
        file_name: str
            A file name for loading variables X, t, Z using numpy.load.

        Returns
        -------

        """
        data = np.load(file_name, allow_pickle=True)
        version = data["version"]
        if version is None:
            version = 1
        else:
            version = int(version)
        old_t = version < 2
        self.X = data["X"]
        self.t = data["t"]
        self.Z = data["Z"]
        self.X = self.__load_helper(self.X)
        self.t = self.__load_helper(self.t, old_t=old_t)
        self.Z = self.__load_helper(self.Z)

        self.check_shape()


    def __load_helper(self, arr, old_t=False):
        if arr is None:
            return None
        if arr.ndim == 0:
            v = arr[()]
            if v is None:
                return None
            return np.array([[v]])
        if arr.ndim == 1:
            if old_t:
                return arr.reshape(-1,1)
            else:
                return arr.reshape(1,-1)
        if arr.ndim == 2:
            return arr
        raise ValueError(f'Invalid array dimension: {arr.ndim}')

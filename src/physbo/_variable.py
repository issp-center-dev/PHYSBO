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

    X: np.ndarray
    """ Points in the search space.
    N x d dimensional array, where N is the number of points and d is the dimension of the search space.
    """

    t: np.ndarray
    """ Values of the objective functions to be maximized for each point.
    N x k dimensional array, where N is the number of points and k is the number of objectives."""

    Z: np.ndarray
    """ Random basis representation of the search candidates for each objective.
    k x N x n dimensional array, where k is the number of objectives, N is the number of points, and n is the dimension of the random basis.
    """

    def __init__(self, X=None, t=None, Z=None):

        if X is not None:
            self.X = np.array(X)
        else:
            self.X = None

        self.t = _normalize_t(t)
        self.number_of_objectives = self.t.shape[1] if self.t is not None else None

        self.Z = _normalize_Z(Z, self.number_of_objectives)
        if self.number_of_objectives is None:
            self.number_of_objectives = self.Z.shape[0] if self.Z is not None else None

        self.check_shape()

    def check_shape(self):
        if self.X is not None:
            assert self.X.ndim == 2, "X must be a 2D array"
            nX = self.X.shape[0]
        else:
            nX = None

        if self.t is not None:
            assert self.t.ndim == 2, "t must be a 2D array"
            nt = self.t.shape[0]
            k = self.t.shape[1]
            if k != self.number_of_objectives:
                raise ValueError(f"The number of objectives in t ({k}) and the number of objectives in the variable ({self.number_of_objectives}) must be the same")
        else:
            nt = None
            k = None

        if self.Z is not None:
            assert self.Z.ndim == 3, "Z must be a 3D array (k, N, n)"
            kZ = self.Z.shape[0]
            nZ = self.Z.shape[1]
            if kZ != self.number_of_objectives:
                raise ValueError(f"The number of objectives in Z ({kZ}) and the number of objectives in the variable ({self.number_of_objectives}) must be the same")
        else:
            kZ = None
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
        if self.Z is not None:
            temp_Z = self.Z[:, index, :]
        else:
            temp_Z = None

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
        self._delete_X(num_row)
        self._delete_t(num_row)
        self._delete_Z(num_row)
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
        self._add_X(X)
        self._add_t(t)
        self._add_Z(Z)
        self.check_shape()

    def _delete_X(self, num_row):
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

    def _delete_t(self, num_row):
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

    def _delete_Z(self, num_row):
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
            # Z is (k, N, n), delete along axis=1 (N dimension)
            self.Z = np.delete(self.Z, num_row, axis=1)

    def _add_X(self, X=None):
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

    def _add_t(self, t=None):
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

        t = _normalize_t(t)

        # Ensure consistent shape for concatenation
        if self.t is not None:
            if t.shape[1] != self.number_of_objectives:
                raise ValueError(f"The number of objectives in t ({t.shape[1]}) and the number of objectives in the variable ({self.number_of_objectives}) must be the same")
            self.t = np.vstack((self.t, t))
        else:
            self.t = t
            self.number_of_objectives = self.t.shape[1]


    def _add_Z(self, Z=None):
        """
        Adding variable Z. If self.Z is None, self.Z is set as Z.

        Parameters
        ----------
        Z: numpy array
            (N, n) or (k, N, n) dimensional array. Will be normalized to (k, N, n) format.

        Returns
        -------

        """
        if Z is not None:
            Z = _normalize_Z(Z, self.number_of_objectives)
            if self.Z is None:
                self.Z = Z
                if self.number_of_objectives is None:
                    self.number_of_objectives = self.Z.shape[0]
            else:
                # Concatenate along axis=1 (N dimension)
                self.Z = np.concatenate((self.Z, Z), axis=1)

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
        np.savez_compressed(file_name, X=self.X, t=self.t, Z=self.Z, version=3)

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
        old_Z = version < 3
        self.X = data["X"]
        self.t = data["t"]
        self.Z = data["Z"]
        self.X = self.__load_helper(self.X)
        self.t = self.__load_helper(self.t, old_t=old_t)
        self.Z = self.__load_helper(self.Z, old_Z=old_Z)

        if self.t is not None:
            self.number_of_objectives = self.t.shape[1]
        elif self.Z is not None:
            self.number_of_objectives = self.Z.shape[0]
        else:
            self.number_of_objectives = None
        self.check_shape()

    def __load_helper(self, arr, old_t=False, old_Z=False):
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
            if old_Z:
                return arr[np.newaxis, :, :]
            else:
                return arr
        if arr.ndim == 3:
            # For Z: this is (k, N, n) format (version >= 3)
            return arr
        raise ValueError(f'Invalid array dimension: {arr.ndim}')


def _normalize_t(t, k=None):
    """
    Normalize t to always be a 2D array with shape (N, k).

    Parameters
    ----------
    t: scalar, numpy.ndarray, or None
        Input value(s) to normalize
    k: int
        Number of objectives
    Returns
    -------
    numpy.ndarray
        Normalized array with shape (N, k), or None if input is None
    """
    if t is None:
        return None

    t = np.array(t)

    # Handle scalar case
    if t.ndim == 0:
        return t.reshape(1, 1)

    # Handle 1D array: (N,) -> (N, 1)
    elif t.ndim == 1:
        return t.reshape(-1, 1)

    # Handle 2D array: (N, k), k should be 1
    elif t.ndim == 2:
        if k is None or k == 1:
            return t
        else:
            raise ValueError(f"given k is {k} > 1 but t is 2D")

    else:
        raise ValueError(f"Unexpected t shape: {t.shape}")


def _normalize_Z(Z, k=None):
    """
    Normalize Z to (k, N, n) format.

    Parameters
    ----------
    Z: numpy array
        (N, n) or (k, N, n) dimensional array
    k: int
        Number of objectives

    Returns
    -------
    Z: numpy array
        (k, N, n) dimensional array
    """
    if Z is None:
        return None

    Z = np.array(Z)

    # Convert Z to (k, N, n) format
    if Z.ndim == 2:
        if k is None or k == 1:
            Z = Z[np.newaxis, :, :]
        else:
            raise ValueError(f"given k is {k} > 1 but Z is 2D")
    elif Z.ndim == 3:
        # Already (k, N, n), check consistency
        if k is not None and Z.shape[0] != k:
            raise ValueError(f"Z.shape[0] ({Z.shape[0]}) must match t.shape[1] ({k})")
    else:
        raise ValueError(f"Z must be 2D (N, n) or 3D (k, N, n), got {Z.ndim}D")

    return Z

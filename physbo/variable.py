import numpy as np


class variable(object):
    def __init__(self, X=None, t=None, Z=None):
        """

        Parameters
        ----------
        X:  numpy array
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of each search candidate.
        t:  numpy array
            N dimensional array. The negative energy of each search candidate (value of the objective function to be optimized).
        Z:

        """
        self.X = X
        self.Z = Z
        self.t = t

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
        temp_X = self.X[index, :] if self.X is not None else None
        temp_t = self.t[index] if self.t is not None else None
        temp_Z = self.Z[index, :] if self.Z is not None else None

        return variable(X=temp_X, t=temp_t, Z=temp_Z)

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

    def add(self, X=None, t=None, Z=None):
        """
        Adding variables of X, t, Z.

        Parameters
        ----------
        X:  numpy array
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of each search candidate.
        t:  numpy array
            N dimensional array. The negative energy of each search candidate (value of the objective function to be optimized).
        Z

        Returns
        -------

        """
        self.add_X(X)
        self.add_t(t)
        self.add_Z(Z)

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
            self.t = np.delete(self.t, num_row)

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
            N dimensional array. The negative energy of each search candidate (value of the objective function to be optimized).

        Returns
        -------

        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        if t is not None:
            if self.t is not None:
                self.t = np.hstack((self.t, t))
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
        np.savez_compressed(file_name, X=self.X, t=self.t, Z=self.Z)

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
        self.X = data['X']
        self.t = data['t']
        self.Z = data['Z']

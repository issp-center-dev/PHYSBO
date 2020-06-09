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
        Z
        """
        self.X = X
        self.Z = Z
        self.t = t

    def get_subset(self, index):
        """

        Parameters
        ----------
        index: int
            Index of selected action.
        Returns
        -------
        variable: combo.variable
        """
        temp_X = self.X[index, :] if self.X is not None else None
        temp_t = self.t[index] if self.t is not None else None
        temp_Z = self.Z[index, :] if self.Z is not None else None

        return variable(X=temp_X, t=temp_t, Z=temp_Z)

    def delete(self, num_row):
        """

        Parameters
        ----------
        num_row

        Returns
        -------

        """
        self.delete_X(num_row)
        self.delete_t(num_row)
        self.delete_Z(num_row)

    def add(self, X=None, t=None, Z=None):
        """

        Parameters
        ----------
        X
        t
        Z

        Returns
        -------

        """
        self.add_X(X)
        self.add_t(t)
        self.add_Z(Z)

    def delete_X(self, num_row):
        """

        Parameters
        ----------
        num_row

        Returns
        -------

        """
        if self.X is not None:
            np.delete(self.X, num_row, 0)

    def delete_t(self, num_row):
        """

        Parameters
        ----------
        num_row

        Returns
        -------

        """
        if self.t is not None:
            np.delete(self.t, num_row)

    def delete_Z(self, num_row):
        """

        Parameters
        ----------
        num_row

        Returns
        -------

        """
        if self.Z is not None:
            np.delete(self.Z, num_row, 0)

    def add_X(self, X=None):
        """

        Parameters
        ----------
        X

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

        Parameters
        ----------
        t

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

        Parameters
        ----------
        file_name

        Returns
        -------

        """
        np.savez_compressed(file_name, X=self.X, t=self.t, Z=self.Z)

    def load(self, file_name):
        """

        Parameters
        ----------
        file_name

        Returns
        -------

        """
        data = np.load(file_name)
        self.X = data['X']
        self.t = data['t']
        self.Z = data['Z']

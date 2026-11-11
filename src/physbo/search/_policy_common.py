# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Code shared by the six policies.

The policies of the discrete and of the range search spaces, and those of
one, several and unified objectives, repeat a good part of their code.
What is literally the same is kept here: a function, and classes that only
hold methods (no state and no constructor of their own) and are listed
among the bases of the policies that share them.
"""

import numpy as np

from .. import _mpi
from .. import gp
from ..gp import Predictor as gp_predictor
from ..blm import Predictor as blm_predictor
from .._variable import Variable
from .._rng import LegacyRNG


def run_simulator(simulator, action, comm=None):
    """
    Run the simulator once for the proposed points of a step.

    In a parallel run the simulator is called on rank 0 only, and its
    result is broadcast; an exception it raises is raised on every process.

    Parameters
    ----------
    simulator: callable
        Function that takes the proposed points (indices of candidates in
        the discrete policies, coordinates in the range policies) and
        returns the value(s) of the objective function(s).
    action: numpy.ndarray
        The proposed points.
    comm: MPI.Comm, optional
        MPI communicator

    Returns
    -------
    The return value of the simulator, unchanged.
    """
    if comm is None:
        return simulator(action)
    if comm.rank == 0:
        t, error = _mpi.call(comm, simulator, action)
    else:
        t, error = 0.0, None
    return _mpi.bcast(comm, t, error, root=0)


class _PolicyCommon:
    """Methods that the policies of both search spaces share.

    The policies of the discrete and of the range search space list this
    class among their bases; the policies of several and of unified
    objectives get its methods through them.
    """

    @property
    def _comm(self):
        """The communicator for collective calls, or None in a serial run
        (also when a communicator of a single process was given)."""
        return self.mpicomm if self.mpisize > 1 else None

    def _init_predictor(self, is_rand_expans):
        """
        Initialize predictor.

        Parameters
        ----------
        is_rand_expans: bool
            If true, physbo.blm.predictor is selected.
            If false, physbo.gp.Predictor is selected.
        """
        if is_rand_expans:
            self.predictor = self._make_blm_predictor()
        else:
            self.predictor = self._make_gp_predictor()

    def _make_blm_predictor(self):
        """Create a BLM predictor, with ARD if self.ard is True."""
        ard = self.ard
        num_dim = self.get_num_dim()
        model = gp.core.Model.create_default(ard=ard, num_dim=num_dim)
        return blm_predictor(self.config, model=model)

    def _make_gp_predictor(self):
        """Create a GP predictor, with ARD if self.ard is True."""
        ard = self.ard
        num_dim = self.get_num_dim()
        model = gp.core.Model.create_default(ard=ard, num_dim=num_dim)
        return gp_predictor(self.config, model=model)

    def _make_variable_X(self, test_X):
        """
        Make a new *Variable* with X=test_X

        Parameters
        ----------
        test_X: numpy.ndarray or physbo.Variable
                The set of candidates. Each row vector represents the feature vector of each search candidate.
        Returns
        -------
        test_X: numpy.ndarray or physbo.Variable
                The set of candidates. Each row vector represents the feature vector of each search candidate.
        """
        if isinstance(test_X, np.ndarray):
            test = Variable(X=test_X)
        elif isinstance(test_X, Variable):
            test = test_X
        else:
            raise TypeError("The type of test_X must be ndarray or physbo.Variable")
        return test

    def _warn_no_predictor(self, method_name):
        if self.mpirank != 0:
            return
        print("Warning: Since policy.predictor is not yet set,")
        print("         a GP predictor (num_rand_basis=0) is used for predicting")
        print("         If you want to use a BLM predictor (num_rand_basis>0),")
        print("         call bayes_search(max_num_probes=0, num_rand_basis=nrb)")
        print("         before calling {}.".format(method_name))

    def export_history(self):
        """
        Returning the information of the history.

        Returns
        -------

        """
        return self.history

    def export_predictor(self):
        """
        Returning the predictor dataset

        Returns
        -------

        """
        return self.predictor

    def export_training(self):
        """
        Returning the training dataset

        Returns
        -------

        """
        return self.training

    def get_kernel_length_scale(self):
        """
        Return the Gaussian kernel length scale(s) (width) of the predictor.

        With ARD, returns one length scale per input dimension; otherwise a single value.

        Returns
        -------
        numpy.ndarray or None
            Length scale(s). Shape (num_dim,) when ARD is used, (1,) otherwise.
            None if the predictor is not set or not a GP with Gaussian kernel.
        """
        if self.predictor is None:
            return None
        self._update_predictor()
        try:
            cov = self.predictor.model.prior.cov
        except AttributeError:
            return None
        if not hasattr(cov, "width"):
            return None
        return np.atleast_1d(np.asarray(cov.width).flatten())

    def set_seed(self, seed):
        """
        Setting a seed parameter.

        In the legacy RNG mode (default), this seeds the global
        ``numpy.random`` state, as before. In the Generator mode
        (``rng=`` given as int or Generator), the policy-owned generator
        is re-created from the seed and the global state is not touched.

        Parameters
        ----------
        seed: int
            seed number
        -------

        """
        self.seed = seed
        if isinstance(self.rng, LegacyRNG):
            self.rng.seed(self.seed)
        else:
            self.rng = np.random.default_rng(self.seed)

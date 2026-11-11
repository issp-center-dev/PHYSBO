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

import pickle

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


class _MultiObjectiveMethods:
    """Methods that the multi-objective policies of both search spaces share.

    They keep one predictor per objective (``predictor_list``). The class
    is listed before the single-objective policy among the bases of
    ``discrete_multi.Policy`` and ``range_multi.Policy``, so that these
    methods replace the single-objective ones.
    """

    @staticmethod
    def _stack_basis(predictors, X):
        """The values at ``X`` of the basis functions of the predictors of
        the objectives, stacked into the (k, N, n) form of the training
        data, or None with Gaussian process predictors, which have no basis
        functions."""
        z = [p.get_basis(X) for p in predictors]
        return np.stack(z, axis=0) if z[0] is not None else None

    def get_post_fmean(self, xs):
        """
        Calculate mean value of predictors (post distribution)

        Parameters
        ----------
        xs: physbo.Variable or np.ndarray
            input parameters to calculate covariance
            shape is (num_points, num_parameters)
        diag: bool
            If true, only variances (diagonal elements) are returned.

        Returns
        -------
        fcov: numpy.ndarray
            Covariance matrix of the post distribution.
            Returned shape is (num_points, num_objectives).
        """
        if self.predictor_list == [None] * self.num_objectives:
            self._warn_no_predictor("get_post_fmean()")
            predictor_list = []
            for i in range(self.num_objectives):
                predictor = self._make_gp_predictor()
                predictor.fit(
                    self.training, 0, comm=self.mpicomm, objective_index=i, rng=self.rng
                )
                predictor.prepare(self.training, objective_index=i)
                predictor_list.append(predictor)
        else:
            self._update_predictor()
            predictor_list = self.predictor_list[:]
        X = self._make_variable_X(xs)
        fmean = [
            predictor.get_post_fmean(self.training, X, objective_index=i)
            for i, predictor in enumerate(predictor_list)
        ]
        return np.array(fmean).T

    def get_post_fcov(self, xs, diag=True):
        """
        Calculate covariance of predictors (post distribution)

        Parameters
        ----------
        xs: physbo.Variable or np.ndarray
            input parameters to calculate covariance
            shape is (num_points, num_parameters)
        diag: bool
            If true, only variances (diagonal elements) are returned.

        Returns
        -------
        fcov: numpy.ndarray
            Covariance matrix of the post distribution.
            Returned shape is (num_points, num_objectives) if diag=true, (num_points, num_points, num_objectives) if diag=false.
        """
        if self.predictor_list == [None] * self.num_objectives:
            self._warn_no_predictor("get_post_fcov()")
            predictor_list = []
            for i in range(self.num_objectives):
                predictor = self._make_gp_predictor()
                predictor.fit(
                    self.training, 0, comm=self.mpicomm, objective_index=i, rng=self.rng
                )
                predictor.prepare(self.training, objective_index=i)
                predictor_list.append(predictor)
        else:
            self._update_predictor()
            predictor_list = self.predictor_list[:]
        X = self._make_variable_X(xs)
        fcov = [
            predictor.get_post_fcov(self.training, X, diag, objective_index=i)
            for i, predictor in enumerate(predictor_list)
        ]
        arr = np.array(fcov)
        if diag:
            return arr.T
        else:
            return np.einsum("nij->ijn", arr)

    def get_kernel_length_scale(self):
        """
        Return the Gaussian kernel length scale(s) for each objective.

        Returns
        -------
        list of numpy.ndarray or None
            List of length num_objectives; each element is the length scale(s)
            for that objective's GP, or None if not available.
        """
        if self.predictor_list is None or all(p is None for p in self.predictor_list):
            return None
        self._update_predictor()
        result = []
        for predictor in self.predictor_list:
            if predictor is None:
                result.append(None)
                continue
            try:
                cov = predictor.model.prior.cov
            except AttributeError:
                result.append(None)
                continue
            if not hasattr(cov, "width"):
                result.append(None)
                continue
            result.append(np.atleast_1d(np.asarray(cov.width).flatten()))
        return result

    def get_permutation_importance(self, n_perm: int, split_features_parallel=False):
        """
        Calculate permutation importance of models

        Parameters
        ----------
        n_perm: int
            The number of permutations
        split_features_parallel: bool
            If true, split features in parallel.

        Returns
        -------
        importance_mean: numpy.ndarray
            importance_mean (num_parameters, num_objectives)
        importance_std: numpy.ndarray
            importance_std (num_parameters, num_objectives)
        """

        if self.predictor_list == [None] * self.num_objectives:
            self._warn_no_predictor("get_permutation_importance()")
            predictor_list = []
            for i in range(self.num_objectives):
                predictor = self._make_gp_predictor()
                predictor.fit(self.training, 0, objective_index=i, rng=self.rng)
                predictor.prepare(self.training, objective_index=i)
                predictor_list.append(predictor)
        else:
            self._update_predictor()
            predictor_list = self.predictor_list[:]

        importance_mean = [None for _ in range(self.num_objectives)]
        importance_std = [None for _ in range(self.num_objectives)]

        for i in range(self.num_objectives):
            importance_mean[i], importance_std[i] = predictor_list[
                i
            ].get_permutation_importance(
                self.training,
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
                objective_index=i,
                rng=self.rng,
            )

        return np.array(importance_mean).T, np.array(importance_std).T

    def save_predictor_list(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.predictor_list, f, 2)

    def save_training_list(self, file_name):
        obj = {"X": self.training.X, "t": self.training.t, "Z": self.training.Z}
        with open(file_name, "wb") as f:
            pickle.dump(obj, f, 2)

    def load_predictor_list(self, file_name):
        with open(file_name, "rb") as f:
            self.predictor_list = pickle.load(f)

    def load_training_list(self, file_name):
        with open(file_name, "rb") as f:
            data = pickle.load(f)

        # Handle both old format (list) and new format (dict/Variable)
        if isinstance(data, list):
            # Old format: list of dicts, convert to single Variable
            X = data[0]["X"]
            Z = np.stack([d["Z"] for d in data], axis=0)
            t = np.stack([d["t"] for d in data], axis=1)
            self.training = Variable(X=X, t=t, Z=Z)
        elif isinstance(data, dict):
            # New format: single dict
            self.training = Variable(X=data["X"], t=data["t"], Z=data["Z"])
        else:
            # Assume it's already a Variable
            self.training = data

    def _update_predictor(self):
        if self.new_data is not None:
            for i in range(self.num_objectives):
                self.predictor_list[i].update(
                    self.training, self.new_data, objective_index=i
                )
            self.new_data = None


class _UnifiedObjectiveMethods:
    """Methods that the unified-objective policies of both search spaces share.

    They turn the objectives into one with ``unify_method`` and keep a
    single predictor for it, with the unified training data in
    ``training_unified``. The class is listed before the single-objective
    policy among the bases of ``discrete_unified.Policy`` and
    ``range_unified.Policy``.
    """

    def _attach_unified_basis(self, variable):
        """Give ``variable`` the values of the basis functions of the unified
        predictor at its points, in the (1, N, n) form of the training data.
        Nothing is done with the Gaussian process predictor, which has no
        basis functions."""
        Z = self.predictor.get_basis(variable.X)
        if Z is not None:
            variable.Z = Z[np.newaxis, :, :]

    def _require_unified_training(self, method_name):
        """The unified training data, or RuntimeError when they are not built yet.

        They are built by bayes_search, which receives ``unify_method``; before
        the first bayes_search the policy cannot unify the objectives. load()
        restores neither them nor unify_method, so after load() bayes_search
        has to be called again as well.
        """
        if self.training_unified is None:
            raise RuntimeError(
                f"ERROR: {method_name} needs the unified training data, which are "
                "built by bayes_search with unify_method. Call bayes_search first "
                "(also after load(), which does not restore them); with "
                "max_num_probes=0 and the default interval it only learns the "
                "predictor without proposing a point."
            )
        return self.training_unified

    def _unify_training(self, training: Variable) -> Variable:
        """
        Wrapper of the unify_method function
        """
        t_unified = self.unify_method(training.t)
        return Variable(X=training.X, t=t_unified.reshape(-1, 1))

    def _update_predictor(self):
        if self.new_data is not None:
            self.training_unified = self._unify_training(self.training)
            # the unified data carry the values of the basis functions, as
            # after _learn_hyperparameter, so that virtual observations with
            # Z can be added to them (several proposals per step)
            self._attach_unified_basis(self.training_unified)
            N = self.training_unified.t.shape[0]
            n = self.new_data.t.shape[0]
            new_data_unified = self.training_unified.get_subset(np.arange(N - n, N))
            assert np.allclose(new_data_unified.X, self.new_data.X)
            self.predictor.update(self.training_unified, new_data_unified)
            self.new_data = None

    def get_permutation_importance(self, n_perm: int, split_features_parallel=False):
        """
        Calculating permutation importance of model

        Parameters
        ==========
        n_perm: int
            number of permutations
        split_features_parallel: bool
            If true, split features in parallel.

        Returns
        =======
        importance_mean: numpy.ndarray
            importance_mean
        importance_std: numpy.ndarray
            importance_std
        """

        self._require_unified_training("get_permutation_importance()")
        if self.predictor is None:
            self._warn_no_predictor("get_permutation_importance()")
            predictor = self._make_gp_predictor()
            predictor.fit(self.training_unified, 0, rng=self.rng)
            predictor.prepare(self.training_unified)
            return predictor.get_permutation_importance(
                self.training_unified,
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
                rng=self.rng,
            )
        else:
            self._update_predictor()
            return self.predictor.get_permutation_importance(
                self.training_unified,
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
                rng=self.rng,
            )

    def _learn_hyperparameter(self, num_rand_basis):
        self.training_unified = self._unify_training(self.training)

        self.predictor.fit(
            self.training_unified, num_rand_basis, comm=self.mpicomm, rng=self.rng
        )
        self.predictor.prepare(self.training_unified)
        self._attach_unified_basis(self.training_unified)
        self.new_data = None

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import copy
import pickle as pickle
import time

from ._history import History
from .. import range as range_single
from .. import utility
from .. import score_multi as search_score
from ..optimize.random import Optimizer as RandomOptimizer
from ...gp import Predictor as gp_predictor
from ...blm import Predictor as blm_predictor
from ...misc import SetConfig
from ..._variable import Variable


class Policy(range_single.Policy):
    """Multi objective Bayesian optimization with continuous search space"""

    def __init__(
        self,
        num_objectives,
        *,
        min_X=None,
        max_X=None,
        comm=None,
        config=None,
        initial_data=None,
    ):
        if min_X is None or max_X is None:
            raise ValueError("min_X and max_X must be specified")
        self.min_X = np.array(min_X)
        self.max_X = np.array(max_X)
        self.L_X = self.max_X - self.min_X
        self.dim = self.min_X.shape[0]

        self.num_objectives = num_objectives
        self.history = History(num_objectives=self.num_objectives, dim=self.dim)

        self.training = Variable()
        self.predictor_list = [None for _ in range(self.num_objectives)]
        self.new_data = None

        if config is None:
            self.config = SetConfig()
        else:
            self.config = config

        self.TS_candidate_num = None

        if initial_data is not None:
            if len(initial_data) != 2:
                msg = "ERROR: initial_data should be 2-elements tuple or list (actions and objectives)"
                raise RuntimeError(msg)
            init_X, fs = initial_data
            assert init_X.shape[0] == len(fs), (
                "The number of initial data must be the same"
            )
            assert init_X.shape[1] == self.dim, (
                "The dimension of initial_data[0] must be the same as the dimension of min_X and max_X"
            )

            ## TODO: add initial data to the history
            ## The following code is for discrete search
            ## self.write(actions, fs)
            ## self.actions = np.array(sorted(list(set(self.actions) - set(actions))))

        if comm is None:
            self.mpicomm = None
            self.mpisize = 1
            self.mpirank = 0
        else:
            self.mpicomm = comm
            self.mpisize = comm.size
            self.mpirank = comm.rank
            self.actions = np.array_split(self.actions, self.mpisize)[self.mpirank]

    def write(
        self,
        X,
        t,
        time_total=None,
        time_update_predictor=None,
        time_get_action=None,
        time_run_simulator=None,
    ):
        self.history.write(
            t,
            X,
            time_total=time_total,
            time_update_predictor=time_update_predictor,
            time_get_action=time_get_action,
            time_run_simulator=time_run_simulator,
        )
        t = np.array(t)

        # Ensure t is 2D: shape (N, num_objectives)
        if t.ndim == 1:
            if len(t) == self.num_objectives:
                t = t.reshape(1, -1)
            else:
                t = t.reshape(-1, 1)

        assert X.shape[0] == t.shape[0], "The number of X and t must be the same"
        assert X.shape[1] == self.dim, (
            "The dimension of X must be the same as the dimension of min_X and max_X"
        )

        if self.predictor_list[0] is not None:
            z = []
            for p in self.predictor_list:
                z.append(p.get_basis(X))
            if z[0] is not None:
                Z = np.stack(z, axis=0)
            else:
                Z = None
        else:
            Z = None

        if self.new_data is None:
            self.new_data = Variable(X=X, t=t, Z=Z)
        else:
            self.new_data.add(X=X, t=t, Z=Z)

        # Add to single training Variable with full 2D t matrix and (k, N, n) Z
        if self.training.X is None:
            self.training = Variable(X=X, t=t, Z=Z)
        else:
            self.training.add(X=X, t=t, Z=Z)

    def random_search(
        self,
        max_num_probes,
        num_search_each_probe=1,
        simulator=None,
        is_disp=True,
        disp_pareto_set=False,
    ):
        if self.mpirank != 0:
            is_disp = False

        N = int(num_search_each_probe)

        if is_disp:
            utility.show_interactive_mode(simulator, self.history)

        for n in range(0, max_num_probes):
            time_total = time.time()
            if is_disp and N > 1:
                utility.show_start_message_multi_search(
                    self.history.num_runs, score="random"
                )

            time_get_action = time.time()
            action_X = self._get_random_action(N)
            time_get_action = time.time() - time_get_action

            if simulator is None:
                return action_X

            time_run_simulator = time.time()
            t = _run_simulator(simulator, action_X, self.mpicomm)
            time_run_simulator = time.time() - time_run_simulator

            time_total = time.time() - time_total
            self.write(
                action_X,
                t,
                time_total=[time_total] * N,
                time_update_predictor=np.zeros(N, dtype=float),
                time_get_action=[time_get_action] * N,
                time_run_simulator=[time_run_simulator] * N,
            )

            if is_disp:
                utility.show_search_results_mo(
                    self.history, N, disp_pareto_set=disp_pareto_set
                )

        return copy.deepcopy(self.history)

    def bayes_search(
        self,
        training_list=None,
        max_num_probes=None,
        num_search_each_probe=1,
        predictor_list=None,
        is_disp=True,
        disp_pareto_set=False,
        simulator=None,
        score="HVPI",
        interval=0,
        num_rand_basis=0,
        optimizer=None,
    ):
        if self.mpirank != 0:
            is_disp = False

        old_disp = self.config.learning.is_disp
        self.config.learning.is_disp = is_disp

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        is_rand_expans = False if num_rand_basis == 0 else True

        if training_list is not None:
            # Convert old format (list of Variables) to new format (single Variable)
            if isinstance(training_list, list) and len(training_list) > 0:
                if isinstance(training_list[0], Variable):
                    # Old format: list of Variables, convert to single Variable
                    X = training_list[0].X
                    Z = training_list[0].Z
                    t = np.column_stack([tr.t for tr in training_list])
                    self.training = Variable(X=X, t=t, Z=Z)
                else:
                    self.training = training_list
            else:
                self.training = training_list

        if predictor_list is None:
            if is_rand_expans:
                self.predictor_list = [
                    blm_predictor(self.config) for i in range(self.num_objectives)
                ]
            else:
                self.predictor_list = [
                    gp_predictor(self.config) for i in range(self.num_objectives)
                ]
        else:
            self.predictor_list = predictor_list

        if max_num_probes == 0 and interval >= 0:
            self._learn_hyperparameter(num_rand_basis)

        if optimizer is None:
            optimizer = RandomOptimizer(
                min_X=self.min_X, max_X=self.max_X, nsamples=1000
            )

        N = int(num_search_each_probe)

        for n in range(max_num_probes):
            time_total = time.time()
            time_update_predictor = time.time()
            if utility.is_learning(n, interval):
                self._learn_hyperparameter(num_rand_basis)
            else:
                self._update_predictor()
            time_update_predictor = time.time() - time_update_predictor

            if num_search_each_probe != 1:
                utility.show_start_message_multi_search(self.history.num_runs, score)

            time_get_action = time.time()
            K = self.config.search.multi_probe_num_sampling
            alpha = self.config.search.alpha
            action_X = self._get_actions(score, N, K, alpha, optimizer=optimizer)
            time_get_action = time.time() - time_get_action

            if simulator is None:
                self.config.learning.is_disp = old_disp
                return action_X

            time_run_simulator = time.time()
            t = _run_simulator(simulator, action_X, self.mpicomm)
            time_run_simulator = time.time() - time_run_simulator

            time_total = time.time() - time_total
            self.write(
                action_X,
                t,
                time_total=[time_total] * N,
                time_update_predictor=[time_update_predictor] * N,
                time_get_action=[time_get_action] * N,
                time_run_simulator=[time_run_simulator] * N,
            )

            if is_disp:
                utility.show_search_results_mo(
                    self.history, N, disp_pareto_set=disp_pareto_set
                )
        self._update_predictor()
        self.config.learning.is_disp = old_disp
        return copy.deepcopy(self.history)

    def _argmax_score(self, mode, predictors, training, virtual_trainings, optimizer):
        """
        Get the action that maximizes the score.

        Arguments
        ----------
        mode: str
            The type of aquision funciton.
            TS (Thompson Sampling), EI (Expected Improvement) and PI (Probability of Improvement) are available.
            These functions are defined in score.py.
        predictors: list[Predictor]
            List of predictors.
        training: Variable
            Training data.
        virtual_trainings: list[Variable]
            List of extra training data.
        optimizer: Function or Optimizer object
            Optimizer object for optimizing the acquisition function.
        """

        K = len(virtual_trainings)
        if K == 0:
            for i, predictor in enumerate(predictors):
                predictor.prepare(training, objective_index=i)

            def fn(x):
                return self.get_score(
                    mode,
                    xs=x.reshape(1, -1),
                    predictor_list=predictors,
                    training_list=training,
                    parallel=False,
                )[0]
        else:  # marginal score
            trains_k = [copy.deepcopy(training) for _ in range(K)]
            predictors_k = [copy.deepcopy(predictors) for _ in range(K)]
            for predictor, training, virtual_training in zip(predictors_k, trains_k, virtual_trainings):
                training.add(X=virtual_training.X, t=virtual_training.t, Z=virtual_training.Z)
                for i in range(self.num_objectives):
                    predictor[i].update(training, virtual_training, objective_index=i)

            def fn(x):
                f = np.zeros(K)
                for k in range(K):
                    f[k] = self.get_score(
                        mode,
                        xs=x.reshape(1, -1),
                        predictor_list=predictors_k[k],
                        training_list=trains_k[k],
                        parallel=False,
                    )[0]
                return np.mean(f)

        X = optimizer(fn, mpicomm=self.mpicomm)
        return X

    def _get_actions(self, mode, N, K, alpha, optimizer, num_rand_basis=0):
        X = np.zeros((N, self.dim))
        self._update_predictor()
        predictors = [copy.deepcopy(predictor) for predictor in self.predictor_list]

        for i, predictor in enumerate(predictors):
            predictor.config.is_disp = False
            predictor.prepare(self.training, objective_index=i)
        X[0, :] = self._argmax_score(
            mode, predictors, self.training, [], optimizer=optimizer
        )

        for n in range(1, N):
            virtual_trainings = [
                Variable(X=X[0:n, :])
                for _ in range(K)
            ]
            virtual_t = np.zeros((K, n, self.num_objectives))
            for i in range(self.num_objectives):
                virtual_t[:, :, i] = predictors[i].get_predict_samples(
                    self.training, virtual_trainings[0], K, objective_index=i
                )
            for k in range(K):
                virtual_trainings[k].t = virtual_t[k, :, :]
            X[n, :] = self._argmax_score(
                mode,
                predictors,
                self.training,
                virtual_trainings,
                optimizer=optimizer,
            )
        return X

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
                predictor = gp_predictor(self.config)
                predictor.fit(self.training, 0, comm=self.mpicomm, objective_index=i)
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
                predictor = gp_predictor(self.config)
                predictor.fit(self.training, 0, comm=self.mpicomm, objective_index=i)
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

    def get_score(
        self,
        mode,
        *,
        xs=None,
        predictor_list=None,
        training_list=None,
        pareto=None,
        parallel=True,
        alpha=1,
    ):
        if training_list is None:
            training = self.training
        else:
            # Handle both old format (list) and new format (single Variable)
            if isinstance(training_list, list) and len(training_list) > 0:
                if isinstance(training_list[0], Variable):
                    # Old format: convert to single Variable
                    X = training_list[0].X
                    Z = training_list[0].Z
                    t = np.column_stack([tr.t for tr in training_list])
                    training = Variable(X=X, t=t, Z=Z)
                else:
                    training = training_list
            else:
                training = training_list

        if pareto is None:
            pareto = self.history.pareto

        if training.X is None or training.X.shape[0] == 0:
            msg = "ERROR: No training data is registered."
            raise RuntimeError(msg)

        if predictor_list is None:
            if self.predictor_list == [None] * self.num_objectives:
                self._warn_no_predictor("get_score()")
                predictor_list = []
                for i in range(self.num_objectives):
                    predictor = gp_predictor(self.config)
                    predictor.fit(training, 0, comm=self.mpicomm, objective_index=i)
                    predictor.prepare(training, objective_index=i)
                    predictor_list.append(predictor)
            else:
                self._update_predictor()
                predictor_list = self.predictor_list

        if xs is not None:
            if isinstance(xs, Variable):
                test = xs
            else:
                test = Variable(X=xs)
            if parallel and self.mpisize > 1:
                actions = np.array_split(np.arange(test.X.shape[0]), self.mpisize)
                test = test.get_subset(actions[self.mpirank])
        else:
            raise RuntimeError("ERROR: xs is not given")

        f = search_score.score(
            mode,
            predictor_list=predictor_list,
            training=training,
            test=test,
            pareto=pareto,
            reduced_candidate_num=self.TS_candidate_num,
            alpha=alpha,
        )
        if parallel and self.mpisize > 1:
            fs = self.mpicomm.allgather(f)
            f = np.hstack(fs)
        return f

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
            self._warn_no_predictor("get_post_fmean()")
            predictor_list = []
            for i in range(self.num_objectives):
                predictor = gp_predictor(self.config)
                predictor.fit(self.training, 0, objective_index=i)
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
            )

        return np.array(importance_mean).T, np.array(importance_std).T

    def save(self, file_history, file_training_list=None, file_predictor_list=None):
        if self.mpirank == 0:
            self.history.save(file_history)
            if file_training_list is not None:
                self.save_training_list(file_training_list)
            if file_predictor_list is not None:
                self.save_predictor_list(file_predictor_list)

    def load(self, file_history, file_training_list=None, file_predictor_list=None):
        self.history.load(file_history)

        if file_training_list is None:
            N = self.history.total_num_search
            X = self.history.action_X[0:N, :]
            t = self.history.fx[0:N, :]
            self.training = Variable(X=X, t=t)
        else:
            self.load_training_list(file_training_list)

        if file_predictor_list is not None:
            self.load_predictor_list(file_predictor_list)

        N = self.history.total_num_search

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
            Z = data[0]["Z"]
            t = np.column_stack([d["t"] for d in data])
            self.training = Variable(X=X, t=t, Z=Z)
        elif isinstance(data, dict):
            # New format: single dict
            self.training = Variable(X=data["X"], t=data["t"], Z=data["Z"])
        else:
            # Assume it's already a Variable
            self.training = data

    def _learn_hyperparameter(self, num_rand_basis):
        # Collect Z for each objective
        Z_list = []
        training = self.training
        for i in range(self.num_objectives):
            predictor = self.predictor_list[i]

            predictor.fit(training, num_rand_basis, comm=self.mpicomm, objective_index=i)

            # Collect Z for training (will be combined into (k, N, n))
            training_Z_basis = predictor.get_basis(training.X)
            Z_list.append(training_Z_basis)
            predictor.prepare(training, objective_index=i)

        # Update training.Z with (k, N, n) format
        if all(z is not None for z in Z_list):
            # Stack along first dimension: (k, N, n)
            self.training.Z = np.stack(Z_list, axis=0)  # Each Z_i is (N, n), stack to (k, N, n)
        self.new_data = None

    def _update_predictor(self):
        if self.new_data is not None:
            for i in range(self.num_objectives):
                self.predictor_list[i].update(
                    self.training, self.new_data, objective_index=i
                )
            self.new_data = None


def _run_simulator(simulator, action_X, comm=None):
    if comm is None:
        return simulator(action_X)
    if comm.rank == 0:
        t = simulator(action_X)
    else:
        t = 0.0
    return comm.bcast(t, root=0)

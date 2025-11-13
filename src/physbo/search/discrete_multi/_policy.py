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
from .. import discrete
from .. import utility
from .. import score_multi as search_score
from ...gp import Predictor as gp_predictor
from ...blm import Predictor as blm_predictor
from ...misc import SetConfig
from ..._variable import Variable


class Policy(discrete.Policy):
    """Multi objective Bayesian optimization with discrete search space"""

    def __init__(
        self, test_X, num_objectives, comm=None, config=None, initial_data=None
    ):
        self.num_objectives = num_objectives
        self.history = History(num_objectives=self.num_objectives)

        self.training = Variable()
        self.predictor_list = [None for _ in range(self.num_objectives)]
        self.test = self._make_variable_X(test_X)
        self.new_data = None

        self.actions = np.arange(0, test_X.shape[0])
        if config is None:
            self.config = SetConfig()
        else:
            self.config = config

        self.TS_candidate_num = None

        if initial_data is not None:
            if len(initial_data) != 2:
                msg = "ERROR: initial_data should be 2-elements tuple or list (actions and objectives)"
                raise RuntimeError(msg)
            actions, fs = initial_data
            if fs.shape[1] != self.num_objectives:
                msg = "ERROR: initial_data[1].shape[1] != num_objectives"
                raise RuntimeError(msg)
            if len(actions) != fs.shape[0]:
                msg = "ERROR: len(initial_data[0]) != initial_data[1].shape[0]"
                raise RuntimeError(msg)
            self.write(actions, fs)
            self.actions = np.array(sorted(list(set(self.actions) - set(actions))))

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
        action,
        t,
        X=None,
        time_total=None,
        time_update_predictor=None,
        time_get_action=None,
        time_run_simulator=None,
    ):
        self.history.write(
            t,
            action,
            time_total=time_total,
            time_update_predictor=time_update_predictor,
            time_get_action=time_get_action,
            time_run_simulator=time_run_simulator,
        )
        action = np.array(action)
        t = np.array(t)

        # Ensure t is 2D: shape (N, num_objectives)
        if t.ndim == 1:
            if len(t) == self.num_objectives:
                t = t.reshape(1, -1)
            else:
                t = t.reshape(-1, 1)

        # Determine X and Z (different for each objective)
        if X is None:
            X = self.test.X[action, :]
            Z = self.test.Z[:, action, :] if self.test.Z is not None else None
        else:
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

        # remove action from candidates if exists
        if len(self.actions) > 0:
            local_index = np.searchsorted(self.actions, action)
            local_index = local_index[
                np.take(self.actions, local_index, mode="clip") == action
            ]
            self.actions = self._delete_actions(local_index)

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
            action = self._get_random_action(N)
            time_get_action = time.time() - time_get_action

            if simulator is None:
                return action

            time_run_simulator = time.time()
            t = _run_simulator(simulator, action, self.mpicomm)
            time_run_simulator = time.time() - time_run_simulator

            time_total = time.time() - time_total
            self.write(
                action,
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
            self.training = _convert_list_of_variables_to_variable(training_list)

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
            action = self._get_actions(score, N, K, alpha)
            time_get_action = time.time() - time_get_action

            N_indeed = len(action)
            if N_indeed == 0:
                if self.mpirank == 0:
                    print("WARNING: All actions have already searched.")
                self.config.learning.is_disp = old_disp
                return copy.deepcopy(self.history)

            if simulator is None:
                self.config.learning.is_disp = old_disp
                return action

            time_run_simulator = time.time()
            t = _run_simulator(simulator, action, self.mpicomm)
            time_run_simulator = time.time() - time_run_simulator

            time_total = time.time() - time_total
            self.write(
                action,
                t,
                time_total=[time_total] * N_indeed,
                time_update_predictor=[time_update_predictor] * N_indeed,
                time_get_action=[time_get_action] * N_indeed,
                time_run_simulator=[time_run_simulator] * N_indeed,
            )

            if is_disp:
                utility.show_search_results_mo(
                    self.history, N, disp_pareto_set=disp_pareto_set
                )
        self._update_predictor()
        self.config.learning.is_disp = old_disp
        return copy.deepcopy(self.history)

    def _get_actions(self, mode, N, K, alpha):
        f = self.get_score(mode=mode, alpha=alpha, parallel=False)
        champion, local_champion, local_index = self._find_champion(f)
        if champion == -1:
            return np.zeros(0, dtype=int)
        if champion == local_champion:
            self.actions = self._delete_actions(local_index)

        chosen_actions = [champion]
        for n in range(1, N):
            f = self._get_marginal_score(mode, chosen_actions[0:n], K, alpha)
            champion, local_champion, local_index = self._find_champion(f)
            if champion == -1:
                break
            if champion == local_champion:
                self.actions = self._delete_actions(local_index)
            chosen_actions.append(champion)
        return np.array(chosen_actions)

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
        actions=None,
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
            training = _convert_list_of_variables_to_variable(training_list)

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
            if actions is not None:
                raise RuntimeError("ERROR: both actions and xs are given")
            if isinstance(xs, Variable):
                test = xs
            else:
                test = Variable(X=xs)
            if parallel and self.mpisize > 1:
                actions = np.array_split(np.arange(test.X.shape[0]), self.mpisize)
                test = test.get_subset(actions[self.mpirank])
        else:
            if actions is None:
                actions = self.actions
            else:
                if isinstance(actions, int):
                    actions = [actions]
                if parallel and self.mpisize > 1:
                    actions = np.array_split(actions, self.mpisize)[self.mpirank]
            test = self.test.get_subset(actions)

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

    def _get_marginal_score(self, mode, chosen_actions, K, alpha):
        """
        Getting marginal scores.

        Parameters
        ----------
        mode: str
            The type of aquision funciton.
            TS (Thompson Sampling), EI (Expected Improvement) and PI (Probability of Improvement) are available.
            These functions are defined in score.py.
        chosen_actions: numpy.ndarray
            Array of selected actions.
        K: int
            The total number of search candidates.
        alpha: float
            not used.

        Returns
        -------
        f: list
            N dimensional scores (score is defined in each mode)
        """
        f = np.zeros((K, len(self.actions)), dtype=float)

        N = len(chosen_actions)
        # draw K samples of the values of objective function of chosen actions
        new_test_local = self.test.get_subset(chosen_actions)
        virtual_t_local = np.zeros((K, N, self.num_objectives))
        for i in range(self.num_objectives):
            virtual_t_local[:, :, i] = self.predictor_list[i].get_predict_samples(
                self.training, new_test_local, K, objective_index=i
            )

        if self.mpisize == 1:
            new_test = new_test_local
            virtual_t = virtual_t_local
        else:
            new_test = Variable()
            virtual_t = np.zeros((K, 0, self.num_objectives))
            for nt in self.mpicomm.allgather(new_test_local):
                new_test.add(X=nt.X, t=nt.t, Z=nt.Z)
            for vt in self.mpicomm.allgather(virtual_t_local):
                virtual_t = np.concatenate((virtual_t, vt), axis=1)

        for k in range(K):
            predictor_list = [copy.deepcopy(p) for p in self.predictor_list]

            virtual_train = copy.deepcopy(new_test)
            virtual_train.t = virtual_t[k, :, :]

            training_k = copy.deepcopy(self.training)
            training_k.add(X=virtual_train.X, t=virtual_train.t, Z=virtual_train.Z)

            for i in range(self.num_objectives):
                predictor_list[i].update(training_k, virtual_train, objective_index=i)

            f[k, :] = self.get_score(
                mode,
                predictor_list=predictor_list,
                training_list=training_k,
                parallel=False,
            )
        return np.mean(f, axis=0)

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
            X = self.test.X[self.history.chosen_actions[0:N], :]
            t = self.history.fx[0:N]
            self.training = Variable(X=X, t=t)
        else:
            self.load_training_list(file_training_list)

        if file_predictor_list is not None:
            self.load_predictor_list(file_predictor_list)

        N = self.history.total_num_search

        visited = self.history.chosen_actions[:N]
        local_index = np.searchsorted(self.actions, visited)
        local_index = local_index[
            np.take(self.actions, local_index, mode="clip") == visited
        ]
        self.actions = self._delete_actions(local_index)

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

    def _learn_hyperparameter(self, num_rand_basis):
        # Collect Z for each objective
        training_Z_list = []
        test_Z_list = []
        for i in range(self.num_objectives):
            predictor = self.predictor_list[i]

            predictor.fit(
                self.training, num_rand_basis, comm=self.mpicomm, objective_index=i
            )
            # Get basis for this objective
            test_Z_basis = predictor.get_basis(self.test.X)
            training_Z_basis = predictor.get_basis(self.training.X)

            # Collect Z for test and training (will be combined into (k, N, n))
            test_Z_list.append(test_Z_basis)
            training_Z_list.append(training_Z_basis)

        # Update test.Z and training.Z with (k, N, n) format
        if all(z is not None for z in test_Z_list):
            self.test.Z = np.stack(
                test_Z_list, axis=0
            )  # Each Z_i is (N, n), stack to (k, N, n)
        if all(z is not None for z in training_Z_list):
            self.training.Z = np.stack(
                training_Z_list, axis=0
            )  # Each Z_i is (N, n), stack to (k, N, n)

        for i in range(self.num_objectives):
            self.predictor_list[i].prepare(self.training, objective_index=i)
        self.new_data = None

    def _update_predictor(self):
        if self.new_data is not None:
            for i in range(self.num_objectives):
                self.predictor_list[i].update(
                    self.training, self.new_data, objective_index=i
                )
            self.new_data = None


def _run_simulator(simulator, action, comm=None):
    if comm is None:
        return simulator(action)
    if comm.rank == 0:
        t = simulator(action)
    else:
        t = 0.0
    return comm.bcast(t, root=0)


def _convert_list_of_variables_to_variable(variables):
    """For backward compatibility"""
    if variables is not None:
        # Convert old format (list of Variables) to new format (single Variable)
        if isinstance(variables, list) and len(variables) > 0:
            if isinstance(variables[0], Variable):
                # Old format: list of Variables, convert to single Variable
                X = variables[0].X
                Z = variables[0].Z
                t = np.column_stack([tr.t for tr in variables])
                return Variable(X=X, t=t, Z=Z)
            else:
                return variables
        else:
            return variables

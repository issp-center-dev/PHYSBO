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

from typing import List, Optional


class Policy(range_single.Policy):
    """Multi objective Bayesian optimization with continuous search space"""
    new_data_list: List[Optional[Variable]]

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

        self.training_list = [Variable() for _ in range(self.num_objectives)]
        self.predictor_list = [None for _ in range(self.num_objectives)]
        self.new_data_list = [None for _ in range(self.num_objectives)]

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

        assert X.shape[0] == len(t), "The number of X and t must be the same"
        assert X.shape[1] == self.dim, (
            "The dimension of X must be the same as the dimension of min_X and max_X"
        )

        for i in range(self.num_objectives):
            predictor = self.predictor_list[i]
            Z = predictor.get_basis(X) if predictor is not None else None

            if self.new_data_list[i] is None:
                self.new_data_list[i] = Variable(X, t[:, i], Z)
            else:
                self.new_data_list[i].add(X=X, t=t[:, i], Z=Z)
            self.training_list[i].add(X=X, t=t[:, i], Z=Z)

    def _model(self, i):
        training = self.training_list[i]
        predictor = self.predictor_list[i]
        # test = self.test_list[i]
        new_data = self.new_data_list[i]
        return {
            "training": training,
            "predictor": predictor,
            # "test": test,
            "new_data": new_data,
        }

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
            self.training_list = training_list

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

    def _argmax_score(self, mode, predictors, trainings, extra_trainings, optimizer):
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
        trainings: list[Variable]
            List of training data.
        extra_trainings: list[list[Variable]]
            List of extra training data.
            The outermost list is for the sample index, and the inner list is for the objective index.
        optimizer: Function or Optimizer object
            Optimizer object for optimizing the acquisition function.

        """
        K = len(extra_trainings)
        if K == 0:
            for predictor, training in zip(predictors, trainings):
                predictor.prepare(training)

            def fn(x):
                return self.get_score(
                    mode,
                    xs=x.reshape(1, -1),
                    predictor_list=predictors,
                    training_list=trainings,
                    parallel=False,
                )[0]
        else:  # marginal score
            trains = [copy.deepcopy(training) for _ in range(K)]
            predictors = [copy.deepcopy(predictors) for _ in range(K)]
            for k in range(K):
                extra_train = extra_trainings[k]
                for i in range(self.num_objectives):
                    trains[k][i].add(X=extra_train[i].X, t=extra_train[i].t)
                    predictors[k][i].update(trains[k][i], extra_train[i])

            def fn(x):
                f = np.zeros(K)
                for k in range(K):
                    f[k] = self.get_score(
                        mode,
                        xs=x.reshape(1, -1),
                        predictor_list=predictors,
                        training_list=trains,
                        parallel=False,
                    )[0]
                return np.mean(f)

        X = optimizer(fn, mpicomm=self.mpicomm)
        return X

    def _get_actions(self, mode, N, K, alpha, optimizer, num_rand_basis=0):
        X = np.zeros((N, self.dim))
        self._update_predictor()
        predictors = [copy.deepcopy(predictor) for predictor in self.predictor_list]

        for predictor, training in zip(predictors, self.training_list):
            predictor.config.is_disp = False
            predictor.prepare(training)
        X[0, :] = self._argmax_score(
            mode, predictors, self.training_list, [], optimizer=optimizer
        )

        for n in range(1, N):
            extra_trainings_list_of_K = []
            ts = [
                predictor.get_predict_samples(self.training_list[i], X[0:n, :], K)
                for i in range(self.num_objectives)
            ]

            for k in range(K):
                et_list = [
                    copy.deepcopy(Variable(X=X[0:n, :]))
                    for _ in range(self.num_objectives)
                ]
                for i in range(self.num_objectives):
                    et_list[i].t = ts[i][k, :]
                extra_trainings_list_of_K.append(et_list)
            X[n, :] = self._argmax_score(
                mode,
                predictors,
                self.training_list,
                extra_trainings_list_of_K,
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
                predictor.fit(self.training_list[i], 0, comm=self.mpicomm)
                predictor.prepare(self.training_list[i])
                predictor_list.append(predictor)
        else:
            self._update_predictor()
            predictor_list = self.predictor_list[:]
        X = self._make_variable_X(xs)
        fmean = [
            predictor.get_post_fmean(training, X)
            for predictor, training in zip(predictor_list, self.training_list)
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
                predictor.fit(self.training_list[i], 0, comm=self.mpicomm)
                predictor.prepare(self.training_list[i])
                predictor_list.append(predictor)
        else:
            self._update_predictor()
            predictor_list = self.predictor_list[:]
        X = self._make_variable_X(xs)
        fcov = [
            predictor.get_post_fcov(training, X, diag)
            for predictor, training in zip(predictor_list, self.training_list)
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
            training_list = self.training_list
        if pareto is None:
            pareto = self.history.pareto

        if training_list[0].X is None or training_list[0].X.shape[0] == 0:
            msg = "ERROR: No training data is registered."
            raise RuntimeError(msg)

        if predictor_list is None:
            if self.predictor_list == [None] * self.num_objectives:
                self._warn_no_predictor("get_score()")
                predictor_list = []
                for i in range(self.num_objectives):
                    predictor = gp_predictor(self.config)
                    predictor.fit(training_list[i], 0, comm=self.mpicomm)
                    predictor.prepare(training_list[i])
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
            training_list=training_list,
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
                predictor.fit(self.training_list[i], 0)
                predictor.prepare(self.training_list[i])
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
                self.training_list[i],
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
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
            self.training_list = [
                Variable(X=X, t=t[:, i]) for i in range(self.num_objectives)
            ]
        else:
            self.load_training_list(file_training_list)

        if file_predictor_list is not None:
            self.load_predictor_list(file_predictor_list)

        N = self.history.total_num_search

    def save_predictor_list(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.predictor_list, f, 2)

    def save_training_list(self, file_name):
        obj = [
            {"X": training.X, "t": training.t, "Z": training.Z}
            for training in self.training_list
        ]
        with open(file_name, "wb") as f:
            pickle.dump(obj, f, 2)

    def load_predictor_list(self, file_name):
        with open(file_name, "rb") as f:
            self.predictor_list = pickle.load(f)

    def load_training_list(self, file_name):
        with open(file_name, "rb") as f:
            data_list = pickle.load(f)

        self.training_list = [Variable() for i in range(self.num_objectives)]
        for data, training in zip(data_list, self.training_list):
            training.X = data["X"]
            training.t = data["t"]
            training.Z = data["Z"]

    def _learn_hyperparameter(self, num_rand_basis):
        for i in range(self.num_objectives):
            m = self._model(i)
            predictor = m["predictor"]
            training = m["training"]

            predictor.fit(training, num_rand_basis, comm=self.mpicomm)
            training.Z = predictor.get_basis(training.X)
            predictor.prepare(training)
            self.new_data_list[i] = None

    def _update_predictor(self):
        for i in range(self.num_objectives):
            if self.new_data_list[i] is not None:
                self.predictor_list[i].update(
                    self.training_list[i], self.new_data_list[i]
                )
                self.new_data_list[i] = None


def _run_simulator(simulator, action_X, comm=None):
    if comm is None:
        return simulator(action_X)
    if comm.rank == 0:
        t = simulator(action_X)
    else:
        t = 0.0
    return comm.bcast(t, root=0)

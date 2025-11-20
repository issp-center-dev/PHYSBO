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
from .. import score as search_score
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
        self.training_unified = None
        self.predictor = None
        self.new_data = None

        if config is None:
            self.config = SetConfig()
        else:
            self.config = config

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
            assert fs.shape[1] == self.num_objectives, (
                "The number of objectives in initial_data[1] must be the same as num_objectives"
            )

            self.write(init_X, fs)

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
        N = X.shape[0]
        t = np.array(t)

        # Ensure t is 2D: shape (N, num_objectives)
        if t.ndim == 1:
            if N == 1:
                t = t.reshape(1, -1)
            else:
                raise ValueError(f"Number of actions is {N} > 1, but t is 1D array")

        assert X.shape[0] == t.shape[0], "The number of X and t must be the same"
        assert X.shape[1] == self.dim, (
            "The dimension of X must be the same as the dimension of min_X and max_X"
        )
        assert t.shape[1] == self.num_objectives, (
            "The number of objectives in t must be the same as num_objectives"
        )

        if self.new_data is None:
            self.new_data = Variable(X=X, t=t, Z=None)
        else:
            self.new_data.add(X=X, t=t, Z=None)

        # Add to single training Variable with full 2D t matrix and (k, N, n) Z
        if self.training.X is None:
            self.training = Variable(X=X, t=t, Z=None)
        else:
            self.training.add(X=X, t=t, Z=None)

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
        predictor=None,
        is_disp=True,
        disp_pareto_set=False,
        simulator=None,
        score="EI",
        unify_method=None,
        interval=0,
        num_rand_basis=0,
        optimizer=None,
    ):
        assert unify_method is not None, "unify_method must be provided"
        self.unify_method = unify_method

        if self.mpirank != 0:
            is_disp = False

        old_disp = self.config.learning.is_disp
        self.config.learning.is_disp = is_disp

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        is_rand_expans = False if num_rand_basis == 0 else True

        if training_list is not None:
            self.training = training_list

        if predictor is None:
            if is_rand_expans:
                self.predictor = blm_predictor(self.config)
            else:
                self.predictor = gp_predictor(self.config)
        else:
            self.predictor = predictor

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

    def _argmax_score(self, mode, predictor, training, extra_trainings, optimizer):
        K = len(extra_trainings)
        if K == 0:
            predictor.prepare(training, objective_index=0)

            def fn(x):
                return self.get_score(
                    mode, xs=x.reshape(1, -1), predictor=predictor, parallel=False
                )[0]
        else:  # marginal score
            trains = [copy.deepcopy(training) for _ in range(K)]
            predictors = [copy.deepcopy(predictor) for _ in range(K)]
            for k in range(K):
                extra_train = extra_trainings[k]
                trains[k].add(X=extra_train.X, t=extra_train.t)
                predictors[k].update(trains[k], extra_train, objective_index=0)

            def fn(x):
                f = np.zeros(K)
                for k in range(K):
                    f[k] = self.get_score(
                        mode,
                        xs=x.reshape(1, -1),
                        predictor=predictors[k],
                        training=trains[k],
                        parallel=False,
                    )[0]
                return np.mean(f)

        X = optimizer(fn, mpicomm=self.mpicomm)
        return X

    def __argmax_score(self, mode, predictors, training, virtual_trainings, optimizer):
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
            for predictor, training, virtual_training in zip(
                predictors_k, trains_k, virtual_trainings
            ):
                training.add(
                    X=virtual_training.X, t=virtual_training.t, Z=virtual_training.Z
                )
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
        """
        Getting next candidates

        Parameters
        ----------
        mode: str
            The type of aquisition funciton.
            TS (Thompson Sampling), EI (Expected Improvement) and PI (Probability of Improvement) are available.
            These functions are defined in score.py.
        N: int
            The total number of actions to return.
        K: int
            The total number of samples to evaluate marginal score
        alpha: float
            Tuning parameter which is used if mode = TS.
            In TS, multi variation is tuned as np.random.multivariate_normal(mean, cov*alpha**2, size).

        Returns
        -------
        chosen_actions: numpy.ndarray
            An N-dimensional array of actions selected in each search process.
        """

        X = np.zeros((N, self.dim))

        self._update_predictor()
        predictor = copy.deepcopy(self.predictor)
        predictor.config.is_disp = False
        X[0, :] = self._argmax_score(
            mode, predictor, self.training_unified, [], optimizer=optimizer
        )

        for n in range(1, N):
            extra_training = Variable(X=X[0:n, :])
            t = self.predictor.get_predict_samples(
                self.training_unified, extra_training, K, objective_index=0
            )
            extra_trainings = [copy.deepcopy(extra_training) for _ in range(K)]
            for k in range(K):
                # Normalize t to (N, 1) shape
                extra_trainings[k].t = t[k, :].reshape(-1, 1)
            X[n, :] = self._argmax_score(
                mode,
                predictor,
                self.training_unified,
                extra_trainings,
                optimizer=optimizer,
            )

        return X

    def get_post_fmean(self, xs):
        """
        Calculate mean value of predictor (post distribution)

        Parameters
        ----------
        xs: physbo.Variable or np.ndarray
            input parameters to calculate mean value
            shape is (num_points, num_parameters)

        Returns
        -------
        fmean: numpy.ndarray
            Mean value of the post distribution.
            Returned shape is (num_points).
        """
        X = self._make_variable_X(xs)
        if self.predictor is None:
            self._warn_no_predictor("get_post_fmean()")
            predictor = gp_predictor(self.config)
            predictor.fit(
                self.training_unified, 0, comm=self.mpicomm, objective_index=0
            )
            predictor.prepare(self.training, objective_index=0)
            return predictor.get_post_fmean(self.training_unified, X, objective_index=0)
        else:
            self._update_predictor()
            return self.predictor.get_post_fmean(
                self.training_unified, X, objective_index=0
            )

    def get_post_fcov(self, xs, diag=True):
        """
        Calculate covariance of predictor (post distribution)

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
            Returned shape is (num_points) if diag=true, (num_points, num_points) if diag=false.
        """
        X = self._make_variable_X(xs)
        if self.predictor is None:
            self._warn_no_predictor("get_post_fcov()")
            predictor = gp_predictor(self.config)
            predictor.fit(
                self.training_unified, 0, comm=self.mpicomm, objective_index=0
            )
            predictor.prepare(self.training_unified, objective_index=0)
            return predictor.get_post_fcov(
                self.training_unified, X, diag, objective_index=0
            )
        else:
            self._update_predictor()
            return self.predictor.get_post_fcov(
                self.training_unified, X, diag, objective_index=0
            )

    def get_score(
        self,
        mode,
        *,
        xs=None,
        predictor=None,
        training=None,
        pareto=None,
        parallel=True,
        alpha=1,
    ):
        if training is None:
            training = self.training_unified

        if pareto is None:
            pareto = self.history.pareto

        if training.X is None or training.X.shape[0] == 0:
            msg = "ERROR: No training data is registered."
            raise RuntimeError(msg)

        if predictor is None:
            if self.predictor is None:
                self._warn_no_predictor("get_score()")
                predictor = gp_predictor(self.config)
                predictor.fit(training, 0, comm=self.mpicomm, objective_index=0)
                predictor.prepare(training, objective_index=0)
            else:
                self._update_predictor()
                predictor = self.predictor

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
            predictor=predictor,
            training=training,
            test=test,
            alpha=alpha,
        )
        if parallel and self.mpisize > 1:
            fs = self.mpicomm.allgather(f)
            f = np.hstack(fs)
        return f

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

        if self.predictor is None:
            self._warn_no_predictor("get_permutation_importance()")
            predictor = gp_predictor(self.config)
            predictor.fit(self.training_unified, 0)
            predictor.prepare(self.training_unified)
            return predictor.get_permutation_importance(
                self.training_unified,
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
            )
        else:
            self._update_predictor()
            return self.predictor.get_permutation_importance(
                self.training_unified,
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
            )

    def save(self, file_history, file_training=None, file_predictor=None):
        if self.mpirank == 0:
            self.history.save(file_history)
            if file_training is not None:
                self.save_training(file_training)
            if file_predictor is not None:
                self.save_predictor(file_predictor)

    def load(self, file_history, file_training=None, file_predictor=None):
        self.history.load(file_history)

        if file_training is None:
            N = self.history.total_num_search
            X = self.history.action_X[0:N, :]
            t = self.history.fx[0:N, :]
            self.training = Variable(X=X, t=t)
        else:
            self.load_training(file_training)

        if file_predictor is not None:
            self.load_predictor(file_predictor)

        N = self.history.total_num_search

    def save_predictor(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.predictor, f, 2)

    def save_training(self, file_name):
        obj = {"X": self.training.X, "t": self.training.t, "Z": self.training.Z}
        with open(file_name, "wb") as f:
            pickle.dump(obj, f, 2)

    def load_predictor(self, file_name):
        with open(file_name, "rb") as f:
            self.predictor = pickle.load(f)

    def load_training(self, file_name):
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
        self.training_unified = self._unify_training(self.training)
        self.predictor.fit(self.training_unified, num_rand_basis, comm=self.mpicomm)
        self.predictor.prepare(self.training_unified)
        Z = self.predictor.get_basis(self.training_unified.X)
        if Z is not None:
            self.training_unified.Z = Z[np.newaxis, :, :]
        self.new_data = None

    def _update_predictor(self):
        if self.new_data is not None:
            self.training_unified = self._unify_training(self.training)
            N = self.training_unified.t.shape[0]
            n = self.new_data.t.shape[0]
            new_data_unified = self.training_unified.get_subset(np.arange(N - n, N))
            assert np.allclose(new_data_unified.X, self.new_data.X)
            self.predictor.update(self.training_unified, new_data_unified)
            self.new_data = None

    def _unify_training(self, training: Variable) -> Variable:
        """
        Wrapper of the unify_method function
        """
        t_unified = self.unify_method(training.t)
        return Variable(X=training.X, t=t_unified.reshape(-1, 1))


def _run_simulator(simulator, action_X, comm=None):
    if comm is None:
        return simulator(action_X)
    if comm.rank == 0:
        t = simulator(action_X)
    else:
        t = 0.0
    return comm.bcast(t, root=0)

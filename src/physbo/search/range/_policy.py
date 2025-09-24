# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import copy
import pickle as pickle
import time

from ._history import History
from .. import utility
from .. import score as search_score
from ..optimize.random import Optimizer as RandomOptimizer
from ...gp import Predictor as gp_predictor
from ...blm import Predictor as blm_predictor
from ...misc import SetConfig
from ..._variable import Variable


class Policy:
    """Single objective Bayesian optimization with continuous search space"""
    def __init__(
        self, *, min_X=None, max_X=None, config=None, initial_data=None, comm=None
    ):
        """

        Parameters
        ----------
        min_X: numpy.ndarray
            The minimum value of each dimension of the search space.
        max_X: numpy.ndarray
            The maximum value of each dimension of the search space.
        config: SetConfig object (physbo.misc.SetConfig)
        initial_data: tuple[np.ndarray, np.ndarray]
            The initial training datasets.
            The first elements is the array of inputs and the second is the array of values of objective functions
        comm: MPI.Comm, optional
            MPI Communicator
        """

        self.predictor = None
        self.training = Variable()
        self.new_data = None

        if min_X is None or max_X is None:
            raise ValueError("min_X and max_X must be specified")
        self.min_X = np.array(min_X)
        self.max_X = np.array(max_X)
        self.dim = self.min_X.shape[0]
        assert self.dim == self.max_X.shape[0], (
            "The dimension of min_X and max_X must be the same"
        )
        assert np.all(self.min_X < self.max_X), (
            "min_X must be less than max_X for each dimension"
        )
        self.L_X = self.max_X - self.min_X
        self.history = History(dim=self.dim)
        if config is None:
            self.config = SetConfig()
        else:
            self.config = config

        if initial_data is not None:
            if len(initial_data) != 2:
                msg = "ERROR: initial_data should be 2-elements tuple or list (X and objectives)"
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
            self.config.learning.is_disp = (
                self.config.learning.is_disp and self.mpirank == 0
            )

    def set_seed(self, seed):
        """
        Setting a seed parameter for np.random.

        Parameters
        ----------
        seed: int
            seed number
        -------

        """
        self.seed = seed
        np.random.seed(self.seed)

    def write(
        self,
        X,
        t,
        time_total=None,
        time_update_predictor=None,
        time_get_action=None,
        time_run_simulator=None,
    ):
        """
        Writing history (update history, not output to a file).

        Parameters
        ----------
        X:  numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of each search candidate.
        t:  numpy.ndarray
            N dimensional array. The negative energy of each search candidate (value of the objective function to be optimized).
        time_total: numpy.ndarray
            N dimenstional array. The total elapsed time in each step.
            If None (default), filled by 0.0.
        time_update_predictor: numpy.ndarray
            N dimenstional array. The elapsed time for updating predictor (e.g., learning hyperparemters) in each step.
            If None (default), filled by 0.0.
        time_get_action: numpy.ndarray
            N dimenstional array. The elapsed time for getting next action in each step.
            If None (default), filled by 0.0.
        time_run_simulator: numpy.ndarray
            N dimenstional array. The elapsed time for running the simulator in each step.
            If None (default), filled by 0.0.

        Returns
        -------

        """

        self.history.write(
            t,
            X,
            time_total=time_total,
            time_update_predictor=time_update_predictor,
            time_get_action=time_get_action,
            time_run_simulator=time_run_simulator,
        )
        Z = self.predictor.get_basis(X) if self.predictor is not None else None
        self.training.add(X=X, t=t, Z=Z)

        if self.new_data is None:
            self.new_data = Variable(X=X, t=t, Z=Z)
        else:
            self.new_data.add(X=X, t=t, Z=Z)

    def random_search(
        self, max_num_probes, num_search_each_probe=1, simulator=None, is_disp=True
    ):
        """
        Performing random search.

        Parameters
        ----------
        max_num_probes: int
            Maximum number of random search process.
        num_search_each_probe: int
            Number of search at each random search process.
        simulator: callable
            Callable (function or object with ``__call__``) from action to t
            Here, action is an integer which represents the index of the candidate.
        is_disp: bool
            If true, process messages are outputted.
        Returns
        -------
        history: history object (physbo.search.discrete.results.history)
        """

        if self.mpirank != 0:
            is_disp = False

        N = int(num_search_each_probe)

        if is_disp:
            utility.show_interactive_mode(simulator, self.history)

        for n in range(0, max_num_probes):
            time_total = time.time()
            if is_disp and N > 1:
                utility.show_start_message_multi_search(self.history.num_runs)

            time_get_action = time.time()
            action = self._get_random_action(N)
            time_get_action = time.time() - time_get_action

            N_indeed = len(action)
            if N_indeed == 0:
                if self.mpirank == 0:
                    print("WARNING: All actions have already searched.")
                return copy.deepcopy(self.history)

            if simulator is None:
                return action

            time_run_simulator = time.time()
            t = _run_simulator(simulator, action, self.mpicomm)
            time_run_simulator = time.time() - time_run_simulator

            time_total = time.time() - time_total
            self.write(
                action,
                t,
                time_total=[time_total] * N_indeed,
                time_update_predictor=np.zeros(N_indeed, dtype=float),
                time_get_action=[time_get_action] * N_indeed,
                time_run_simulator=[time_run_simulator] * N_indeed,
            )

            if is_disp:
                self.history.show_search_results(N_indeed)

        return copy.deepcopy(self.history)

    def bayes_search(
        self,
        training=None,
        max_num_probes=None,
        num_search_each_probe=1,
        predictor=None,
        is_disp=True,
        simulator=None,
        score="TS",
        interval=0,
        num_rand_basis=0,
        optimizer=None,
    ):
        """
        Performing Bayesian optimization.

        Parameters
        ----------
        training: physbo.Variable
            Training dataset.
        max_num_probes: int
            Maximum number of searching process by Bayesian optimization.
        num_search_each_probe: int
            Number of searching by Bayesian optimization at each process.
        predictor: predictor object
            Base class is defined in physbo.predictor.
            If None, blm_predictor is defined.
        is_disp: bool
            If true, process messages are outputted.
        simulator: callable
            Callable (function or object with ``__call__``)
            Here, action is an integer which represents the index of the candidate.
        score: str
            The type of aquision funciton.
            TS (Thompson Sampling), EI (Expected Improvement) and PI (Probability of Improvement) are available.
        interval: int
            The interval number of learning the hyper parameter.
            If you set the negative value to interval, the hyper parameter learning is not performed.
            If you set zero to interval, the hyper parameter learning is performed only at the first step.
        num_rand_basis: int
            The number of basis function. If you choose 0, ordinary Gaussian process run.
        optimizer: Optimizer object
            Optimizer object for optimizing the acquisition function.
            If None, the default optimizer is used.

        Returns
        -------
        history: history object (physbo.search.discrete.results.history)
        """

        if self.mpirank != 0:
            is_disp = False

        old_disp = self.config.learning.is_disp
        self.config.learning.is_disp = is_disp

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        is_rand_expans = num_rand_basis != 0

        if training is not None:
            self.training = training

        if predictor is not None:
            self.predictor = predictor
        elif self.predictor is None:
            self._init_predictor(is_rand_expans)

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
            action = self._get_actions(
                score, N, K, alpha, optimizer, num_rand_basis=num_rand_basis
            )
            time_get_action = time.time() - time_get_action

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
                time_total=[time_total] * N,
                time_update_predictor=[time_update_predictor] * N,
                time_get_action=[time_get_action] * N,
                time_run_simulator=[time_run_simulator] * N,
            )

            if is_disp:
                self.history.show_search_results(N)
        self._update_predictor()
        self.config.learning.is_disp = old_disp
        return copy.deepcopy(self.history)

    @staticmethod
    def _warn_no_predictor(method_name):
        print("Warning: Since policy.predictor is not yet set,")
        print("         a GP predictor (num_rand_basis=0) is used for predicting")
        print("         If you want to use a BLM predictor (num_rand_basis>0),")
        print("         call bayes_search(max_num_probes=0, num_rand_basis=nrb)")
        print("         before calling {}.".format(method_name))

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
            predictor.fit(self.training, 0, comm=self.mpicomm)
            predictor.prepare(self.training)
            return predictor.get_post_fmean(self.training, X)
        else:
            self._update_predictor()
            return self.predictor.get_post_fmean(self.training, X)

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
            predictor.fit(self.training, 0, comm=self.mpicomm)
            predictor.prepare(self.training)
            return predictor.get_post_fcov(self.training, X, diag)
        else:
            self._update_predictor()
            return self.predictor.get_post_fcov(self.training, X, diag)

    def get_score(
        self, mode, *, xs=None, predictor=None, training=None, parallel=True, alpha=1
    ):
        """
        Calcualte score (acquisition function)

        Parameters
        ----------
        mode: str
            The type of aquisition funciton. TS, EI and PI are available.
            These functions are defined in score.py.
        xs: physbo.Variable or np.ndarray
            input parameters to calculate score
        predictor: predictor object
            predictor used to calculate score.
            If not given, self.predictor will be used.
        training:physbo.Variable
            Training dataset.
            If not given, self.training will be used.
        parallel: bool
            Calculate scores in parallel by MPI (default: True)
        alpha: float
            Tuning parameter which is used if mode = TS.
            In TS, multi variation is tuned as np.random.multivariate_normal(mean, cov*alpha**2, size).

        Returns
        -------
        f: float or list of float
            Score defined in each mode.

        Raises
        ------
        RuntimeError
            If both *actions* and *xs* are given

        Notes
        -----
        When neither *actions* nor *xs* are given, scores for actions not yet searched will be calculated.

        When *parallel* is True, it is assumed that the function receives the same input (*actions* or *xs*) for all the ranks.
        If you want to split the input array itself, set *parallel* be False and merge results by yourself.
        """
        if training is None:
            training = self.training

        if training.X is None or training.X.shape[0] == 0:
            msg = "ERROR: No training data is registered."
            raise RuntimeError(msg)

        if predictor is None:
            if self.predictor is None:
                self._warn_no_predictor("get_score()")
                predictor = gp_predictor(self.config)
                predictor.fit(training, 0, comm=self.mpicomm)
                predictor.prepare(training)
            else:
                self._update_predictor()
                predictor = self.predictor

        if xs is not None:
            test = self._make_variable_X(xs)
            if parallel and self.mpisize > 1:
                actions = np.array_split(np.arange(test.X.shape[0]), self.mpisize)
                test = test.get_subset(actions[self.mpirank])
        else:
            raise RuntimeError("ERROR: xs is not given")

        f = search_score.score(
            mode, predictor=predictor, training=training, test=test, alpha=alpha
        )
        if parallel and self.mpisize > 1:
            fs = self.mpicomm.allgather(f)
            f = np.hstack(fs)
        return f

    def _argmax_score(self, mode, predictor, training, extra_trainings, optimizer):
        K = len(extra_trainings)
        if K == 0:
            predictor.prepare(training)

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
                predictors[k].update(trains[k], extra_train)

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
            mode, predictor, self.training, [], optimizer=optimizer
        )

        for n in range(1, N):
            extra_training = Variable(X=X[0:n, :])
            t = self.predictor.get_predict_samples(self.training, extra_training, K)
            extra_trainings = [copy.deepcopy(extra_training) for _ in range(K)]
            for k in range(K):
                extra_trainings[k].t = t[k, :]
            X[n, :] = self._argmax_score(
                mode, predictor, self.training, extra_trainings, optimizer=optimizer
            )

        return X

    def _get_random_action(self, N):
        """
        Getting indexes of actions randomly.

        Parameters
        ----------
        N: int
            Total number of search candidates.
        Returns
        -------
        action: numpy.ndarray
            Indexes of actions selected randomly from search candidates.
        """
        action = np.random.rand(N, self.dim) * self.L_X.reshape(
            1, -1
        ) + self.min_X.reshape(1, -1)
        if self.mpisize > 1:
            self.mpicomm.Bcast(action, root=0)
        return action

    def save(self, file_history, file_training=None, file_predictor=None):
        """

        Saving history, training and predictor into the corresponding files.

        Parameters
        ----------
        file_history: str
            The name of the file that stores the information of the history.
        file_training: str
            The name of the file that stores the training dataset.
        file_predictor: str
            The name of the file that stores the predictor dataset.

        Returns
        -------

        """
        if self.mpirank == 0:
            self.history.save(file_history)

            if file_training is not None:
                self.training.save(file_training)

            if file_predictor is not None:
                with open(file_predictor, "wb") as f:
                    pickle.dump(self.predictor, f)

    def load(self, file_history, file_training=None, file_predictor=None):
        """

        Loading files about history, training and predictor.

        Parameters
        ----------
        file_history: str
            The name of the file that stores the information of the history.
        file_training: str
            The name of the file that stores the training dataset.
        file_predictor: str
            The name of the file that stores the predictor dataset.

        Returns
        -------

        """
        self.history.load(file_history)

        if file_training is None:
            N = self.history.total_num_search
            X = self.history.action_X[0:N, :]
            t = self.history.fx[0:N]
            self.training = Variable(X=X, t=t)
        else:
            self.training = Variable()
            self.training.load(file_training)

        if file_predictor is not None:
            with open(file_predictor, "rb") as f:
                self.predictor = pickle.load(f)

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

    def export_history(self):
        """
        Returning the information of the history.

        Returns
        -------

        """
        return self.history

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
            self.predictor = blm_predictor(self.config)
        else:
            self.predictor = gp_predictor(self.config)

    def _learn_hyperparameter(self, num_rand_basis):
        self.predictor.fit(self.training, num_rand_basis, comm=self.mpicomm)
        # self.test.Z = self.predictor.get_basis(self.test.X)
        self.training.Z = self.predictor.get_basis(self.training.X)
        self.predictor.prepare(self.training)
        self.new_data = None

    def _update_predictor(self):
        if self.new_data is not None:
            self.predictor.update(self.training, self.new_data)
            self.new_data = None

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


def _run_simulator(simulator, action, comm=None):
    if comm is None:
        return simulator(action)
    if comm.rank == 0:
        t = simulator(action)
    else:
        t = 0.0
    return comm.bcast(t, root=0)

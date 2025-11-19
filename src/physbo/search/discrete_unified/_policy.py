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
from .. import score as search_score
from ...gp import Predictor as gp_predictor
from ...blm import Predictor as blm_predictor
from ...misc import SetConfig
from ..._variable import Variable, normalize_t


class Policy(discrete.Policy):
    """Multi objective Bayesian optimization with discrete search space by using unified objective function"""

    def __init__(
        self, test_X, num_objectives, comm=None, config=None, initial_data=None
    ):
        """
        Initialize the Policy object

        Parameters
        ----------
        test_X: numpy.ndarray
            The set of candidates. Each row vector represents the feature vector of each search candidate.
        num_objectives: int
            The number of objectives
        comm: MPI.Comm, optional
            MPI Communicator
        config: physbo.misc.SetConfig, optional
        initial_data: tuple[np.ndarray, np.ndarray], optional
            The initial training datasets.
            The first elements is the array of actions and the second is the array of value of objective functions
        """
        self.num_objectives = num_objectives
        self.history = History(num_objectives=self.num_objectives)

        self.training = Variable()
        self.training_unified = None
        self.predictor = None
        self.test = self._make_variable_X(test_X)
        self.new_data = None

        self.actions = np.arange(0, test_X.shape[0])
        if config is None:
            self.config = SetConfig()
        else:
            self.config = config

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
        N = len(action)

        t = np.array(t)
        # Ensure t is 2D: shape (N, num_objectives)
        if t.ndim == 1:
            if N == 1:
                t = t.reshape(1, -1)
            else:
                raise ValueError(f"Number of actions is {N} > 1, but t is 1D array")
        assert action.shape[0] == t.shape[0], "The number of actions and t must be the same"
        assert t.shape[1] == self.num_objectives, "The number of objectives in t must be the same as num_objectives"

        # Determine X and Z (different for each objective)
        if X is None:
            X = self.test.X[action, :]
            Z = self.test.Z[:, action, :] if self.test.Z is not None else None
        else:
            if self.predictor is not None:
                z = [self.predictor.get_basis(X) for _ in range(self.num_objectives)]
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
        predictor=None,
        is_disp=True,
        disp_pareto_set=False,
        simulator=None,
        score="EI",
        unify_method=None,
        interval=0,
        num_rand_basis=0,
    ):
        """
        Performing Bayesian optimization by using unified objective function

        Parameters
        ----------
        training_list: list of physbo.Variable, optional
            The training datasets.
        max_num_probes: int, optional
            The maximum number of searching process by Bayesian optimization.
        num_search_each_probe: int, optional
            The number of searching by Bayesian optimization at each process.
        predictor: predictor object, optional
            The predictor object.
        is_disp: bool, optional
            If true, process messages are outputted.
        disp_pareto_set: bool, optional
            If true, Pareto set is displayed.
        simulator: callable, optional
            The simulator function.
        score: str, optional
            The type of aquisition funciton.
            TS (Thompson Sampling), EI (Expected Improvement) and PI (Probability of Improvement) are available.
        unify_method: callable
            The unified objective function. It is a function or a callable object that maps a (N, num_objectives) numpy.ndarray of original objective functions to a (N, 1) numpy.ndarray of unified objective functions.
            See physbo.search.unify for examples.
        interval: int, optional
            The interval number of learning the hyper parameter.
            If you set the negative value to interval, the hyper parameter learning is not performed.
            If you set zero to interval, the hyper parameter learning is performed only at the first step.
        num_rand_basis: int, optional
            The number of basis function. If you choose 0, ordinary Gaussian process run.

        Returns
        -------
        history: history object (physbo.search.discrete_unified.results.history)
        """
        assert unify_method is not None, "unify_method must be provided"

        if self.mpirank != 0:
            is_disp = False

        old_disp = self.config.learning.is_disp
        self.config.learning.is_disp = is_disp

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        self.unify_method = unify_method
        if num_rand_basis < 0:
            raise ValueError("num_rand_basis must be non-negative")
        is_rand_expans = (num_rand_basis > 0)

        if training_list is not None:
            self.training = training_list

        if predictor is None:
            self._initialize_predictor(is_rand_expans)
        else:
            self.predictor = predictor

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
            predictor.fit(self.training_unified, 0, comm=self.mpicomm)
            predictor.prepare(self.training_unified)
            return predictor.get_post_fmean(self.training_unified, X)
        else:
            self._update_predictor()
            return self.predictor.get_post_fmean(self.training_unified, X)

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
            predictor.fit(self.training_unified, 0, comm=self.mpicomm)
            predictor.prepare(self.training_unified)
            return predictor.get_post_fcov(self.training_unified, X, diag)
        else:
            self._update_predictor()
            return self.predictor.get_post_fcov(self.training_unified, X, diag)


    def get_score(
        self,
        mode,
        *,
        actions=None,
        xs=None,
        predictor=None,
        training=None,
        parallel=True,
        alpha=1,
    ):
        """
        Calcualte score (acquisition function)

        Parameters
        ----------
        mode: str
            The type of aquisition funciton. TS, EI and PI are available.
            These functions are defined in score.py.
        actions: array of int
            actions to calculate score
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
            training = self.training_unified

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
            if actions is not None:
                raise RuntimeError("ERROR: both actions and xs are given")
            test = self._make_variable_X(xs)
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
            mode, predictor=predictor, training=training, test=test, alpha=alpha
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
            self._warn_no_predictor("get_post_fmean()")
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
            The number of samples for evaluating score.
        alpha: float
            not used.

        Returns
        -------
        f: list
            N dimensional scores (score is defined in each mode)
        """
        f = np.zeros((K, len(self.actions)), dtype=float)

        # draw K samples of the values of objective function of chosen actions
        new_test_local = self.test.get_subset(chosen_actions)
        virtual_t_local = self.predictor.get_predict_samples(
            self.training_unified, new_test_local, K
        )
        if self.mpisize == 1:
            new_test = new_test_local
            virtual_t = virtual_t_local
        else:
            new_test = Variable()
            for nt in self.mpicomm.allgather(new_test_local):
                new_test.add(X=nt.X, t=nt.t, Z=nt.Z)
            virtual_t = np.concatenate(self.mpicomm.allgather(virtual_t_local), axis=1)
        # virtual_t = self.predictor.get_predict_samples(self.training, new_test, K)

        for k in range(K):
            predictor = copy.deepcopy(self.predictor)
            train = copy.deepcopy(self.training_unified)
            virtual_train = new_test
            # Normalize virtual_t[k, :] to (N, 1) shape
            virtual_train.t = normalize_t(virtual_t[k, :], k=1)

            if virtual_train.Z is None:
                train.add(virtual_train.X, virtual_train.t)
            else:
                train.add(virtual_train.X, virtual_train.t, virtual_train.Z)

            predictor.update(train, virtual_train)

            f[k, :] = self.get_score(
                mode, predictor=predictor, training=train, parallel=False
            )
        return np.mean(f, axis=0)

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
            X = self.test.X[self.history.chosen_actions[0:N], :]
            t = self.history.fx[0:N]
            self.training = Variable(X=X, t=t)
        else:
            self.training = Variable()
            self.training.load(file_training)

        if file_predictor is not None:
            with open(file_predictor, "rb") as f:
                self.predictor = pickle.load(f)

        N = self.history.total_num_search

        visited = self.history.chosen_actions[:N]
        local_index = np.searchsorted(self.actions, visited)
        local_index = local_index[
            np.take(self.actions, local_index, mode="clip") == visited
        ]
        self.actions = self._delete_actions(local_index)

    def _learn_hyperparameter(self, num_rand_basis):
        self.training_unified = self._unify_training(self.training)

        self.predictor.fit(
            self.training_unified, num_rand_basis, comm=self.mpicomm
        )
        self.predictor.prepare(self.training_unified)
        self.new_data = None

    def _initialize_predictor(self, is_rand_expans):
        if is_rand_expans:
            self.predictor = blm_predictor(self.config)
        else:
            self.predictor = gp_predictor(self.config)

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

def _run_simulator(simulator, action, comm=None):
    if comm is None:
        return simulator(action)
    if comm.rank == 0:
        t = simulator(action)
    else:
        t = 0.0
    return comm.bcast(t, root=0)

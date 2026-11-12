# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import copy
import pickle as pickle

from ._history import History
from ... import _mpi
from .. import discrete
from .. import utility
from .. import score_multi as search_score
from ..._variable import Variable
from .._policy_common import _MultiObjectiveMethods


class Policy(_MultiObjectiveMethods, discrete.Policy):
    """Multi objective Bayesian optimization with discrete search space"""

    def __init__(
        self,
        test_X,
        num_objectives,
        comm=None,
        config=None,
        initial_data=None,
        rng=None,
    ):
        self.num_objectives = num_objectives
        super().__init__(
            test_X, config=config, initial_data=initial_data, comm=comm, rng=rng
        )

    def _new_history(self):
        return History(num_objectives=self.num_objectives)

    def _write_initial_data(self, initial_data):
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
            if self.predictor_list[0] is not None:
                Z = self._stack_basis(self.predictor_list, X)
            else:
                Z = None

        # Failed observations (any non-finite objective) are kept in the
        # history and their actions are consumed, but they are not used
        # for training.
        valid = utility.finite_mask(t)
        if np.any(valid):
            X_ok = np.asarray(X)[valid]
            t_ok = t[valid]
            Z_ok = utility.mask_rows(Z, valid)
            if self.new_data is None:
                self.new_data = Variable(X=X_ok, t=t_ok, Z=Z_ok)
            else:
                self.new_data.add(X=X_ok, t=t_ok, Z=Z_ok)

            # Add to single training Variable with full 2D t matrix and (k, N, n) Z
            if self.training.X is None:
                self.training = Variable(X=X_ok, t=t_ok, Z=Z_ok)
            else:
                self.training.add(X=X_ok, t=t_ok, Z=Z_ok)

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
        return self._random_search(
            max_num_probes,
            num_search_each_probe,
            simulator,
            is_disp,
            disp_pareto_set=disp_pareto_set,
        )

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
        unify_method=None,
        ard=False,
    ):
        """
        Performing Bayesian optimization by using multi objective function

        Parameters
        ----------
        training_list: list of physbo.Variable, optional
            The training datasets.
        max_num_probes: int, optional
            The maximum number of searching process by Bayesian optimization.
        num_search_each_probe: int, optional
            The number of searching by Bayesian optimization at each process.
        predictor_list: list of predictor objects, optional
            The predictor objects.
        is_disp: bool, optional
            If true, process messages are outputted.
        disp_pareto_set: bool, optional
            If true, Pareto set is displayed.
        simulator: callable, optional
            The simulator function.
        score: str or callable
            EHVI, HVPI, TS, or a callable called positionally as
            ``score(predictor_list, training, test, pareto)`` returning a
            numpy.ndarray of shape (test.X.shape[0],), with larger values
            preferred. pareto is self.history.pareto and must be treated as
            read-only; marginal scores do not update it with virtual data.
            Empty test data skips the callable. Under MPI, the callable
            evaluates local points and must not perform collective communication.
        interval: int, optional
            The interval number of learning the hyper parameter.
            If you set the negative value to interval, the hyper parameter learning is not performed.
            If you set zero to interval, the hyper parameter learning is performed only at the first step.
            A negative interval needs a learned predictor: the one learned by an
            earlier bayes_search, or one given as an argument, read by load() or
            set by hand. Otherwise, or if another kind of predictor is asked for
            (see num_rand_basis), a RuntimeError is raised.
        num_rand_basis: int, optional
            The number of basis function. If you choose 0, ordinary Gaussian process run.
            The predictor is kept between calls of bayes_search. A new one is made,
            to be learned, when num_rand_basis or ard differ from those of the
            previous call and the predictor is not of the kind asked for.
        optimizer: optimizer object, optional
            This is for compatibility with the range-based Policies.
        unify_method: callable, optional
            This is for compatibility with the unified-optimization Policies.
        ard: bool, optional
            If True, use Automatic Relevance Determination (ARD) for the Gaussian kernel.
            Default is False.

        Returns
        -------
        history: history object (physbo.search.discrete_multi.results.history)
        """
        score = search_score.resolve(
            score,
            alpha=self.config.search.alpha,
            reduced_candidate_num=self.TS_candidate_num,
            rng=self.rng,
        )

        def propose(N, K, alpha):
            return self._get_actions(score, N, K, alpha)

        return self._bayes_search(
            score,
            propose,
            training=training_list,
            predictor=predictor_list,
            max_num_probes=max_num_probes,
            num_search_each_probe=num_search_each_probe,
            simulator=simulator,
            is_disp=is_disp,
            interval=interval,
            num_rand_basis=num_rand_basis,
            ard=ard,
            disp_pareto_set=disp_pareto_set,
        )

    def _get_actions(self, mode, N, K, alpha):
        f, error = _mpi.call(
            self.mpicomm, self.get_score, mode=mode, alpha=alpha, parallel=False
        )
        champion, local_champion, local_index = self._find_champion(f, error)
        if champion == -1:
            return np.zeros(0, dtype=int)
        if champion == local_champion:
            self.actions = self._delete_actions(local_index)

        chosen_actions = [champion]
        for n in range(1, N):
            f, error = self._get_marginal_score(mode, chosen_actions[0:n], K, alpha)
            champion, local_champion, local_index = self._find_champion(f, error)
            if champion == -1:
                break
            if champion == local_champion:
                self.actions = self._delete_actions(local_index)
            chosen_actions.append(champion)
        return np.array(chosen_actions)

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
        """
        Calculate multi-objective scores (acquisition function).

        Parameters
        ----------
        mode: str or callable
            EHVI, HVPI, TS, or a callable called positionally as
            ``mode(predictor_list, training, test, pareto)`` returning a
            numpy.ndarray of shape (test.X.shape[0],), with larger values
            preferred. Empty test data skips the callable but still
            participates in MPI gathering. The callable must not perform
            collective communication. alpha and self.TS_candidate_num apply
            only to string modes.
        actions: array of int, optional
            Candidate indices to evaluate. If neither actions nor xs is given,
            evaluate candidates not yet searched.
        xs: physbo.Variable or numpy.ndarray, optional
            Input parameters to calculate scores.
        predictor_list: list of predictor objects, optional
            Predictors for each objective. Defaults to self.predictor_list.
        training_list: physbo.Variable, optional
            Training dataset containing all objectives. Defaults to self.training.
        pareto: Pareto, optional
            Passed unchanged when provided; otherwise self.history.pareto is
            used. Treat it as read-only. Marginal scores use the observed
            history's Pareto front without adding virtual observations.
        parallel: bool
            Calculate scores in parallel by MPI (default: True).
        alpha: float
            Posterior sampling scale for string "TS" (default: 1).
            Ignored for callable modes.

        Returns
        -------
        f: numpy.ndarray
            Acquisition scores of shape (num_points,).

        Raises
        ------
        RuntimeError
            If both actions and xs are given.

        Notes
        -----
        When parallel is True, all ranks must receive the same input. To
        supply separate local inputs, set parallel=False and gather results
        yourself.
        """
        mode = search_score.resolve(
            mode,
            alpha=alpha,
            reduced_candidate_num=self.TS_candidate_num,
            rng=self.rng,
        )

        if training_list is None:
            training = self.training
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
                    predictor = self._make_gp_predictor()
                    predictor.fit(training, 0, comm=self.mpicomm, objective_index=i, rng=self.rng)
                    predictor.prepare(training, objective_index=i)
                    predictor_list.append(predictor)
            else:
                self._update_predictor()
                predictor_list = self.predictor_list

        def scores(actions):
            # the share of this process of the candidates
            if xs is not None:
                if actions is not None:
                    raise RuntimeError("ERROR: both actions and xs are given")
                if isinstance(xs, Variable):
                    test = xs
                else:
                    test = Variable(X=xs)
                if parallel and self.mpisize > 1:
                    actions = np.array_split(
                        np.arange(test.X.shape[0]), self.mpisize
                    )
                    test = test.get_subset(actions[self.mpirank])
            else:
                if actions is None:
                    actions = self.actions
                else:
                    if isinstance(actions, int):
                        actions = [actions]
                    if parallel and self.mpisize > 1:
                        actions = np.array_split(actions, self.mpisize)[
                            self.mpirank
                        ]
                test = self.test.get_subset(actions)
            return search_score.score(
                mode,
                predictor_list=predictor_list,
                training=training,
                test=test,
                pareto=pareto,
                reduced_candidate_num=self.TS_candidate_num,
                alpha=alpha,
            )

        comm = self.mpicomm if parallel and self.mpisize > 1 else None
        f, error = _mpi.call(comm, scores, actions)
        if comm is not None:
            fs = _mpi.allgather(comm, f, error)
            f = np.hstack(fs)
        return f

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
        f: numpy.ndarray or None
            The marginal scores of the candidates of this process (score is
            defined in each mode), or None if ``error`` is not None.
        error: Exception or None
            An exception raised on this process while computing ``f`` in a
            parallel run, to be passed to :meth:`_find_champion` (in a serial
            run it is raised).
        """

        comm = self._comm

        def predict_samples():
            N = len(chosen_actions)
            # draw K samples of the values of objective function of chosen
            # actions
            virtual_t = np.zeros((K, N, self.num_objectives))
            for i in range(self.num_objectives):
                virtual_t[:, :, i] = self.predictor_list[i].get_predict_samples(
                    self.training, new_test, K, objective_index=i, rng=self.rng
                )
            return virtual_t

        # self.test is not partitioned over ranks, so the chosen points are
        # identical on every rank; the virtual values are drawn on rank 0 and
        # broadcast so that all ranks fantasize the same observations.
        new_test = self.test.get_subset(chosen_actions)
        if comm is None or self.mpirank == 0:
            virtual_t, error = _mpi.call(comm, predict_samples)
        else:
            virtual_t, error = None, None
        if comm is not None:
            virtual_t = _mpi.bcast(comm, virtual_t, error, root=0)
        return _mpi.call(comm, self._marginal_score, mode, new_test, virtual_t, K)

    def _marginal_score(self, mode, new_test, virtual_t, K):
        """The marginal scores of this process's candidates for the K virtual
        observations ``virtual_t`` at the points ``new_test`` (the same on
        every process) from :meth:`_get_marginal_score`."""
        f = np.zeros((K, len(self.actions)), dtype=float)
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
            # rebuild the training data from the valid observations only
            actions, t = self.history.export_valid()
            X = self.test.X[actions, :]
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

        self._after_load()

    def _learn_hyperparameter(self, num_rand_basis):
        for i in range(self.num_objectives):
            self.predictor_list[i].fit(
                self.training,
                num_rand_basis,
                comm=self.mpicomm,
                objective_index=i,
                rng=self.rng,
            )
        self._prepare_statistics()

    def _prepare_statistics(self):
        # The basis values of the candidates and of the training data, in
        # (k, N, n) format, are those of the current predictors (None with
        # Gaussian process predictors) before prepare() reads them.
        test_Z = self._stack_basis(self.predictor_list, self.test.X)
        training_Z = self._stack_basis(self.predictor_list, self.training.X)
        if test_Z is not None:
            self.test.Z = test_Z
        if training_Z is not None:
            self.training.Z = training_Z

        for i in range(self.num_objectives):
            self.predictor_list[i].prepare(self.training, objective_index=i)
        self.new_data = None

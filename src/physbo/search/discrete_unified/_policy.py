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
from .. import score as search_score
from ..._variable import Variable, normalize_t
from ..._rng import attach_rng
from .._policy_common import _UnifiedObjectiveMethods


class Policy(_UnifiedObjectiveMethods, discrete.Policy):
    """Multi objective Bayesian optimization with discrete search space by using unified objective function"""

    def __init__(
        self,
        test_X,
        num_objectives,
        comm=None,
        config=None,
        initial_data=None,
        rng=None,
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
        # unlike the single-objective policies, this one does not stop when a
        # step proposes no point; kept as it was
        return self._random_search(
            max_num_probes,
            num_search_each_probe,
            simulator,
            is_disp,
            stop_when_exhausted=False,
            disp_pareto_set=disp_pareto_set,
        )

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
        interval=0,
        num_rand_basis=0,
        unify_method=None,
        optimizer=None,
        ard=False,
        ucb_beta=1.0,
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
        score: str or callable
            TS, EI, PI, UCB, or a callable called positionally as
            ``score(predictor, training, test)`` returning a numpy.ndarray of
            shape (test.X.shape[0],), with larger values preferred. Empty test
            data skips the callable. Under MPI, the callable evaluates local
            points and must not perform collective communication.
            training contains unified targets of shape (N, 1).
        interval: int, optional
            The interval number of learning the hyper parameter.
            If you set the negative value to interval, the hyper parameter learning is not performed.
            If you set zero to interval, the hyper parameter learning is performed only at the first step.
        num_rand_basis: int, optional
            The number of basis function. If you choose 0, ordinary Gaussian process run.
        unify_method: callable
            The unified objective function. It is a function or a callable object that maps a (N, num_objectives) numpy.ndarray of original objective functions to a (N, 1) numpy.ndarray of unified objective functions.
            See physbo.search.unify for examples.
            If it draws random numbers through an ``rng`` attribute (as
            ``ParEGO`` does) and was not given a generator of its own, it
            uses the random number generator of the policy.
        optimizer: Optimizer object, optional
            This is for compatibility with the range-based Policies.
            This is not used.
        ard: bool, optional
            If True, use Automatic Relevance Determination (ARD) for the Gaussian kernel.
            Default is False.
        ucb_beta: float
            Trade-off parameter between exploitation and exploration used for
            string "UCB". The score is ``fmean + ucb_beta * fstd``.
            Default is 1.0; ignored for callable scores.

        Returns
        -------
        history: history object (physbo.search.discrete_unified.results.history)
        """
        score = search_score.resolve(
            score, alpha=self.config.search.alpha, ucb_beta=ucb_beta, rng=self.rng
        )

        assert unify_method is not None, "unify_method must be provided"
        self.unify_method = unify_method
        # a unify_method without a generator of its own (e.g. ParEGO) draws
        # from the policy's rng, so that the Generator mode covers it
        attach_rng(unify_method, self.rng)
        if num_rand_basis < 0:
            raise ValueError("num_rand_basis must be non-negative")

        def propose(N, K, alpha):
            return self._get_actions(score, N, K, alpha, ucb_beta)

        return self._bayes_search(
            score,
            propose,
            training=training_list,
            predictor=predictor,
            max_num_probes=max_num_probes,
            num_search_each_probe=num_search_each_probe,
            simulator=simulator,
            is_disp=is_disp,
            interval=interval,
            num_rand_basis=num_rand_basis,
            ard=ard,
            disp_pareto_set=disp_pareto_set,
        )

    def _get_actions(self, mode, N, K, alpha, ucb_beta=1.0):
        self._update_predictor()
        comm = self._comm
        # TS: one posterior sample for every process, drawn on rank 0
        first_mode = search_score.share_sample(
            mode, self.predictor, self.training_unified, comm
        )
        f, error = _mpi.call(
            self.mpicomm,
            self.get_score,
            mode=first_mode,
            alpha=alpha,
            ucb_beta=ucb_beta,
            parallel=False,
        )
        champion, local_champion, local_index = self._find_champion(f, error)
        if champion == -1:
            return np.zeros(0, dtype=int)
        if champion == local_champion:
            self.actions = self._delete_actions(local_index)

        chosen_actions = [champion]
        for n in range(1, N):
            f, error = self._get_marginal_score(
                mode, chosen_actions[0:n], K, alpha, ucb_beta
            )
            champion, local_champion, local_index = self._find_champion(f, error)
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
        self._require_unified_training("get_post_fmean()")
        if self.predictor is None:
            self._warn_no_predictor("get_post_fmean()")
            predictor = self._make_gp_predictor()
            predictor.fit(self.training_unified, 0, comm=self.mpicomm, rng=self.rng)
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
        self._require_unified_training("get_post_fcov()")
        if self.predictor is None:
            self._warn_no_predictor("get_post_fcov()")
            predictor = self._make_gp_predictor()
            predictor.fit(self.training_unified, 0, comm=self.mpicomm, rng=self.rng)
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
        ucb_beta=1.0,
    ):
        """
        Calculate score (acquisition function)

        Parameters
        ----------
        mode: str or callable
            TS, EI, PI, UCB, or a callable called positionally as
            ``mode(predictor, training, test)`` returning a numpy.ndarray of
            shape (test.X.shape[0],), with larger values preferred. Empty test
            data skips the callable but still participates in MPI gathering.
            The callable must not perform collective communication. alpha and
            ucb_beta apply only to string modes.
            training contains unified targets of shape (N, 1).
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
        ucb_beta: float
            Trade-off parameter between exploitation and exploration used for
            string "UCB". The score is ``fmean + ucb_beta * fstd``.
            Default is 1.0; ignored for callable scores.

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
        mode = search_score.resolve(
            mode, alpha=alpha, ucb_beta=ucb_beta, rng=self.rng
        )

        if training is None:
            training = self._require_unified_training("get_score()")

        if training.X is None or training.X.shape[0] == 0:
            msg = "ERROR: No training data is registered."
            raise RuntimeError(msg)

        if predictor is None:
            if self.predictor is None:
                self._warn_no_predictor("get_score()")
                predictor = self._make_gp_predictor()
                predictor.fit(training, 0, comm=self.mpicomm, rng=self.rng)
                predictor.prepare(training)
            else:
                self._update_predictor()
                predictor = self.predictor

        def scores(actions):
            # the share of this process of the candidates
            if xs is not None:
                if actions is not None:
                    raise RuntimeError("ERROR: both actions and xs are given")
                test = self._make_variable_X(xs)
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
                predictor=predictor,
                training=training,
                test=test,
                alpha=alpha,
                ucb_beta=ucb_beta,
            )

        comm = self.mpicomm if parallel and self.mpisize > 1 else None
        # TS: one posterior sample for every process, drawn on rank 0
        mode = search_score.share_sample(mode, predictor, training, comm)
        f, error = _mpi.call(comm, scores, actions)
        if comm is not None:
            fs = _mpi.allgather(comm, f, error)
            f = np.hstack(fs)
        return f


    def _get_marginal_score(self, mode, chosen_actions, K, alpha, ucb_beta=1.0):
        """
        Getting marginal scores.

        Parameters
        ----------
        mode: str
            The type of acquisition function.
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
        f: numpy.ndarray or None
            The marginal scores of the candidates of this process (score is
            defined in each mode), or None if ``error`` is not None.
        error: Exception or None
            An exception raised on this process while computing ``f`` in a
            parallel run, to be passed to :meth:`_find_champion` (in a serial
            run it is raised).
        """

        comm = self._comm

        # Draw K samples of the values of objective function of chosen actions.
        # self.test is not partitioned over ranks, so the chosen points are
        # identical on every rank; the virtual values are drawn on rank 0 and
        # broadcast so that all ranks fantasize the same observations.
        new_test = self.test.get_subset(chosen_actions)
        if comm is None or self.mpirank == 0:
            virtual_t, error = _mpi.call(
                comm,
                self.predictor.get_predict_samples,
                self.training_unified,
                new_test,
                K,
                rng=self.rng,
            )
        else:
            virtual_t, error = None, None
        if comm is not None:
            virtual_t = _mpi.bcast(comm, virtual_t, error, root=0)

        def model(k):
            # the predictor and the training data with the k-th virtual
            # observations added (the same on every process)
            predictor = copy.deepcopy(self.predictor)
            train = copy.deepcopy(self.training_unified)
            virtual_train = new_test
            # Normalize virtual_t[k, :] to (N, 1) shape
            virtual_train.t = normalize_t(virtual_t[k, :], k=1)

            if virtual_train.Z is None:
                # the candidates carry no basis values (the basis of the
                # unified predictor belongs to the unified data, not to the
                # multi-objective candidates), so compute them for the virtual
                # observations as the unified training data have them; None
                # with the GP predictor
                self._attach_unified_basis(virtual_train)
            train.add(virtual_train.X, virtual_train.t, virtual_train.Z)

            predictor.update(train, virtual_train)
            return predictor, train

        # TS: one posterior sample per virtual data set for every process,
        # drawn on rank 0
        samples = search_score.share_samples(mode, self.predictor, model, K, comm)
        return _mpi.call(comm, self._marginal_score, mode, model, K, ucb_beta, samples)

    def _marginal_score(self, mode, model, K, ucb_beta, samples=None):
        """The marginal scores of this process's candidates: the mean of the
        scores over the K models ``model(k)`` of :meth:`_get_marginal_score`.
        ``samples`` are their posterior samples for Thompson sampling (see
        :func:`physbo.search.score.share_samples`), or None."""
        f = np.zeros((K, len(self.actions)), dtype=float)
        for k in range(K):
            predictor, train = model(k)
            mode_k = mode if samples is None else mode.with_sample(samples[k])
            f[k, :] = self.get_score(
                mode_k,
                predictor=predictor,
                training=train,
                ucb_beta=ucb_beta,
                parallel=False,
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
            # rebuild the training data from the valid observations only
            actions, t = self.history.export_valid()
            X = self.test.X[actions, :]
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


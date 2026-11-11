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
from .. import range as range_single
from .. import utility
from .. import score as search_score
from ..._variable import Variable
from ..._rng import attach_rng
from .._policy_common import _UnifiedObjectiveMethods


class Policy(_UnifiedObjectiveMethods, range_single.Policy):
    """Multi objective Bayesian optimization with continuous search space"""

    def __init__(
        self,
        num_objectives,
        *,
        min_X=None,
        max_X=None,
        constraint=None,
        comm=None,
        config=None,
        initial_data=None,
        rng=None,
    ):
        """

        Parameters
        ----------
        num_objectives: int
            The number of objective functions.
        min_X: numpy.ndarray
            The minimum value of each dimension of the search space.
        max_X: numpy.ndarray
            The maximum value of each dimension of the search space.
        constraint: callable, optional
            Inequality constraints on the search space, as in physbo.search.range.Policy:
            a function from an array of shape (N, d) to a boolean array of N elements
            (True for feasible points). Only feasible points are proposed.
            Not saved by save(); pickled by save_checkpoint(), so it must be picklable.
        comm: MPI.Comm, optional
            MPI Communicator
        config: SetConfig object (physbo.misc.SetConfig)
        initial_data: tuple[np.ndarray, np.ndarray]
            The initial training datasets.
            The first element is the array of inputs and the second is the array of
            values of the objective functions, of shape (N, num_objectives).
            These points are not checked against the constraint.
        rng: None, "legacy", int, or numpy.random.Generator, optional
            Random number generator specification (default: "legacy").
            See physbo.search.discrete.Policy for details.
        """
        self.num_objectives = num_objectives
        super().__init__(
            min_X=min_X,
            max_X=max_X,
            constraint=constraint,
            config=config,
            initial_data=initial_data,
            comm=comm,
            rng=rng,
        )

    def _new_history(self):
        return History(num_objectives=self.num_objectives, dim=self.dim)

    def _write_initial_data(self, initial_data):
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

    def write(
        self,
        X,
        t,
        time_total=None,
        time_update_predictor=None,
        time_get_action=None,
        time_run_simulator=None,
    ):
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

        # Failed evaluations are not supported in continuous search spaces;
        # raise before anything is written
        utility.require_finite(X, t)

        self.history.write(
            t,
            X,
            time_total=time_total,
            time_update_predictor=time_update_predictor,
            time_get_action=time_get_action,
            time_run_simulator=time_run_simulator,
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
        unify_method=None,
        interval=0,
        num_rand_basis=0,
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
        optimizer: Optimizer object, optional
            Optimizer object for optimizing the acquisition function.
            If None, the default optimizer is used.
        ard: bool, optional
            If True, use Automatic Relevance Determination (ARD) for the Gaussian kernel.
            Default is False.
        ucb_beta: float
            Trade-off parameter between exploitation and exploration used for
            string "UCB". The score is ``fmean + ucb_beta * fstd``.
            Default is 1.0; ignored for callable scores.

        Returns
        -------
        history: history object (physbo.search.range_unified.results.history)
        """
        score = search_score.resolve(
            score, alpha=self.config.search.alpha, ucb_beta=ucb_beta, rng=self.rng
        )

        assert unify_method is not None, "unify_method must be provided"
        self.unify_method = unify_method
        # a unify_method without a generator of its own (e.g. ParEGO) draws
        # from the policy's rng, so that the Generator mode covers it
        attach_rng(unify_method, self.rng)
        if optimizer is None:
            optimizer = self._default_optimizer()

        def propose(N, K, alpha):
            return self._get_actions(score, N, K, alpha, optimizer=optimizer)

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

    def _argmax_score(
        self,
        mode,
        predictor,
        training,
        extra_trainings,
        optimizer,
        error=None,
        samples_drawn=False,
    ):
        """``error``: an exception raised on this process while drawing the
        samples in ``extra_trainings`` (see :func:`physbo._mpi.call`);
        ``samples_drawn``: whether they were drawn (on every process alike),
        so that the processes exchange whether one of them raised."""
        if error is None and len(extra_trainings) == 0:
            predictor.prepare(training, objective_index=0)

            def fn(x):
                return self.get_score(
                    mode, xs=x.reshape(1, -1), predictor=predictor, parallel=False
                )[0]
        elif error is None:  # marginal score

            def marginal_score_function():
                K = len(extra_trainings)
                trains = [copy.deepcopy(training) for _ in range(K)]
                predictors = [copy.deepcopy(predictor) for _ in range(K)]
                for k in range(K):
                    extra_train = extra_trainings[k]
                    trains[k].add(X=extra_train.X, t=extra_train.t, Z=extra_train.Z)
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

                return fn

            fn, error = _mpi.call(self.mpicomm, marginal_score_function)
        if self.mpisize > 1 and (samples_drawn or len(extra_trainings) > 0):
            # the samples, and so the predictors, differ between the processes
            _mpi.allgather(self.mpicomm, None, error)

        return self._optimize_score(fn, optimizer)

    def _get_actions(self, mode, N, K, alpha, optimizer, num_rand_basis=0):
        """
        Getting next candidates

        Parameters
        ----------
        mode: str
            The type of acquisition function.
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

            def draw_samples():
                # with a Bayesian linear model the training data carry the
                # values of the basis functions, so the virtual observations
                # must too (the GP predictor has no basis: None)
                extra_training = Variable(
                    X=X[0:n, :], Z=self.predictor.get_basis(X[0:n, :])
                )
                t = self.predictor.get_predict_samples(
                    self.training_unified,
                    extra_training,
                    K,
                    objective_index=0,
                    rng=self.rng,
                )
                extra_trainings = [copy.deepcopy(extra_training) for _ in range(K)]
                for k in range(K):
                    # Normalize t to (N, 1) shape
                    extra_trainings[k].t = t[k, :].reshape(-1, 1)
                return extra_trainings

            extra_trainings, error = _mpi.call(self.mpicomm, draw_samples)
            X[n, :] = self._argmax_score(
                mode,
                predictor,
                self.training_unified,
                extra_trainings,
                optimizer=optimizer,
                error=error,
                samples_drawn=True,
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
        self._require_unified_training("get_post_fmean()")
        if self.predictor is None:
            self._warn_no_predictor("get_post_fmean()")
            predictor = self._make_gp_predictor()
            predictor.fit(
                self.training_unified,
                0,
                comm=self.mpicomm,
                objective_index=0,
                rng=self.rng,
            )
            predictor.prepare(self.training_unified, objective_index=0)
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
        self._require_unified_training("get_post_fcov()")
        if self.predictor is None:
            self._warn_no_predictor("get_post_fcov()")
            predictor = self._make_gp_predictor()
            predictor.fit(
                self.training_unified,
                0,
                comm=self.mpicomm,
                objective_index=0,
                rng=self.rng,
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
        xs: physbo.Variable or np.ndarray
            input parameters to calculate score
        predictor: predictor object
            predictor used to calculate score.
            If not given, self.predictor will be used.
        training:physbo.Variable
            Training dataset.
            If not given, self.training_unified will be used.
        pareto: Pareto, optional
            Retained for API compatibility; unused for unified scoring.
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
        f: numpy.ndarray
            Acquisition scores of shape (num_points,).

        Raises
        ------
        RuntimeError
            If *xs* is not given

        Notes
        -----
        When *parallel* is True, it is assumed that the function receives the same input (*xs*) for all the ranks.
        If you want to split the input array itself, set *parallel* be False and merge results by yourself.
        """
        mode = search_score.resolve(
            mode, alpha=alpha, ucb_beta=ucb_beta, rng=self.rng
        )

        if training is None:
            training = self._require_unified_training("get_score()")

        if pareto is None:
            pareto = self.history.pareto

        # `pareto` is currently unused in scoring but kept for API compatibility.
        # This dummy read prevents it from being flagged as an unused variable.
        _ = pareto
        if training.X is None or training.X.shape[0] == 0:
            msg = "ERROR: No training data is registered."
            raise RuntimeError(msg)

        if predictor is None:
            if self.predictor is None:
                self._warn_no_predictor("get_score()")
                predictor = self._make_gp_predictor()
                predictor.fit(training, 0, comm=self.mpicomm, objective_index=0, rng=self.rng)
                predictor.prepare(training, objective_index=0)
            else:
                self._update_predictor()
                predictor = self.predictor

        def scores():
            # the share of this process of the candidates
            if xs is not None:
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
                raise RuntimeError("ERROR: xs is not given")
            return search_score.score(
                mode,
                predictor=predictor,
                training=training,
                test=test,
                alpha=alpha,
            )

        comm = self.mpicomm if parallel and self.mpisize > 1 else None
        f, error = _mpi.call(comm, scores)
        if comm is not None:
            fs = _mpi.allgather(comm, f, error)
            f = np.hstack(fs)
        return f

    def save(self, file_history, file_training=None, file_predictor=None):
        """Save the history, the training data and the predictor into files.

        The constraint is not saved; give it to the constructor of the policy
        that loads the files.
        """
        if self.mpirank == 0:
            self.history.save(file_history)
            if file_training is not None:
                self.save_training(file_training)
            if file_predictor is not None:
                self.save_predictor(file_predictor)

    def load(self, file_history, file_training=None, file_predictor=None):
        """Load the history, the training data and the predictor from files.

        The constraint is not restored by load(); the policy keeps the one
        given to its constructor.
        """
        self.history.load(file_history)

        if file_training is None:
            # rebuild the training data from the valid observations only
            X, t = self.history.export_valid()
            self.training = Variable(X=X, t=t)
        else:
            self.load_training(file_training)

        if file_predictor is not None:
            self.load_predictor(file_predictor)


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

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import copy
import pickle as pickle
import warnings

from ._history import History
from .. import utility
from .. import score as search_score
from ..optimize.random import Optimizer as RandomOptimizer
from ... import _mpi
from ..._variable import Variable, normalize_t
from .._policy_common import _PolicyBase


class Policy(_PolicyBase):
    """Single objective Bayesian optimization with continuous search space"""

    #: No constraint, for subclasses whose constructor does not set it.
    constraint = None

    #: Rejection sampling of N feasible points is given up after max_trials_per_point * N samples.
    max_trials_per_point = 10000

    def __init__(
        self,
        *,
        min_X=None,
        max_X=None,
        constraint=None,
        config=None,
        initial_data=None,
        comm=None,
        rng=None,
    ):
        """

        Parameters
        ----------
        min_X: numpy.ndarray
            The minimum value of each dimension of the search space.
        max_X: numpy.ndarray
            The maximum value of each dimension of the search space.
        constraint: callable, optional
            Inequality constraints on the search space.
            A function from an array of shape (N, d) to a boolean array of N elements (True for feasible points).
            Only feasible points are proposed by random_search and bayes_search.
            If None (default), the whole box [min_X, max_X] is searched.
            This is not saved by save(); pass it again to the constructor.
            It is part of the state pickled by save_checkpoint(), so it must be
            picklable (a function defined at module level or a method of a
            picklable object, not a lambda or a local function).
        config: SetConfig object (physbo.misc.SetConfig)
        initial_data: tuple[np.ndarray, np.ndarray]
            The initial training datasets.
            The first elements is the array of inputs and the second is the array of values of objective functions
            These points are not checked against the constraint.
        comm: MPI.Comm, optional
            MPI Communicator
        rng: None, "legacy", int, or numpy.random.Generator, optional
            Random number generator specification (default: "legacy").
            See physbo.search.discrete.Policy for details.
        """
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
        self.constraint = constraint

        super().__init__(config=config, initial_data=initial_data, comm=comm, rng=rng)

    def _new_history(self):
        return History(dim=self.dim)

    def _write_initial_data(self, initial_data):
        if len(initial_data) != 2:
            msg = "ERROR: initial_data should be 2-elements tuple or list (X and objectives)"
            raise RuntimeError(msg)
        init_X, fs = initial_data
        fs_normalized = normalize_t(fs, k=1)
        assert init_X.shape[0] == fs_normalized.shape[0], (
            "The number of initial data must be the same"
        )
        assert init_X.shape[1] == self.dim, (
            "The dimension of initial_data[0] must be the same as the dimension of min_X and max_X"
        )
        self.write(init_X, fs_normalized)

    def _show_search_results(self, N_indeed, disp_pareto_set):
        self.history.show_search_results(N_indeed)

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
            N dimensional array (1D) or N x 1 dimensional array (2D).
            The negative energy of each search candidate (value of the objective function to be optimized).
            Will be normalized to (N, 1) shape internally.
            All the values must be finite. Failed evaluations are not
            supported in continuous search spaces.
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

        Raises
        ------
        ValueError
            If t contains a non-finite value (NaN or +-Inf).
            Nothing is written to the history in this case.
        """
        # Normalize t to (N, 1) shape
        t_normalized = normalize_t(t, k=1)

        utility.require_finite(X, t_normalized)

        self.history.write(
            t.flatten(),
            X,
            time_total=time_total,
            time_update_predictor=time_update_predictor,
            time_get_action=time_get_action,
            time_run_simulator=time_run_simulator,
        )
        # Get basis and convert to (1, N, n) format for single-objective
        Z_basis = self.predictor.get_basis(X) if self.predictor is not None else None
        if Z_basis is not None:
            Z = Z_basis[np.newaxis, :, :]  # (N, n) -> (1, N, n)
        else:
            Z = None
        self.training.add(X=X, t=t_normalized, Z=Z)

        if self.new_data is None:
            self.new_data = Variable(X=X, t=t_normalized, Z=Z)
        else:
            self.new_data.add(X=X, t=t_normalized, Z=Z)

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
        return self._random_search(
            max_num_probes, num_search_each_probe, simulator, is_disp
        )

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
        ard=False,
        ucb_beta=1.0,
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
        score: str or callable
            TS, EI, PI, UCB, or a callable called positionally as
            ``score(predictor, training, test)`` returning a numpy.ndarray of
            shape (test.X.shape[0],), with larger values preferred. Empty test
            data skips the callable. Under MPI, the callable evaluates local
            points and must not perform collective communication.
        interval: int
            The interval number of learning the hyper parameter.
            If you set the negative value to interval, the hyper parameter learning is not performed.
            If you set zero to interval, the hyper parameter learning is performed only at the first step.
            A negative interval needs a learned predictor: the one learned by an
            earlier bayes_search, or one given as an argument, read by load() or
            set by hand. Otherwise, or if another kind of predictor is asked for
            (see num_rand_basis), a RuntimeError is raised.
        num_rand_basis: int
            The number of basis function. If you choose 0, ordinary Gaussian process run.
            The predictor is kept between calls of bayes_search. A new one is made,
            to be learned, when num_rand_basis or ard differ from those of the
            previous call and the predictor is not of the kind asked for.
        optimizer: Optimizer object
            Optimizer object for optimizing the acquisition function.
            If None, the default optimizer is used.
        ard: bool
            If True, use Automatic Relevance Determination (ARD) for the Gaussian kernel.
            Default is False.
        ucb_beta: float
            Trade-off parameter between exploitation and exploration used for
            string "UCB". The score is ``fmean + ucb_beta * fstd``.
            Default is 1.0; ignored for callable scores.

        Returns
        -------
        history: history object (physbo.search.discrete.results.history)
        """
        score = search_score.resolve(
            score, alpha=self.config.search.alpha, ucb_beta=ucb_beta, rng=self.rng
        )
        if optimizer is None:
            optimizer = self._default_optimizer()

        def propose(N, K, alpha):
            return self._get_actions(
                score,
                N,
                K,
                alpha,
                optimizer,
                num_rand_basis=num_rand_basis,
                ucb_beta=ucb_beta,
            )

        return self._bayes_search(
            score,
            propose,
            training=training,
            predictor=predictor,
            max_num_probes=max_num_probes,
            num_search_each_probe=num_search_each_probe,
            simulator=simulator,
            is_disp=is_disp,
            interval=interval,
            num_rand_basis=num_rand_basis,
            ard=ard,
        )

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
            predictor = self._make_gp_predictor()
            predictor.fit(self.training, 0, comm=self.mpicomm, objective_index=0, rng=self.rng)
            predictor.prepare(self.training, objective_index=0)
            return predictor.get_post_fmean(self.training, X, objective_index=0)
        else:
            self._update_predictor()
            return self.predictor.get_post_fmean(self.training, X, objective_index=0)

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
            predictor = self._make_gp_predictor()
            predictor.fit(self.training, 0, comm=self.mpicomm, objective_index=0, rng=self.rng)
            predictor.prepare(self.training, objective_index=0)
            return predictor.get_post_fcov(self.training, X, diag, objective_index=0)
        else:
            self._update_predictor()
            return self.predictor.get_post_fcov(
                self.training, X, diag, objective_index=0
            )

    def get_num_dim(self):
        """
        Return the input dimension (number of features) of the search space.

        Returns
        -------
        int
            The number of dimensions (from min_X / max_X).
        """
        return self.dim

    def get_score(
        self,
        mode,
        *,
        xs=None,
        predictor=None,
        training=None,
        parallel=True,
        alpha=1,
        ucb_beta=1.0,
    ):
        """
        Calcualte score (acquisition function)

        Parameters
        ----------
        mode: str or callable
            TS, EI, PI, UCB, or a callable called positionally as
            ``mode(predictor, training, test)`` returning a numpy.ndarray of
            shape (test.X.shape[0],), with larger values preferred. Empty test
            data skips the callable but still participates in MPI gathering.
            The callable must not perform collective communication. alpha and
            ucb_beta apply only to string modes.
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
            training = self.training

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
                test = self._make_variable_X(xs)
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
                ucb_beta=ucb_beta,
            )

        comm = self.mpicomm if parallel and self.mpisize > 1 else None
        f, error = _mpi.call(comm, scores)
        if comm is not None:
            fs = _mpi.allgather(comm, f, error)
            f = np.hstack(fs)
        return f

    def _argmax_score(
        self,
        mode,
        predictor,
        training,
        extra_trainings,
        optimizer,
        ucb_beta=1.0,
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
                    mode,
                    xs=x.reshape(1, -1),
                    predictor=predictor,
                    ucb_beta=ucb_beta,
                    parallel=False,
                )[0]
        elif error is None:  # marginal score

            def marginal_score_function():
                K = len(extra_trainings)
                trains = [copy.deepcopy(training) for _ in range(K)]
                predictors = [copy.deepcopy(predictor) for _ in range(K)]
                for k in range(K):
                    extra_train = extra_trainings[k]
                    # Ensure t is normalized
                    if extra_train.t is not None:
                        extra_train.t = normalize_t(extra_train.t, k=1)
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
                            ucb_beta=ucb_beta,
                            parallel=False,
                        )[0]
                    return np.mean(f)

                return fn

            fn, error = _mpi.call(self.mpicomm, marginal_score_function)
        if self.mpisize > 1 and (samples_drawn or len(extra_trainings) > 0):
            # the samples, and so the predictors, differ between the processes
            _mpi.allgather(self.mpicomm, None, error)

        return self._optimize_score(fn, optimizer)

    def _default_optimizer(self):
        """Optimizer of the acquisition function used when none is given."""
        return RandomOptimizer(
            min_X=self.min_X,
            max_X=self.max_X,
            nsamples=1000,
            rng=self.rng,
            constraint=self.constraint,
            max_trials_per_point=self.max_trials_per_point,
        )

    def _constrained(self, fn):
        """Make the acquisition function -inf at infeasible points."""
        if self.constraint is None:
            return fn

        def wrapped(x):
            x = np.asarray(x)
            if utility.is_feasible(self.constraint, x.reshape(1, -1))[0]:
                return fn(x)
            return -np.inf

        return wrapped

    def _optimize_score(self, fn, optimizer):
        """
        Maximize the acquisition function under the constraint.

        Parameters
        ----------
        fn: callable
            Acquisition function, from a point of shape (dim,) to float.
        optimizer: Optimizer object
            Called as optimizer(fn, mpicomm=...).

        Returns
        -------
        X: numpy.ndarray
            The point to be searched next.

        Notes
        -----
        Infeasible points are hidden from the optimizer by -inf, but the optimizer can
        still fail to return a feasible point (no feasible sample, or a local search
        started from an infeasible point). The returned point is therefore checked,
        and replaced with a feasible random point if it is None, not finite or not
        feasible. A point of the wrong dimension is an error of the optimizer and
        raises ValueError, as without a constraint.
        """
        if isinstance(optimizer, RandomOptimizer) and (
            # == rather than "is": a bound method (fn.constraint) is a new
            # object at every attribute access
            optimizer.constraint == self.constraint
        ):
            # the optimizer draws feasible samples only; no need to judge
            # them one by one again
            X = optimizer(fn, mpicomm=self.mpicomm)
        else:
            X = optimizer(self._constrained(fn), mpicomm=self.mpicomm)
        if self.constraint is None:
            return X

        def check():
            if X is None:
                return False
            x = np.asarray(X, dtype=float).reshape(-1)
            if x.shape[0] != self.dim:
                raise ValueError(
                    f"ERROR: the optimizer returned a point of {x.shape[0]} "
                    f"dimension(s), but the search space has {self.dim}"
                )
            x = x.reshape(1, -1)
            return bool(
                np.all(np.isfinite(x)) and utility.is_feasible(self.constraint, x)[0]
            )

        # the verdict of rank 0 decides, so that all the ranks take the same
        # branch; an exception while judging is raised on every rank
        comm = self._comm
        if comm is None or self.mpirank == 0:
            is_ok, error = _mpi.call(comm, check)
        else:
            is_ok, error = None, None
        if comm is not None:
            is_ok = _mpi.bcast(comm, is_ok, error, root=0)
        if is_ok:
            return X

        if self.mpirank == 0:
            warnings.warn(
                "The optimizer of the acquisition function did not return a feasible point. "
                "A feasible random point is proposed instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._get_random_action(1)[0]

    def _get_actions(
        self, mode, N, K, alpha, optimizer, num_rand_basis=0, ucb_beta=1.0
    ):
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
            mode, predictor, self.training, [], optimizer=optimizer, ucb_beta=ucb_beta
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
                    self.training, extra_training, K, objective_index=0, rng=self.rng
                )
                extra_trainings = [copy.deepcopy(extra_training) for _ in range(K)]
                for k in range(K):
                    # Normalize t to (N, 1) shape
                    t_normalized = normalize_t(t[k, :], k=1)
                    extra_trainings[k].t = t_normalized
                return extra_trainings

            extra_trainings, error = _mpi.call(self.mpicomm, draw_samples)
            X[n, :] = self._argmax_score(
                mode,
                predictor,
                self.training,
                extra_trainings,
                optimizer=optimizer,
                ucb_beta=ucb_beta,
                error=error,
                samples_drawn=True,
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
        if self.constraint is None:
            action = self._sample_uniform(N)
            if self.mpisize > 1:
                self.mpicomm.Bcast(action, root=0)
            return action

        # Only rank 0 samples, since the number of trials depends on the random
        # numbers; the other ranks then draw the same number of points, so that
        # the generators of all the ranks stay in step as without a constraint.
        num_drawn = 0

        def draw(n):
            nonlocal num_drawn
            num_drawn += n
            return self._sample_uniform(n)

        comm = self._comm
        action, error = None, None
        if comm is None or self.mpirank == 0:
            # an exception of the constraint is raised on every rank below
            action, error = _mpi.call(
                comm,
                utility.sample_feasible,
                draw,
                N,
                self.constraint,
                max_trials_per_point=self.max_trials_per_point,
            )
        if comm is not None:
            # all the ranks must raise the error together
            found, num_drawn = _mpi.bcast(
                comm, (action is not None, num_drawn), error, root=0
            )
            if found:
                if self.mpirank != 0:
                    self._sample_uniform(num_drawn)
                    action = np.zeros((N, self.dim))
                comm.Bcast(action, root=0)
            else:
                action = None
        if action is None:
            raise RuntimeError(
                "ERROR: feasible points were not found in {} random samples. "
                "The feasible region may be empty or too small "
                "(equality constraints are not supported).".format(
                    self.max_trials_per_point * N
                )
            )
        return action

    def _sample_uniform(self, N):
        """Draw N points uniformly from the box [min_X, max_X]."""
        return self.rng.random((N, self.dim)) * self.L_X.reshape(
            1, -1
        ) + self.min_X.reshape(1, -1)

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

        Note
        ----
        The constraint is not saved; give it to the constructor of the policy
        that loads the files.
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

        Note
        ----
        The constraint is not restored by load(); the policy keeps the one
        given to its constructor.
        """
        self.history.load(file_history)

        if file_training is None:
            # rebuild the training data from the valid observations only
            X, t = self.history.export_valid()
            # Normalize t to (N, 1) shape
            t_normalized = normalize_t(t, k=1)
            self.training = Variable(X=X, t=t_normalized)
        else:
            self.training = Variable()
            self.training.load(file_training)
            # Ensure t is normalized to (N, 1) shape after loading
            if self.training.t is not None:
                self.training.t = normalize_t(self.training.t, k=1)

        if file_predictor is not None:
            with open(file_predictor, "rb") as f:
                self.predictor = pickle.load(f)

        self._after_load()

    def _learn_hyperparameter(self, num_rand_basis):
        self.predictor.fit(
            self.training,
            num_rand_basis,
            comm=self.mpicomm,
            objective_index=0,
            rng=self.rng,
        )
        self._prepare_statistics()

    def _prepare_statistics(self):
        # Get basis and convert to (1, N, n) format for single-objective
        training_Z_basis = self.predictor.get_basis(self.training.X)
        if training_Z_basis is not None:
            self.training.Z = training_Z_basis[np.newaxis, :, :]  # (N, n) -> (1, N, n)
        self.predictor.prepare(self.training, objective_index=0)
        self.new_data = None

    def _update_predictor(self):
        if self.new_data is not None:
            self.predictor.update(self.training, self.new_data, objective_index=0)
            self.new_data = None

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
from .. import score_multi as search_score
from ..._variable import Variable
from .._policy_common import _MultiObjectiveMethods


class Policy(_MultiObjectiveMethods, range_single.Policy):
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
        assert t.shape[1] == self.num_objectives, "The number of objectives in t must be the same as num_objectives"

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

        if self.predictor_list[0] is not None:
            Z = self._stack_basis(self.predictor_list, X)
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
            A negative interval needs a learned predictor of the kind asked for
            (num_rand_basis, ard): the one learned by an earlier bayes_search, or
            one given as an argument. Otherwise a RuntimeError is raised.
        num_rand_basis: int, optional
            The number of basis function. If you choose 0, ordinary Gaussian process run.
            The predictor is kept between calls of bayes_search; it is replaced by
            a new one, to be learned, when num_rand_basis or ard differ from those
            it was made with.
        optimizer: Optimizer object, optional
            Optimizer object for optimizing the acquisition function.
            If None, the default optimizer is used.
        unify_method: callable, optional
            This is for compatibility with the unified-optimization Policies.
            This is not used.
        ard: bool, optional
            If True, use Automatic Relevance Determination (ARD) for the Gaussian kernel.
            Default is False.

        Returns
        -------
        history: history object (physbo.search.range_multi.results.history)
        """
        score = search_score.resolve(
            score,
            alpha=self.config.search.alpha,
            reduced_candidate_num=self.TS_candidate_num,
            rng=self.rng,
        )
        if optimizer is None:
            optimizer = self._default_optimizer()

        def propose(N, K, alpha):
            return self._get_actions(score, N, K, alpha, optimizer=optimizer)

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

    def _argmax_score(
        self,
        mode,
        predictors,
        training,
        virtual_trainings,
        optimizer,
        error=None,
        samples_drawn=False,
    ):
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
        error: Exception or None
            An exception raised on this process while drawing the samples in
            ``virtual_trainings`` (see :func:`physbo._mpi.call`).
        samples_drawn: bool
            Whether the samples were drawn (on every process alike), so that
            the processes exchange whether one of them raised.
        """

        if error is None and len(virtual_trainings) == 0:
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
        elif error is None:  # marginal score

            def marginal_score_function():
                K = len(virtual_trainings)
                trains_k = [copy.deepcopy(training) for _ in range(K)]
                predictors_k = [copy.deepcopy(predictors) for _ in range(K)]
                for predictor_k, training_k, virtual_training in zip(
                    predictors_k, trains_k, virtual_trainings
                ):
                    training_k.add(
                        X=virtual_training.X, t=virtual_training.t, Z=virtual_training.Z
                    )
                    for i in range(self.num_objectives):
                        predictor_k[i].update(
                            training_k, virtual_training, objective_index=i
                        )

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

                return fn

            fn, error = _mpi.call(self.mpicomm, marginal_score_function)
        if self.mpisize > 1 and (samples_drawn or len(virtual_trainings) > 0):
            # the samples, and so the predictors, differ between the processes
            _mpi.allgather(self.mpicomm, None, error)

        pareto = self.history.pareto
        upper = search_score._reference_points_upper(mode)
        if upper is not None and pareto.front.shape[0] > 0:
            # on every process before the optimizer, which may evaluate fn on
            # some processes only, so that they all use the same reference
            # points and warn (or raise) together
            pareto.prepare_reference_points(upper=upper)
        return self._optimize_score(fn, optimizer)

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

            def draw_samples():
                # with Bayesian linear models the training data carry the
                # values of the basis functions of every objective, so the
                # virtual observations must too (as in write)
                Z = self._stack_basis(predictors, X[0:n, :])
                virtual_trainings = [Variable(X=X[0:n, :], Z=Z) for _ in range(K)]
                virtual_t = np.zeros((K, n, self.num_objectives))
                for i in range(self.num_objectives):
                    virtual_t[:, :, i] = predictors[i].get_predict_samples(
                        self.training,
                        virtual_trainings[0],
                        K,
                        objective_index=i,
                        rng=self.rng,
                    )
                for k in range(K):
                    virtual_trainings[k].t = virtual_t[k, :, :]
                return virtual_trainings

            virtual_trainings, error = _mpi.call(self.mpicomm, draw_samples)
            X[n, :] = self._argmax_score(
                mode,
                predictors,
                self.training,
                virtual_trainings,
                optimizer=optimizer,
                error=error,
                samples_drawn=True,
            )
        return X

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
            If xs is not given.

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
                predictor_list=predictor_list,
                training=training,
                test=test,
                pareto=pareto,
                reduced_candidate_num=self.TS_candidate_num,
                alpha=alpha,
            )

        comm = self.mpicomm if parallel and self.mpisize > 1 else None
        f, error = _mpi.call(comm, scores)
        if comm is not None:
            fs = _mpi.allgather(comm, f, error)
            f = np.hstack(fs)
        return f

    def save(self, file_history, file_training_list=None, file_predictor_list=None):
        """Save the history, the training data and the predictors into files.

        The constraint is not saved; give it to the constructor of the policy
        that loads the files.
        """
        if self.mpirank == 0:
            self.history.save(file_history)
            if file_training_list is not None:
                self.save_training_list(file_training_list)
            if file_predictor_list is not None:
                self.save_predictor_list(file_predictor_list)

    def load(self, file_history, file_training_list=None, file_predictor_list=None):
        """Load the history, the training data and the predictors from files.

        The constraint is not restored by load(); the policy keeps the one
        given to its constructor.
        """
        self.history.load(file_history)

        if file_training_list is None:
            # rebuild the training data from the valid observations only
            X, t = self.history.export_valid()
            self.training = Variable(X=X, t=t)
        else:
            self.load_training_list(file_training_list)

        if file_predictor_list is not None:
            self.load_predictor_list(file_predictor_list)

        N = self.history.total_num_search

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
        # The basis values of the training data, in (k, N, n) format, are
        # those of the current predictors (None with Gaussian process
        # predictors) before prepare() reads them.
        training_Z = self._stack_basis(self.predictor_list, self.training.X)
        if training_Z is not None:
            self.training.Z = training_Z

        for i in range(self.num_objectives):
            self.predictor_list[i].prepare(self.training, objective_index=i)
        self.new_data = None

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pickle as pickle
from typing import Optional, Any

from ._history import History
from .. import utility
from .. import score as search_score
from ... import _mpi
from ...misc import SetConfig

from ..._variable import Variable, normalize_t
from ..._rng import get_rng
from .._policy_common import _PolicyBase
from .._space import _DiscreteSpace, _space_attribute


class Policy(_PolicyBase):
    """Single objective Bayesian optimization with discrete search space"""

    predictor: Optional[Any]
    """Predictor used for Bayesian optimization.

    Base class is defined in physbo.predictor.
    If None, predictor is not yet initialized.
    """

    training: Variable
    """Training dataset containing pairs of input (X) and output (t).

    Stores all evaluated data points.
    """

    new_data: Optional[Variable]
    """New data that has been added to training but not yet to predictor.

    Set to None after predictor is updated.
    """

    test = _space_attribute(
        "test",
        """The set of candidate search points (physbo.Variable).

        Each row of X represents the feature vector of each search candidate.
        """,
    )

    actions = _space_attribute(
        "actions",
        """Array of action indices that have not been searched yet.

        Initially contains all candidate indices, and decreases as actions are evaluated.
        """,
    )

    history: History
    """History object storing search results including chosen actions,

    objective function values, and timing information.
    """

    config: SetConfig
    """Configuration object containing settings for learning and search."""

    mpicomm: Optional[Any]
    """MPI communicator for parallel computation.

    If None, MPI is not used (single process).
    """

    mpisize: int
    """Number of MPI processes. Set to 1 if MPI is not used."""

    mpirank: int
    """Rank of the current MPI process. Set to 0 if MPI is not used."""

    seed: Optional[int]
    """Seed parameter for np.random. Set by set_seed() method."""

    def __init__(self, test_X, config=None, initial_data=None, comm=None, rng=None):
        """

        Parameters
        ----------
        test_X: numpy.ndarray or physbo.Variable
            The set of candidates. Each row vector represents the feature vector of each search candidate.
        config: SetConfig object (physbo.misc.SetConfig)
        initial_data: tuple[np.ndarray, np.ndarray]
            The initial training datasets.
            The first elements is the array of actions and the second is the array of value of objective functions
        comm: MPI.Comm, optional
            MPI Communicator
        rng: None, "legacy", int, or numpy.random.Generator, optional
            Random number generator specification (default: "legacy").
            None or "legacy" uses the global numpy.random state and is
            bit-compatible with the historical behavior. An int seed or a
            Generator switches to a policy-owned generator whose state is
            stored on the policy (and thus included when the policy is
            pickled). Under MPI, pass the same specification on all ranks.
        """
        super().__init__(
            space=_DiscreteSpace(self._make_variable_X(test_X)),
            config=config,
            initial_data=initial_data,
            comm=comm,
            rng=rng,
        )

    def _new_history(self):
        return History()

    def _share_sample(self, mode, comm):
        # TS: one posterior sample for every process, drawn on rank 0
        predictor, training = self._current_model()
        return search_score.share_sample(mode, predictor, training, comm)

    def _share_samples(self, mode, model, K, comm):
        # TS: one posterior sample per model for every process, drawn on
        # rank 0
        predictor, _ = self._current_model()
        return search_score.share_samples(mode, predictor, model, K, comm)

    def _write_initial_data(self, initial_data):
        if len(initial_data) != 2:
            msg = "ERROR: initial_data should be 2-elements tuple or list (actions and objectives)"
            raise RuntimeError(msg)
        actions, fs = initial_data
        if len(actions) != len(fs):
            msg = "ERROR: len(initial_data[0]) != len(initial_data[1])"
            raise RuntimeError(msg)
        # Normalize fs to (N, 1) shape before passing to write()
        fs_normalized = normalize_t(fs, k=1)
        self.write(actions, fs_normalized)

    def _show_search_results(self, N_indeed, disp_pareto_set):
        utility.show_search_results(self.history, N_indeed)

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
        """
        Writing history (update history, not output to a file).

        Parameters
        ----------
        action: numpy.ndarray
            Indexes of actions.
        t:  numpy.ndarray
            N dimensional array (1D) or N x k dimensional array (2D).
            The negative energy of each search candidate (value of the objective function to be optimized).
            Will be normalized to (N, 1) shape internally.
            A non-finite value (NaN or +-Inf) marks a failed evaluation:
            the action is recorded in the history and consumed, but the
            observation is excluded from the training data and the
            best-value tracking.
        X:  numpy.ndarray
            N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of each search candidate.
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

        Note
        ----
        In this function, (X, t, Z) is added to the training data set (self.training), but not yet to predictor (self.predictor).
        self.newdata means such data.
        To add (X, t, Z) to the predictor, call self._update_predictor() after self.write().

        """
        if X is None:
            X = self.test.X[action, :]
            # Z is (1, N, n), extract (N, n) for objective_index=0
            Z = self.test.Z[0, action, :] if self.test.Z is not None else None
        else:
            # Get basis and convert to (1, N, n) format
            Z_basis = (
                self.predictor.get_basis(X) if self.predictor is not None else None
            )
            if Z_basis is not None:
                Z = Z_basis[np.newaxis, :, :]  # (N, n) -> (1, N, n)
            else:
                Z = None

        # Normalize t to (N, 1) shape
        t_normalized = normalize_t(t, k=1)

        self.history.write(
            t_normalized,
            action,
            time_total=time_total,
            time_update_predictor=time_update_predictor,
            time_get_action=time_get_action,
            time_run_simulator=time_run_simulator,
        )

        # Failed observations (non-finite t) are kept in the history and
        # their actions are consumed, but they are not used for training.
        valid = utility.finite_mask(t_normalized)
        X_ok = np.asarray(X)[valid]
        t_ok = t_normalized[valid]
        Z_ok = utility.mask_rows(Z, valid)
        if np.any(valid):
            self.training.add(X=X_ok, t=t_ok, Z=Z_ok)

        # remove the selected actions from the list of candidates if exists
        self.space.exclude(action)

        if np.any(valid):
            if self.new_data is None:
                self.new_data = Variable(X=X_ok, t=t_ok, Z=Z_ok)
            else:
                self.new_data.add(X=X_ok, t=t_ok, Z=Z_ok)

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
        num_rand_basis=None,
        optimizer=None,
        unify_method=None,
        ard=None,
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
        num_rand_basis: int, optional
            The number of basis function. If you choose 0, ordinary Gaussian process run.
            If None (default), the predictor the policy has is used, whatever its
            kind; a Gaussian process is made if it has none yet.
            The predictor is kept between calls of bayes_search. A new one is made,
            to be learned, when the predictor is not of the kind asked for by
            num_rand_basis or ard.
        optimizer: optimizer object, optional
            This is for compatibility with the range-based Policies.
            This is not used.
        ard: bool, optional
            If True, use Automatic Relevance Determination (ARD) for the Gaussian kernel.
            If None (default), as in the predictor the policy has; without ARD if
            it has none yet.
        ucb_beta: float
            Trade-off parameter between exploitation and exploration used when
            ``score == "UCB"``. The score is ``fmean + ucb_beta * fstd``;
            larger values favor exploration. Default is 1.0.

        Returns
        -------
        history: history object (physbo.search.discrete.results.history)
        """
        score = search_score.resolve(
            score, alpha=self.config.search.alpha, ucb_beta=ucb_beta, rng=self.rng
        )

        return self._bayes_search(
            score,
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
            predictor.fit(self.training, 0, comm=self.mpicomm, rng=self.rng)
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
            predictor = self._make_gp_predictor()
            predictor.fit(self.training, 0, comm=self.mpicomm, rng=self.rng)
            predictor.prepare(self.training)
            return predictor.get_post_fcov(self.training, X, diag)
        else:
            self._update_predictor()
            return self.predictor.get_post_fcov(self.training, X, diag)

    def get_num_dim(self):
        """
        Return the input dimension (number of features) of the search space.

        Returns
        -------
        int or None
            The number of dimensions. None if neither training nor test data is set.
        """
        if (
            self.training.X is not None
            and self.training.X.shape[0] > 0
        ):
            return self.training.X.shape[1]
        if self.test.X is not None and self.test.X.shape[0] > 0:
            return self.test.X.shape[1]
        return None

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
            predictor = self._make_gp_predictor()
            predictor.fit(self.training, 0, rng=self.rng)
            predictor.prepare(self.training)
            return predictor.get_permutation_importance(
                self.training,
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
                rng=self.rng,
            )
        else:
            self._update_predictor()
            return self.predictor.get_permutation_importance(
                self.training,
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
                rng=self.rng,
            )

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
            Trade-off parameter between exploitation and exploration used if mode = UCB.
            The score is ``fmean + ucb_beta * fstd``. Default is 1.0.

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

    def save(self, file_history, file_training=None, file_predictor=None):
        """

        Saving history, training and predictor into the corresponding files.

        This persists the search *results* in a portable form (the files
        can be loaded by runs with a different number of MPI processes).
        To suspend and later resume an execution bit-exactly (including
        the random number generator state), use ``save_checkpoint`` /
        ``load_checkpoint`` instead.

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

        See ``save`` for the difference from the checkpoint API
        (``save_checkpoint`` / ``load_checkpoint``).

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
            # Normalize t to (N, 1) shape
            t_normalized = normalize_t(t, k=1)
            self.training = Variable(X=X, t=t_normalized)
        else:
            self.training = Variable()
            self.training.load(file_training)

        if file_predictor is not None:
            with open(file_predictor, "rb") as f:
                self.predictor = pickle.load(f)

        N = self.history.total_num_search

        visited = self.history.chosen_actions[:N]
        self.space.exclude(visited)

        self._after_load()

    def _learn_hyperparameter(self, num_rand_basis):
        self.predictor.fit(
            self.training, num_rand_basis, comm=self.mpicomm, rng=self.rng
        )
        self._prepare_statistics()

    def _prepare_statistics(self):
        # Get basis and convert to (1, N, n) format for single-objective
        test_Z_basis = self.predictor.get_basis(self.test.X)
        training_Z_basis = self.predictor.get_basis(self.training.X)
        if test_Z_basis is not None:
            self.test.Z = test_Z_basis[np.newaxis, :, :]  # (N, n) -> (1, N, n)
        if training_Z_basis is not None:
            self.training.Z = training_Z_basis[np.newaxis, :, :]  # (N, n) -> (1, N, n)
        self.predictor.prepare(self.training)
        self.new_data = None

    def _clear_basis_values(self):
        self.training.Z = None
        self.test.Z = None

    def _update_predictor(self):
        """
        Updating predictor by using new_data, which means a subset of the trained data (self.training) that is not yet added to the predictor.
        """

        if self.new_data is not None:
            self.predictor.update(self.training, self.new_data)
            self.new_data = None

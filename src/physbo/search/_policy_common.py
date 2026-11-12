# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Code shared by the six policies.

``_PolicyBase`` is the base class of the six policies: the state they all
have, the loop of random_search and bayes_search, and the helpers that do
not depend on the kind of policy. What depends on it is asked from the
policy through a few methods (hooks), named in the docstrings below.

``_MultiObjectiveMethods`` and ``_UnifiedObjectiveMethods`` hold what the
policies of several and of unified objectives share between the two search
spaces. They have no constructor and are listed before the single-objective
policy among the bases of the policies that use them.
"""

import copy
import pickle
import time

import numpy as np

from . import utility
from ._checkpoint import CheckpointMixin
from .. import _mpi
from .. import gp
from ..gp import Predictor as gp_predictor
from ..blm import Predictor as blm_predictor
from ..misc import SetConfig
from .._variable import Variable
from .._rng import LegacyRNG, attach_rng, make_rng


def _mark_unlearned(predictor):
    """Mark a predictor a policy made (or is about to learn) as not learned."""
    predictor._unlearned = True


def _mark_learned(predictor):
    predictor.__dict__.pop("_unlearned", None)


def _is_learned(predictor):
    """Whether a predictor can be used without learning its hyperparameters.

    Any predictor can, except one that a policy made and has not learned
    (the mark above travels with the predictor, also through pickle), and a
    Bayesian linear model that was never fitted, which has no basis.
    """
    if predictor is None or getattr(predictor, "_unlearned", False):
        return False
    if isinstance(predictor, blm_predictor) and predictor.blm is None:
        return False
    return True


def _num_basis_of(predictors):
    """The number of basis functions of the Bayesian linear model(s) among
    ``predictors``: 0 if there is none (Gaussian processes), None if the
    model was never fitted (it is then fitted with the number of the
    config)."""
    for predictor in predictors:
        if isinstance(predictor, blm_predictor):
            return None if predictor.blm is None else predictor.blm.nbasis
    return 0


def _ard_of(predictors):
    """Whether the kernel of the first of ``predictors`` that tells has ARD;
    False if none does."""
    for predictor in predictors:
        model = getattr(predictor, "model", None)
        cov = getattr(getattr(model, "prior", None), "cov", None)
        if hasattr(cov, "ard"):
            return bool(cov.ard)
    return False


def run_simulator(simulator, action, comm=None):
    """
    Run the simulator once for the proposed points of a step.

    In a parallel run the simulator is called on rank 0 only, and its
    result is broadcast; an exception it raises is raised on every process.

    Parameters
    ----------
    simulator: callable
        Function that takes the proposed points (indices of candidates in
        the discrete policies, coordinates in the range policies) and
        returns the value(s) of the objective function(s).
    action: numpy.ndarray
        The proposed points.
    comm: MPI.Comm, optional
        MPI communicator

    Returns
    -------
    The return value of the simulator, unchanged.
    """
    if comm is None:
        return simulator(action)
    if comm.rank == 0:
        t, error = _mpi.call(comm, simulator, action)
    else:
        t, error = 0.0, None
    return _mpi.bcast(comm, t, error, root=0)


class _PolicyBase(CheckpointMixin):
    """Base class of the six policies.

    The constructor of a policy makes its search space (an object of
    ``physbo.search._space``, kept as ``space``), sets ``num_objectives``,
    and calls the constructor of this class, which sets up the rest and asks
    the policy for what depends on its kind:

    - ``_init_predictor_state()``: the attributes that hold the predictor(s);
    - ``_new_history()``: the history object;
    - ``_write_initial_data(initial_data)``: check and record the initial data.

    random_search and bayes_search of the policies keep their signatures and
    docstrings and call ``_random_search`` and ``_bayes_search``, which run
    the loop ``_search_loop`` and use:

    - ``space.random(N, rng, mpicomm)`` (through ``_get_random_action(N)``)
      and the proposal function given by the policy;
    - ``_has_predictors()``, ``_current_predictors()``,
      ``_use_predictor(predictor)`` and ``_new_predictor(is_rand_expans)`` to
      keep, take or create the predictor(s);
    - ``_learn_hyperparameter(num_rand_basis)``, ``_update_predictor()``,
      ``_prepare_statistics()`` and ``_clear_basis_values()``;
    - ``write(...)`` to record a step;
    - ``_show_search_results(N_indeed, disp_pareto_set)`` to print it.
    """

    # What the statistics of the predictor(s), and the basis values kept with
    # the data, were built from: the predictor(s) and the training data (and
    # the unify method in the unified policies), as a tuple of the objects
    # themselves. Before a search that does not learn, bayes_search compares
    # it with the objects in use and builds the statistics again if one of
    # them is another object: a predictor that was given, read by load() or
    # set by hand, other training data. () if nothing was built yet. None in
    # a policy restored from a state saved before the attribute existed; its
    # statistics are taken as they are.
    _statistics_sources = None

    def __init__(self, *, space, config=None, initial_data=None, comm=None, rng=None):
        self.space = space
        self.rng = make_rng(rng)
        self.training = Variable()
        self.new_data = None
        self._statistics_sources = ()
        self._init_predictor_state()
        self.history = self._new_history()
        if config is None:
            self.config = SetConfig()
        else:
            self.config = config

        self.ard = False
        if initial_data is not None:
            self._write_initial_data(initial_data)

        if comm is None:
            self.mpicomm = None
            self.mpisize = 1
            self.mpirank = 0
        else:
            self.mpicomm = comm
            self.mpisize = comm.size
            self.mpirank = comm.rank
            self.space.split(self.mpisize, self.mpirank)
            self.config.learning.is_disp = (
                self.config.learning.is_disp and self.mpirank == 0
            )

    def _init_predictor_state(self):
        """Create the attributes that hold the predictor(s), not built yet."""
        self.predictor = None

    def _new_history(self):
        """The (empty) history object of the policy."""
        raise NotImplementedError

    def _write_initial_data(self, initial_data):
        """Check ``initial_data`` and record it with write()."""
        raise NotImplementedError

    def _get_random_action(self, N):
        """``N`` points of the search space drawn at random (indices of
        candidates, removed from those not searched yet, or points of the
        box), the same on every process."""
        return self.space.random(N, self.rng, self.mpicomm)

    @property
    def _comm(self):
        """The communicator for collective calls, or None in a serial run
        (also when a communicator of a single process was given)."""
        return self.mpicomm if self.mpisize > 1 else None

    def _show_search_results(self, N_indeed, disp_pareto_set):
        """Print the results of the step just recorded, of ``N_indeed``
        points. ``disp_pareto_set`` is the argument of the multi-objective
        and unified policies; the single-objective ones have no use for it."""
        raise NotImplementedError

    def _search_loop(
        self,
        max_num_probes,
        N,
        simulator,
        is_disp,
        propose,
        *,
        update_predictor=None,
        score=None,
        disp_pareto_set=False,
    ):
        """
        The steps of random_search and bayes_search.

        Each step proposes points with ``propose()``, runs the simulator on
        them and records the result with write(). A step may propose fewer
        points than asked for (the candidates of a discrete search space run
        out); when it proposes none, the loop stops with a warning.

        Parameters
        ----------
        max_num_probes: int
            Number of steps.
        N: int
            Number of points asked for in each step.
        simulator: callable or None
            None for the interactive mode, in which the first proposal is
            returned instead of being evaluated.
        is_disp: bool
            Whether to print the progress.
        propose: callable
            Returns the proposal of a step.
        update_predictor: callable, optional
            Called with the index of the step before the proposal
            (bayes_search); its time is recorded.
        score: optional
            The acquisition function, for the header of a step of several
            points; None in random_search.
        disp_pareto_set: bool
            Passed to ``_show_search_results``.

        Returns
        -------
        The proposal in the interactive mode, None otherwise.
        """
        for n in range(0, max_num_probes):
            time_total = time.time()

            if update_predictor is not None:
                time_update_predictor = time.time()
                update_predictor(n)
                time_update_predictor = time.time() - time_update_predictor

            if is_disp and N > 1:
                utility.show_start_message_multi_search(
                    self.history.num_runs,
                    None if score is None else utility.score_name(score),
                )

            time_get_action = time.time()
            action = propose()
            time_get_action = time.time() - time_get_action

            N_indeed = len(action)
            if N_indeed == 0:
                if self.mpirank == 0:
                    print("WARNING: All actions have already searched.")
                break

            if simulator is None:
                return action

            time_run_simulator = time.time()
            t = run_simulator(simulator, action, self.mpicomm)
            time_run_simulator = time.time() - time_run_simulator

            time_total = time.time() - time_total
            if update_predictor is None:
                times_update_predictor = np.zeros(N_indeed, dtype=float)
            else:
                times_update_predictor = [time_update_predictor] * N_indeed
            self.write(
                action,
                t,
                time_total=[time_total] * N_indeed,
                time_update_predictor=times_update_predictor,
                time_get_action=[time_get_action] * N_indeed,
                time_run_simulator=[time_run_simulator] * N_indeed,
            )

            if is_disp:
                self._show_search_results(N_indeed, disp_pareto_set)
        return None

    def _random_search(
        self,
        max_num_probes,
        num_search_each_probe,
        simulator,
        is_disp,
        disp_pareto_set=False,
    ):
        """The body of random_search of the six policies."""
        self._require_comm()
        if self.mpirank != 0:
            is_disp = False

        N = int(num_search_each_probe)

        if is_disp:
            utility.show_interactive_mode(simulator, self.history)

        proposal = self._search_loop(
            max_num_probes,
            N,
            simulator,
            is_disp,
            lambda: self._get_random_action(N),
            disp_pareto_set=disp_pareto_set,
        )
        if proposal is not None:
            return proposal
        return copy.deepcopy(self.history)

    def _has_predictors(self):
        """Whether the policy has the predictor(s) it needs."""
        return self.predictor is not None

    def _current_predictors(self):
        """The predictor(s) the policy has, as a list (empty if none yet)."""
        return [] if self.predictor is None else [self.predictor]

    def _as_predictor_list(self, predictor):
        """The predictor(s) given to bayes_search, as a list."""
        return [predictor]

    def _use_predictor(self, predictor):
        """Take the predictor(s) given to bayes_search."""
        self.predictor = predictor

    def _new_predictor(self, is_rand_expans):
        """Create the predictor(s), not learned: Bayesian linear model(s) if
        ``is_rand_expans``, Gaussian process(es) otherwise."""
        self._init_predictor(is_rand_expans)

    def _predictor_matches(self, num_rand_basis, ard):
        """Whether the predictor(s) the policy has are of the kind asked for:
        Gaussian process (``num_rand_basis == 0``) or Bayesian linear model
        with ``num_rand_basis`` basis functions, with ARD or without. None
        asks for nothing, so that any predictor matches (None, None).

        The kind is read from the predictor itself. What cannot be read (a
        predictor of another class, a model without the ``ard`` attribute, the
        number of basis functions of a model that was never fitted) is taken
        to match.
        """
        for predictor in self._current_predictors():
            if not isinstance(predictor, (gp_predictor, blm_predictor)):
                continue
            if num_rand_basis is not None:
                is_blm = isinstance(predictor, blm_predictor)
                if is_blm != (num_rand_basis != 0):
                    return False
                if is_blm and predictor.blm is not None:
                    if predictor.blm.nbasis != num_rand_basis:
                        return False
            if ard is not None:
                cov = getattr(getattr(predictor.model, "prior", None), "cov", None)
                if hasattr(cov, "ard") and bool(cov.ard) != bool(ard):
                    return False
        return True

    def _clear_basis_values(self):
        """Forget the values of the basis functions kept with the data: they
        belong to a predictor that is being replaced."""
        self.training.Z = None

    def _prepare_statistics(self):
        """Build the statistics of the predictor(s) from the training data
        (and the basis values the data carry), without learning."""
        raise NotImplementedError

    def _sources_of_statistics(self):
        """The objects the statistics are built from (see
        ``_statistics_sources``)."""
        return (*self._current_predictors(), self.training)

    def _statistics_missing(self):
        """Whether the statistics have to be built again before a search
        that does not learn: they were built from other objects than those
        in use, or not at all."""
        built_from = self._statistics_sources
        if built_from is None:
            return False
        in_use = self._sources_of_statistics()
        return len(built_from) != len(in_use) or any(
            a is not b for a, b in zip(built_from, in_use)
        )

    def _after_load(self):
        """Called at the end of load(): the training data, and the
        predictor(s) if their file was given, were read from files."""
        self._statistics_sources = ()
        self.new_data = None

    def _bayes_search(
        self,
        score,
        propose,
        *,
        training,
        predictor,
        max_num_probes,
        num_search_each_probe,
        simulator,
        is_disp,
        interval,
        num_rand_basis,
        ard,
        unify_method=None,
        disp_pareto_set=False,
    ):
        """
        The body of bayes_search of the six policies.

        ``score`` is the acquisition function already resolved, and
        ``propose(N, K, alpha)`` returns the proposal of a step (``K`` and
        ``alpha`` are read from the config). ``training`` and ``predictor``
        are what the caller gave (``training_list`` / ``predictor_list`` in
        the multi-objective policies) or None. The other arguments are those
        of bayes_search; ``unify_method`` is that of the unified policies,
        and ``disp_pareto_set`` is passed to ``_show_search_results``.
        """
        self._require_comm()

        # The predictor(s) of the policy are kept between calls. New ones are
        # made, to be learned, when the policy has none, or when this call
        # asks for a kind (num_rand_basis, ard; None asks for nothing) that
        # the predictor(s) are not of. A predictor that was given, read by
        # load() or set by hand therefore stays until a call asks for another
        # kind.
        given = predictor is not None
        if given:
            keep = True
            in_use = self._as_predictor_list(predictor)
        else:
            keep = self._has_predictors() and self._predictor_matches(
                num_rand_basis, ard
            )
            in_use = self._current_predictors() if keep else []
        if interval < 0 and not (in_use and all(_is_learned(p) for p in in_use)):
            # checked before anything is changed; the condition is the same
            # on every process
            raise RuntimeError(
                "ERROR: bayes_search with interval < 0 does not learn the "
                "hyperparameters, but there is no learned predictor of the "
                "kind asked for (num_rand_basis, ard). Call bayes_search with "
                "interval >= 0 first, or give a learned predictor."
            )

        if self.mpirank != 0:
            is_disp = False

        old_disp = self.config.learning.is_disp
        self.config.learning.is_disp = is_disp

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        if ard is not None:
            self.ard = ard
        if unify_method is not None:
            self._set_unify_method(unify_method)

        if training is not None:
            self.training = training
            # the points not given to the predictor yet belonged to the
            # training data just replaced
            self.new_data = None

        if given:
            self._use_predictor(predictor)
            # a Bayesian linear model is fitted with the number of basis
            # functions asked for or, if none was, with the number it has
            num_basis = num_rand_basis or _num_basis_of(in_use)
        elif keep:
            num_basis = num_rand_basis or _num_basis_of(in_use)
        else:
            # what was not asked for is as in the predictor(s) replaced
            replaced = self._current_predictors()
            num_basis = num_rand_basis
            if num_basis is None:
                num_basis = _num_basis_of(replaced)
            if ard is None:
                self.ard = _ard_of(replaced)
            self._new_predictor(num_basis != 0)
            for made in self._current_predictors():
                _mark_unlearned(made)

        if given or not keep or training is not None or self._statistics_missing():
            # The basis values kept with the data are those of a predictor
            # that was replaced (here, by load() or by hand), and the
            # statistics of a predictor that was given may be those of other
            # data: both are built again, by the learning or below.
            self._clear_basis_values()
            self._statistics_sources = ()

        if max_num_probes > 0 or interval >= 0:
            utility.require_training_data(self.training)

        def learn():
            # not learned until the learning has gone through
            predictors = self._current_predictors()
            for p in predictors:
                _mark_unlearned(p)
            self._learn_hyperparameter(num_basis)
            for p in predictors:
                _mark_learned(p)
            self._statistics_sources = self._sources_of_statistics()

        if max_num_probes == 0 and interval >= 0:
            learn()
        elif interval < 0 and self._statistics_missing():
            # nothing is learned in this call, so the statistics of the
            # predictor(s) and the basis values kept with the data are
            # built here from the predictor(s) and the training data in use
            if self.training.X is not None and self.training.X.shape[0] > 0:
                self._prepare_statistics()
                self._statistics_sources = self._sources_of_statistics()

        N = int(num_search_each_probe)

        def update_predictor(n):
            if utility.is_learning(n, interval):
                learn()
            else:
                self._update_predictor()

        def propose_step():
            K = self.config.search.multi_probe_num_sampling
            alpha = self.config.search.alpha
            return propose(N, K, alpha)

        proposal = self._search_loop(
            max_num_probes,
            N,
            simulator,
            is_disp,
            propose_step,
            update_predictor=update_predictor,
            score=score,
            disp_pareto_set=disp_pareto_set,
        )
        if proposal is not None:
            self.config.learning.is_disp = old_disp
            return proposal

        self._update_predictor()
        self.config.learning.is_disp = old_disp
        return copy.deepcopy(self.history)

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
            self.predictor = self._make_blm_predictor()
        else:
            self.predictor = self._make_gp_predictor()

    def _make_predictor(self, predictor_class):
        """Create a predictor of the given class with the default model,
        with ARD if self.ard is True."""
        ard = self.ard
        num_dim = self.get_num_dim()
        model = gp.core.Model.create_default(ard=ard, num_dim=num_dim)
        return predictor_class(self.config, model=model)

    def _make_blm_predictor(self):
        """Create a BLM predictor, with ARD if self.ard is True."""
        return self._make_predictor(blm_predictor)

    def _make_gp_predictor(self):
        """Create a GP predictor, with ARD if self.ard is True."""
        return self._make_predictor(gp_predictor)

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

    def _warn_no_predictor(self, method_name):
        if self.mpirank != 0:
            return
        print("Warning: Since policy.predictor is not yet set,")
        print("         a GP predictor (num_rand_basis=0) is used for predicting")
        print("         If you want to use a BLM predictor (num_rand_basis>0),")
        print("         call bayes_search(max_num_probes=0, num_rand_basis=nrb)")
        print("         before calling {}.".format(method_name))

    def export_history(self):
        """
        Returning the information of the history.

        Returns
        -------

        """
        return self.history

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

    def get_kernel_length_scale(self):
        """
        Return the Gaussian kernel length scale(s) (width) of the predictor.

        With ARD, returns one length scale per input dimension; otherwise a single value.

        Returns
        -------
        numpy.ndarray or None
            Length scale(s). Shape (num_dim,) when ARD is used, (1,) otherwise.
            None if the predictor is not set or not a GP with Gaussian kernel.
        """
        if self.predictor is None:
            return None
        self._update_predictor()
        try:
            cov = self.predictor.model.prior.cov
        except AttributeError:
            return None
        if not hasattr(cov, "width"):
            return None
        return np.atleast_1d(np.asarray(cov.width).flatten())

    def set_seed(self, seed):
        """
        Setting a seed parameter.

        In the legacy RNG mode (default), this seeds the global
        ``numpy.random`` state, as before. In the Generator mode
        (``rng=`` given as int or Generator), the policy-owned generator
        is re-created from the seed and the global state is not touched.

        Parameters
        ----------
        seed: int
            seed number
        -------

        """
        self.seed = seed
        if isinstance(self.rng, LegacyRNG):
            self.rng.seed(self.seed)
        else:
            self.rng = np.random.default_rng(self.seed)


class _MultiObjectiveMethods:
    """Methods that the multi-objective policies of both search spaces share.

    They keep one predictor per objective (``predictor_list``). The class
    is listed before the single-objective policy among the bases of
    ``discrete_multi.Policy`` and ``range_multi.Policy``, so that these
    methods replace the single-objective ones.
    """

    def _init_predictor_state(self):
        self.predictor_list = [None for _ in range(self.num_objectives)]
        self.TS_candidate_num = None

    def _has_predictors(self):
        # a list with a hole is not kept: new predictors are made
        return len(self.predictor_list) > 0 and all(
            p is not None for p in self.predictor_list
        )

    def _current_predictors(self):
        return [p for p in self.predictor_list if p is not None]

    def _as_predictor_list(self, predictor_list):
        return list(predictor_list)

    def _use_predictor(self, predictor_list):
        self.predictor_list = predictor_list

    def _new_predictor(self, is_rand_expans):
        if is_rand_expans:
            self.predictor_list = [
                self._make_blm_predictor() for i in range(self.num_objectives)
            ]
        else:
            self.predictor_list = [
                self._make_gp_predictor() for i in range(self.num_objectives)
            ]

    def _show_search_results(self, N_indeed, disp_pareto_set):
        utility.show_search_results_mo(
            self.history, N_indeed, disp_pareto_set=disp_pareto_set
        )

    @staticmethod
    def _stack_basis(predictors, X):
        """The values at ``X`` of the basis functions of the predictors of
        the objectives, stacked into the (k, N, n) form of the training
        data, or None if any of them is a Gaussian process predictor, which
        has no basis functions."""
        z = [p.get_basis(X) for p in predictors]
        return np.stack(z, axis=0) if all(b is not None for b in z) else None

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
                predictor = self._make_gp_predictor()
                predictor.fit(
                    self.training, 0, comm=self.mpicomm, objective_index=i, rng=self.rng
                )
                predictor.prepare(self.training, objective_index=i)
                predictor_list.append(predictor)
        else:
            self._update_predictor()
            predictor_list = self.predictor_list[:]
        X = self._make_variable_X(xs)
        fmean = [
            predictor.get_post_fmean(self.training, X, objective_index=i)
            for i, predictor in enumerate(predictor_list)
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
                predictor = self._make_gp_predictor()
                predictor.fit(
                    self.training, 0, comm=self.mpicomm, objective_index=i, rng=self.rng
                )
                predictor.prepare(self.training, objective_index=i)
                predictor_list.append(predictor)
        else:
            self._update_predictor()
            predictor_list = self.predictor_list[:]
        X = self._make_variable_X(xs)
        fcov = [
            predictor.get_post_fcov(self.training, X, diag, objective_index=i)
            for i, predictor in enumerate(predictor_list)
        ]
        arr = np.array(fcov)
        if diag:
            return arr.T
        else:
            return np.einsum("nij->ijn", arr)

    def get_kernel_length_scale(self):
        """
        Return the Gaussian kernel length scale(s) for each objective.

        Returns
        -------
        list of numpy.ndarray or None
            List of length num_objectives; each element is the length scale(s)
            for that objective's GP, or None if not available.
        """
        if self.predictor_list is None or all(p is None for p in self.predictor_list):
            return None
        self._update_predictor()
        result = []
        for predictor in self.predictor_list:
            if predictor is None:
                result.append(None)
                continue
            try:
                cov = predictor.model.prior.cov
            except AttributeError:
                result.append(None)
                continue
            if not hasattr(cov, "width"):
                result.append(None)
                continue
            result.append(np.atleast_1d(np.asarray(cov.width).flatten()))
        return result

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
            self._warn_no_predictor("get_permutation_importance()")
            predictor_list = []
            for i in range(self.num_objectives):
                predictor = self._make_gp_predictor()
                predictor.fit(self.training, 0, objective_index=i, rng=self.rng)
                predictor.prepare(self.training, objective_index=i)
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
                self.training,
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
                objective_index=i,
                rng=self.rng,
            )

        return np.array(importance_mean).T, np.array(importance_std).T

    def save_predictor_list(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.predictor_list, f, 2)

    def save_training_list(self, file_name):
        obj = {"X": self.training.X, "t": self.training.t, "Z": self.training.Z}
        with open(file_name, "wb") as f:
            pickle.dump(obj, f, 2)

    def load_predictor_list(self, file_name):
        with open(file_name, "rb") as f:
            self.predictor_list = pickle.load(f)

    def load_training_list(self, file_name):
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

    def _update_predictor(self):
        if self.new_data is not None:
            for i in range(self.num_objectives):
                self.predictor_list[i].update(
                    self.training, self.new_data, objective_index=i
                )
            self.new_data = None


class _UnifiedObjectiveMethods:
    """Methods that the unified-objective policies of both search spaces share.

    They turn the objectives into one with ``unify_method`` and keep a
    single predictor for it, with the unified training data in
    ``training_unified``. The class is listed before the single-objective
    policy among the bases of ``discrete_unified.Policy`` and
    ``range_unified.Policy``.
    """

    def _init_predictor_state(self):
        self.training_unified = None
        self.predictor = None

    def _show_search_results(self, N_indeed, disp_pareto_set):
        utility.show_search_results_mo(
            self.history, N_indeed, disp_pareto_set=disp_pareto_set
        )

    def _attach_unified_basis(self, variable):
        """Give ``variable`` the values of the basis functions of the unified
        predictor at its points, in the (1, N, n) form of the training data.
        Nothing is done with the Gaussian process predictor, which has no
        basis functions."""
        Z = self.predictor.get_basis(variable.X)
        if Z is not None:
            variable.Z = Z[np.newaxis, :, :]

    def _require_unified_training(self, method_name):
        """The unified training data, or RuntimeError when they are not built yet.

        They are built by bayes_search, which receives ``unify_method``; before
        the first bayes_search the policy cannot unify the objectives. load()
        restores neither them nor unify_method, so after load() bayes_search
        has to be called again as well.
        """
        if self.training_unified is None:
            raise RuntimeError(
                f"ERROR: {method_name} needs the unified training data, which are "
                "built by bayes_search with unify_method. Call bayes_search first "
                "(also after load(), which does not restore them); with "
                "max_num_probes=0 and the default interval it only learns the "
                "predictor without proposing a point."
            )
        return self.training_unified

    def _unify_training(self, training: Variable) -> Variable:
        """
        Wrapper of the unify_method function
        """
        t_unified = self.unify_method(training.t)
        return Variable(X=training.X, t=t_unified.reshape(-1, 1))

    def _set_unify_method(self, unify_method):
        """Take the unify method given to bayes_search."""
        self.unify_method = unify_method
        # a unify_method without a generator of its own (e.g. ParEGO) draws
        # from the policy's rng, so that the Generator mode covers it
        attach_rng(unify_method, self.rng)

    def _sources_of_statistics(self):
        # the unified data are those of the unify method
        return (*super()._sources_of_statistics(), getattr(self, "unify_method", None))

    def _statistics_missing(self):
        return super()._statistics_missing() or self.training_unified is None

    def _after_load(self):
        super()._after_load()
        # load() restores neither the unified training data nor unify_method
        self.training_unified = None

    def _prepare_statistics(self):
        self.training_unified = self._unify_training(self.training)
        # the unified data carry the values of the basis functions, as
        # after _learn_hyperparameter, so that virtual observations with
        # Z can be added to them (several proposals per step)
        self._attach_unified_basis(self.training_unified)
        self.predictor.prepare(self.training_unified)
        self.new_data = None

    def _update_predictor(self):
        if self.new_data is not None:
            # Unifying again may change the unified objective of every point,
            # not only of the new ones (ParEGO draws new weights at each
            # call, the ranks of NDS move with the front), so the statistics
            # are built again from all of them rather than updated with the
            # new points.
            self._prepare_statistics()

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

        self._require_unified_training("get_permutation_importance()")
        if self.predictor is None:
            self._warn_no_predictor("get_permutation_importance()")
            predictor = self._make_gp_predictor()
            predictor.fit(self.training_unified, 0, rng=self.rng)
            predictor.prepare(self.training_unified)
            return predictor.get_permutation_importance(
                self.training_unified,
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
                rng=self.rng,
            )
        else:
            self._update_predictor()
            return self.predictor.get_permutation_importance(
                self.training_unified,
                n_perm,
                comm=self.mpicomm,
                split_features_parallel=split_features_parallel,
                rng=self.rng,
            )

    def _learn_hyperparameter(self, num_rand_basis):
        self.training_unified = self._unify_training(self.training)

        self.predictor.fit(
            self.training_unified, num_rand_basis, comm=self.mpicomm, rng=self.rng
        )
        self.predictor.prepare(self.training_unified)
        self._attach_unified_basis(self.training_unified)
        self.new_data = None

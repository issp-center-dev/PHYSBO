# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The proposal of a step of bayes_search, written once for the six policies.

``_get_actions`` chooses the points one after another: the first maximizes
the acquisition function, each following one its mean over K models in
which the points chosen so far are observed with values drawn from the
posterior. What depends on the kind of objective (the model, the virtual
observations, the scores) and on the search space (the maximization) is
asked from the policy and from its space.
"""

import copy
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from .test_predictor_consistency import (  # noqa: E402
    X,
    make_policy,
    max_difference_from_fresh_statistics,
    search_kwargs,
)

KINDS = [
    "discrete",
    "discrete_multi",
    "discrete_unified",
    "range",
    "range_multi",
    "range_unified",
]
K = 4


def learned(kind, num_rand_basis=0):
    """A policy with five random observations and learned predictors, its
    simulator and the arguments of bayes_search."""
    policy, simulator = make_policy(kind)
    policy.config.search.multi_probe_num_sampling = K
    policy.random_search(max_num_probes=5, simulator=simulator, is_disp=False)
    unify = None
    if kind.endswith("unified"):
        unify = physbo.search.unify.NDS(num_objectives=2)
    kwargs = search_kwargs(kind, policy, unify)
    kwargs["num_rand_basis"] = num_rand_basis
    policy.bayes_search(max_num_probes=0, interval=0, **kwargs)
    return policy, simulator, kwargs


def num_objectives(kind):
    return 2 if kind.endswith("multi") else 1


def predictors_of(model):
    predictors, _ = model
    return predictors if isinstance(predictors, list) else [predictors]


def pickle_of_space(policy):
    import pickle

    return pickle.dumps(policy.space)


def count_calls(monkeypatch, owner, name):
    calls = []
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return calls


# ---------------------------------------------------------------------------
# the model


@pytest.mark.parametrize("kind", KINDS)
def test_current_model_is_what_the_scores_are_computed_from(kind):
    policy, _, _ = learned(kind)
    predictors, training = policy._current_model()
    if kind.endswith("multi"):
        assert predictors is policy.predictor_list and training is policy.training
    elif kind.endswith("unified"):
        assert predictors is policy.predictor
        assert training is policy.training_unified
    else:
        assert predictors is policy.predictor and training is policy.training


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
def test_proposal_model(kind, num_rand_basis, monkeypatch):
    """The discrete policies score with their model as it is; the range
    policies with copies of their predictors prepared from the training
    data, which are not copied."""
    policy, _, _ = learned(kind, num_rand_basis)
    current = policy._current_model()
    prepared = []
    for cls in (physbo.gp.Predictor, physbo.blm.Predictor):
        original = cls.prepare

        def wrapper(self, *args, original=original, **kwargs):
            prepared.append(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "prepare", wrapper)
    model = policy._proposal_model()
    assert model[1] is current[1]
    if kind.startswith("discrete"):
        assert prepared == []
    else:
        # each copy once, and no predictor of the policy
        assert len(prepared) == len(predictors_of(model))
        assert all(a is b for a, b in zip(prepared, predictors_of(model)))
    for mine, used in zip(predictors_of(current), predictors_of(model)):
        if kind.startswith("discrete"):
            assert used is mine
        else:
            assert used is not mine and type(used) is type(mine)
    # the predictors of the policy are untouched
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
def test_virtual_observations(kind, num_rand_basis):
    """K samples of every objective at the points given, drawn from the
    generator of the policy, with the basis values of the predictors."""
    policy, _, _ = learned(kind, num_rand_basis)
    points = (
        [3, 11, 20]
        if kind.startswith("discrete")
        else [np.array([0.2, 0.3]), np.array([0.6, 0.1]), np.array([0.9, 0.9])]
    )
    variable = policy.space.to_variable(points)
    if variable.Z is None:
        policy._attach_basis(variable)
    if num_rand_basis:
        assert variable.Z.shape == (num_objectives(kind), 3, num_rand_basis)
    else:
        assert variable.Z is None

    state = copy.deepcopy(policy.rng)
    rng = copy.deepcopy(policy.rng)
    first = policy._draw_virtual_observations(variable, K)
    assert first.shape == (K, 3, num_objectives(kind))
    assert np.all(np.isfinite(first))
    again = policy._draw_virtual_observations(variable, K)
    assert not np.array_equal(first, again)
    policy.rng = state
    np.testing.assert_array_equal(policy._draw_virtual_observations(variable, K), first)

    # the samples of each objective are those of its predictor, drawn in the
    # order of the objectives
    predictors, training = policy._current_model()
    for i, predictor in enumerate(predictors_of((predictors, training))):
        expected = predictor.get_predict_samples(
            training, variable, K, objective_index=i, rng=rng
        )
        np.testing.assert_array_equal(first[:, :, i], expected)


@pytest.mark.parametrize("kind", KINDS)
def test_score_function_is_get_score_under_the_model(kind):
    policy, _, kwargs = learned(kind)
    mode = "HVPI" if kind.endswith("multi") else "EI"
    if kind.endswith("multi"):
        resolved = physbo.search.score_multi.resolve(mode, rng=policy.rng)
        expected_kwargs = {}
    else:
        resolved = physbo.search.score.resolve(mode, rng=policy.rng)
        expected_kwargs = {}
    test = physbo.Variable(X=X[:6])
    model = policy._current_model()
    scores = policy._score_function(resolved, model)(test)
    assert scores.shape == (6,)
    np.testing.assert_array_equal(
        scores, policy.get_score(mode, xs=X[:6], parallel=False, **expected_kwargs)
    )


# ---------------------------------------------------------------------------
# the proposal


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("N", [1, 3])
def test_proposal_has_the_form_of_the_search_space(kind, N):
    policy, _, kwargs = learned(kind)
    proposal = policy.bayes_search(
        max_num_probes=None, num_search_each_probe=N, interval=-1, **kwargs
    )
    if kind.startswith("discrete"):
        assert proposal.shape == (N,) and proposal.dtype.kind == "i"
        assert len(set(proposal.tolist())) == N
        # proposed candidates are taken from those not searched yet
        assert not set(proposal.tolist()) & set(policy.actions.tolist())
        assert len(policy.actions) == 25 - 5 - N
    else:
        assert proposal.shape == (N, 2) and proposal.dtype == float
        assert np.all((proposal >= 0.0) & (proposal <= 1.0))
        assert len(np.unique(proposal, axis=0)) == N


@pytest.mark.parametrize("kind", ["discrete", "discrete_multi", "discrete_unified"])
def test_proposal_is_shorter_when_the_candidates_run_out(kind, monkeypatch):
    policy, simulator, kwargs = learned(kind)
    policy.random_search(max_num_probes=18, simulator=simulator, is_disp=False)
    assert len(policy.actions) == 2
    draws = count_calls(monkeypatch, type(policy), "_draw_virtual_observations")
    proposal = policy.bayes_search(
        max_num_probes=None, num_search_each_probe=4, interval=-1, **kwargs
    )
    # for the second point and for the third, which does not exist: the
    # proposal stops there
    assert len(draws) == 2
    assert sorted(proposal.tolist()) == sorted(
        set(range(25)) - set(policy.history.chosen_actions[:23].tolist())
    )
    assert len(policy.actions) == 0
    # with none left the proposal is empty, and the search stops
    empty = policy._get_actions(lambda *args: None, 4, K)
    assert empty.shape == (0,) and empty.dtype.kind == "i"
    again = policy.bayes_search(
        max_num_probes=None, num_search_each_probe=4, interval=-1, **kwargs
    )
    assert isinstance(again, type(policy.history))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("N", [1, 2, 4])
def test_virtual_observations_are_drawn_once_for_each_point_after_the_first(
    kind, N, monkeypatch
):
    policy, _, kwargs = learned(kind)
    draws = count_calls(monkeypatch, type(policy), "_draw_virtual_observations")
    observed = count_calls(monkeypatch, type(policy), "_observe")
    policy.bayes_search(
        max_num_probes=None, num_search_each_probe=N, interval=-1, **kwargs
    )
    assert len(draws) == N - 1
    # at the points chosen so far, K sets each
    assert [call[0][1].X.shape[0] for call in draws] == list(range(1, N))
    assert all(call[0][2] == K for call in draws)
    # K models for each of these points, built once also where the scores
    # are asked for point by point (range)
    assert len(observed) == K * (N - 1)


@pytest.mark.parametrize("kind", KINDS)
def test_proposal_uses_the_observations_not_given_to_the_predictor_yet(kind):
    """_get_actions on its own, after write(): the predictor is updated
    first."""
    policy, simulator, kwargs = learned(kind)
    if kind.startswith("discrete"):
        action = policy.actions[:1].copy()
        policy.write(action, simulator(action))
        mode_module = physbo.search.score_multi if kind.endswith("multi") else None
    else:
        point = np.array([[0.4, 0.4]])
        policy.write(point, simulator(point))
        mode_module = physbo.search.score_multi if kind.endswith("multi") else None
    assert policy.new_data is not None
    module = mode_module or physbo.search.score
    mode = module.resolve(kwargs["score"], rng=policy.rng)
    policy._get_actions(mode, 1, K, optimizer=kwargs.get("optimizer"))
    assert policy.new_data is None
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
def test_virtual_observations_come_from_the_predictors_of_the_policy(
    kind, num_rand_basis, monkeypatch
):
    """Not from the copies the range policies score with, also after the
    statistics of the predictors were updated with a new observation."""
    policy, simulator, kwargs = learned(kind, num_rand_basis)
    kwargs.pop("num_rand_basis")
    policy.bayes_search(max_num_probes=1, simulator=simulator, interval=-1, **kwargs)
    drawing = []
    for cls in (physbo.gp.Predictor, physbo.blm.Predictor):
        original = cls.get_predict_samples

        def wrapper(self, *args, original=original, **kwargs):
            drawing.append(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "get_predict_samples", wrapper)
    policy.bayes_search(
        max_num_probes=None, num_search_each_probe=3, interval=-1, **kwargs
    )
    mine = predictors_of(policy._current_model())
    assert len(drawing) == 2 * len(mine)
    assert all(any(used is p for p in mine) for used in drawing)


@pytest.mark.parametrize("kind", KINDS)
def test_models_of_a_point_are_let_go_before_those_of_the_next(kind, monkeypatch):
    """The K models of a point are not kept while those of the next point
    are built."""
    policy, _, kwargs = learned(kind)
    built = []  # (number of points observed, weak reference to the training data)
    alive_from_earlier_points = []
    original = type(policy)._observe

    def observe(self, model, observed):
        n = observed.X.shape[0]
        gc.collect()
        alive_from_earlier_points.append(
            sum(1 for m, ref in built if m < n and ref() is not None)
        )
        built.append((n, weakref.ref(model[1])))
        return original(self, model, observed)

    monkeypatch.setattr(type(policy), "_observe", observe)
    policy.bayes_search(
        max_num_probes=None, num_search_each_probe=4, interval=-1, **kwargs
    )
    assert len(built) == 3 * K
    assert alive_from_earlier_points == [0] * (3 * K)


@pytest.mark.parametrize("kind", KINDS)
def test_without_virtual_observations_every_point_maximizes_the_score(
    kind, monkeypatch
):
    """multi_probe_num_sampling = 0: nothing is drawn and no model is built;
    each point maximizes the acquisition function itself."""
    policy, _, kwargs = learned(kind)
    policy.config.search.multi_probe_num_sampling = 0
    draws = count_calls(monkeypatch, type(policy), "_draw_virtual_observations")
    observed = count_calls(monkeypatch, type(policy), "_observe")
    if kind.startswith("discrete"):
        remaining = policy.actions.copy()
        scores = policy.get_score(kwargs["score"], parallel=False)
    proposal = policy.bayes_search(
        max_num_probes=None, num_search_each_probe=3, interval=-1, **kwargs
    )
    assert draws == [] and observed == []
    if kind.startswith("discrete"):
        # the three best candidates, the best first
        expected = remaining[np.argsort(-scores, kind="stable")[:3]]
        np.testing.assert_array_equal(proposal, expected)
    else:
        assert proposal.shape == (3, 2) and np.all(np.isfinite(proposal))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("N", [0, -1])
def test_number_of_points_per_step_must_be_positive(kind, N):
    policy, simulator, kwargs = learned(kind)
    candidates = pickle_of_space(policy)
    for max_num_probes in (1, None):
        with pytest.raises(ValueError, match="num_search_each_probe"):
            policy.bayes_search(
                max_num_probes=max_num_probes,
                num_search_each_probe=N,
                simulator=simulator if max_num_probes else None,
                interval=-1,
                **kwargs,
            )
    with pytest.raises(ValueError, match="num_search_each_probe"):
        policy.random_search(
            max_num_probes=1, num_search_each_probe=N, simulator=simulator
        )
    assert policy.history.total_num_search == 5
    assert pickle_of_space(policy) == candidates


@pytest.mark.parametrize("kind", KINDS)
def test_search_is_marked_as_running_only_while_it_runs(kind):
    """bayes_search is where the processes of a parallel run may communicate
    about the unified objective; the mark is not part of the state of the
    policy, also after a search that raised."""
    policy, simulator, kwargs = learned(kind)
    seen = []

    def watching(action):
        seen.append(policy._in_search)
        return simulator(action)

    assert policy._in_search is False
    policy.bayes_search(max_num_probes=1, simulator=watching, interval=-1, **kwargs)
    assert seen == [True]
    assert policy._in_search is False and "_in_search" not in vars(policy)

    def failing(action):
        raise KeyError("simulator")

    with pytest.raises(KeyError, match="simulator"):
        policy.bayes_search(max_num_probes=1, simulator=failing, interval=-1, **kwargs)
    assert policy._in_search is False and "_in_search" not in vars(policy)
    policy.random_search(max_num_probes=1, simulator=watching, is_disp=False)
    assert seen == [True, False]


@pytest.mark.parametrize("kind", KINDS)
def test_models_of_a_proposal_leave_the_policy_as_it_was(kind):
    """The virtual observations go into copies: the training data and the
    statistics of the predictors of the policy are those of the real
    observations."""
    policy, _, kwargs = learned(kind, 20)
    before = policy.training.X.copy(), policy.training.t.copy()
    policy.bayes_search(
        max_num_probes=None, num_search_each_probe=3, interval=-1, **kwargs
    )
    np.testing.assert_array_equal(policy.training.X, before[0])
    np.testing.assert_array_equal(policy.training.t, before[1])
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
def test_proposal_goes_through_the_space(kind, monkeypatch):
    """One maximization per point; the range policies pass their optimizer
    and the random point for the fallback, and the multi-objective range
    policy prepares its reference points before each."""
    policy, _, kwargs = learned(kind)
    calls = count_calls(monkeypatch, type(policy.space), "argmax")
    prepared = count_calls(monkeypatch, type(policy), "_prepare_scoring")
    policy.bayes_search(
        max_num_probes=None, num_search_each_probe=3, interval=-1, **kwargs
    )
    assert len(calls) == 3 and len(prepared) == 3
    for args, _ in calls:
        _, score_function, optimizer, mpicomm, random_point = args
        assert callable(score_function) and mpicomm is None
        if kind.startswith("range"):
            assert optimizer is kwargs["optimizer"]
            assert random_point().shape == (2,)


# ---------------------------------------------------------------------------
# the unified objective in a parallel run


class TwoProcesses:
    """A communicator of two processes, as far as the unified objective uses
    it: allgather returns what the other process would have sent."""

    size = 2

    def __init__(self, rank, from_the_other):
        self.rank = rank
        self.from_the_other = from_the_other
        self.sent = []

    def Get_size(self):
        return 2

    def Get_rank(self):
        return self.rank

    def allgather(self, message):
        self.sent.append(message)
        if self.rank == 0:
            return [message, self.from_the_other]
        return [self.from_the_other, message]


@pytest.mark.parametrize("kind", ["discrete_unified", "range_unified"])
def test_objectives_are_unified_on_rank_0_during_a_search(kind):
    """A unify method may draw random numbers (the weights of ParEGO), and
    the generators of the processes do not stay in step: during a search,
    which every process runs, rank 0 unifies and the others take its
    result. Outside a search each process unifies by itself, so that one
    process alone may ask for the posterior."""
    training = physbo.Variable(X=X[:4], t=np.arange(8.0).reshape(4, 2))
    calls = []

    def unify(t):
        calls.append(t)
        return t[:, 0] + 10 * t[:, 1]

    # rank 0 calls the unify method and sends its result
    comm = TwoProcesses(0, (None, None))
    policy, _ = make_policy(kind)
    policy.mpicomm, policy.mpisize, policy.mpirank = comm, 2, 0
    policy.unify_method = unify
    policy._in_search = True
    unified = policy._unify_training(training)
    assert len(calls) == 1
    np.testing.assert_array_equal(unified.t, [[10.0], [32.0], [54.0], [76.0]])
    np.testing.assert_array_equal(comm.sent[0][0], [10.0, 32.0, 54.0, 76.0])
    np.testing.assert_array_equal(unified.X, training.X)

    # another rank does not call it and takes what rank 0 sent
    comm = TwoProcesses(1, (np.array([1.0, 2.0, 3.0, 4.0]), None))
    policy, _ = make_policy(kind)
    policy.mpicomm, policy.mpisize, policy.mpirank = comm, 2, 1
    policy.unify_method = unify
    policy._in_search = True
    unified = policy._unify_training(training)
    assert len(calls) == 1
    np.testing.assert_array_equal(unified.t, [[1.0], [2.0], [3.0], [4.0]])

    # outside a search no process waits for another: each calls the unify
    # method, without communication
    del policy._in_search
    assert type(policy)._in_search is False
    comm.sent.clear()
    unified = policy._unify_training(training)
    assert len(calls) == 2 and comm.sent == []
    np.testing.assert_array_equal(unified.t, [[10.0], [32.0], [54.0], [76.0]])
    calls.pop()

    # in a serial run, also with a communicator of one process, it is called
    for comm in (None, SimpleNamespace(size=1, rank=0)):
        policy, _ = make_policy(kind)
        if comm is not None:
            policy.mpicomm = comm
        policy.unify_method = unify
        np.testing.assert_array_equal(
            policy._unify_training(training).t, [[10.0], [32.0], [54.0], [76.0]]
        )
    assert len(calls) == 3


@pytest.mark.parametrize("kind", ["discrete_unified", "range_unified"])
def test_processes_unify_anew_together_when_their_unified_data_differ(kind):
    """At the start of a search that does not learn, the processes compare
    their unified data, which each may have built by itself outside a
    search, and rebuild them if one differs or has none."""
    policy, _, kwargs = learned(kind)
    assert not policy._statistics_missing()
    mine = policy.training_unified.t.tobytes()
    other = (policy.training_unified.t + 1.0).tobytes()

    def answer_of_the_other_process(missing, unified):
        comm = TwoProcesses(0, (missing, unified))
        policy.mpicomm, policy.mpisize = comm, 2
        policy._in_search = True
        try:
            return policy._statistics_missing(), comm.sent
        finally:
            del policy._in_search
            policy.mpicomm, policy.mpisize = None, 1

    missing, sent = answer_of_the_other_process(False, mine)
    assert missing is False and sent == [(False, mine)]
    assert answer_of_the_other_process(False, other)[0] is True
    assert answer_of_the_other_process(False, None)[0] is True
    assert answer_of_the_other_process(True, mine)[0] is True

    # outside a search nothing is asked from the other processes
    comm = TwoProcesses(0, (True, other))
    policy.mpicomm, policy.mpisize = comm, 2
    assert policy._statistics_missing() is False and comm.sent == []

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The search-space objects of the policies (physbo.search._space).

A policy has its search space as ``policy.space``: the candidates of a
discrete policy and those not searched yet, or the box and the constraint
of a range policy, with the operations on them. The public attributes of
the policies (``test``, ``actions``, ``min_X``, ...) read and write it.
"""

import copy
import os
import pickle
import threading
from types import SimpleNamespace

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from physbo.search._checkpoint import FORMAT_VERSION  # noqa: E402
from physbo.search._space import _DiscreteSpace, _RangeSpace  # noqa: E402
from physbo.search.optimize.random import Optimizer as RandomOptimizer  # noqa: E402

X = np.array(
    [[a, b] for a in np.linspace(0.0, 1.0, 5) for b in np.linspace(0.0, 1.0, 5)]
)
DISCRETE = ["discrete", "discrete_multi", "discrete_unified"]
RANGE = ["range", "range_multi", "range_unified"]


def in_lower_half(x):
    return np.sum(np.atleast_2d(x), axis=1) <= 1.0


def nowhere(x):
    return np.zeros(np.atleast_2d(x).shape[0], dtype=bool)


def make_policy(kind, **kwargs):
    kwargs.setdefault("rng", 5)
    if kind not in ("discrete", "range"):
        kwargs["num_objectives"] = 2
    if kind.startswith("discrete"):
        return getattr(physbo.search, kind).Policy(test_X=X, **kwargs)
    return getattr(physbo.search, kind).Policy(
        min_X=np.zeros(2), max_X=np.ones(2), **kwargs
    )


def objectives(x, kind):
    x = np.atleast_2d(x)
    f1 = -np.sum((x - 0.25) ** 2, axis=1)
    if kind in ("discrete", "range"):
        return f1
    return np.c_[f1, -np.sum((x - 0.75) ** 2, axis=1)]


def simulator(kind):
    if kind.startswith("discrete"):
        return lambda action: objectives(X[action], kind)
    return lambda x: objectives(x, kind)


# ---------------------------------------------------------------------------
# the policy and its space


@pytest.mark.parametrize("kind", DISCRETE + RANGE)
def test_policy_has_the_space_of_its_kind(kind):
    policy = make_policy(kind)
    expected = _DiscreteSpace if kind.startswith("discrete") else _RangeSpace
    assert type(policy.space) is expected


@pytest.mark.parametrize("kind", DISCRETE)
def test_candidates_of_a_discrete_policy_live_in_its_space(kind):
    policy = make_policy(kind)
    assert set(vars(policy.space)) == {"test", "actions"}
    assert "test" not in vars(policy) and "actions" not in vars(policy)
    assert policy.test is policy.space.test
    assert policy.actions is policy.space.actions
    np.testing.assert_array_equal(policy.test.X, X)
    np.testing.assert_array_equal(policy.actions, np.arange(25))

    # assigning to the policy writes the space
    policy.actions = np.arange(3)
    np.testing.assert_array_equal(policy.space.actions, np.arange(3))
    other = physbo.Variable(X=X[:4])
    policy.test = other
    assert policy.space.test is other

    assert not hasattr(policy, "min_X")
    assert "candidate" in type(policy).test.__doc__


@pytest.mark.parametrize("kind", RANGE)
def test_box_of_a_range_policy_lives_in_its_space(kind):
    policy = make_policy(kind, constraint=in_lower_half)
    assert set(vars(policy.space)) == {"min_X", "max_X", "dim", "L_X", "constraint"}
    for name in ("min_X", "max_X", "dim", "L_X", "constraint"):
        assert name not in vars(policy)
        assert getattr(policy, name) is getattr(policy.space, name)
    np.testing.assert_array_equal(policy.min_X, np.zeros(2))
    np.testing.assert_array_equal(policy.max_X, np.ones(2))
    np.testing.assert_array_equal(policy.L_X, np.ones(2))
    assert policy.dim == 2
    assert policy.constraint is in_lower_half

    policy.constraint = None
    assert policy.space.constraint is None

    assert not hasattr(policy, "actions") and not hasattr(policy, "test")


@pytest.mark.parametrize("kind", DISCRETE + RANGE)
def test_space_is_copied_and_pickled_with_the_policy(kind):
    policy = make_policy(kind)
    policy.random_search(max_num_probes=3, simulator=simulator(kind), is_disp=False)
    for clone in (copy.deepcopy(policy), pickle.loads(pickle.dumps(policy))):
        assert clone.space is not policy.space
        assert set(vars(clone.space)) == set(vars(policy.space))
        if kind.startswith("discrete"):
            np.testing.assert_array_equal(clone.actions, policy.actions)
            assert len(clone.actions) == 22
        else:
            np.testing.assert_array_equal(clone.max_X, policy.max_X)


@pytest.mark.parametrize("kind", DISCRETE + RANGE)
def test_random_points_of_the_policy_are_those_of_its_space(kind):
    first, second = make_policy(kind), make_policy(kind)
    np.testing.assert_array_equal(
        first._get_random_action(3), second.space.random(3, second.rng, None)
    )


# ---------------------------------------------------------------------------
# a list of candidates


def test_discrete_space_starts_with_every_candidate():
    space = _DiscreteSpace(physbo.Variable(X=X))
    np.testing.assert_array_equal(space.actions, np.arange(25))


def test_discrete_space_splits_the_candidates_between_processes():
    shares = []
    for rank in range(3):
        space = _DiscreteSpace(physbo.Variable(X=X))
        space.split(3, rank)
        shares.append(space.actions)
    assert [len(s) for s in shares] == [9, 8, 8]
    np.testing.assert_array_equal(np.concatenate(shares), np.arange(25))


def test_discrete_space_excludes_candidates_by_index():
    space = _DiscreteSpace(physbo.Variable(X=X))
    space.exclude(np.array([3, 0, 24]))
    np.testing.assert_array_equal(
        space.actions, np.setdiff1d(np.arange(25), [0, 3, 24])
    )
    # candidates the process does not have (any more) are skipped
    space.exclude(np.array([3, 4]))
    assert 4 not in space.actions and len(space.actions) == 21

    part = _DiscreteSpace(physbo.Variable(X=X))
    part.split(3, 1)
    part.exclude(np.array([0, 9, 24, 30]))
    np.testing.assert_array_equal(part.actions, np.arange(10, 17))

    empty = _DiscreteSpace(physbo.Variable(X=X))
    empty.actions = np.array([], dtype=int)
    empty.exclude(np.array([1, 2]))
    assert len(empty.actions) == 0


def test_discrete_space_removes_by_position():
    space = _DiscreteSpace(physbo.Variable(X=X))
    space.actions = np.array([4, 8, 15])
    space.remove(1)
    np.testing.assert_array_equal(space.actions, [4, 15])


def test_discrete_space_draws_candidates_without_repeating():
    space = _DiscreteSpace(physbo.Variable(X=X[:6]))
    rng = np.random.default_rng(0)
    first = space.random(4, rng, None)
    assert len(first) == 4 and len(set(first.tolist())) == 4
    assert set(space.actions.tolist()) == set(range(6)) - set(first.tolist())
    # fewer left than asked for: those that are left, then none
    rest = space.random(4, rng, None)
    assert sorted(rest.tolist() + first.tolist()) == list(range(6))
    assert len(space.random(4, rng, None)) == 0


def test_discrete_space_finds_the_best_candidate():
    space = _DiscreteSpace(physbo.Variable(X=X))
    space.actions = np.array([4, 8, 15, 16])
    champion, local_champion, local_index = space.find_champion(
        np.array([0.1, 0.7, 0.3, 0.2]), None, None
    )
    assert (champion, local_champion, local_index) == (8, 8, 1)

    space.actions = np.array([], dtype=int)
    assert space.find_champion(np.zeros(0), None, None) == (-1, -1, -1)

    with pytest.raises(KeyError, match="scores"):
        space.find_champion(None, KeyError("scores"), None)


def test_discrete_space_has_a_local_state():
    space = _DiscreteSpace(physbo.Variable(X=X))
    space.split(2, 1)
    state = space.local_state()
    other = _DiscreteSpace(physbo.Variable(X=X))
    other.restore_local_state(state)
    np.testing.assert_array_equal(other.actions, np.arange(13, 25))


# ---------------------------------------------------------------------------
# a box


def test_range_space_checks_its_bounds():
    with pytest.raises(ValueError, match="min_X and max_X"):
        _RangeSpace(None, np.ones(2))
    with pytest.raises(AssertionError, match="dimension"):
        _RangeSpace(np.zeros(2), np.ones(3))
    with pytest.raises(AssertionError, match="less than"):
        _RangeSpace(np.ones(2), np.ones(2))
    space = _RangeSpace([0.0, -1.0], [2.0, 1.0])
    assert space.dim == 2 and space.constraint is None
    np.testing.assert_array_equal(space.L_X, [2.0, 2.0])
    assert space.local_state() == {}


def test_range_space_draws_points_of_the_box():
    space = _RangeSpace([0.0, -1.0], [2.0, 1.0])
    rng = np.random.default_rng(0)
    points = space.sample_uniform(200, rng)
    assert points.shape == (200, 2)
    assert np.all(points >= space.min_X) and np.all(points <= space.max_X)
    assert points[:, 0].max() > 1.5 and points[:, 1].min() < -0.5

    # without a constraint, random() is the uniform sample
    np.testing.assert_array_equal(
        space.random(5, np.random.default_rng(1), None),
        space.sample_uniform(5, np.random.default_rng(1)),
    )


def test_range_space_draws_feasible_points():
    space = _RangeSpace(np.zeros(2), np.ones(2), in_lower_half)
    points = space.random(50, np.random.default_rng(0), None)
    assert points.shape == (50, 2) and np.all(in_lower_half(points))

    # the sampling is given up after max_trials_per_point * N points
    judged = []

    def nowhere_counting(x):
        judged.append(np.atleast_2d(x).shape[0])
        return nowhere(x)

    space = _RangeSpace(np.zeros(2), np.ones(2), nowhere_counting)
    with pytest.raises(RuntimeError, match="not found in 60 random samples"):
        space.random(3, np.random.default_rng(0), None, max_trials_per_point=20)
    assert sum(judged) == 60


def test_range_space_hides_infeasible_points_from_the_optimizer():
    def fn(x):
        return float(np.sum(x))

    free = _RangeSpace(np.zeros(2), np.ones(2))
    assert free.constrained(fn) is fn

    space = _RangeSpace(np.zeros(2), np.ones(2), in_lower_half)
    wrapped = space.constrained(fn)
    assert wrapped(np.array([0.2, 0.3])) == 0.5
    assert wrapped(np.array([0.9, 0.9])) == -np.inf


def test_range_space_default_optimizer_draws_from_the_space():
    space = _RangeSpace(np.zeros(2), np.ones(2), in_lower_half)
    rng = np.random.default_rng(0)
    optimizer = space.default_optimizer(rng, 30)
    assert isinstance(optimizer, RandomOptimizer)
    assert optimizer.constraint is in_lower_half
    assert optimizer.max_trials_per_point == 30
    assert optimizer.rng is rng
    assert space.default_optimizer(rng).max_trials_per_point == 10000


def test_range_space_maximizes_with_the_optimizer():
    def fn(x):
        return -float(np.sum((np.asarray(x) - 0.2) ** 2))

    def corner(f, mpicomm=None):
        return np.array([0.1, 0.1])

    def outside(f, mpicomm=None):
        return np.array([0.9, 0.9])

    def wrong_dimension(f, mpicomm=None):
        return np.array([0.1, 0.1, 0.1])

    asked = []

    def random_point():
        asked.append(1)
        return np.array([0.3, 0.3])

    # without a constraint the point of the optimizer is taken as it is
    free = _RangeSpace(np.zeros(2), np.ones(2))
    np.testing.assert_array_equal(
        free.argmax(fn, outside, None, random_point), [0.9, 0.9]
    )

    space = _RangeSpace(np.zeros(2), np.ones(2), in_lower_half)
    np.testing.assert_array_equal(
        space.argmax(fn, corner, None, random_point), [0.1, 0.1]
    )
    assert asked == []
    # an infeasible point is replaced by the random point it is given
    with pytest.warns(RuntimeWarning, match="feasible random point"):
        point = space.argmax(fn, outside, None, random_point)
    np.testing.assert_array_equal(point, [0.3, 0.3])
    assert asked == [1]
    with pytest.raises(ValueError, match="3 dimension"):
        space.argmax(fn, wrong_dimension, None, random_point)

    # the default optimizer proposes feasible points by itself
    rng = np.random.default_rng(0)
    best = space.argmax(fn, space.default_optimizer(rng), None, random_point)
    assert in_lower_half(best)[0] and np.linalg.norm(best - 0.2) < 0.1
    assert asked == [1]


@pytest.mark.parametrize("bad", [np.inf, np.nan, None], ids=["inf", "nan", "none"])
def test_range_space_replaces_a_point_that_is_not_finite(bad):
    """Also when the constraint itself would accept it."""

    def anything(x):
        return np.ones(np.atleast_2d(x).shape[0], dtype=bool)

    def broken(f, mpicomm=None):
        return None if bad is None else np.array([bad, 0.5])

    space = _RangeSpace(np.zeros(2), np.ones(2), anything)
    with pytest.warns(RuntimeWarning, match="feasible random point"):
        point = space.argmax(lambda x: 0.0, broken, None, lambda: np.full(2, 0.3))
    np.testing.assert_array_equal(point, [0.3, 0.3])


def test_range_space_gives_the_communicator_to_the_optimizer_as_it_is():
    """Also a communicator of one process, with which the rest is serial."""
    one_process = SimpleNamespace(size=1, rank=0)
    seen = []

    def optimizer(f, mpicomm=None):
        seen.append(mpicomm)
        return np.array([0.1, 0.1])

    for constraint in (None, in_lower_half):
        space = _RangeSpace(np.zeros(2), np.ones(2), constraint)
        space.argmax(lambda x: 0.0, optimizer, one_process, lambda: None)
    assert seen == [one_process, one_process]


def test_discrete_space_takes_what_is_left_in_order():
    """With no more candidates left than asked for, they are returned as
    they are, without using the generator."""
    space = _DiscreteSpace(physbo.Variable(X=X[:6]))
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    np.testing.assert_array_equal(space.random(6, rng, None), np.arange(6))
    assert rng.bit_generator.state == before
    assert len(space.actions) == 0


# ---------------------------------------------------------------------------
# what stays with the policy


def counting(constraint):
    judged = []

    def wrapped(x):
        judged.append(np.atleast_2d(x).shape[0])
        return constraint(x)

    return wrapped, judged


@pytest.mark.parametrize("kind", RANGE)
def test_max_trials_per_point_is_an_attribute_of_the_policy(kind, monkeypatch):
    """Set on the class, in a subclass or on a policy, it is the limit of
    the rejection sampling and the one of the default optimizer."""
    cls = getattr(physbo.search, kind).Policy
    assert cls.max_trials_per_point == 10000

    def gives_up_after(policy, judged, expected):
        assert policy.max_trials_per_point == expected
        assert policy._default_optimizer().max_trials_per_point == expected
        with pytest.raises(RuntimeError, match=f"not found in {3 * expected} "):
            policy._get_random_action(3)
        # the points are drawn in batches of at least 3
        assert 3 * expected <= sum(judged) < 3 * expected + 3

    constraint, judged = counting(nowhere)
    policy = make_policy(kind, constraint=constraint)
    policy.max_trials_per_point = 7
    gives_up_after(policy, judged, 7)
    assert "max_trials_per_point" not in vars(policy.space)

    class Impatient(cls):
        max_trials_per_point = 5

    constraint, judged = counting(nowhere)
    kwargs = {} if kind == "range" else {"num_objectives": 2}
    policy = Impatient(
        min_X=np.zeros(2), max_X=np.ones(2), constraint=constraint, rng=5, **kwargs
    )
    gives_up_after(policy, judged, 5)

    monkeypatch.setattr(cls, "max_trials_per_point", 3)
    constraint, judged = counting(nowhere)
    gives_up_after(make_policy(kind, constraint=constraint), judged, 3)


@pytest.mark.parametrize("kind", DISCRETE + RANGE)
def test_attributes_of_the_space_need_the_constructor_of_the_policy(kind):
    """A subclass that reads or sets them before super().__init__() is told
    so, and a policy that was not constructed does not seem to have them."""
    cls = getattr(physbo.search, kind).Policy
    name = "actions" if kind.startswith("discrete") else "constraint"

    class TooEarly(cls):
        def __init__(self, **kwargs):
            setattr(self, name, None)
            super().__init__(**kwargs)

    kwargs = {} if kind in ("discrete", "range") else {"num_objectives": 2}
    if kind.startswith("discrete"):
        kwargs["test_X"] = X
    else:
        kwargs.update(min_X=np.zeros(2), max_X=np.ones(2))
    with pytest.raises(AttributeError, match=rf"'{name}' yet.*super\(\)\.__init__\(\)"):
        TooEarly(**kwargs)

    bare = cls.__new__(cls)
    assert not hasattr(bare, name)
    with pytest.raises(AttributeError, match=r"super\(\)\.__init__\(\)"):
        getattr(bare, name)


def bayes_kwargs(kind, policy):
    kwargs = dict(is_disp=False)
    if kind.endswith("unified"):
        kwargs["unify_method"] = physbo.search.unify.NDS(num_objectives=2)
    if kind.endswith("multi"):
        kwargs["score"] = "HVPI"
    return kwargs


@pytest.mark.parametrize("kind", RANGE)
def test_point_proposed_when_the_optimizer_fails_is_the_random_action_of_the_policy(
    kind, monkeypatch
):
    def outside(f, mpicomm=None):
        return np.array([0.9, 0.9])

    policy = make_policy(kind, constraint=in_lower_half)
    policy.random_search(max_num_probes=4, simulator=simulator(kind), is_disp=False)
    monkeypatch.setattr(policy, "_get_random_action", lambda N: np.full((N, 2), 0.25))
    with pytest.warns(RuntimeWarning, match="feasible random point") as record:
        proposal = policy.bayes_search(
            max_num_probes=None, optimizer=outside, **bayes_kwargs(kind, policy)
        )
    np.testing.assert_array_equal(proposal, [[0.25, 0.25]])
    # the warning points at the proposal of the policy, not at the helpers
    (warning,) = [w for w in record if "feasible random point" in str(w.message)]
    assert warning.filename.endswith(os.path.join("search", kind, "_policy.py"))


@pytest.mark.parametrize("kind", RANGE)
def test_policy_gives_its_communicator_to_the_optimizer_as_it_is(kind):
    one_process = SimpleNamespace(size=1, rank=0)
    policy = make_policy(kind, comm=one_process)
    seen = []

    def optimizer(f, mpicomm=None):
        seen.append(mpicomm)
        return np.array([0.1, 0.1])

    policy._optimize_score(lambda x: 0.0, optimizer)
    assert seen == [one_process]


@pytest.mark.parametrize("kind", DISCRETE + RANGE)
@pytest.mark.parametrize("how", ["restored", "copied", "assigned"])
def test_policy_without_the_communicator_of_its_state_does_not_search(
    kind, how, tmp_path
):
    """The state of a parallel run without a communicator (after pickle or
    copy, before set_comm), or a communicator the state does not know of,
    would be searched as a serial run on each process."""
    if how == "assigned":
        policy = make_policy(kind)
        policy.mpicomm = SimpleNamespace(size=2, rank=0)
    else:
        policy = make_policy(kind, comm=SimpleNamespace(size=2, rank=0))
        if how == "restored":
            policy = pickle.loads(pickle.dumps(policy))
        else:
            policy = copy.deepcopy(policy)
        assert (policy.mpicomm, policy.mpisize) == (None, 2)
    before = pickle.dumps(policy.space)
    with pytest.raises(RuntimeError, match="set_comm"):
        policy.random_search(max_num_probes=1, simulator=simulator(kind), is_disp=False)
    with pytest.raises(RuntimeError, match="set_comm"):
        policy.bayes_search(
            max_num_probes=1, simulator=simulator(kind), **bayes_kwargs(kind, policy)
        )
    assert policy.history.total_num_search == 0
    assert pickle.dumps(policy.space) == before
    # nor is it written to a checkpoint, which would hold one process only
    fname = tmp_path / "policy.ckpt"
    with pytest.raises(RuntimeError, match="set_comm"):
        policy.save_checkpoint(str(fname))
    assert not fname.exists()


# ---------------------------------------------------------------------------
# checkpoint


@pytest.mark.parametrize("kind", DISCRETE + RANGE)
def test_checkpoint_keeps_the_space(kind, tmp_path):
    assert FORMAT_VERSION == 2
    policy = make_policy(kind)
    policy.random_search(max_num_probes=3, simulator=simulator(kind), is_disp=False)
    fname = str(tmp_path / "policy.ckpt")
    policy.save_checkpoint(fname)

    with open(fname, "rb") as f:
        header = pickle.load(f)
        state = pickle.load(f)
    assert header["format_version"] == 2
    local = state["per_rank"][0]
    assert set(local) == {"rng", "space"}
    if kind.startswith("discrete"):
        np.testing.assert_array_equal(local["space"]["actions"], policy.actions)
    else:
        assert local["space"] == {}

    restored = type(policy).load_checkpoint(fname)
    assert type(restored.space) is type(policy.space)
    if kind.startswith("discrete"):
        np.testing.assert_array_equal(restored.actions, policy.actions)
        np.testing.assert_array_equal(restored.test.X, X)
    else:
        np.testing.assert_array_equal(restored.min_X, policy.min_X)
        np.testing.assert_array_equal(restored.max_X, policy.max_X)
    np.testing.assert_array_equal(
        restored._get_random_action(2), policy._get_random_action(2)
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a named pipe")
def test_checkpoint_is_read_in_one_pass(tmp_path):
    """The file is opened once and read from front to back, so that it may
    be a pipe."""
    policy = make_policy("discrete")
    policy.random_search(
        max_num_probes=3, simulator=simulator("discrete"), is_disp=False
    )
    fname = str(tmp_path / "policy.ckpt")
    policy.save_checkpoint(fname)
    with open(fname, "rb") as f:
        content = f.read()

    pipe = str(tmp_path / "pipe")
    os.mkfifo(pipe)

    def feed():
        with open(pipe, "wb") as f:
            f.write(content)

    # a daemon, so that it cannot keep the interpreter alive if the pipe is
    # never opened for reading
    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    try:
        restored = type(policy).load_checkpoint(pipe)
    finally:
        writer.join(timeout=10)
    np.testing.assert_array_equal(restored.actions, policy.actions)


def test_checkpoint_restores_the_candidates_of_each_process(tmp_path):
    """The candidates not searched yet are read from the part of the file
    that belongs to the process."""
    policy = make_policy("discrete")
    fname = str(tmp_path / "policy.ckpt")
    policy.save_checkpoint(fname)
    with open(fname, "rb") as f:
        header = pickle.load(f)
        state = pickle.load(f)
    state["per_rank"][0]["space"]["actions"] = np.array([3, 4, 5])
    with open(fname, "wb") as f:
        pickle.dump(header, f)
        pickle.dump(state, f)
    restored = type(policy).load_checkpoint(fname)
    np.testing.assert_array_equal(restored.actions, [3, 4, 5])

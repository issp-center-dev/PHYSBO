# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Constraints in the range (continuous) search: every proposal must be feasible."""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from physbo.search.optimize.random import Optimizer as RandomOptimizer  # noqa: E402
from physbo.search.utility import is_feasible, sample_feasible  # noqa: E402
from physbo.test_functions import multi_objective as mo  # noqa: E402

KINDS = ["single", "multi", "unified"]
SCORES = {
    "single": ["EI", "PI", "TS"],
    "multi": ["HVPI", "EHVI", "TS"],
    "unified": ["EI", "PI", "TS"],
}
TEST_FUNCTIONS = ["BinhKorn", "ChankongHaimes", "OsyczkaKundu", "ConstrEX"]


class Disk:
    """Two objectives on [-1, 1]^2 restricted to a disk (21% of the box for r = 0.5)."""

    nobj = 2
    dim = 2
    min_X = np.array([-1.0, -1.0])
    max_X = np.array([1.0, 1.0])

    def __init__(self, radius=0.5, column=False):
        self.radius = radius
        self.column = column

    def constraint(self, X):
        ok = np.sum(X**2, axis=1) <= self.radius**2
        return ok.reshape(-1, 1) if self.column else ok

    def __call__(self, X):
        X = np.atleast_2d(X)
        f1 = -np.sum((X - 1.0) ** 2, axis=1)
        f2 = -np.sum((X + 1.0) ** 2, axis=1)
        return np.c_[f1, f2]


def make_policy(kind, fn, constraint="default"):
    if constraint == "default":
        constraint = fn.constraint
    kwargs = dict(min_X=fn.min_X, max_X=fn.max_X, constraint=constraint)
    if kind == "single":
        return physbo.search.range.Policy(**kwargs)
    if kind == "multi":
        return physbo.search.range_multi.Policy(num_objectives=fn.nobj, **kwargs)
    return physbo.search.range_unified.Policy(num_objectives=fn.nobj, **kwargs)


def make_simulator(kind, fn):
    if kind == "single":
        return lambda X: np.sum(fn(X), axis=1)
    return fn


def bayes_kwargs(kind, fn, unify_method=None):
    if kind != "unified":
        return {}
    if unify_method is None:
        unify_method = physbo.search.unify.ParEGO(num_objectives=fn.nobj)
    return {"unify_method": unify_method}


def searched_X(policy):
    return policy.history.action_X[: policy.history.total_num_search]


class ReferenceRandomOptimizer:
    """The random optimizer before constraints were introduced."""

    def __init__(self, min_X, max_X, nsamples):
        self.min_X = np.array(min_X)
        self.max_X = np.array(max_X)
        self.nsamples = nsamples

    def __call__(self, fn, mpicomm=None):
        result_fx = -np.inf
        result_x = None
        for _ in range(self.nsamples):
            x = self.min_X + (self.max_X - self.min_X) * np.random.rand(self.min_X.size)
            fx = fn(x)
            if fx > result_fx:
                result_fx = fx
                result_x = x
        return result_x


# ---------------------------------------------------------------------------
# helpers in physbo.search.utility


def test_is_feasible_accepts_column_output():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert is_feasible(None, X).tolist() == [True, True]
    assert is_feasible(Disk().constraint, X).tolist() == [True, False]
    assert is_feasible(Disk(column=True).constraint, X).tolist() == [True, False]
    with pytest.raises(ValueError):
        is_feasible(lambda X: True, X)


def test_sample_feasible():
    disk = Disk()

    def draw(n):
        return 2 * np.random.rand(n, 2) - 1

    np.random.seed(0)
    X = sample_feasible(draw, 50, disk.constraint)
    assert X.shape == (50, 2)
    assert np.all(disk.constraint(X))
    assert sample_feasible(draw, 3, lambda X: np.zeros(len(X), dtype=bool), 100) is None


def test_sample_feasible_consumes_same_random_numbers_if_always_feasible():
    def draw(n):
        return np.random.rand(n, 3)

    np.random.seed(1)
    expected = draw(7)
    np.random.seed(1)
    X = sample_feasible(draw, 7, lambda X: np.ones(len(X), dtype=bool))
    np.testing.assert_array_equal(X, expected)


# ---------------------------------------------------------------------------
# random_search


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", TEST_FUNCTIONS + ["Disk", "DiskColumn"])
def test_random_search_is_feasible(kind, name):
    if name.startswith("Disk"):
        fn = Disk(column=(name == "DiskColumn"))
    else:
        fn = getattr(mo, name)()
    policy = make_policy(kind, fn)
    policy.set_seed(0)
    policy.random_search(
        max_num_probes=10,
        num_search_each_probe=3,
        simulator=make_simulator(kind, fn),
        is_disp=False,
    )
    X = searched_X(policy)
    assert X.shape == (30, fn.dim)
    assert np.all(fn.constraint(X))
    assert np.all((fn.min_X <= X) & (X <= fn.max_X))


@pytest.mark.parametrize("kind", KINDS)
def test_random_search_interactive_is_feasible(kind):
    fn = Disk()
    policy = make_policy(kind, fn)
    policy.set_seed(0)
    X = policy.random_search(max_num_probes=1, num_search_each_probe=20)
    assert X.shape == (20, 2)
    assert np.all(fn.constraint(X))


# ---------------------------------------------------------------------------
# bayes_search


@pytest.mark.parametrize(
    "kind,score", [(kind, score) for kind in KINDS for score in SCORES[kind]]
)
def test_bayes_search_is_feasible(kind, score):
    fn = Disk()
    policy = make_policy(kind, fn)
    policy.set_seed(0)
    sim = make_simulator(kind, fn)
    policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
    policy.bayes_search(
        max_num_probes=2,
        simulator=sim,
        score=score,
        is_disp=False,
        **bayes_kwargs(kind, fn),
    )
    X = searched_X(policy)
    assert X.shape == (7, 2)
    assert np.all(fn.constraint(X))


@pytest.mark.parametrize("kind", KINDS)
def test_bayes_search_multi_probe_is_feasible(kind):
    """The marginal score of the second and later points goes through the same path."""
    fn = Disk()
    policy = make_policy(kind, fn)
    policy.config.search.multi_probe_num_sampling = 3
    policy.set_seed(0)
    sim = make_simulator(kind, fn)
    policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
    policy.bayes_search(
        max_num_probes=1,
        num_search_each_probe=3,
        simulator=sim,
        score=SCORES[kind][0],
        is_disp=False,
        **bayes_kwargs(kind, fn),
    )
    X = searched_X(policy)
    assert X.shape == (8, 2)
    assert np.all(fn.constraint(X))


@pytest.mark.parametrize("kind", ["multi", "unified"])
@pytest.mark.parametrize("name", TEST_FUNCTIONS)
def test_bayes_search_is_feasible_for_test_functions(kind, name):
    fn = getattr(mo, name)()
    policy = make_policy(kind, fn)
    policy.set_seed(0)
    policy.random_search(max_num_probes=5, simulator=fn, is_disp=False)
    policy.bayes_search(
        max_num_probes=2,
        simulator=fn,
        score="HVPI" if kind == "multi" else "EI",
        is_disp=False,
        **bayes_kwargs(kind, fn),
    )
    assert np.all(fn.constraint(searched_X(policy)))


@pytest.mark.parametrize("unify", ["ParEGO", "NDS"])
def test_bayes_search_is_feasible_for_unify_methods(unify):
    fn = Disk()
    if unify == "ParEGO":
        unify_method = physbo.search.unify.ParEGO(num_objectives=fn.nobj)
    else:
        unify_method = physbo.search.unify.NDS(num_objectives=fn.nobj)
    policy = make_policy("unified", fn)
    policy.set_seed(0)
    policy.random_search(max_num_probes=5, simulator=fn, is_disp=False)
    policy.bayes_search(
        max_num_probes=2,
        simulator=fn,
        score="EI",
        is_disp=False,
        unify_method=unify_method,
    )
    assert np.all(fn.constraint(searched_X(policy)))


@pytest.mark.parametrize("kind", KINDS)
def test_bayes_search_interactive_is_feasible(kind):
    fn = Disk()
    policy = make_policy(kind, fn)
    policy.set_seed(0)
    sim = make_simulator(kind, fn)
    policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
    X = policy.bayes_search(
        max_num_probes=1,
        simulator=None,
        score=SCORES[kind][0],
        is_disp=False,
        **bayes_kwargs(kind, fn),
    )
    assert X.shape == (1, 2)
    assert np.all(fn.constraint(X))


@pytest.mark.parametrize("kind", KINDS)
def test_user_optimizer_without_constraint_is_feasible(kind):
    """An optimizer which does not know the constraint sees -inf at infeasible points."""
    fn = Disk()
    policy = make_policy(kind, fn)
    policy.set_seed(0)
    sim = make_simulator(kind, fn)
    seen = []

    def optimizer(f, mpicomm=None):
        best_x, best_f = None, -np.inf
        for x in 2 * np.random.rand(200, 2) - 1:
            fx = f(x)
            seen.append((fn.constraint(x.reshape(1, -1))[0], fx))
            if fx > best_f:
                best_x, best_f = x, fx
        return best_x

    policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
    policy.bayes_search(
        max_num_probes=2,
        simulator=sim,
        score=SCORES[kind][0],
        optimizer=optimizer,
        is_disp=False,
        **bayes_kwargs(kind, fn),
    )
    assert np.all(fn.constraint(searched_X(policy)))
    assert any(not ok for ok, _ in seen)
    assert all(np.isneginf(fx) for ok, fx in seen if not ok)
    assert all(np.isfinite(fx) for ok, fx in seen if ok)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "result",
    [None, np.array([0.9, 0.9]), np.array([[0.9, 0.9]]), np.array([np.nan, 0.0])],
    ids=["none", "infeasible", "infeasible-2d", "nan"],
)
def test_fallback_if_optimizer_fails(kind, result):
    """None (no feasible candidate) or an infeasible point is replaced, with a warning."""
    fn = Disk()
    policy = make_policy(kind, fn)
    policy.set_seed(0)
    sim = make_simulator(kind, fn)
    policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
    with pytest.warns(RuntimeWarning, match="feasible"):
        policy.bayes_search(
            max_num_probes=2,
            num_search_each_probe=2,
            simulator=sim,
            score=SCORES[kind][0],
            optimizer=lambda f, mpicomm=None: result,
            is_disp=False,
            **bayes_kwargs(kind, fn),
        )
    X = searched_X(policy)
    assert X.shape == (9, 2)
    assert np.all(fn.constraint(X))


def test_random_optimizer_returns_none_if_no_feasible_sample_is_found():
    optimizer = RandomOptimizer(
        min_X=[0.0, 0.0],
        max_X=[1.0, 1.0],
        nsamples=10,
        constraint=lambda X: np.zeros(X.shape[0], dtype=bool),
    )
    optimizer.max_trials_per_point = 10
    assert optimizer(lambda x: 1.0) is None


def test_random_optimizer_returns_feasible_sample_if_all_scores_are_minus_inf():
    """As without a constraint (the first sample, with a warning), and feasible."""
    disk = Disk()
    optimizer = RandomOptimizer(
        min_X=disk.min_X, max_X=disk.max_X, nsamples=10, constraint=disk.constraint
    )
    with pytest.warns(RuntimeWarning, match="-inf"):
        x = optimizer(lambda x: -np.inf)
    assert disk.constraint(x.reshape(1, -1))[0]


def test_random_optimizer_with_constraint():
    disk = Disk()
    optimizer = RandomOptimizer(
        min_X=disk.min_X, max_X=disk.max_X, nsamples=50, constraint=disk.constraint
    )
    np.random.seed(0)
    xs = []

    def f(x):
        xs.append(x)
        return x[0]

    best = optimizer(f)
    xs = np.array(xs)
    assert xs.shape == (50, 2)  # nsamples feasible points, not fewer
    assert np.all(disk.constraint(xs))
    np.testing.assert_array_equal(best, xs[np.argmax(xs[:, 0])])


# ---------------------------------------------------------------------------
# infeasible problems


@pytest.mark.parametrize("kind", KINDS)
def test_never_satisfied_constraint_raises(kind, monkeypatch):
    fn = Disk()
    policy = make_policy(
        kind, fn, constraint=lambda X: np.zeros(X.shape[0], dtype=bool)
    )
    policy.max_trials_per_point = 100
    monkeypatch.setattr(RandomOptimizer, "max_trials_per_point", 10)
    with pytest.raises(RuntimeError, match="feasible"):
        policy.random_search(max_num_probes=1, simulator=make_simulator(kind, fn))

    # initial data are not checked, so that bayes_search can be started
    X0 = 2 * np.random.rand(5, 2) - 1
    policy.write(X0, make_simulator(kind, fn)(X0))
    with pytest.warns(RuntimeWarning, match="feasible"):
        with pytest.raises(RuntimeError, match="feasible"):
            policy.bayes_search(
                max_num_probes=1,
                simulator=make_simulator(kind, fn),
                score=SCORES[kind][0],
                is_disp=False,
                **bayes_kwargs(kind, fn),
            )


# ---------------------------------------------------------------------------
# constraint=None must not change anything


@pytest.mark.parametrize("kind", KINDS)
def test_without_constraint_random_search_is_unchanged(kind):
    fn = Disk()
    policy = make_policy(kind, fn, constraint=None)
    assert policy.constraint is None
    policy.set_seed(3)
    X = policy.random_search(max_num_probes=1, num_search_each_probe=4)
    np.random.seed(3)
    expected = np.random.rand(4, 2) * (fn.max_X - fn.min_X) + fn.min_X
    np.testing.assert_array_equal(X, expected)


@pytest.mark.parametrize("kind", KINDS)
def test_always_satisfied_constraint_consumes_same_random_numbers(kind):
    fn = Disk()
    Xs = []
    for constraint in [None, lambda X: np.ones(X.shape[0], dtype=bool)]:
        policy = make_policy(kind, fn, constraint=constraint)
        policy.set_seed(3)
        Xs.append(policy.random_search(max_num_probes=1, num_search_each_probe=4))
    np.testing.assert_array_equal(Xs[0], Xs[1])


@pytest.mark.parametrize(
    "kind,score", [(kind, score) for kind in KINDS for score in SCORES[kind]]
)
def test_without_constraint_bayes_search_is_unchanged(kind, score):
    """The default optimizer draws the same random numbers as it did before."""
    fn = Disk()
    sim = make_simulator(kind, fn)
    Xs = []
    for optimizer in [None, ReferenceRandomOptimizer(fn.min_X, fn.max_X, 1000)]:
        policy = make_policy(kind, fn, constraint=None)
        policy.set_seed(5)
        policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
        policy.bayes_search(
            max_num_probes=2,
            simulator=sim,
            score=score,
            optimizer=optimizer,
            is_disp=False,
            **bayes_kwargs(kind, fn),
        )
        Xs.append(searched_X(policy).copy())
    np.testing.assert_array_equal(Xs[0], Xs[1])


# ---------------------------------------------------------------------------
# ODAT-SE optimizers


@pytest.mark.parametrize("algorithm", ["mapper", "minsearch", "exchange", "pamc"])
def test_odatse_optimizer_is_feasible(algorithm, tmp_path, monkeypatch):
    """ODAT-SE minimizes -fn, so that infeasible points have +inf.

    "bayes" is not tested: its Gaussian process cannot be trained with inf.
    """
    pytest.importorskip("odatse")
    from physbo.search.optimize.odatse import Optimizer, default_alg_dict

    import warnings

    monkeypatch.chdir(tmp_path)
    fn = Disk()
    policy = make_policy("multi", fn)
    policy.set_seed(0)
    policy.random_search(max_num_probes=5, simulator=fn, is_disp=False)
    optimizer = Optimizer(default_alg_dict(fn.min_X, fn.max_X, algorithm))
    with warnings.catch_warnings():
        # inf - inf in ODAT-SE, and the fallback of this package
        warnings.simplefilter("ignore", RuntimeWarning)
        policy.bayes_search(
            max_num_probes=2,
            simulator=fn,
            score="HVPI",
            optimizer=optimizer,
            is_disp=False,
        )
    assert np.all(fn.constraint(searched_X(policy)))

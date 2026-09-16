# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Sanity tests for the bundled benchmark (test) functions.

For the single-objective functions, the value at the reported global
minimum is compared with the documented optimum, and random points inside
the search domain are verified never to go below it.

Each function is written in the sense (minimization or maximization) of the
reference it follows, declared by ``is_maximization``; with
test_maximizer=True (the default) the returned values always describe a
maximization problem, and with test_maximizer=False a minimization problem.
"""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from physbo.test_functions import single_objective, multi_objective


# (class, documented global minimum value, or None to use the value at the
# reported minimum point only as a lower bound)
SINGLE_CASES = [
    (single_objective.Sphere, 0.0),
    (single_objective.Rastrigin, 0.0),
    (single_objective.Ackley, 0.0),
    (single_objective.Rosenbrock, 0.0),
    (single_objective.Beale, 0.0),
    (single_objective.Booth, 0.0),
    (single_objective.Matyas, 0.0),
    (single_objective.Himmelblau, 0.0),
    (single_objective.ThreeHumpCamel, 0.0),
    # note: physbo's Easom is shifted by +1, so its minimum value is 0
    (single_objective.Easom, 0.0),
    (single_objective.StyblinskiTang, None),
    (single_objective.Schaffer2, 0.0),
]


@pytest.mark.parametrize(
    "cls, fmin_ref", SINGLE_CASES, ids=[c.__name__ for c, _ in SINGLE_CASES]
)
def test_single_objective_minimum(cls, fmin_ref):
    fn = cls()
    Xopt = fn.global_minimum_point()
    assert Xopt.ndim == 2
    assert Xopt.shape[1] == fn.dim

    # the minimum points are inside the search domain
    assert np.all(Xopt >= fn.min_X - 1e-12)
    assert np.all(Xopt <= fn.max_X + 1e-12)

    # value at the reported minimum matches the documented optimum
    # (fn is negated by default: minimization value = -fn(x))
    fopt = -fn(Xopt)
    if fmin_ref is not None:
        np.testing.assert_allclose(
            fopt, np.full_like(fopt, fmin_ref), atol=1e-8
        )
    fmin = fopt.min()

    # no point in the domain goes below the reported minimum
    rng = np.random.RandomState(12345)
    X = rng.uniform(fn.min_X, fn.max_X, size=(3000, fn.dim))
    f = -fn(X)
    assert np.all(f >= fmin - 1e-8)

    # the reported minimum is a local minimum: small perturbations only
    # increase the function value
    for x0 in Xopt:
        perturbed = x0 + 1e-4 * rng.randn(200, fn.dim)
        perturbed = np.clip(perturbed, fn.min_X, fn.max_X)
        assert np.all(-fn(perturbed) >= fmin - 1e-8)


@pytest.mark.parametrize(
    "cls, fmin_ref", SINGLE_CASES, ids=[c.__name__ for c, _ in SINGLE_CASES]
)
def test_single_objective_maximizer_flag(cls, fmin_ref):
    fn_max = cls()
    fn_min = cls(test_maximizer=False)
    rng = np.random.RandomState(0)
    X = rng.uniform(fn_max.min_X, fn_max.max_X, size=(100, fn_max.dim))
    np.testing.assert_allclose(fn_max(X), -fn_min(X))


def test_single_objective_dim_mismatch():
    fn = single_objective.Sphere(dim=2)
    with pytest.raises(ValueError):
        fn(np.zeros((5, 3)))


MULTI_NAMES = [
    "FonsecaFleming",
    "Viennet",
    "BinhKorn",
    "ChankongHaimes",
    "KitaYabumotoMoriNishikawa",
    "Binh1",
    "Binh2",
    "Binh3",
    "Binh4",
    "Binh5",
    "Binh6",
    "Binh8",
    "Binh9",
    "Kursawe",
    "Schaffer1",
    "Schaffer2",
    "Poloni",
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "ZDT6",
    "OsyczkaKundu",
    "ConstrEX",
    "SRN",
    "VLMOP1",
    "VLMOP2",
    "VLMOP3",
]

# functions whose reference defines a maximization problem
MULTI_MAXIMIZATION = {"KitaYabumotoMoriNishikawa", "Binh4", "Poloni"}

# functions with (non-trivial) constraints
MULTI_CONSTRAINED = [
    "BinhKorn",
    "ChankongHaimes",
    "KitaYabumotoMoriNishikawa",
    "OsyczkaKundu",
    "ConstrEX",
]


def _nondominated_mask(Y):
    """Boolean mask of the non-dominated rows of Y (maximization)."""
    n = Y.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        ge = np.all(Y >= Y[i], axis=1)
        gt = np.any(Y > Y[i], axis=1)
        if np.any(ge & gt):
            mask[i] = False
    return mask


@pytest.mark.parametrize("name", MULTI_NAMES)
def test_multi_objective_smoke(name):
    fn = getattr(multi_objective, name)()

    assert fn.nobj >= 2
    assert fn.min_X.shape == (fn.dim,)
    assert fn.max_X.shape == (fn.dim,)
    assert np.all(fn.min_X < fn.max_X)

    rng = np.random.RandomState(12345)
    X = rng.uniform(fn.min_X, fn.max_X, size=(200, fn.dim))
    mask = np.asarray(fn.constraint(X)).reshape(-1)
    assert mask.dtype == bool
    X = X[mask]
    assert len(X) > 0, "constraint rejected all random points"

    f = fn(X)
    assert f.shape == (len(X), fn.nobj)
    assert np.all(np.isfinite(f))

    # the reference box used for hypervolume calculations is well-formed
    ref_min = np.asarray(fn.reference_min).reshape(-1)
    ref_max = np.asarray(fn.reference_max).reshape(-1)
    assert ref_min.shape == (fn.nobj,)
    assert ref_max.shape == (fn.nobj,)
    assert np.all(ref_min < ref_max)


@pytest.mark.parametrize("name", MULTI_NAMES)
def test_multi_objective_maximizer_flag(name):
    # test_maximizer only selects the sense of the returned values:
    # True gives a maximization problem, False the negated (minimization) one,
    # whichever sense the function is written in.
    fn_max = getattr(multi_objective, name)()
    fn_min = getattr(multi_objective, name)(test_maximizer=False)
    assert fn_max.test_maximizer is True
    assert fn_min.test_maximizer is False
    assert fn_max.is_maximization == (name in MULTI_MAXIMIZATION)

    rng = np.random.RandomState(0)
    X = rng.uniform(fn_max.min_X, fn_max.max_X, size=(100, fn_max.dim))
    np.testing.assert_allclose(fn_max(X), -fn_min(X))

    # f itself is written in the declared sense
    if fn_max.is_maximization:
        np.testing.assert_allclose(fn_max(X), fn_max.f(X))
    else:
        np.testing.assert_allclose(fn_min(X), fn_min.f(X))

    # the reference box is flipped and swapped together with the values
    np.testing.assert_allclose(fn_max.reference_min, -fn_min.reference_max)
    np.testing.assert_allclose(fn_max.reference_max, -fn_min.reference_min)


@pytest.mark.parametrize("name", MULTI_CONSTRAINED)
def test_multi_objective_constraint_is_active(name):
    # the constraints must actually cut the default search space; otherwise
    # the problem degenerates into an unconstrained box problem
    fn = getattr(multi_objective, name)()
    num = 11
    X = fn.make_grid(num)
    assert 0 < len(X) < num**fn.dim


@pytest.mark.parametrize(
    "name",
    [
        "KitaYabumotoMoriNishikawa",
        "Binh1",
        "BinhKorn",
        "Poloni",
        "FonsecaFleming",
        "Schaffer1",
        "Schaffer2",
        "Viennet",
    ],
)
def test_multi_objective_reference_box_covers_range(name):
    # for these functions the reference box is the range of the objectives
    # over the (feasible) search space, so every feasible value lies inside
    fn = getattr(multi_objective, name)()
    X = fn.make_grid(41)
    f = fn(X)
    assert np.all(f >= fn.reference_min - 1e-9)
    assert np.all(f <= fn.reference_max + 1e-9)


def test_schaffer1_reference_box_follows_domain():
    # Deb (2001) uses -A <= x <= A with A up to 1e5; the box must follow A
    fn = multi_objective.Schaffer1(min_X=-1000.0, max_X=1000.0, test_maximizer=False)
    np.testing.assert_allclose(fn.reference_min, [0.0, 0.0])
    np.testing.assert_allclose(fn.reference_max, [1000.0**2, 1002.0**2])


def test_kita_pareto_set():
    # Kita et al. (1996): both objectives increase with x2, so the
    # Pareto-optimal set lies on the boundary g1: x2 = 6.5 - x1/6, x1 in [0, 3]
    fn = multi_objective.KitaYabumotoMoriNishikawa()
    num = 71  # grid spacing 0.1 on [0, 7]
    h = 7.0 / (num - 1)
    X = fn.make_grid(num)
    Y = fn(X)
    P = X[_nondominated_mask(Y)]
    assert len(P) > 0
    assert np.all(P[:, 0] <= 3.0 + h + 1e-12)
    assert np.all(P[:, 1] >= 6.5 - P[:, 0] / 6.0 - h - 1e-12)
    # the Pareto-optimal set is outside the box [-7, 4]^2 used by the
    # widely circulated variant of this problem
    assert P[:, 1].max() > 6.0


def test_binh1_pareto_set():
    # Binh (1999) case 1 (unconstrained): the Pareto-optimal set is the
    # segment x1 = x2 in [0, 5]
    fn = multi_objective.Binh1()
    num = 61  # grid spacing 0.25 on [-5, 10]
    h = 15.0 / (num - 1)
    X = fn.make_grid(num)
    P = X[_nondominated_mask(fn(X))]
    assert len(P) > 0
    np.testing.assert_allclose(P[:, 0], P[:, 1], atol=h + 1e-12)
    assert np.all(P[:, 0] >= -h - 1e-12)
    assert np.all(P[:, 0] <= 5.0 + h + 1e-12)
    assert P[:, 0].max() > 4.0  # the segment is not truncated like BinhKorn


def test_fonseca_fleming_domains():
    assert np.all(multi_objective.FonsecaFleming().min_X == -4.0)
    assert np.all(multi_objective.FonsecaFleming().max_X == 4.0)
    # VLMOP2 uses the search space of Van Veldhuizen and Lamont (1999)
    fn = multi_objective.VLMOP2()
    assert fn.name == "VLMOP2"
    assert np.all(fn.min_X == -2.0)
    assert np.all(fn.max_X == 2.0)
    # explicit arguments still win
    fn = multi_objective.VLMOP2(dim=3, min_X=-1.0, max_X=1.5)
    assert fn.dim == 3
    assert np.all(fn.min_X == -1.0)
    assert np.all(fn.max_X == 1.5)
    fn = multi_objective.VLMOP2(3, -1.0)
    assert np.all(fn.min_X == -1.0)
    assert np.all(fn.max_X == 2.0)


def _zdt_reference(name, X):
    # independent transcription of Zitzler, Deb, Thiele (2000)
    n = X.shape[1]
    x1 = X[:, 0]
    rest = X[:, 1:]
    if name == "ZDT4":
        g = 1.0 + 10.0 * (n - 1) + np.sum(rest**2 - 10.0 * np.cos(4.0 * np.pi * rest), axis=1)
    elif name == "ZDT6":
        g = 1.0 + 9.0 * (np.sum(rest, axis=1) / (n - 1)) ** 0.25
    else:
        g = 1.0 + 9.0 * np.sum(rest, axis=1) / (n - 1)
    if name == "ZDT6":
        f1 = 1.0 - np.exp(-4.0 * x1) * np.sin(6.0 * np.pi * x1) ** 6
    else:
        f1 = x1
    if name in ("ZDT1", "ZDT4"):
        h = 1.0 - np.sqrt(f1 / g)
    elif name in ("ZDT2", "ZDT6"):
        h = 1.0 - (f1 / g) ** 2
    elif name == "ZDT3":
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
    return np.c_[f1, g * h]


@pytest.mark.parametrize("name", ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"])
def test_zdt_matches_definition(name):
    fn = getattr(multi_objective, name)(test_maximizer=False)
    rng = np.random.RandomState(7)
    X = rng.uniform(fn.min_X, fn.max_X, size=(200, fn.dim))
    np.testing.assert_allclose(fn(X), _zdt_reference(name, X), rtol=1e-12, atol=1e-12)


def test_multi_objective_gaussian():
    centers = np.array([[1.0, 0.0], [-1.0, 0.0]])
    fn = multi_objective.Gaussian(centers=centers)
    assert fn.nobj == 2
    assert fn.dim == 2

    # each objective is maximal at its own center
    f_centers = fn(centers)
    rng = np.random.RandomState(12345)
    X = rng.uniform(fn.min_X, fn.max_X, size=(500, 2))
    f = fn(X)
    for k in range(2):
        assert np.all(f[:, k] <= f_centers[k, k] + 1e-12)


def test_vlmop2_optimum():
    # the first objective of VLMOP2 attains its minimum 0 at
    # x = (1/sqrt(n), ..., 1/sqrt(n)) and the second at its negation
    fn = multi_objective.VLMOP2(test_maximizer=False)
    n = fn.dim
    x1 = np.full((1, n), 1.0 / np.sqrt(n))
    f = fn(np.r_[x1, -x1])
    assert f[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert f[1, 1] == pytest.approx(0.0, abs=1e-12)

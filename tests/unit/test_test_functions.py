# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Sanity tests for the bundled benchmark (test) functions.

For the single-objective functions, the value at the reported global
minimum is compared with the documented optimum, and random points inside
the search domain are verified never to go below it. All functions are
defined as minimization problems and negated when test_maximizer=True
(the default).
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
    "VLMOP1",
    "VLMOP2",
    "VLMOP3",
]


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

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Several proposals per step with a Bayesian linear model (issue #131).

The virtual observations of the points already chosen in a step need the
values Z of the basis functions, like the training data they are added
to: in the range policies they had none, and the unified policies dropped
Z from the unified training data when updating the predictor, so the
second step failed.
"""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

KINDS = ["range", "range_multi", "range_unified", "discrete_unified"]
GRID = np.array([[x, y] for x in np.linspace(0, 1, 6) for y in np.linspace(0, 1, 6)])


def f_single(x):
    return -np.sum(np.atleast_2d(x) ** 2, axis=1)


def f_multi(x):
    x = np.atleast_2d(x)
    return np.c_[-np.sum(x**2, axis=1), -np.sum((x - 1.0) ** 2, axis=1)]


def unify():
    return dict(score="EI", unify_method=physbo.search.unify.ParEGO(num_objectives=2))


def make_policy(kind):
    kwargs = dict(min_X=np.zeros(2), max_X=np.ones(2), rng=0)
    if kind == "range":
        return physbo.search.range.Policy(**kwargs), f_single, dict(score="EI")
    if kind == "range_multi":
        policy = physbo.search.range_multi.Policy(num_objectives=2, **kwargs)
        return policy, f_multi, dict(score="EHVI")
    if kind == "range_unified":
        policy = physbo.search.range_unified.Policy(num_objectives=2, **kwargs)
        return policy, f_multi, unify()
    policy = physbo.search.discrete_unified.Policy(test_X=GRID, num_objectives=2, rng=0)
    return policy, lambda actions: f_multi(GRID[actions]), unify()


def searched_X(policy, N):
    if hasattr(policy.history, "action_X"):
        return policy.history.action_X[:N]
    return GRID[policy.history.chosen_actions[:N]]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20])
def test_several_proposals_per_step(kind, num_rand_basis):
    """Three steps of two proposals: the second step adds virtual observations
    to training data that the predictor update has rebuilt."""
    policy, sim, kwargs = make_policy(kind)
    policy.random_search(max_num_probes=3, simulator=sim, is_disp=False)
    policy.bayes_search(
        max_num_probes=3,
        num_search_each_probe=2,
        num_rand_basis=num_rand_basis,
        simulator=sim,
        is_disp=False,
        **kwargs,
    )
    N = policy.history.total_num_search
    assert N == 3 + 3 * 2
    X = searched_X(policy, N)
    assert np.all((0.0 <= X) & (X <= 1.0))
    assert np.all(np.isfinite(policy.history.fx[:N]))

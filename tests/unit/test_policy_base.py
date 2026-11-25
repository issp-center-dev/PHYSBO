# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""What the six policies get from their common base class: the constructor
and the search loop (physbo/search/_policy_common.py)."""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

KINDS = [
    "discrete",
    "discrete_multi",
    "discrete_unified",
    "range",
    "range_multi",
    "range_unified",
]
DISCRETE_KINDS = KINDS[:3]
RANGE_KINDS = KINDS[3:]

X = np.linspace(0.0, 1.0, 8)[:, None]


def objectives(x, kind):
    """One objective, or two of which neither dominates everywhere."""
    x = np.atleast_2d(x)
    f1 = -np.sum((x - 0.3) ** 2, axis=1)
    if kind in ("discrete", "range"):
        return f1
    return np.c_[f1, -np.sum((x - 0.6) ** 2, axis=1)]


def make_config(is_disp):
    config = physbo.misc.SetConfig()
    config.learning.is_disp = is_disp
    config.learning.num_init_params_search = 2
    config.learning.max_epoch = 50
    config.learning.max_epoch_init_params_search = 10
    return config


def make_policy(kind, is_disp=False, **space):
    """A policy and its simulator; ``space`` replaces the default search space."""
    kwargs = dict(config=make_config(is_disp))
    if kind not in ("discrete", "range"):
        kwargs["num_objectives"] = 2
    if kind in DISCRETE_KINDS:
        kwargs["test_X"] = space.get("test_X", X)
        policy = getattr(physbo.search, kind).Policy(**kwargs)
        return policy, lambda action: objectives(X[action], kind)
    kwargs["min_X"] = space.get("min_X", np.zeros(1))
    kwargs["max_X"] = space.get("max_X", np.ones(1))
    policy = getattr(physbo.search, kind).Policy(**kwargs)
    return policy, lambda x: objectives(x, kind)


def bayes_kwargs(kind):
    kwargs = {}
    if kind.endswith("unified"):
        kwargs["unify_method"] = physbo.search.unify.NDS(num_objectives=2)
    if kind in RANGE_KINDS:
        kwargs["optimizer"] = physbo.search.optimize.random.Optimizer(
            np.zeros(1), np.ones(1), nsamples=20
        )
    return kwargs


# ---------------------------------------------------------------------------
# the search loop


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["simulator", "interactive", "no probe"])
@pytest.mark.parametrize("configured", [True, False])
def test_bayes_search_restores_the_display_of_the_learning(
    kind, mode, configured, capsys
):
    """bayes_search sets config.learning.is_disp to its own is_disp while it
    runs; afterwards the config says what it said before, however the call
    ends: after the last step, with a proposal (interactive mode), or
    without any step."""
    policy, simulator = make_policy(kind, is_disp=configured)
    policy.set_seed(0)
    policy.random_search(max_num_probes=3, simulator=simulator, is_disp=False)

    kwargs = dict(is_disp=not configured, **bayes_kwargs(kind))
    if mode == "simulator":
        policy.bayes_search(max_num_probes=1, simulator=simulator, **kwargs)
    elif mode == "interactive":
        proposal = policy.bayes_search(max_num_probes=1, **kwargs)
        assert len(proposal) == 1
    else:
        policy.bayes_search(max_num_probes=0, **kwargs)
    capsys.readouterr()

    assert policy.config.learning.is_disp is configured


@pytest.mark.parametrize("kind", DISCRETE_KINDS)
def test_random_search_takes_what_is_left_and_then_stops(kind, capsys):
    """With fewer candidates left than asked for, a step takes them all; with
    none left, random_search warns and returns the history, also in the
    interactive mode. The three discrete policies behave alike."""
    policy, simulator = make_policy(kind)
    policy.set_seed(0)
    n = X.shape[0]  # 8 candidates, 3 per step
    history = policy.random_search(
        max_num_probes=3, num_search_each_probe=3, simulator=simulator, is_disp=False
    )
    assert history.total_num_search == n
    assert sorted(history.chosen_actions[:n].tolist()) == list(range(n))
    assert len(policy.actions) == 0
    assert "All actions have already searched" not in capsys.readouterr().out

    for interactive in (False, True):
        # asked for two more steps, it stops at the first
        result = policy.random_search(
            max_num_probes=2,
            simulator=None if interactive else simulator,
            is_disp=False,
        )
        assert capsys.readouterr().out.count("All actions have already searched") == 1
        assert result.total_num_search == n


# ---------------------------------------------------------------------------
# the constructor


@pytest.mark.parametrize("kind", RANGE_KINDS)
def test_range_bounds_are_checked(kind):
    """The three range policies go through the same constructor, so the
    multi-objective and unified ones check the bounds like range.Policy."""
    with pytest.raises(ValueError, match="min_X and max_X"):
        make_policy(kind, min_X=None)
    with pytest.raises(AssertionError, match="dimension"):
        make_policy(kind, min_X=np.zeros(2), max_X=np.ones(1))
    with pytest.raises(AssertionError, match="less than"):
        make_policy(kind, min_X=np.array([0.0, 1.0]), max_X=np.array([1.0, 1.0]))


@pytest.mark.parametrize("kind", DISCRETE_KINDS)
def test_candidates_can_be_given_as_a_variable(kind):
    """The three discrete policies accept the candidates as an array or as a
    physbo.Variable."""
    policy, _ = make_policy(kind, test_X=physbo.Variable(X=X))
    np.testing.assert_array_equal(policy.test.X, X)
    np.testing.assert_array_equal(policy.actions, np.arange(X.shape[0]))

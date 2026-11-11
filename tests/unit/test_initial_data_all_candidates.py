# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""initial_data of the discrete policies and the remaining candidates (issue #130)."""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

X = np.linspace(0.0, 1.0, 5)[:, None]


def values(actions, kind):
    """One objective, or two whose optima (x = 0.25 and x = 0.5) are close
    enough for some candidates to be dominated: also among the first two,
    so that the unified objective of any set used here is not constant."""
    x = X[actions, 0]
    if kind == "discrete":
        return -(x**2)
    return np.c_[-((x - 0.25) ** 2), -((x - 0.5) ** 2)]


def make_policy(kind, actions):
    data = (np.asarray(actions), values(actions, kind))
    if kind == "discrete":
        return physbo.search.discrete.Policy(test_X=X, initial_data=data)
    if kind == "discrete_multi":
        return physbo.search.discrete_multi.Policy(
            test_X=X, num_objectives=2, initial_data=data
        )
    return physbo.search.discrete_unified.Policy(
        test_X=X, num_objectives=2, initial_data=data
    )


KINDS = ["discrete", "discrete_multi", "discrete_unified"]


@pytest.mark.parametrize("kind", KINDS)
def test_initial_data_of_every_candidate_leaves_an_integer_empty_array(kind):
    policy = make_policy(kind, np.arange(5))
    assert policy.actions.shape == (0,)
    assert policy.actions.dtype.kind == "i"
    assert policy.history.total_num_search == 5


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("actions", [[3, 1], [1, 3, 1]], ids=["unsorted", "repeated"])
def test_remaining_candidates_are_sorted_integers(kind, actions):
    """write() alone removes the candidates: unsorted and repeated actions included."""
    policy = make_policy(kind, actions)
    np.testing.assert_array_equal(policy.actions, [0, 2, 4])
    assert policy.actions.dtype.kind == "i"
    assert policy.history.total_num_search == len(actions)


def score_name(kind):
    return "EHVI" if kind == "discrete_multi" else "EI"


def build_unified_data(policy, kind):
    """A unified policy builds its unified training data, and its predictor,
    in bayes_search (issue #132): call it once without proposing a point,
    so that get_score can be asked. The other two kinds need nothing; with
    no predictor yet, their get_score fits a Gaussian process of its own, so
    the three kinds do not take the same branch of get_score."""
    if kind == "discrete_unified":
        policy.bayes_search(
            max_num_probes=0,
            unify_method=physbo.search.unify.NDS(num_objectives=2),
            is_disp=False,
        )


def assert_empty_score_and_usable_state(policy, kind):
    """No candidate remains, so get_score returns an empty array; the state
    it worked with is sound, since explicit points get finite scores."""
    assert policy.actions.shape == (0,) and policy.actions.dtype.kind == "i"
    assert policy.get_score(score_name(kind)).shape == (0,)
    f = policy.get_score(score_name(kind), xs=X[:2])
    assert f.shape == (2,)
    assert np.all(np.isfinite(f))


@pytest.mark.parametrize("kind", KINDS)
def test_get_score_without_remaining_candidates_returns_an_empty_array(kind):
    """Whether initial_data covered every candidate or write() used up the rest."""
    policy = make_policy(kind, np.arange(5))
    build_unified_data(policy, kind)
    assert_empty_score_and_usable_state(policy, kind)

    # the same state reached through write(); in the unified policy after
    # its unified data exist, so that get_score has new data to take in
    other = make_policy(kind, [0, 1])
    build_unified_data(other, kind)
    other.write(np.array([2, 3, 4]), values([2, 3, 4], kind))
    assert_empty_score_and_usable_state(other, kind)
    assert other.training.X.shape[0] == 5


@pytest.mark.parametrize("kind", KINDS)
def test_bayes_search_without_remaining_candidates_ends_with_a_warning(kind, capsys):
    """A script that reloads its results as initial_data and keeps calling
    bayes_search ends like random_search does, instead of an IndexError."""
    policy = make_policy(kind, np.arange(5))
    kwargs = {}
    if kind == "discrete_unified":
        kwargs["unify_method"] = physbo.search.unify.NDS(num_objectives=2)
    capsys.readouterr()
    policy.bayes_search(
        max_num_probes=1,
        simulator=lambda actions: values(actions, kind),
        is_disp=False,
        **kwargs,
    )
    assert "All actions have already searched" in capsys.readouterr().out
    assert policy.history.total_num_search == 5

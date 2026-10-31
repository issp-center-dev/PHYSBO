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
    x = X[actions, 0]
    if kind == "discrete":
        return -(x**2)
    return np.c_[-(x**2), -((x - 1.0) ** 2)]


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
def test_remaining_candidates_are_sorted_integers(kind):
    policy = make_policy(kind, [3, 1])
    np.testing.assert_array_equal(policy.actions, [0, 2, 4])
    assert policy.actions.dtype.kind == "i"


@pytest.mark.parametrize("kind", ["discrete", "discrete_multi"])
def test_get_score_without_remaining_candidates_returns_an_empty_array(kind):
    """As when the candidates were used up by write() or a search."""
    policy = make_policy(kind, np.arange(5))
    score = "EI" if kind == "discrete" else "EHVI"
    f = policy.get_score(score)
    assert f.shape == (0,)

    # the same state reached through write(): the candidates are consumed
    other = make_policy(kind, [0, 1])
    other.write(np.array([2, 3, 4]), values([2, 3, 4], kind))
    assert other.actions.shape == (0,) and other.actions.dtype.kind == "i"
    assert other.get_score(score).shape == (0,)

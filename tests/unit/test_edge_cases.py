# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Edge-case and error-path tests for the search policies.

Some of these tests document the *current* behavior rather than an ideal
one (see the individual comments); if the behavior is deliberately
improved later, update the corresponding assertions.
"""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")


@pytest.fixture
def X():
    return np.linspace(0.0, 1.0, 5).reshape(-1, 1)


@pytest.fixture
def sim(X):
    return lambda action: -np.sum((X[action] - 0.5) ** 2, axis=1)


def test_random_search_after_exhaustion(X, sim):
    # searching more probes than candidates stops with a warning instead
    # of raising or looping forever
    policy = physbo.search.discrete.Policy(test_X=X)
    policy.set_seed(1)
    policy.random_search(max_num_probes=len(X), simulator=sim, is_disp=False)
    assert len(policy.actions) == 0

    res = policy.random_search(max_num_probes=1, simulator=sim, is_disp=False)
    assert res.num_runs == len(X)


def test_bayes_search_after_exhaustion(X, sim):
    policy = physbo.search.discrete.Policy(test_X=X)
    policy.set_seed(1)
    policy.random_search(max_num_probes=len(X), simulator=sim, is_disp=False)

    res = policy.bayes_search(
        max_num_probes=1, simulator=sim, score="EI", is_disp=False
    )
    assert res.num_runs == len(X)


def test_bayes_search_with_single_training_point(X, sim):
    # a single observation is enough to start a Bayes search
    policy = physbo.search.discrete.Policy(test_X=X)
    policy.set_seed(1)
    policy.random_search(max_num_probes=1, simulator=sim, is_disp=False)
    res = policy.bayes_search(
        max_num_probes=1, simulator=sim, score="EI", is_disp=False
    )
    assert res.num_runs == 2


def test_nan_objective_value(X):
    # a NaN objective value is a failed observation: it is recorded in the
    # history and the candidate is consumed, but it is excluded from the
    # training data, so the subsequent GP fit is unaffected
    # (see test_failed_observations.py for the full contract)
    def simnan(action):
        action = np.asarray(action)
        val = -np.sum((X[action] - 0.5) ** 2, axis=1)
        return np.where(action == 2, np.nan, val)

    policy = physbo.search.discrete.Policy(test_X=X)
    policy.set_seed(1)
    policy.random_search(max_num_probes=len(X) - 1, simulator=simnan, is_disp=False)
    n = policy.history.total_num_search
    assert np.isnan(policy.history.fx[:n]).any()
    assert not policy.history.valid_mask.all()
    assert np.all(np.isfinite(policy.training.t))

    res = policy.bayes_search(
        max_num_probes=1, simulator=simnan, score="EI", is_disp=False
    )
    assert res.total_num_search == len(X)
    assert np.all(np.isfinite(policy.get_post_fmean(X)))


def test_get_score_dimension_mismatch():
    rng = np.random.RandomState(12345)
    policy = physbo.search.discrete.Policy(test_X=rng.rand(10, 3))
    policy.set_seed(1)
    policy.random_search(
        max_num_probes=2,
        simulator=lambda a: rng.rand(len(np.atleast_1d(a))),
        is_disp=False,
    )
    with pytest.raises(ValueError):
        policy.get_score("EI", xs=rng.rand(4, 2))


def test_range_policy_inverted_bounds():
    with pytest.raises(AssertionError):
        physbo.search.range.Policy(
            min_X=np.array([1.0, 1.0]), max_X=np.array([0.0, 0.0])
        )


def test_initial_data_length_mismatch(X):
    with pytest.raises(RuntimeError):
        physbo.search.discrete.Policy(
            test_X=X, initial_data=(np.array([0, 1]), np.array([1.0]))
        )


def test_write_out_of_range_action(X):
    # actions outside the candidate set raise an IndexError from numpy
    policy = physbo.search.discrete.Policy(test_X=X)
    with pytest.raises(IndexError):
        policy.write(np.array([len(X) + 1]), np.array([1.0]))

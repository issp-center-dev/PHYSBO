# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The unified policies before their first bayes_search (issue #132).

Their unified training data are built by bayes_search, which receives
unify_method, so the methods that need them report that with a
RuntimeError instead of failing on the missing data.
"""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

X = np.linspace(0.0, 1.0, 6)[:, None]


def f(x):
    x = np.atleast_2d(x)
    return np.c_[-np.sum(x**2, axis=1), -np.sum((x - 1.0) ** 2, axis=1)]


def make_policy(kind):
    if kind == "discrete_unified":
        policy = physbo.search.discrete_unified.Policy(test_X=X, num_objectives=2)
        simulator = lambda action: f(X[action])  # noqa: E731
    else:
        policy = physbo.search.range_unified.Policy(
            min_X=np.zeros(1), max_X=np.ones(1), num_objectives=2
        )
        simulator = f
    policy.set_seed(0)
    policy.random_search(max_num_probes=3, simulator=simulator, is_disp=False)
    return policy, simulator


KINDS = ["discrete_unified", "range_unified"]


@pytest.mark.parametrize("kind", KINDS)
def test_get_score_before_bayes_search_raises_runtime_error(kind):
    policy, _ = make_policy(kind)
    with pytest.raises(RuntimeError, match="bayes_search"):
        policy.get_score("EI", xs=np.array([[0.3]]))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("method", ["get_post_fmean", "get_post_fcov"])
def test_posterior_before_bayes_search_raises_runtime_error(kind, method):
    policy, _ = make_policy(kind)
    with pytest.raises(RuntimeError, match="bayes_search"):
        getattr(policy, method)(np.array([[0.3]]))


@pytest.mark.parametrize("kind", KINDS)
def test_get_score_works_after_bayes_search_with_no_probe(kind):
    """bayes_search(max_num_probes=0) builds the unified data and the predictor."""
    policy, simulator = make_policy(kind)
    policy.bayes_search(
        max_num_probes=0,
        simulator=simulator,
        score="EI",
        unify_method=physbo.search.unify.ParEGO(num_objectives=2),
        is_disp=False,
    )
    f_score = policy.get_score("EI", xs=np.array([[0.3], [0.6]]))
    assert f_score.shape == (2,)
    assert np.all(np.isfinite(f_score))
    assert np.isfinite(policy.get_post_fmean(np.array([[0.3]]))).all()

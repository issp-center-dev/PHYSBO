# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The predictor of a policy agrees with its training data.

A Bayesian linear model keeps statistics of the training data and, in the
training data, the values of its basis functions. After any number of
steps of bayes_search, whether the hyperparameters were learned once or at
every step, the posterior of each predictor must be the one obtained by
preparing a copy of it afresh from the training data it is used with.
"""

import copy

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

X = np.array(
    [[a, b] for a in np.linspace(0.0, 1.0, 5) for b in np.linspace(0.0, 1.0, 5)]
)
POINTS = X[::4]

KINDS = ["discrete", "discrete_multi", "range", "range_multi"]


def objectives(x, kind):
    x = np.atleast_2d(x)
    f1 = -np.sum((x - 0.25) ** 2, axis=1)
    if kind in ("discrete", "range"):
        return f1
    return np.c_[f1, -np.sum((x - 0.75) ** 2, axis=1)]


def make_policy(kind):
    config = physbo.misc.SetConfig()
    config.learning.is_disp = False
    config.learning.num_init_params_search = 2
    config.learning.max_epoch = 50
    config.learning.max_epoch_init_params_search = 10
    kwargs = dict(config=config, rng=3)
    if kind not in ("discrete", "range"):
        kwargs["num_objectives"] = 2
    if kind.startswith("discrete"):
        policy = getattr(physbo.search, kind).Policy(test_X=X, **kwargs)
        return policy, lambda action: objectives(X[action], kind)
    policy = getattr(physbo.search, kind).Policy(
        min_X=np.zeros(2), max_X=np.ones(2), **kwargs
    )
    return policy, lambda x: objectives(x, kind)


def search_kwargs(kind, policy, unify_method=None):
    kwargs = dict(is_disp=False, num_rand_basis=30)
    if kind.endswith("multi"):
        kwargs["score"] = "HVPI"
    else:
        kwargs["score"] = "EI"
    if kind.endswith("unified"):
        kwargs["unify_method"] = unify_method
    if kind.startswith("range"):
        kwargs["optimizer"] = physbo.search.optimize.random.Optimizer(
            min_X=np.zeros(2), max_X=np.ones(2), nsamples=20, rng=policy.rng
        )
    return kwargs


def predictors_and_training(policy, kind):
    """(predictor, the training data it is used with, objective index) of a policy."""
    if kind.endswith("multi"):
        return [(p, policy.training, i) for i, p in enumerate(policy.predictor_list)]
    if kind.endswith("unified"):
        return [(policy.predictor, policy.training_unified, 0)]
    return [(policy.predictor, policy.training, 0)]


def max_difference_from_fresh_statistics(policy, kind):
    """Largest difference of the posterior mean at POINTS between each
    predictor as it is and a copy prepared afresh from the training data."""
    test = physbo.Variable(X=POINTS)
    worst = 0.0
    for predictor, training, index in predictors_and_training(policy, kind):
        as_is = predictor.get_post_fmean(training, test)
        fresh = copy.deepcopy(predictor)
        fresh.delete_stats()
        # without Z, the basis values are computed from X with the basis of
        # the predictor
        bare = physbo.Variable(X=training.X, t=training.t)
        fresh.prepare(bare, objective_index=index)
        difference = np.abs(as_is - fresh.get_post_fmean(bare, test))
        worst = max(worst, float(np.max(difference)))
    return worst


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("interval", [0, 1], ids=["learn once", "learn every step"])
@pytest.mark.parametrize("num_search_each_probe", [1, 2])
def test_statistics_agree_with_the_training_data(kind, interval, num_search_each_probe):
    policy, simulator = make_policy(kind)
    policy.random_search(max_num_probes=5, simulator=simulator, is_disp=False)
    policy.bayes_search(
        max_num_probes=3,
        num_search_each_probe=num_search_each_probe,
        simulator=simulator,
        interval=interval,
        **search_kwargs(kind, policy),
    )
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8

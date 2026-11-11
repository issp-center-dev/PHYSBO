# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The predictor of a policy between calls of bayes_search.

The six policies keep their predictor(s) from one call to the next and
replace them only when another kind is asked for (Gaussian process or
Bayesian linear model, the number of basis functions, ARD). A call that
does not learn the hyperparameters (interval < 0) needs a learned
predictor of the kind asked for.
"""

import copy

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from .test_predictor_consistency import (  # noqa: E402
    X,
    make_policy,
    max_difference_from_fresh_statistics,
    search_kwargs,
)

KINDS = [
    "discrete",
    "discrete_multi",
    "discrete_unified",
    "range",
    "range_multi",
    "range_unified",
]


def predictors(policy, kind):
    if kind.endswith("multi"):
        return list(policy.predictor_list)
    return [policy.predictor]


def predictor_argument(kind, value):
    """The keyword of bayes_search that takes the predictor(s)."""
    return {"predictor_list" if kind.endswith("multi") else "predictor": value}


def training_argument(kind, value):
    return {"training" if kind in ("discrete", "range") else "training_list": value}


def started(kind):
    """A policy with five random observations, its simulator and the
    arguments of bayes_search for a Gaussian process."""
    policy, simulator = make_policy(kind)
    policy.random_search(max_num_probes=5, simulator=simulator, is_disp=False)
    unify = None
    if kind.endswith("unified"):
        unify = physbo.search.unify.NDS(num_objectives=2)
    kwargs = search_kwargs(kind, policy, unify)
    kwargs.update(simulator=simulator, num_rand_basis=0)
    return policy, kwargs


# ---------------------------------------------------------------------------
# kept, or replaced


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
def test_predictor_is_kept_between_calls(kind, num_rand_basis):
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    policy.bayes_search(max_num_probes=1, **kwargs)
    first = predictors(policy, kind)
    policy.bayes_search(max_num_probes=1, **kwargs)
    second = predictors(policy, kind)
    assert all(a is b for a, b in zip(first, second))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "before,after",
    [
        (dict(num_rand_basis=0), dict(num_rand_basis=20)),
        (dict(num_rand_basis=20), dict(num_rand_basis=0)),
        (dict(num_rand_basis=20), dict(num_rand_basis=30)),
        (dict(ard=False), dict(ard=True)),
    ],
    ids=["gp to blm", "blm to gp", "more basis functions", "ard"],
)
def test_predictor_is_replaced_when_another_kind_is_asked_for(kind, before, after):
    policy, kwargs = started(kind)
    policy.bayes_search(max_num_probes=1, **{**kwargs, **before})
    first = predictors(policy, kind)
    policy.bayes_search(max_num_probes=1, **{**kwargs, **after})
    second = predictors(policy, kind)
    assert all(a is not b for a, b in zip(first, second))

    asked = {**kwargs, **after}
    for predictor in second:
        is_blm = isinstance(predictor, physbo.blm.Predictor)
        assert is_blm == (asked["num_rand_basis"] > 0)
        if is_blm:
            assert predictor.blm.nbasis == asked["num_rand_basis"]
        assert predictor.model.prior.cov.ard == asked.get("ard", False)
    # the new predictor is the one in use: its statistics are those of the data
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
def test_basis_values_do_not_outlive_their_predictor(kind):
    """After a Bayesian linear model is replaced by a Gaussian process, the
    data carry no basis values; replaced by a model with another number of
    basis functions, they carry those of the new one."""
    policy, kwargs = started(kind)
    policy.bayes_search(max_num_probes=1, **{**kwargs, "num_rand_basis": 20})

    policy.bayes_search(max_num_probes=1, **{**kwargs, "num_rand_basis": 30})
    for data in (policy.training, getattr(policy, "test", None)):
        if data is not None and data.Z is not None:
            assert data.Z.shape[-1] == 30

    policy.bayes_search(max_num_probes=1, **{**kwargs, "num_rand_basis": 0})
    assert policy.training.Z is None
    if kind.startswith("discrete"):
        assert policy.test.Z is None


# ---------------------------------------------------------------------------
# interval < 0


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
def test_negative_interval_continues_with_the_learned_predictor(kind, num_rand_basis):
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    first = predictors(policy, kind)
    history = policy.bayes_search(max_num_probes=2, interval=-1, **kwargs)
    assert history.total_num_search == 8
    assert all(a is b for a, b in zip(first, predictors(policy, kind)))
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
def test_negative_interval_needs_a_learned_predictor(kind, num_rand_basis):
    """Also with a Gaussian process, which used to run with its default
    hyperparameters; and nothing is changed by the refused call (#143)."""
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    is_disp = policy.config.learning.is_disp
    for max_num_probes in (1, 0, None):
        with pytest.raises(RuntimeError, match="interval < 0"):
            policy.bayes_search(max_num_probes=max_num_probes, interval=-1, **kwargs)
    assert all(p is None for p in predictors(policy, kind))
    assert policy.history.total_num_search == 5
    assert policy.config.learning.is_disp == is_disp


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "change", [dict(num_rand_basis=20), dict(ard=True)], ids=["blm", "ard"]
)
def test_negative_interval_refuses_another_kind_of_predictor(kind, change):
    policy, kwargs = started(kind)
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    first = predictors(policy, kind)
    with pytest.raises(RuntimeError, match="interval < 0"):
        policy.bayes_search(max_num_probes=1, interval=-1, **{**kwargs, **change})
    assert all(a is b for a, b in zip(first, predictors(policy, kind)))
    assert policy.history.total_num_search == 6
    # and the policy goes on with the predictor it has
    policy.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert policy.history.total_num_search == 7


@pytest.mark.parametrize("kind", KINDS)
def test_given_predictor_counts_as_learned(kind):
    """A predictor given to bayes_search is used as it is, without learning."""
    donor, kwargs = started(kind)
    donor.bayes_search(max_num_probes=1, interval=0, **kwargs)
    given = copy.deepcopy(predictors(donor, kind))
    given = given if kind.endswith("multi") else given[0]

    policy, kwargs = started(kind)
    policy.bayes_search(
        max_num_probes=1, interval=-1, **predictor_argument(kind, given), **kwargs
    )
    assert policy.history.total_num_search == 6
    used = predictors(policy, kind)
    assert all(
        a is b for a, b in zip(used, given if kind.endswith("multi") else [given])
    )
    # later calls go on with it
    policy.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert all(a is b for a, b in zip(used, predictors(policy, kind)))


# ---------------------------------------------------------------------------
# the statistics follow the training data


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
def test_training_data_given_without_learning_are_the_ones_used(kind, num_rand_basis):
    """bayes_search(training=..., interval < 0) keeps the hyperparameters and
    builds the statistics of the predictor from the data given."""
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    policy.bayes_search(max_num_probes=2, interval=0, **kwargs)

    keep = np.arange(0, policy.training.X.shape[0], 2)
    subset = physbo.Variable(X=policy.training.X[keep], t=policy.training.t[keep])
    policy.bayes_search(
        max_num_probes=0, interval=-1, **training_argument(kind, subset), **kwargs
    )
    assert policy.training.X.shape[0] == len(keep)
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


# ---------------------------------------------------------------------------
# saved and restored


@pytest.mark.parametrize("kind", KINDS)
def test_loaded_predictor_counts_as_learned(kind, tmp_path):
    """After save() and load(), a policy goes on without learning; the
    unified policies, whose unified data are not saved, build them again."""
    policy, kwargs = started(kind)
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    files = [
        str(tmp_path / name)
        for name in ("history.npz", "training.npz", "predictor.pickle")
    ]
    policy.save(*files)

    loaded, _ = make_policy(kind)
    loaded.load(*files)
    loaded.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert loaded.history.total_num_search == 7
    assert max_difference_from_fresh_statistics(loaded, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
def test_checkpoint_remembers_whether_the_predictor_was_learned(kind, tmp_path):
    policy, kwargs = started(kind)
    fresh = str(tmp_path / "fresh.ckpt")
    policy.save_checkpoint(fresh)
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    learned = str(tmp_path / "learned.ckpt")
    policy.save_checkpoint(learned)

    restored = type(policy).load_checkpoint(fresh)
    with pytest.raises(RuntimeError, match="interval < 0"):
        restored.bayes_search(max_num_probes=1, interval=-1, **kwargs)

    restored = type(policy).load_checkpoint(learned)
    restored.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert restored.history.total_num_search == 7


def test_candidates_are_shared_with_the_helpers():
    """The helpers imported from the consistency test use a 5 x 5 grid."""
    assert X.shape == (25, 2)

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The predictor of a policy between calls of bayes_search.

The six policies keep their predictor(s) from one call to the next and
replace them only when a call asks for another kind (Gaussian process or
Bayesian linear model, the number of basis functions, ARD); a call that
leaves num_rand_basis and ard at None asks for nothing. A call that does
not learn the hyperparameters (interval < 0) needs a learned predictor of
the kind asked for. The statistics of the predictor(s) are those of the
predictor(s) and the training data in use, however these came to the policy.
"""

import copy
import inspect

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


def set_predictors(policy, kind, value):
    """Assign the predictor(s) by hand."""
    if kind.endswith("multi"):
        policy.predictor_list = value
    else:
        policy.predictor = value


def kinds_of(policy, kind):
    """(class, number of basis functions, ARD) of the predictor(s)."""
    return [
        (
            "blm" if isinstance(p, physbo.blm.Predictor) else "gp",
            p.blm.nbasis if isinstance(p, physbo.blm.Predictor) else 0,
            bool(p.model.prior.cov.ard),
        )
        for p in predictors(policy, kind)
    ]


def save_files(policy, tmp_path):
    files = [
        str(tmp_path / name)
        for name in ("history.npz", "training.npz", "predictor.pickle")
    ]
    policy.save(*files)
    return files


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
def test_defaults_ask_for_nothing(kind):
    parameters = inspect.signature(
        getattr(physbo.search, kind).Policy.bayes_search
    ).parameters
    assert parameters["num_rand_basis"].default is None
    assert parameters["ard"].default is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "first", [dict(num_rand_basis=0), dict(num_rand_basis=20), dict(ard=True)]
)
@pytest.mark.parametrize("interval", [0, -1])
def test_call_that_asks_for_nothing_keeps_the_predictor(kind, first, interval):
    """Without num_rand_basis and ard, the predictor the policy has is used
    and, if the call learns, learned as what it is: a Bayesian linear model
    keeps its number of basis functions."""
    policy, kwargs = started(kind)
    policy.bayes_search(max_num_probes=1, interval=0, **{**kwargs, **first})
    kept = predictors(policy, kind)
    before = kinds_of(policy, kind)
    kwargs.pop("num_rand_basis")
    policy.bayes_search(max_num_probes=2, interval=interval, **kwargs)
    assert all(a is b for a, b in zip(kept, predictors(policy, kind)))
    assert kinds_of(policy, kind) == before
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
def test_what_is_not_asked_for_is_as_in_the_predictor_replaced(kind):
    """A call that asks for ARD only replaces a Bayesian linear model by one
    with as many basis functions; one that asks for a number of basis
    functions only keeps the ARD."""
    policy, kwargs = started(kind)
    kwargs.pop("num_rand_basis")
    policy.bayes_search(max_num_probes=1, interval=0, num_rand_basis=20, **kwargs)
    first = predictors(policy, kind)

    policy.bayes_search(max_num_probes=1, interval=0, ard=True, **kwargs)
    second = predictors(policy, kind)
    assert all(a is not b for a, b in zip(first, second))
    assert set(kinds_of(policy, kind)) == {("blm", 20, True)}

    policy.bayes_search(max_num_probes=1, interval=0, num_rand_basis=30, **kwargs)
    assert all(a is not b for a, b in zip(second, predictors(policy, kind)))
    assert set(kinds_of(policy, kind)) == {("blm", 30, True)}
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
def test_first_call_makes_a_gaussian_process_without_ard(kind):
    policy, kwargs = started(kind)
    kwargs.pop("num_rand_basis")
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    assert set(kinds_of(policy, kind)) == {("gp", 0, False)}


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
    # the unified policies have not taken the unify method either
    assert getattr(policy, "unify_method", None) is None


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
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
@pytest.mark.parametrize("training_file", [True, False])
def test_loaded_predictor_counts_as_learned(
    kind, num_rand_basis, training_file, tmp_path
):
    """After save() and load(), a policy goes on without learning, whether or
    not the training data were read from their file: the statistics, the
    basis values kept with the data and the unified data are built again for
    the predictor and the data read."""
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    files = [
        str(tmp_path / name)
        for name in ("history.npz", "training.npz", "predictor.pickle")
    ]
    policy.save(*files)

    loaded, _ = make_policy(kind)
    loaded.load(files[0], files[1] if training_file else None, files[2])
    # without num_rand_basis and ard, the call does not ask for another predictor
    kwargs.pop("num_rand_basis")
    loaded.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert loaded.history.total_num_search == 7
    assert max_difference_from_fresh_statistics(loaded, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
def test_loaded_predictor_stays_whatever_was_asked_for_before(kind, tmp_path):
    """load() replaces the predictor, and what the policy asked for before
    says nothing about the one it has now: it is kept by a call that does
    not ask for a kind."""
    saved, kwargs = started(kind)
    saved.bayes_search(max_num_probes=1, interval=0, **{**kwargs, "num_rand_basis": 20})
    files = [
        str(tmp_path / name)
        for name in ("history.npz", "training.npz", "predictor.pickle")
    ]
    saved.save(*files)

    policy, kwargs = started(kind)
    policy.bayes_search(
        max_num_probes=1, interval=0, **{**kwargs, "num_rand_basis": 30}
    )
    policy.load(*files)
    loaded = predictors(policy, kind)
    kwargs.pop("num_rand_basis")
    policy.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert all(a is b for a, b in zip(loaded, predictors(policy, kind)))
    assert all(p.blm.nbasis == 20 for p in loaded)
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("predictor_file", [True, False])
@pytest.mark.parametrize(
    "before,after", [(0, 20), (20, 0), (20, 30)], ids=["gp to blm", "blm to gp", "more"]
)
def test_kind_asked_for_after_load_is_the_kind_used(
    kind, predictor_file, before, after, tmp_path
):
    """The first call after load() that asks for a kind gets it, whether the
    predictor was read from its file or is the one the policy had."""
    policy, kwargs = started(kind)
    policy.bayes_search(
        max_num_probes=1, interval=0, **{**kwargs, "num_rand_basis": before}
    )
    files = save_files(policy, tmp_path)
    policy.load(files[0], files[1], files[2] if predictor_file else None)
    old = predictors(policy, kind)
    with pytest.raises(RuntimeError, match="interval < 0"):
        policy.bayes_search(
            max_num_probes=1, interval=-1, **{**kwargs, "num_rand_basis": after}
        )
    policy.bayes_search(
        max_num_probes=1, interval=0, **{**kwargs, "num_rand_basis": after}
    )
    assert all(a is not b for a, b in zip(old, predictors(policy, kind)))
    assert set(kinds_of(policy, kind)) == {("blm" if after else "gp", after, False)}
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
@pytest.mark.parametrize("interval", [0, -1])
def test_load_without_predictor_file_goes_on_with_the_predictor(
    kind, num_rand_basis, interval, tmp_path
):
    """load() of the history alone, on a policy that has searched on, rolls
    the data back; the predictor of the policy is then used with the data
    read, and a Bayesian linear model keeps its number of basis functions."""
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    files = save_files(policy, tmp_path)
    policy.bayes_search(max_num_probes=2, interval=0, **kwargs)
    kept = predictors(policy, kind)
    before = kinds_of(policy, kind)
    policy.load(files[0])
    kwargs.pop("num_rand_basis")
    policy.bayes_search(max_num_probes=1, interval=interval, **kwargs)
    assert policy.history.total_num_search == 7
    assert all(a is b for a, b in zip(kept, predictors(policy, kind)))
    assert kinds_of(policy, kind) == before
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", ["discrete_unified", "range_unified"])
def test_unified_data_follow_load_and_the_unify_method(kind, tmp_path):
    """load() on a policy that has searched on rolls its unified data back
    with the rest, and a call that does not learn uses the unify method it
    is given."""
    policy, kwargs = started(kind)
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    files = [
        str(tmp_path / name)
        for name in ("history.npz", "training.npz", "predictor.pickle")
    ]
    policy.save(*files)
    policy.bayes_search(max_num_probes=2, interval=0, **kwargs)
    policy.load(*files)
    # the unified data of the searches rolled back are gone, not used
    with pytest.raises(RuntimeError, match="load\\(\\)"):
        policy.get_post_fmean(X[:3])
    policy.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert policy.history.total_num_search == 7
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8

    other = physbo.search.unify.NDS(num_objectives=2, rank_max=2)
    expected = other(policy.training.t).reshape(-1, 1)
    policy.bayes_search(
        max_num_probes=0, interval=-1, **{**kwargs, "unify_method": other}
    )
    np.testing.assert_array_equal(policy.training_unified.t, expected)
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


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


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
def test_checkpoint_keeps_the_statistics(kind, num_rand_basis, tmp_path):
    """The predictor(s) and the training data restored from a checkpoint are
    those the statistics were built from: nothing is built again."""
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    assert not policy._statistics_missing()
    fname = str(tmp_path / "policy.ckpt")
    policy.save_checkpoint(fname)
    restored = type(policy).load_checkpoint(fname)
    assert not restored._statistics_missing()
    assert not copy.deepcopy(policy)._statistics_missing()


@pytest.mark.parametrize("kind", KINDS)
def test_state_saved_before_the_bookkeeping_existed(kind):
    """A policy restored from a state without the attribute this rule added
    goes on with the predictor it has and the statistics as they are; when
    the predictor is replaced, the basis values go with it."""
    policy, kwargs = started(kind)
    policy.bayes_search(
        max_num_probes=1, interval=0, **{**kwargs, "num_rand_basis": 20}
    )
    first = predictors(policy, kind)
    del policy.__dict__["_statistics_sources"]
    assert not policy._statistics_missing()
    kwargs.pop("num_rand_basis")
    policy.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert all(a is b for a, b in zip(first, predictors(policy, kind)))
    assert policy.history.total_num_search == 7
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8

    assert "_statistics_sources" not in policy.__dict__
    policy.bayes_search(max_num_probes=0, interval=0, num_rand_basis=0, **kwargs)
    assert set(kinds_of(policy, kind)) == {("gp", 0, False)}
    assert policy.training.Z is None
    if kind.startswith("discrete"):
        assert policy.test.Z is None


# ---------------------------------------------------------------------------
# learned, or not


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
def test_predictor_made_but_not_learned_is_refused(kind, num_rand_basis, tmp_path):
    """bayes_search makes the predictor before it finds that there are no
    training data. The predictor is not learned then: a later call with
    interval < 0 is refused, also after a checkpoint, until one has learned."""
    policy, simulator = make_policy(kind)
    unify = None
    if kind.endswith("unified"):
        unify = physbo.search.unify.NDS(num_objectives=2)
    kwargs = search_kwargs(kind, policy, unify)
    kwargs.update(simulator=simulator, num_rand_basis=num_rand_basis)
    with pytest.raises(RuntimeError, match="No valid observation"):
        policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    made = predictors(policy, kind)
    assert all(p is not None for p in made)

    policy.random_search(max_num_probes=5, simulator=simulator, is_disp=False)
    with pytest.raises(RuntimeError, match="interval < 0"):
        policy.bayes_search(max_num_probes=1, interval=-1, **kwargs)

    fname = str(tmp_path / "policy.ckpt")
    policy.save_checkpoint(fname)
    restored = type(policy).load_checkpoint(fname)
    with pytest.raises(RuntimeError, match="interval < 0"):
        restored.bayes_search(max_num_probes=1, interval=-1, **kwargs)

    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    policy.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert policy.history.total_num_search == 7
    assert all(a is b for a, b in zip(made, predictors(policy, kind)))


@pytest.mark.parametrize("kind", KINDS)
def test_learning_that_fails_leaves_the_predictor_unlearned(kind, monkeypatch):
    policy, kwargs = started(kind)
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)

    def fail(num_rand_basis):
        raise ValueError("the learning failed")

    monkeypatch.setattr(policy, "_learn_hyperparameter", fail)
    with pytest.raises(ValueError, match="the learning failed"):
        policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    monkeypatch.undo()

    with pytest.raises(RuntimeError, match="interval < 0"):
        policy.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    policy.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert policy.history.total_num_search == 8


@pytest.mark.parametrize("kind", KINDS)
def test_unfitted_bayesian_linear_model_given_is_refused(kind):
    """A Bayesian linear model that was never fitted has no basis; given
    with interval < 0 it is refused like any unlearned predictor."""
    policy, kwargs = started(kind)
    model = physbo.gp.core.Model.create_default(ard=False, num_dim=2)
    unfitted = physbo.blm.Predictor(policy.config, model=model)
    given = [unfitted, copy.deepcopy(unfitted)] if kind.endswith("multi") else unfitted
    with pytest.raises(RuntimeError, match="interval < 0"):
        policy.bayes_search(
            max_num_probes=1, interval=-1, **predictor_argument(kind, given), **kwargs
        )
    assert all(p is None for p in predictors(policy, kind))


# ---------------------------------------------------------------------------
# predictors that come from outside


def learned_elsewhere(kind, num_rand_basis):
    """The predictor(s) of another policy, learned on other data and, for a
    Bayesian linear model, with another basis."""
    donor, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    donor.random_search(max_num_probes=2, simulator=kwargs["simulator"], is_disp=False)
    donor.bayes_search(max_num_probes=1, interval=0, **kwargs)
    given = copy.deepcopy(predictors(donor, kind))
    return given if kind.endswith("multi") else given[0]


def as_list(kind, given):
    return given if kind.endswith("multi") else [given]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
def test_given_predictor_is_used_with_the_data_of_the_policy(kind, num_rand_basis):
    """A predictor given with interval < 0 replaces the one of the policy
    entirely: its statistics are built from the training data of the policy,
    and the basis values kept with the data are its own."""
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    given = learned_elsewhere(kind, num_rand_basis)
    policy.bayes_search(
        max_num_probes=1, interval=-1, **predictor_argument(kind, given), **kwargs
    )
    assert all(a is b for a, b in zip(predictors(policy, kind), as_list(kind, given)))
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
def test_given_predictor_stays_until_another_kind_is_asked_for(kind):
    """A Bayesian linear model given once is used by the calls that follow
    and say nothing about the kind of predictor (the defaults of
    num_rand_basis and ard), and by a call that asks for what it is. It is
    replaced when a call asks for something else."""
    policy, kwargs = started(kind)
    kwargs.pop("num_rand_basis")
    given = learned_elsewhere(kind, 20)
    policy.bayes_search(
        max_num_probes=1, interval=-1, **predictor_argument(kind, given), **kwargs
    )
    policy.bayes_search(max_num_probes=1, interval=-1, **kwargs)
    assert all(a is b for a, b in zip(predictors(policy, kind), as_list(kind, given)))

    policy.bayes_search(max_num_probes=1, interval=0, num_rand_basis=20, **kwargs)
    assert all(a is b for a, b in zip(predictors(policy, kind), as_list(kind, given)))
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8

    with pytest.raises(RuntimeError, match="interval < 0"):
        policy.bayes_search(max_num_probes=1, interval=-1, num_rand_basis=30, **kwargs)
    policy.bayes_search(max_num_probes=1, interval=0, num_rand_basis=30, **kwargs)
    replaced = predictors(policy, kind)
    assert all(a is not b for a, b in zip(replaced, as_list(kind, given)))
    assert all(p.blm.nbasis == 30 for p in replaced)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
@pytest.mark.parametrize("has_searched", [False, True])
def test_predictor_set_by_hand_is_used(kind, num_rand_basis, has_searched):
    """A learned predictor assigned to the policy is used by a call that
    does not learn, with the training data of the policy: its statistics,
    and the basis values kept with the data, are built for it, also when it
    takes the place of a predictor of the same kind."""
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    if has_searched:
        policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    given = learned_elsewhere(kind, num_rand_basis)
    set_predictors(policy, kind, given)
    kwargs.pop("num_rand_basis")
    policy.bayes_search(max_num_probes=2, interval=-1, **kwargs)
    assert policy.history.total_num_search == 7 + has_searched
    assert all(a is b for a, b in zip(predictors(policy, kind), as_list(kind, given)))
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8
    # built once for the predictor, not at every call
    assert not policy._statistics_missing()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("before,after", [(20, 0), (0, 20)], ids=["gp", "blm"])
@pytest.mark.parametrize("interval", [0, -1])
def test_predictor_set_by_hand_replaces_one_of_another_kind(
    kind, before, after, interval
):
    """The basis values kept with the data go with the predictor they
    belonged to, also when it is replaced by hand."""
    policy, kwargs = started(kind)
    kwargs.pop("num_rand_basis")
    policy.bayes_search(max_num_probes=1, interval=0, num_rand_basis=before, **kwargs)
    set_predictors(policy, kind, learned_elsewhere(kind, after))
    policy.bayes_search(max_num_probes=2, interval=interval, **kwargs)
    assert policy.history.total_num_search == 8
    assert set(kinds_of(policy, kind)) == {("blm" if after else "gp", after, False)}
    if not after:
        assert policy.training.Z is None
        if kind.startswith("discrete"):
            assert policy.test.Z is None
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("how", ["by hand", "loaded"])
def test_predictor_from_outside_is_replaced_when_another_kind_is_asked_for(
    kind, how, tmp_path
):
    """Also in the first call after the predictor came to the policy."""
    if how == "by hand":
        policy, kwargs = started(kind)
        set_predictors(policy, kind, learned_elsewhere(kind, 0))
    else:
        saved, kwargs = started(kind)
        saved.bayes_search(max_num_probes=1, interval=0, **kwargs)
        files = save_files(saved, tmp_path)
        policy, kwargs = started(kind)
        policy.load(*files)
    old = predictors(policy, kind)
    policy.bayes_search(
        max_num_probes=1, interval=0, **{**kwargs, "num_rand_basis": 20}
    )
    assert all(a is not b for a, b in zip(old, predictors(policy, kind)))
    assert set(kinds_of(policy, kind)) == {("blm", 20, False)}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "asked,expected", [(None, 20), (0, 20), (30, 30)], ids=["nothing", "0", "30"]
)
def test_given_bayesian_linear_model_is_learned_with_basis_functions(
    kind, asked, expected
):
    """A Bayesian linear model given to a call that learns is fitted with the
    number of basis functions asked for or, if none (or 0, which no Bayesian
    linear model can have) was, with the number it has."""
    policy, kwargs = started(kind)
    kwargs.pop("num_rand_basis")
    if asked is not None:
        kwargs["num_rand_basis"] = asked
    given = learned_elsewhere(kind, 20)
    policy.bayes_search(
        max_num_probes=2, interval=0, **predictor_argument(kind, given), **kwargs
    )
    assert all(a is b for a, b in zip(predictors(policy, kind), as_list(kind, given)))
    assert set(kinds_of(policy, kind)) == {("blm", expected, False)}
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
def test_predictor_of_the_policy_given_back_is_prepared_again(kind):
    """A predictor given may have been changed since the policy used it
    (here it is fitted again, with another basis), also when it is the
    object the policy has."""
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = 20
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    own = predictors(policy, kind)
    data = policy.training_unified if kind.endswith("unified") else policy.training
    for index, predictor in enumerate(own):
        predictor.fit(data, 20, objective_index=index, rng=policy.rng)
    given = own if kind.endswith("multi") else own[0]
    policy.bayes_search(
        max_num_probes=1, interval=-1, **predictor_argument(kind, given), **kwargs
    )
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("num_rand_basis", [0, 20], ids=["gp", "blm"])
@pytest.mark.parametrize("how", ["by hand", "given back"])
def test_training_data_from_outside_are_the_ones_used(kind, num_rand_basis, how):
    """Training data assigned to the policy, and the data of the policy
    changed in place and given back, are those the statistics are built
    from in a call that does not learn."""
    policy, kwargs = started(kind)
    kwargs["num_rand_basis"] = num_rand_basis
    policy.bayes_search(max_num_probes=2, interval=0, **kwargs)
    if how == "by hand":
        keep = np.arange(0, policy.training.X.shape[0], 2)
        policy.training = physbo.Variable(
            X=policy.training.X[keep], t=policy.training.t[keep]
        )
        policy.new_data = None
        policy.bayes_search(max_num_probes=0, interval=-1, **kwargs)
        assert policy.training.X.shape[0] == len(keep)
    else:
        policy.training.t[:] = policy.training.t[::-1].copy()
        policy.bayes_search(
            max_num_probes=0,
            interval=-1,
            **training_argument(kind, policy.training),
            **kwargs,
        )
    assert max_difference_from_fresh_statistics(policy, kind) < 1e-8


@pytest.mark.parametrize("kind", ["discrete_multi", "range_multi"])
def test_predictor_list_of_mixed_kinds(kind):
    """A list with a Gaussian process among Bayesian linear models can be
    given and learned; the data then carry no basis values."""
    policy, kwargs = started(kind)
    mixed = [learned_elsewhere(kind, 20)[0], learned_elsewhere(kind, 0)[1]]
    policy.bayes_search(max_num_probes=2, interval=0, predictor_list=mixed, **kwargs)
    assert policy.history.total_num_search == 7
    assert policy.training.Z is None


@pytest.mark.parametrize("kind", ["discrete_multi", "range_multi"])
def test_predictor_list_with_a_hole_is_not_kept(kind):
    policy, kwargs = started(kind)
    holed = [learned_elsewhere(kind, 0)[0], None]
    with pytest.raises(AttributeError):
        policy.bayes_search(
            max_num_probes=1, interval=0, predictor_list=holed, **kwargs
        )
    policy.bayes_search(max_num_probes=1, interval=0, **kwargs)
    assert policy.history.total_num_search == 6
    assert all(p is not None for p in policy.predictor_list)


class PredictorWithoutDeleteStats(physbo.gp.Predictor):
    """A predictor of a user's own that implements what the policies need,
    and not delete_stats."""

    def fit(self, training, num_basis=None, comm=None, objective_index=0, rng=None):
        t = training.t[:, objective_index] if training.t.ndim == 2 else training.t
        if self.model.prior.cov.num_dim is None:
            self.model.prior.cov.num_dim = training.X.shape[1]
        self.model.fit(training.X, t, self.config, comm=comm, rng=rng)

    def delete_stats(self):
        raise NotImplementedError


@pytest.mark.parametrize("kind", ["discrete_unified", "range_unified"])
def test_unified_policies_do_not_need_delete_stats(kind):
    policy, kwargs = started(kind)
    model = physbo.gp.core.Model.create_default(ard=False, num_dim=2)
    predictor = PredictorWithoutDeleteStats(policy.config, model=model)
    policy.bayes_search(max_num_probes=2, interval=0, predictor=predictor, **kwargs)
    assert policy.history.total_num_search == 7


def test_candidates_are_shared_with_the_helpers():
    """The helpers imported from the consistency test use a 5 x 5 grid."""
    assert X.shape == (25, 2)

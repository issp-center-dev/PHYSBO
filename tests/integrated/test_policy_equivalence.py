# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Equivalences between ways of driving the six policies.

Regression tests for the restructuring of the policies: the searches below
must give identical results whichever way they are driven, in both random
number modes (legacy: global numpy.random seeded by set_seed; generator:
rng= given to the constructor) and with both predictors (the Gaussian
process and the Bayesian linear model, whose training data carry the values
of the basis functions).

- the interactive mode (simulator=None, the caller evaluates and writes)
  equals the simulator-driven mode;
- initial_data equals write() of the same data;
- a policy saved with save() and loaded into a fresh policy continues like
  the original (single-objective policies; the multi-objective ones are
  covered by tests/unit/test_saveload_multi.py);
- a policy restored from a checkpoint continues bit-exactly.

Several proposals per step with the Bayesian linear model are broken in the
range and unified policies (issue #131); those cases are expected to fail
until the fix is in, strictly, so that the marks have to be removed then.
"""

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

from ._policy_fixtures import (  # noqa: E402
    MAX_X,
    MIN_X,
    f_multi,
    f_single,
    grid_X,
    is_range,
    make_config,
    make_simulator,
)

KINDS = [
    "discrete",
    "discrete_multi",
    "discrete_unified",
    "range",
    "range_multi",
    "range_unified",
]
SINGLE_KINDS = ["discrete", "range"]
RNG_MODES = ["legacy", "generator"]
PREDICTORS = ["gp", "blm"]
SEED = 2468
NUM_RAND_BASIS = {"gp": 0, "blm": 20}

# policies in which several proposals per step fail with the BLM (#131)
MULTI_PROPOSAL_BLM_BROKEN = (
    "range",
    "range_multi",
    "range_unified",
    "discrete_unified",
)


def proposals_per_step(kind, predictor):
    """Two proposals per step, or one where #131 breaks it (see the module docstring)."""
    if predictor == "blm" and kind in MULTI_PROPOSAL_BLM_BROKEN:
        return 1
    return 2


def new_policy(kind, rng_mode, initial_data=None):
    config = make_config()
    rng = None if rng_mode == "legacy" else SEED
    if kind == "discrete":
        return physbo.search.discrete.Policy(
            test_X=grid_X(), config=config, initial_data=initial_data, rng=rng
        )
    if kind == "range":
        return physbo.search.range.Policy(
            min_X=MIN_X, max_X=MAX_X, config=config, initial_data=initial_data, rng=rng
        )
    if kind.startswith("discrete"):
        return getattr(physbo.search, kind).Policy(
            test_X=grid_X(),
            num_objectives=2,
            config=config,
            initial_data=initial_data,
            rng=rng,
        )
    return getattr(physbo.search, kind).Policy(
        num_objectives=2,
        min_X=MIN_X,
        max_X=MAX_X,
        config=config,
        initial_data=initial_data,
        rng=rng,
    )


def seed(policy, rng_mode, seed=SEED):
    """Seed the legacy global state; a generator was seeded by the constructor."""
    if rng_mode == "legacy":
        policy.set_seed(seed)


def reseed(policy, rng_mode, seed):
    """Give the policy a freshly seeded generator (in both modes)."""
    if rng_mode == "legacy":
        policy.set_seed(seed)
    else:
        policy.rng = np.random.default_rng(seed)


def search_kwargs(kind, policy, score=None, predictor="gp"):
    """Keyword arguments of bayes_search for ``kind``, with a cheap optimizer
    for the range policies that draws from the policy's generator. The
    hyperparameters are learned before every proposal (interval=1), so that
    one call of bayes_search and a sequence of calls learn alike."""
    kwargs = dict(is_disp=False, num_rand_basis=NUM_RAND_BASIS[predictor], interval=1)
    if kind.endswith("multi"):
        kwargs["score"] = score or "EHVI"
    else:
        kwargs["score"] = score or "EI"
    if kind.endswith("unified"):
        kwargs["unify_method"] = physbo.search.unify.NDS(num_objectives=2)
    if is_range(kind):
        kwargs["optimizer"] = physbo.search.optimize.random.Optimizer(
            min_X=MIN_X, max_X=MAX_X, nsamples=20, rng=policy.rng
        )
    return kwargs


def proposals(policy):
    """The searched points (indices or X) and their objective values."""
    N = policy.history.total_num_search
    if hasattr(policy.history, "action_X"):
        return np.array(policy.history.action_X[:N]), np.array(policy.history.fx[:N])
    return np.array(policy.history.chosen_actions[:N]), np.array(policy.history.fx[:N])


def assert_same_search(p, q):
    xp, fp = proposals(p)
    xq, fq = proposals(q)
    assert xp.shape == xq.shape and xp.shape[0] > 0
    np.testing.assert_array_equal(xp, xq)
    np.testing.assert_array_equal(fp, fq)


def initial_points(kind):
    """(points as given to initial_data / write, their objective values)."""
    X = grid_X()
    f = f_single if kind in SINGLE_KINDS else f_multi
    if is_range(kind):
        X0 = np.array([[0.1, 0.1], [0.9, 0.2], [0.3, 0.8], [0.5, 0.5]])
        return X0, f(X0)
    actions = np.array([0, 7, 13, 19])
    return actions, f(X[actions])


# ---------------------------------------------------------------------------
# interactive mode


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rng_mode", RNG_MODES)
@pytest.mark.parametrize("predictor", PREDICTORS)
def test_interactive_mode_matches_simulator_mode(kind, rng_mode, predictor):
    sim = make_simulator(kind)
    N = proposals_per_step(kind, predictor)

    p = new_policy(kind, rng_mode)
    seed(p, rng_mode)
    p.random_search(
        max_num_probes=2, num_search_each_probe=2, simulator=sim, is_disp=False
    )
    p.bayes_search(
        max_num_probes=2,
        num_search_each_probe=N,
        simulator=sim,
        **search_kwargs(kind, p, predictor=predictor),
    )

    q = new_policy(kind, rng_mode)
    seed(q, rng_mode)
    for _ in range(2):
        a = q.random_search(max_num_probes=1, num_search_each_probe=2, is_disp=False)
        q.write(a, sim(a))
    kwargs = search_kwargs(kind, q, predictor=predictor)
    for _ in range(2):
        a = q.bayes_search(max_num_probes=1, num_search_each_probe=N, **kwargs)
        q.write(a, sim(a))

    assert_same_search(p, q)


# ---------------------------------------------------------------------------
# initial_data


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rng_mode", RNG_MODES)
@pytest.mark.parametrize("predictor", PREDICTORS)
def test_initial_data_matches_write(kind, rng_mode, predictor):
    sim = make_simulator(kind)
    points, values = initial_points(kind)
    N = proposals_per_step(kind, predictor)

    p = new_policy(kind, rng_mode, initial_data=(points, values))
    q = new_policy(kind, rng_mode)
    q.write(points, values)
    for policy in (p, q):
        seed(policy, rng_mode)
        policy.bayes_search(
            max_num_probes=2,
            num_search_each_probe=N,
            simulator=sim,
            **search_kwargs(kind, policy, predictor=predictor),
        )
    if not is_range(kind):
        np.testing.assert_array_equal(p.actions, q.actions)
    assert_same_search(p, q)


# ---------------------------------------------------------------------------
# save / load of the single-objective policies


@pytest.mark.parametrize("kind", SINGLE_KINDS)
@pytest.mark.parametrize("rng_mode", RNG_MODES)
@pytest.mark.parametrize("predictor", PREDICTORS)
def test_save_load_restores_the_state(kind, rng_mode, predictor, tmp_path):
    sim = make_simulator(kind)
    kwargs = dict(simulator=sim, max_num_probes=2)
    p = new_policy(kind, rng_mode)
    seed(p, rng_mode)
    p.random_search(max_num_probes=3, simulator=sim, is_disp=False)
    p.bayes_search(**kwargs, **search_kwargs(kind, p, predictor=predictor))

    file_history = str(tmp_path / "history.npz")
    file_training = str(tmp_path / "training.npz")
    file_predictor = str(tmp_path / "predictor.pickle")
    p.save(file_history, file_training, file_predictor)

    q = new_policy(kind, rng_mode)
    q.load(file_history, file_training, file_predictor)
    assert_same_search(p, q)
    np.testing.assert_array_equal(q.training.X, p.training.X)
    np.testing.assert_array_equal(q.training.t, p.training.t)
    if kind == "discrete":
        np.testing.assert_array_equal(q.actions, p.actions)
    xs = grid_X()[:5]
    np.testing.assert_array_equal(q.get_post_fmean(xs), p.get_post_fmean(xs))
    np.testing.assert_array_equal(q.get_post_fcov(xs), p.get_post_fcov(xs))

    # the generator is not part of save(): with a freshly seeded one on both
    # sides, the loaded policy continues like the original
    for policy in (p, q):
        reseed(policy, rng_mode, SEED + 1)
        policy.bayes_search(
            **kwargs, **search_kwargs(kind, policy, predictor=predictor)
        )
    assert_same_search(p, q)


# ---------------------------------------------------------------------------
# checkpoint of the six policies


def continue_search(kind, policy, N):
    policy.bayes_search(
        max_num_probes=2,
        num_search_each_probe=N,
        simulator=make_simulator(kind),
        **search_kwargs(kind, policy, score="TS", predictor="blm"),
    )
    return proposals(policy)


def checkpoint_cases():
    for kind in KINDS:
        for rng_mode in RNG_MODES:
            yield pytest.param(kind, rng_mode, 1, id=f"{rng_mode}-{kind}-1")
            marks = []
            if kind in MULTI_PROPOSAL_BLM_BROKEN:
                marks.append(pytest.mark.xfail(strict=True, reason="issue #131"))
            yield pytest.param(
                kind, rng_mode, 2, marks=marks, id=f"{rng_mode}-{kind}-2"
            )


@pytest.mark.parametrize("kind,rng_mode,N", list(checkpoint_cases()))
def test_checkpoint_resume_is_bit_exact_for_every_policy(kind, rng_mode, N, tmp_path):
    """Thompson sampling with a Bayesian linear model exercises the generator
    and the basis values through the checkpoint."""
    sim = make_simulator(kind)
    p = new_policy(kind, rng_mode)
    seed(p, rng_mode)
    p.random_search(max_num_probes=3, simulator=sim, is_disp=False)
    p.bayes_search(
        max_num_probes=1,
        num_search_each_probe=N,
        simulator=sim,
        **search_kwargs(kind, p, score="TS", predictor="blm"),
    )
    fname = str(tmp_path / "policy.ckpt")
    p.save_checkpoint(fname)

    reference = continue_search(kind, p, N)
    restored = type(p).load_checkpoint(fname)
    assert restored.mpicomm is None
    resumed = continue_search(kind, restored, N)
    np.testing.assert_array_equal(reference[0], resumed[0])
    np.testing.assert_array_equal(reference[1], resumed[1])

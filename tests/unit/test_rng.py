# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the RNG adapter (physbo._rng).

The legacy adapter must draw exactly the same values from the global
numpy.random stream as the historical np.random.* calls it replaced.
This is the bit-compatibility contract of the RNG plumbing refactor.
"""

import pickle

import numpy as np
import pytest

import physbo
from physbo._rng import LegacyRNG, get_rng, make_rng, _default_rng

SEED = 12345


def _reseed():
    np.random.seed(SEED)


def test_get_rng_default_is_shared_legacy_singleton():
    assert get_rng() is _default_rng
    assert isinstance(get_rng(), LegacyRNG)


def test_get_rng_passthrough():
    gen = np.random.default_rng(0)
    assert get_rng(gen) is gen


@pytest.mark.parametrize(
    "legacy_draw, adapter_draw",
    [
        (lambda: np.random.randn(3, 2), lambda r: r.standard_normal((3, 2))),
        (lambda: np.random.randn(4), lambda r: r.standard_normal(4)),
        (lambda: np.random.rand(3, 2), lambda r: r.random((3, 2))),
        (lambda: np.random.rand(5), lambda r: r.random(5)),
        (lambda: np.random.randint(0, 10, 4), lambda r: r.integers(0, 10, 4)),
        (lambda: np.random.randint(7), lambda r: r.integers(7)),
        (lambda: np.random.choice(20, 5, replace=False),
         lambda r: r.choice(20, 5, replace=False)),
        (lambda: np.random.permutation(10), lambda r: r.permutation(10)),
        (lambda: np.random.uniform(0.4, 0.8), lambda r: r.uniform(0.4, 0.8)),
        (lambda: np.random.multivariate_normal(np.zeros(3), np.eye(3), 2),
         lambda r: r.multivariate_normal(np.zeros(3), np.eye(3), 2)),
    ],
)
def test_legacy_rng_matches_global_numpy_random(legacy_draw, adapter_draw):
    _reseed()
    expected = legacy_draw()
    _reseed()
    actual = adapter_draw(LegacyRNG())
    np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_legacy_rng_seed_seeds_global_state():
    # Hosts embedding PHYSBO (e.g. ODAT-SE) rely on set_seed() seeding the
    # global numpy.random state as a side effect.
    LegacyRNG().seed(SEED)
    a = np.random.rand(3)
    np.random.seed(SEED)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)


def test_policy_set_seed_keeps_global_seeding_side_effect():
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    policy = physbo.search.discrete.Policy(test_X=X)
    policy.set_seed(SEED)
    a = np.random.rand(3)
    np.random.seed(SEED)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)


def test_make_rng_normalization():
    assert make_rng() is _default_rng
    assert make_rng("legacy") is _default_rng
    gen = make_rng(42)
    assert isinstance(gen, np.random.Generator)
    gen2 = np.random.default_rng(7)
    assert make_rng(gen2) is gen2
    assert isinstance(make_rng(np.random.SeedSequence(3)), np.random.Generator)
    with pytest.raises(ValueError):
        make_rng("unknown")


def _run_generator_mode_search(rng):
    X = np.linspace(0, 1, 60).reshape(-1, 1)

    def sim(actions):
        return -((X[actions, 0] - 0.5) ** 2)

    policy = physbo.search.discrete.Policy(test_X=X, rng=rng)
    policy.random_search(max_num_probes=4, simulator=sim, is_disp=False)
    res = policy.bayes_search(max_num_probes=3, simulator=sim, score="TS",
                              interval=1, num_rand_basis=50, is_disp=False)
    return res.chosen_actions[: res.total_num_search]


def test_generator_mode_is_reproducible_and_isolated_from_global_state():
    np.random.seed(1)
    a = _run_generator_mode_search(rng=777)
    np.random.seed(2)  # different global state must not matter
    b = _run_generator_mode_search(rng=777)
    np.testing.assert_array_equal(a, b)


def test_generator_mode_set_seed_recreates_generator():
    X = np.linspace(0, 1, 60).reshape(-1, 1)
    p1 = physbo.search.discrete.Policy(test_X=X, rng=1)
    p1.set_seed(42)
    p2 = physbo.search.discrete.Policy(test_X=X, rng=42)
    np.testing.assert_array_equal(
        p1.rng.standard_normal(5), p2.rng.standard_normal(5)
    )
    assert isinstance(p1.rng, np.random.Generator)


def test_generator_mode_rng_state_is_pickled_with_policy():
    X = np.linspace(0, 1, 60).reshape(-1, 1)
    policy = physbo.search.discrete.Policy(test_X=X, rng=123)
    policy.rng.standard_normal(3)  # advance the stream
    clone = pickle.loads(pickle.dumps(policy))
    np.testing.assert_array_equal(
        policy.rng.standard_normal(4), clone.rng.standard_normal(4)
    )


def test_blm_draw_evaluate_matches_get_post_samples():
    # Thompson sampling is split into draw (random) and evaluate
    # (deterministic); the composition must reproduce get_post_samples
    # exactly for the BLM predictor.
    X = np.linspace(0, 1, 40).reshape(-1, 1)

    def sim(actions):
        return -((X[actions, 0] - 0.5) ** 2)

    policy = physbo.search.discrete.Policy(test_X=X)
    policy.set_seed(SEED)
    policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
    policy.bayes_search(max_num_probes=0, num_rand_basis=30, is_disp=False)
    predictor = policy.predictor
    test = policy.test.get_subset([3, 10, 20])

    np.random.seed(999)
    expected = predictor.get_post_samples(policy.training, test, alpha=1.0)
    np.random.seed(999)
    w_hat = predictor.draw_post_sample_params(policy.training, alpha=1.0)
    actual = predictor.evaluate_post_sample(w_hat, test)
    np.testing.assert_array_equal(np.asarray(expected).ravel(),
                                  np.asarray(actual).ravel())


def test_policy_search_is_bit_compatible_with_global_stream():
    # random_search draws candidates through the adapter; the selection must
    # be identical to drawing directly from the seeded global stream.
    X = np.linspace(0, 1, 50).reshape(-1, 1)

    policy = physbo.search.discrete.Policy(test_X=X)
    policy.set_seed(SEED)
    actions = policy.random_search(max_num_probes=1, num_search_each_probe=5)

    np.random.seed(SEED)
    index = np.random.choice(50, 5, replace=False)
    np.testing.assert_array_equal(actions, np.arange(50)[index])

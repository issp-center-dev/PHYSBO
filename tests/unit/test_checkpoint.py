# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for pickling and the checkpoint API (serial).

A checkpoint must capture the complete execution state, so that a
restored policy continues bit-exactly as if the run had never stopped.
"""

import pickle

import numpy as np
import pytest

import physbo
from physbo._rng import _default_rng
from physbo.search._checkpoint import FORMAT_VERSION

X = np.linspace(0, 1, 100).reshape(-1, 1)


def sim(actions):
    return -((X[actions, 0] - 0.5) ** 2)


def _make_policy(rng=None):
    policy = physbo.search.discrete.Policy(test_X=X, rng=rng)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=4, simulator=sim, is_disp=False)
    policy.bayes_search(max_num_probes=2, simulator=sim, score="TS",
                        interval=1, num_rand_basis=50, is_disp=False)
    return policy


def _continue_policy(policy):
    res = policy.bayes_search(max_num_probes=2, simulator=sim, score="TS",
                              interval=1, num_rand_basis=50, is_disp=False)
    return res.chosen_actions[: res.total_num_search]


def test_pickle_roundtrip_drops_comm_and_keeps_state():
    policy = _make_policy()
    clone = pickle.loads(pickle.dumps(policy))
    assert clone.mpicomm is None
    assert clone.rng is _default_rng  # legacy adapter is normalized back
    np.testing.assert_array_equal(policy.actions, clone.actions)
    np.testing.assert_array_equal(policy.training.X, clone.training.X)
    N = policy.history.total_num_search
    np.testing.assert_array_equal(
        policy.history.chosen_actions[:N], clone.history.chosen_actions[:N]
    )


@pytest.mark.parametrize("rng", [None, 4242])
def test_checkpoint_resume_is_bit_exact(tmp_path, rng):
    fname = str(tmp_path / "ckpt.pkl")
    policy = _make_policy(rng=rng)
    policy.save_checkpoint(fname)
    reference = _continue_policy(policy)  # uninterrupted continuation

    restored = physbo.search.discrete.Policy.load_checkpoint(fname)
    assert restored.mpicomm is None
    resumed = _continue_policy(restored)
    np.testing.assert_array_equal(reference, resumed)


def test_plain_pickle_resume_generator_mode():
    # In Generator mode the RNG state lives on the policy, so a plain
    # pickle (e.g. embedded in a host application's own checkpoint dict,
    # as ODAT-SE does) is sufficient for bit-exact resumption.
    policy = _make_policy(rng=999)
    host_ckpt = pickle.dumps({"policy": policy, "step": 6})
    reference = _continue_policy(policy)

    restored = pickle.loads(host_ckpt)["policy"]
    restored.set_comm(None)
    resumed = _continue_policy(restored)
    np.testing.assert_array_equal(reference, resumed)


def test_set_comm_rejects_wrong_size():
    policy = _make_policy()
    policy.mpisize = 2  # pretend the state came from a 2-rank run
    with pytest.raises(RuntimeError, match="mpisize=2"):
        policy.set_comm(None)


def test_load_checkpoint_rejects_wrong_mpisize(tmp_path):
    fname = str(tmp_path / "ckpt.pkl")
    policy = _make_policy()
    policy.save_checkpoint(fname)

    with open(fname, "rb") as f:
        ckpt = pickle.load(f)
    ckpt["mpisize"] = 2
    ckpt["per_rank"] = ckpt["per_rank"] * 2
    with open(fname, "wb") as f:
        pickle.dump(ckpt, f)

    with pytest.raises(RuntimeError, match="mpisize=2"):
        physbo.search.discrete.Policy.load_checkpoint(fname)


def test_load_checkpoint_rejects_wrong_format_version(tmp_path):
    fname = str(tmp_path / "ckpt.pkl")
    policy = _make_policy()
    policy.save_checkpoint(fname)

    with open(fname, "rb") as f:
        ckpt = pickle.load(f)
    ckpt["format_version"] = FORMAT_VERSION + 1
    with open(fname, "wb") as f:
        pickle.dump(ckpt, f)

    with pytest.raises(RuntimeError, match="format version"):
        physbo.search.discrete.Policy.load_checkpoint(fname)


def test_load_checkpoint_rejects_wrong_class(tmp_path):
    fname = str(tmp_path / "ckpt.pkl")
    policy = _make_policy()
    policy.save_checkpoint(fname)
    with pytest.raises(RuntimeError, match="saved by"):
        physbo.search.range.Policy.load_checkpoint(fname)


def test_load_checkpoint_warns_on_version_mismatch(tmp_path):
    fname = str(tmp_path / "ckpt.pkl")
    policy = _make_policy()
    policy.save_checkpoint(fname)

    with open(fname, "rb") as f:
        ckpt = pickle.load(f)
    ckpt["physbo_version"] = "0.0.1"
    with open(fname, "wb") as f:
        pickle.dump(ckpt, f)

    with pytest.warns(RuntimeWarning, match="0.0.1"):
        physbo.search.discrete.Policy.load_checkpoint(fname)


def test_range_policy_checkpoint_roundtrip(tmp_path):
    fname = str(tmp_path / "ckpt.pkl")

    def sim_range(x):
        return -((x[0] - 0.5) ** 2)

    policy = physbo.search.range.Policy(
        min_X=np.array([0.0]), max_X=np.array([1.0]), rng=7
    )
    policy.random_search(max_num_probes=4, simulator=sim_range, is_disp=False)
    policy.bayes_search(max_num_probes=1, simulator=sim_range, score="EI",
                        interval=0, is_disp=False)
    policy.save_checkpoint(fname)

    def cont(p):
        res = p.bayes_search(max_num_probes=2, simulator=sim_range, score="EI",
                             interval=0, is_disp=False)
        return res.action_X[: res.total_num_search]

    reference = cont(policy)
    restored = physbo.search.range.Policy.load_checkpoint(fname)
    resumed = cont(restored)
    np.testing.assert_array_equal(reference, resumed)


def test_multi_policy_pickle_smoke():
    t1 = -((X[:, 0] - 0.3) ** 2)
    t2 = -((X[:, 0] - 0.7) ** 2)

    def sim_multi(actions):
        return np.column_stack([t1[actions], t2[actions]])

    policy = physbo.search.discrete_multi.Policy(
        test_X=X, num_objectives=2, rng=11
    )
    policy.random_search(max_num_probes=4, simulator=sim_multi, is_disp=False)
    policy.bayes_search(max_num_probes=1, simulator=sim_multi, score="TS",
                        interval=1, num_rand_basis=50, is_disp=False)
    clone = pickle.loads(pickle.dumps(policy))
    assert clone.mpicomm is None

    def cont(p):
        res = p.bayes_search(max_num_probes=2, simulator=sim_multi, score="TS",
                             interval=1, num_rand_basis=50, is_disp=False)
        return res.chosen_actions[: res.total_num_search]

    reference = cont(policy)
    resumed = cont(clone)
    np.testing.assert_array_equal(reference, resumed)

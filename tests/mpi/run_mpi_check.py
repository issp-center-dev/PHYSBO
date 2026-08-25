# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""MPI consistency checks for the search policies.

Run under MPI, e.g.:

    mpirun -np 2 python run_mpi_check.py

The script exits with a non-zero status if any check fails; it is
invoked by tests/mpi/test_mpi.py via subprocess.
"""

import sys
from itertools import product

import numpy as np
from mpi4py import MPI

import physbo

comm = MPI.COMM_WORLD


def log(msg):
    if comm.rank == 0:
        print(msg, flush=True)


def assert_identical_over_ranks(arr, label):
    """All ranks must hold an identical copy of arr."""
    gathered = comm.allgather(np.asarray(arr))
    for other in gathered[1:]:
        if not np.array_equal(gathered[0], other):
            raise AssertionError(f"{label} differs between ranks")


def make_grid():
    a = np.linspace(0.0, 1.0, 11)
    return np.array(list(product(a, a)))


def f(x):
    return -np.sum((x - 0.5) ** 2, axis=-1)


def check_discrete(score):
    """Parallel discrete search: ranks must agree, and the deterministic
    scores (EI/PI) must reproduce the serial result exactly."""
    X = make_grid()
    sim = lambda action: f(X[action])

    policy = physbo.search.discrete.Policy(test_X=X, comm=comm)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=10, simulator=sim, is_disp=False)
    res = policy.bayes_search(
        max_num_probes=5, simulator=sim, score=score, is_disp=False, interval=0
    )

    N = res.total_num_search
    assert_identical_over_ranks(res.chosen_actions[:N], f"chosen_actions ({score})")
    assert_identical_over_ranks(res.fx[:N], f"fx ({score})")

    best_fx, best_action = res.export_all_sequence_best_fx()
    if not np.isclose(best_fx[-1], 0.0, atol=1e-3):
        raise AssertionError(
            f"parallel search ({score}) missed the optimum: {best_fx[-1]}"
        )

    if score in ("EI", "PI"):
        # deterministic scores: the parallel run must reproduce the
        # serial run exactly
        serial = physbo.search.discrete.Policy(test_X=X)
        serial.set_seed(12345)
        serial.random_search(max_num_probes=10, simulator=sim, is_disp=False)
        sres = serial.bayes_search(
            max_num_probes=5, simulator=sim, score=score, is_disp=False, interval=0
        )
        M = sres.total_num_search
        if M != N or not np.array_equal(
            sres.chosen_actions[:M], res.chosen_actions[:N]
        ):
            raise AssertionError(f"parallel != serial ({score})")

    log(f"discrete ({score}): OK")


def _run_discrete_ts_blm(comm_, rng, num_search_each_probe=1):
    """Run a fixed-seed random+bayes (TS, BLM) search and return chosen actions."""
    X = make_grid()
    sim = lambda action: f(X[action])
    policy = physbo.search.discrete.Policy(test_X=X, comm=comm_, rng=rng)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=6, simulator=sim, is_disp=False)
    res = policy.bayes_search(
        max_num_probes=3,
        num_search_each_probe=num_search_each_probe,
        simulator=sim,
        score="TS",
        interval=0,
        num_rand_basis=100,
        is_disp=False,
    )
    N = res.total_num_search
    return res.chosen_actions[:N]


def check_discrete_ts_blm_vs_serial(rng, num_search_each_probe, label):
    """TS with the BLM predictor must reproduce the serial run exactly:
    the posterior sample is drawn on rank 0 and broadcast, so the search
    is independent of the number of ranks (single-sample Thompson
    sampling). This also covers the multi-probe (marginal score) path,
    where the virtual observations must not be duplicated over ranks."""
    parallel = _run_discrete_ts_blm(comm, rng, num_search_each_probe)
    assert_identical_over_ranks(parallel, f"chosen_actions ({label})")
    serial = _run_discrete_ts_blm(None, rng, num_search_each_probe)
    if not np.array_equal(parallel, serial):
        raise AssertionError(f"parallel != serial ({label})")
    log(f"{label}: OK")


def check_ts_sample_coherence():
    """All ranks must score identical candidates identically in TS:
    they must share a single posterior sample (regression test for the
    rank-divergent Thompson sampling bug)."""
    import physbo.search.score as search_score

    X = make_grid()
    sim = lambda action: f(X[action])
    policy = physbo.search.discrete.Policy(test_X=X, comm=comm)
    policy.set_seed(4321)
    # rank 0 alone consumes RNG in random_search, so the global streams
    # of the ranks diverge here; the broadcast must hide that.
    policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
    policy.bayes_search(max_num_probes=0, num_rand_basis=100, is_disp=False)
    test = policy.test.get_subset([10, 50, 100])
    fscore = search_score.score(
        "TS",
        predictor=policy.predictor,
        training=policy.training,
        test=test,
        alpha=1.0,
        rng=policy.rng,
        comm=comm,
    )
    assert_identical_over_ranks(fscore, "TS scores of identical candidates")
    log("TS sample coherence: OK")


def check_random_optimizer():
    """The range random optimizer must return the same point on all ranks."""
    opt = physbo.search.optimize.random.Optimizer(
        min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=100
    )
    np.random.seed(12345 + comm.rank)  # deliberately different streams
    x = opt(lambda x: f(x), mpicomm=comm)
    assert_identical_over_ranks(x, "random optimizer result")
    log("random optimizer: OK")


def check_range():
    """Parallel range search: ranks must agree on the history."""
    policy = physbo.search.range.Policy(
        min_X=np.array([0.0, 0.0]), max_X=np.array([1.0, 1.0]), comm=comm
    )
    policy.set_seed(12345)
    sim = lambda x: f(np.atleast_2d(x))
    policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
    res = policy.bayes_search(
        max_num_probes=2, simulator=sim, score="EI", is_disp=False, interval=0
    )
    N = res.total_num_search
    assert_identical_over_ranks(res.action_X[:N], "range action_X")
    assert_identical_over_ranks(res.fx[:N], "range fx")
    log("range (EI): OK")


def main():
    log(f"running on {comm.size} MPI process(es)")
    check_discrete("EI")
    check_discrete("PI")
    check_discrete("TS")
    check_discrete_ts_blm_vs_serial("legacy", 1, "discrete TS-BLM vs serial (legacy)")
    check_discrete_ts_blm_vs_serial(9999, 1, "discrete TS-BLM vs serial (generator)")
    check_discrete_ts_blm_vs_serial(
        "legacy", 2, "discrete TS-BLM multi-probe vs serial (legacy)"
    )
    check_discrete_ts_blm_vs_serial(
        9999, 2, "discrete TS-BLM multi-probe vs serial (generator)"
    )
    check_ts_sample_coherence()
    check_random_optimizer()
    check_range()
    log("all MPI checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        comm.Abort(1)

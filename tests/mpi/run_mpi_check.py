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
import warnings
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


def check_random_optimizer():
    """The range random optimizer must return the same point on all ranks."""
    opt = physbo.search.optimize.random.Optimizer(
        min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=100
    )
    np.random.seed(12345 + comm.rank)  # deliberately different streams
    x = opt(lambda x: f(x), mpicomm=comm)
    assert_identical_over_ranks(x, "random optimizer result")

    # fewer samples than processes: some processes evaluate none
    opt = physbo.search.optimize.random.Optimizer(
        min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=1
    )
    x = opt(lambda x: f(x), mpicomm=comm)
    assert x is not None and x.shape == (2,)
    assert_identical_over_ranks(x, "random optimizer result, one sample")

    # scores of nan on all processes but rank 0: they keep no sample
    opt = physbo.search.optimize.random.Optimizer(
        min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=10
    )
    x = opt(lambda x: f(x) if comm.rank == 0 else np.nan, mpicomm=comm)
    assert x is not None and np.all(np.isfinite(x))
    assert_identical_over_ranks(x, "random optimizer result, nan elsewhere")

    # the point returned is the best sample over all processes, also when it
    # lies on a process other than rank 0 (which here keeps no sample)
    best = [-np.inf]

    def score(x):
        fx = np.nan if comm.rank == 0 else f(x)
        if fx > best[0]:
            best[0] = fx
        return fx

    x = opt(score, mpicomm=comm)
    assert f(x) == max(comm.allgather(best[0]))
    assert_identical_over_ranks(x, "random optimizer result, nan on rank 0")

    # extended precision bounds and a score of another type: the point is
    # received in the type of the samples
    opt = physbo.search.optimize.random.Optimizer(
        min_X=np.zeros(2, dtype=np.longdouble),
        max_X=np.ones(2, dtype=np.longdouble),
        nsamples=1,
    )
    x = opt(lambda x: np.float32(f(x)), mpicomm=comm)
    assert x.dtype == np.longdouble
    assert_identical_over_ranks(x, "random optimizer result, longdouble")

    # integer scores, larger on the last rank by 1 beyond 2**53 (equal as
    # float64): its point wins
    opt = physbo.search.optimize.random.Optimizer(
        min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=comm.size
    )
    last = comm.size - 1
    sample = []

    def integer_score(x):
        if comm.rank == last:
            sample.append(x.copy())  # x itself is overwritten by the broadcast
        return 2**53 + 1 if comm.rank == last else 2**53

    x = opt(integer_score, mpicomm=comm)
    expected = comm.bcast(sample[0] if comm.rank == last else None, root=last)
    assert np.array_equal(x, expected)
    assert_identical_over_ranks(x, "random optimizer result, integer scores")

    # equal scores everywhere: the point of the first process, like argmax
    first = []

    def equal_score(x):
        if comm.rank == 0:
            first.append(x.copy())
        return 0.0

    x = opt(equal_score, mpicomm=comm)
    expected = comm.bcast(first[0] if comm.rank == 0 else None, root=0)
    assert np.array_equal(x, expected)

    # no score larger than -inf anywhere (nan on rank 0, -inf elsewhere):
    # the first sample of the first process, with a warning on every process
    first.clear()

    def no_score(x):
        if comm.rank == 0:
            first.append(x.copy())
            return np.nan
        return -np.inf

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = opt(no_score, mpicomm=comm)
    expected = comm.bcast(first[0] if comm.rank == 0 else None, root=0)
    assert np.array_equal(x, expected)
    assert [w.category for w in caught] == [RuntimeWarning]
    assert_identical_over_ranks(x, "random optimizer result, no finite score")
    log("random optimizer: OK")


def check_range_few_samples():
    """A range search whose optimizer has fewer samples than processes."""
    policy = physbo.search.range.Policy(
        min_X=np.array([0.0, 0.0]), max_X=np.array([1.0, 1.0]), comm=comm
    )
    policy.set_seed(12345)
    sim = lambda x: f(np.atleast_2d(x))
    policy.random_search(max_num_probes=3, simulator=sim, is_disp=False)
    optimizer = physbo.search.optimize.random.Optimizer(
        min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=1
    )
    res = policy.bayes_search(
        max_num_probes=2,
        simulator=sim,
        score="EI",
        is_disp=False,
        interval=0,
        optimizer=optimizer,
    )
    N = res.total_num_search
    assert N == 5
    assert_identical_over_ranks(res.action_X[:N], "range action_X, one sample")
    log("range with one optimizer sample: OK")


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


def f_multi(x):
    x = np.atleast_2d(x)
    return np.stack(
        [-np.sum((x - 0.25) ** 2, axis=-1), -np.sum((x - 0.75) ** 2, axis=-1)],
        axis=-1,
    )


def check_range_multi():
    """Parallel multi-objective range search: ranks must agree on the history."""
    policy = physbo.search.range_multi.Policy(
        num_objectives=2,
        min_X=np.array([0.0, 0.0]),
        max_X=np.array([1.0, 1.0]),
        comm=comm,
    )
    policy.set_seed(12345)
    policy.random_search(max_num_probes=5, simulator=f_multi, is_disp=False)
    res = policy.bayes_search(
        max_num_probes=2, simulator=f_multi, score="HVPI", is_disp=False, interval=0
    )
    N = res.total_num_search
    assert_identical_over_ranks(res.action_X[:N], "range_multi action_X")
    assert_identical_over_ranks(res.fx[:N], "range_multi fx")
    log("range_multi (HVPI): OK")


def check_range_unified():
    """Parallel unified-objective range search: ranks must agree on the history."""
    policy = physbo.search.range_unified.Policy(
        num_objectives=2,
        min_X=np.array([0.0, 0.0]),
        max_X=np.array([1.0, 1.0]),
        comm=comm,
    )
    policy.set_seed(12345)
    policy.random_search(max_num_probes=5, simulator=f_multi, is_disp=False)
    res = policy.bayes_search(
        max_num_probes=2,
        simulator=f_multi,
        score="EI",
        unify_method=physbo.search.unify.ParEGO(num_objectives=2),
        is_disp=False,
        interval=0,
    )
    N = res.total_num_search
    assert_identical_over_ranks(res.action_X[:N], "range_unified action_X")
    assert_identical_over_ranks(res.fx[:N], "range_unified fx")
    log("range_unified (EI, ParEGO): OK")


def main():
    log(f"running on {comm.size} MPI process(es)")
    check_discrete("EI")
    check_discrete("PI")
    check_discrete("TS")
    check_random_optimizer()
    check_range_few_samples()
    check_range()
    check_range_multi()
    check_range_unified()
    log("all MPI checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        comm.Abort(1)

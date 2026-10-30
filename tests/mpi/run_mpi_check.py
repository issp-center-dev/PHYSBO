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

import os
import pickle
import shutil
import sys
import tempfile
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
    # as the policies do: the sample of rank 0 is shared, and the returned
    # score evaluates it
    mode = search_score.share_sample(
        search_score.resolve("TS", rng=policy.rng),
        policy.predictor,
        policy.training,
        comm,
    )
    fscore = mode(policy.predictor, policy.training, test)
    assert_identical_over_ranks(fscore, "TS scores of identical candidates")
    # without sharing, each rank draws its own sample
    own = search_score.resolve("TS", rng=policy.rng)(
        policy.predictor, policy.training, test
    )
    gathered = comm.allgather(np.asarray(own))
    if all(np.array_equal(gathered[0], other) for other in gathered[1:]):
        raise AssertionError("TS scores without a shared sample agree by chance")
    log("TS sample coherence: OK")


def check_checkpoint_resume(rng, label):
    """A checkpoint saved mid-run must resume bit-exactly: the continued
    run after load_checkpoint must equal the uninterrupted run, in both
    the legacy and the Generator RNG modes."""
    X = make_grid()
    sim = lambda action: f(X[action])

    tmpdir = tempfile.mkdtemp() if comm.rank == 0 else None
    tmpdir = comm.bcast(tmpdir, root=0)
    fname = os.path.join(tmpdir, "ckpt.pkl")

    try:
        policy = physbo.search.discrete.Policy(test_X=X, comm=comm, rng=rng)
        policy.set_seed(24680)
        policy.random_search(max_num_probes=6, simulator=sim, is_disp=False)
        policy.bayes_search(
            max_num_probes=2, simulator=sim, score="TS", interval=0,
            num_rand_basis=100, is_disp=False,
        )
        policy.save_checkpoint(fname)
        comm.barrier()

        def cont(p):
            res = p.bayes_search(
                max_num_probes=2, num_search_each_probe=2, simulator=sim,
                score="TS", interval=0, num_rand_basis=100, is_disp=False,
            )
            return res.chosen_actions[: res.total_num_search]

        reference = cont(policy)  # uninterrupted continuation

        restored = physbo.search.discrete.Policy.load_checkpoint(fname, comm=comm)
        resumed = cont(restored)

        assert_identical_over_ranks(resumed, f"resumed chosen_actions ({label})")
        if not np.array_equal(reference, resumed):
            raise AssertionError(f"resumed != uninterrupted ({label})")

        # a checkpoint taken with mpisize>1 must refuse to load serially
        if comm.size > 1:
            try:
                physbo.search.discrete.Policy.load_checkpoint(fname, comm=None)
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    f"load_checkpoint accepted a wrong mpisize ({label})"
                )
    finally:
        comm.barrier()
        if comm.rank == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
    log(f"checkpoint resume ({label}): OK")


def check_checkpoint_with_dup_comm():
    """Policies holding a non-predefined communicator (Dup) must pickle
    and checkpoint correctly: the communicator is excluded from the
    state and re-attached on load."""
    dup = comm.Dup()
    X = make_grid()
    sim = lambda action: f(X[action])

    tmpdir = tempfile.mkdtemp() if comm.rank == 0 else None
    tmpdir = comm.bcast(tmpdir, root=0)
    fname = os.path.join(tmpdir, "ckpt.pkl")

    try:
        policy = physbo.search.discrete.Policy(test_X=X, comm=dup, rng=13579)
        policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)

        # plain pickle must succeed even though dup is not picklable
        blob = pickle.dumps(policy)
        clone = pickle.loads(blob)
        if clone.mpicomm is not None:
            raise AssertionError("mpicomm leaked into the pickled state")
        clone.set_comm(dup)

        policy.save_checkpoint(fname)
        dup.barrier()
        restored = physbo.search.discrete.Policy.load_checkpoint(fname, comm=dup)
        res = restored.bayes_search(
            max_num_probes=1, simulator=sim, score="EI", interval=0,
            is_disp=False,
        )
        N = res.total_num_search
        assert_identical_over_ranks(
            res.chosen_actions[:N], "chosen_actions (dup comm)"
        )
    finally:
        dup.barrier()
        if comm.rank == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
        dup.Free()
    log("checkpoint with Dup() comm: OK")


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

    # scores of nan on all processes but rank 0
    opt = physbo.search.optimize.random.Optimizer(
        min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=10
    )
    x = opt(lambda x: f(x) if comm.rank == 0 else np.nan, mpicomm=comm)
    assert x is not None and np.all(np.isfinite(x))
    assert_identical_over_ranks(x, "random optimizer result, nan elsewhere")

    # the point returned is the best sample over all processes, also when it
    # lies on a process other than rank 0 (whose scores are all nan)
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

    check_random_optimizer_without_score()
    log("random optimizer: OK")


def run_recording_warnings(opt, fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = opt(fn, mpicomm=comm)
    return x, [w.category for w in caught]


def recording(score_of):
    """fn(x) = score_of(x), recording copies of the samples of this process."""
    samples = []

    def fn(x):
        samples.append(x.copy())  # x itself may be overwritten by the broadcast
        return score_of(x)

    return fn, samples


def first_sample_of(samples, rank=0):
    return comm.bcast(samples[0] if comm.rank == rank else None, root=rank)


def check_random_optimizer_without_score():
    """No score larger than -inf on any process: the first sample of rank 0,
    with a warning on every process; otherwise no warning anywhere."""
    Optimizer = physbo.search.optimize.random.Optimizer
    np.random.seed(777 + comm.rank)

    # several samples per process: nan on rank 0, -inf elsewhere
    fn, samples = recording(lambda x: np.nan if comm.rank == 0 else -np.inf)
    x, caught = run_recording_warnings(Optimizer([0, 0], [1, 1], 2 * comm.size), fn)
    assert np.array_equal(x, first_sample_of(samples))
    assert comm.allgather(caught) == [[RuntimeWarning]] * comm.size
    assert_identical_over_ranks(x, "random optimizer result, no score")

    # fewer samples than processes: processes without samples warn too
    fn, samples = recording(lambda x: np.nan)
    x, caught = run_recording_warnings(Optimizer([0, 0], [1, 1], 1), fn)
    assert np.array_equal(x, first_sample_of(samples))
    assert comm.allgather(caught) == [[RuntimeWarning]] * comm.size

    # the warning turned into an error is raised on every process
    for nsamples in [1, 2 * comm.size]:
        opt = Optimizer([0, 0], [1, 1], nsamples)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                opt(lambda x: np.nan, mpicomm=comm)
                raised = False
            except RuntimeWarning:
                raised = True
        assert comm.allgather(raised) == [True] * comm.size

    # a score larger than -inf on some process: no warning on any process,
    # also not on rank 0, whose scores are all nan
    x, caught = run_recording_warnings(
        Optimizer([0, 0], [1, 1], 10), lambda x: np.nan if comm.rank == 0 else f(x)
    )
    assert comm.allgather(caught) == [[]] * comm.size

    # +inf on the last process only: its first sample wins
    last = comm.size - 1
    fn, samples = recording(lambda x: np.inf if comm.rank == last else f(x))
    x, caught = run_recording_warnings(Optimizer([0, 0], [1, 1], 2 * comm.size), fn)
    assert np.array_equal(x, first_sample_of(samples, last))
    assert comm.allgather(caught) == [[]] * comm.size

    # equal scores, several samples per process: the first sample of rank 0
    fn, samples = recording(lambda x: 0.0)
    x, caught = run_recording_warnings(Optimizer([0, 0], [1, 1], 2 * comm.size), fn)
    assert np.array_equal(x, first_sample_of(samples))
    assert comm.allgather(caught) == [[]] * comm.size


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
    if comm.size < 2:
        # several checks give some processes other scores than rank 0
        raise RuntimeError("run with at least 2 MPI processes")
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
    check_checkpoint_resume("legacy", "legacy")
    check_checkpoint_resume(97531, "generator")
    check_checkpoint_with_dup_comm()
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

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

import contextlib
import io
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


def in_disk(X):
    return np.sum((X - 0.5) ** 2, axis=1) <= 0.3**2


def never(X):
    return np.zeros(X.shape[0], dtype=bool)


def check_random_optimizer_constraint():
    """Constrained random optimizer: a common feasible point on every rank."""
    opt = physbo.search.optimize.random.Optimizer(
        min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=100, constraint=in_disk
    )
    np.random.seed(12345 + comm.rank)  # deliberately different streams
    x = opt(lambda x: f(x), mpicomm=comm)
    assert_identical_over_ranks(x, "constrained random optimizer result")
    assert in_disk(x.reshape(1, -1))[0]

    # scores of -inf on all ranks but rank 0
    x = opt(lambda x: f(x) if comm.rank == 0 else -np.inf, mpicomm=comm)
    assert_identical_over_ranks(x, "constrained random optimizer result, rank 0")
    assert in_disk(x.reshape(1, -1))[0]

    # no score larger than -inf anywhere: the first (feasible) sample is
    # returned, with a warning on every rank
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = opt(lambda x: -np.inf, mpicomm=comm)
    assert [w.category for w in caught] == [RuntimeWarning]
    assert_identical_over_ranks(x, "constrained random optimizer result, -inf")
    assert in_disk(x.reshape(1, -1))[0]

    # no feasible sample on any rank: None on every rank
    opt = physbo.search.optimize.random.Optimizer(
        min_X=[0.0, 0.0], max_X=[1.0, 1.0], nsamples=10, constraint=never
    )
    opt.max_trials_per_point = 10
    x = opt(lambda x: f(x), mpicomm=comm)
    assert x is None

    # feasible samples on the last rank only: the other ranks have no sample
    # (and no buffer of their own to receive the point)
    last = comm.size - 1
    opt = physbo.search.optimize.random.Optimizer(
        min_X=[0.0, 0.0],
        max_X=[1.0, 1.0],
        nsamples=10,
        constraint=in_disk if comm.rank == last else never,
    )
    opt.max_trials_per_point = 10
    x = opt(lambda x: f(x), mpicomm=comm)
    assert_identical_over_ranks(x, "constrained random optimizer result, last")
    assert in_disk(x.reshape(1, -1))[0]
    log("random optimizer (constraint): OK")


def make_range_policy(kind, constraint, **kwargs):
    kwargs = dict(
        min_X=np.array([0.0, 0.0]),
        max_X=np.array([1.0, 1.0]),
        constraint=constraint,
        comm=comm,
        **kwargs,
    )
    if kind == "range":
        return physbo.search.range.Policy(**kwargs)
    if kind == "range_multi":
        return physbo.search.range_multi.Policy(num_objectives=2, **kwargs)
    return physbo.search.range_unified.Policy(num_objectives=2, **kwargs)


def check_range_constraint(kind):
    """Constrained range search: ranks must agree on feasible proposals."""
    policy = make_range_policy(kind, in_disk)
    if kind == "range":
        search_args = dict(score="EI")
    elif kind == "range_multi":
        search_args = dict(score="HVPI")
    else:
        search_args = dict(
            score="EI", unify_method=physbo.search.unify.ParEGO(num_objectives=2)
        )
    np.random.seed(12345 + comm.rank)  # deliberately different streams

    def sim(x):
        return f(np.atleast_2d(x)) if kind == "range" else f_multi(x)

    policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
    policy.bayes_search(
        max_num_probes=2, simulator=sim, is_disp=False, interval=0, **search_args
    )
    # the optimizer fails only on rank 0: all the ranks must fall back together
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = policy.bayes_search(
            max_num_probes=2,
            simulator=sim,
            is_disp=False,
            interval=-1,
            **search_args,
            optimizer=lambda fn, mpicomm=None: (
                None if mpicomm.rank == 0 else np.array([0.5, 0.5])
            ),
        )
    N = res.total_num_search
    assert N == 9
    assert_identical_over_ranks(res.action_X[:N], f"constrained {kind} action_X")
    assert np.all(in_disk(res.action_X[:N]))

    # an error while judging the optimizer's result on rank 0 is raised on
    # every rank (no deadlock in the broadcast of the verdict)
    try:
        policy.bayes_search(
            max_num_probes=1,
            simulator=sim,
            is_disp=False,
            interval=-1,
            **search_args,
            optimizer=lambda fn, mpicomm=None: (
                np.array(["a", "b"]) if mpicomm.rank == 0 else np.array([0.5, 0.5])
            ),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a non-numeric optimizer result did not raise")
    comm.Barrier()

    # an error of the constraint itself (here: values instead of booleans)
    # during random_search is raised on every rank, although only rank 0
    # evaluates it
    policy = make_range_policy(kind, lambda X: X[:, 0] - 0.5)
    try:
        policy.random_search(max_num_probes=1, simulator=sim, is_disp=False)
    except ValueError:
        pass
    else:
        raise AssertionError("a constraint returning values did not raise")
    comm.Barrier()

    # the generators of all the ranks stay in step: rank 0 draws the random
    # points, the other ranks draw the same number of points
    policy = make_range_policy(kind, in_disk, rng=4321)
    policy.random_search(
        max_num_probes=3, num_search_each_probe=2, simulator=sim, is_disp=False
    )
    states = comm.allgather(policy.rng.bit_generator.state["state"])
    assert all(s == states[0] for s in states), (
        f"{kind}: rng states differ between ranks after a constrained random_search"
    )
    # the same in the legacy mode (global numpy.random state)
    np.random.seed(2468)
    policy = make_range_policy(kind, in_disk)
    policy.random_search(
        max_num_probes=3, num_search_each_probe=2, simulator=sim, is_disp=False
    )
    key, pos = np.random.get_state()[1:3]
    states = comm.allgather((key.tobytes(), pos))
    assert all(s == states[0] for s in states), (
        f"{kind}: global numpy.random states differ between ranks after a "
        "constrained random_search"
    )

    # an empty feasible region must raise on all ranks (no deadlock)
    policy = make_range_policy(kind, never)
    policy.max_trials_per_point = 100
    try:
        policy.random_search(max_num_probes=1, simulator=sim, is_disp=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("an empty feasible region did not raise")
    comm.Barrier()
    log(f"{kind} (constraint): OK")


def check_no_predictor_warning_on_rank_0_only():
    """The warning about a missing predictor (printed by get_post_fmean etc.
    before bayes_search) appears on rank 0 only."""
    import io
    from contextlib import redirect_stdout

    X = make_grid()
    sim = lambda action: f(X[action])
    policy = physbo.search.discrete.Policy(test_X=X, comm=comm)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=4, simulator=sim, is_disp=False)
    policy.config.learning.is_disp = False
    buf = io.StringIO()
    with redirect_stdout(buf):
        policy.get_post_fmean(X[:3])
    counts = comm.allgather(
        buf.getvalue().count("Warning: Since policy.predictor is not yet set")
    )
    if counts != [1] + [0] * (comm.size - 1):
        raise AssertionError(f"no-predictor warning printed {counts} times per rank")
    log("no-predictor warning on rank 0 only: OK")


def check_multi_probe_header_obeys_is_disp():
    """The header of a multiple-probe step of bayes_search is printed on
    rank 0 only, and not at all with is_disp=False."""
    import io
    from contextlib import redirect_stdout

    X = make_grid()
    sim = lambda action: f(X[action])
    counts = {}
    for is_disp in (False, True):
        policy = physbo.search.discrete.Policy(test_X=X, comm=comm)
        policy.set_seed(12345)
        policy.random_search(max_num_probes=4, simulator=sim, is_disp=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            policy.bayes_search(
                max_num_probes=1,
                num_search_each_probe=2,
                simulator=sim,
                score="EI",
                is_disp=is_disp,
                interval=0,
            )
        counts[is_disp] = comm.allgather(buf.getvalue().count("multiple probe search"))
    if counts[False] != [0] * comm.size:
        raise AssertionError(
            f"multi-probe header printed with is_disp=False: {counts[False]} per rank"
        )
    if counts[True] != [1] + [0] * (comm.size - 1):
        raise AssertionError(
            f"multi-probe header printed {counts[True]} times per rank with is_disp=True"
        )
    log("multi-probe header obeys is_disp: OK")


def check_negative_interval_needs_a_learned_predictor():
    """bayes_search with interval < 0 and no learned predictor is refused on
    every process before anything is communicated, and the policy is usable
    afterwards."""
    X = make_grid()
    sim = lambda action: f(X[action])
    policy = physbo.search.discrete.Policy(test_X=X, comm=comm)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=4, simulator=sim, is_disp=False)
    try:
        policy.bayes_search(max_num_probes=1, simulator=sim, interval=-1, is_disp=False)
    except RuntimeError:
        refused = True
    else:
        refused = False
    if not all(comm.allgather(refused)):
        raise AssertionError("interval < 0 without a learned predictor was not refused")
    res = policy.bayes_search(max_num_probes=1, simulator=sim, interval=0, is_disp=False)
    res = policy.bayes_search(max_num_probes=1, simulator=sim, interval=-1, is_disp=False)
    N = res.total_num_search
    assert N == 6
    assert_identical_over_ranks(res.chosen_actions[:N], "actions after interval < 0")
    log("interval < 0 needs a learned predictor: OK")


def _random_search_to_the_end(comm_, kind, num_candidates):
    """Ask for 3 candidates per step until none is left; the candidates
    searched, in order, and the state of the generator afterwards."""
    X = make_grid()[:num_candidates]
    kwargs = {} if kind == "discrete" else {"num_objectives": 2}
    objective = f if kind == "discrete" else f_multi
    policy = getattr(physbo.search, kind).Policy(
        test_X=X, comm=comm_, rng=2468, **kwargs
    )
    with contextlib.redirect_stdout(io.StringIO()):
        res = policy.random_search(
            max_num_probes=4,
            num_search_each_probe=3,
            simulator=lambda action: objective(X[action]),
            is_disp=False,
        )
    searched = res.chosen_actions[: res.total_num_search]
    return searched, str(policy.rng.bit_generator.state["state"])


def check_random_search_until_exhausted():
    """random_search takes the candidates that are left when they are no
    more than asked for (6 candidates: 3, then exactly 3; 7 candidates: 3,
    3, then 1), stops when none is left, and searches at each step the
    candidates a serial run searches (in the order of the processes that
    have them), with the same use of the generator on rank 0."""
    for kind in ["discrete", "discrete_multi", "discrete_unified"]:
        for num_candidates in (6, 7):
            label = f"{kind} random_search of all {num_candidates} candidates"
            parallel, state = _random_search_to_the_end(comm, kind, num_candidates)
            assert_identical_over_ranks(parallel, label)
            if sorted(parallel.tolist()) != list(range(num_candidates)):
                raise AssertionError(f"{label}: searched {parallel.tolist()}")
            serial, serial_state = _random_search_to_the_end(
                None, kind, num_candidates
            )
            for i in range(0, num_candidates, 3):
                if sorted(parallel[i : i + 3]) != sorted(serial[i : i + 3]):
                    raise AssertionError(f"parallel != serial ({label})")
            # only rank 0 draws, as often as the serial run
            if comm.rank == 0 and state != serial_state:
                raise AssertionError(f"generator of rank 0 != serial ({label})")
    log("random_search until no candidate is left: OK")


def check_random_proposals_do_not_repeat():
    """Candidates proposed at random in the interactive mode are taken from
    the candidates of the process that has them, so that the next proposal
    is made of other ones."""
    X = make_grid()[:8]
    policy = physbo.search.discrete.Policy(test_X=X, comm=comm, rng=1357)
    first = policy.random_search(max_num_probes=1, num_search_each_probe=3, is_disp=False)
    second = policy.random_search(max_num_probes=1, num_search_each_probe=3, is_disp=False)
    assert_identical_over_ranks(first, "first random proposal")
    assert_identical_over_ranks(second, "second random proposal")
    if len(set(first.tolist()) | set(second.tolist())) != 6:
        raise AssertionError(
            f"random proposals overlap: {first.tolist()} and {second.tolist()}"
        )
    log("random proposals do not repeat: OK")


def _negative_score(predictor, training, test):
    return -1.0 - np.arange(test.X.shape[0], dtype=float)


def check_process_without_candidates_does_not_win():
    """When the scores are negative, a process that has no candidate left
    must not be taken for the one with the best candidate."""
    X = make_grid()[:3]
    sim = lambda action: f(X[action])
    policy = physbo.search.discrete.Policy(test_X=X, comm=comm, rng=97)
    policy.random_search(max_num_probes=2, simulator=sim, is_disp=False)
    with contextlib.redirect_stdout(io.StringIO()):
        res = policy.bayes_search(
            max_num_probes=1,
            simulator=sim,
            score=_negative_score,
            interval=0,
            is_disp=False,
        )
    searched = res.chosen_actions[: res.total_num_search]
    assert_identical_over_ranks(searched, "last candidate")
    if sorted(searched.tolist()) != [0, 1, 2]:
        raise AssertionError(f"the last candidate was not proposed: {searched}")
    log("a process without candidates does not win: OK")


def check_range_random_points_are_those_of_rank_0():
    """The random points of a range policy are broadcast from rank 0, so
    that the processes agree even if their generators do not."""
    policy = physbo.search.range.Policy(
        min_X=[-1.0, -1.0], max_X=[1.0, 1.0], comm=comm, rng=1000 + comm.rank
    )
    X = policy.random_search(max_num_probes=1, num_search_each_probe=4, is_disp=False)
    assert_identical_over_ranks(X, "random points of a range policy")
    log("random points of a range policy are those of rank 0: OK")


class _GridOptimizer:
    """Maximize over the points of a grid, without random numbers and
    without communication: every process evaluates every point."""

    def __init__(self):
        a = np.linspace(-1.0, 1.0, 9)
        self.points = np.array(list(product(a, a)))

    def __call__(self, fn, mpicomm=None):
        return self.points[int(np.argmax([fn(x) for x in self.points]))]


def _range_multi_point_search(kind, comm_, rng, num_rand_basis, unify="NDS"):
    """Two steps of three points each with the grid optimizer, after the
    hyperparameters were learned; the points searched, and whether the
    generator of this process was used by these steps."""
    kwargs = {} if kind == "range" else {"num_objectives": 2}
    policy = getattr(physbo.search, kind).Policy(
        min_X=[-1.0, -1.0], max_X=[1.0, 1.0], comm=comm_, rng=rng, **kwargs
    )
    objective = f if kind == "range" else f_multi
    search = {"score": "HVPI"} if kind == "range_multi" else {"score": "EI"}
    if kind == "range_unified":
        search["unify_method"] = getattr(physbo.search.unify, unify)(num_objectives=2)
    policy.random_search(max_num_probes=6, simulator=objective, is_disp=False)
    policy.bayes_search(
        max_num_probes=0,
        interval=0,
        num_rand_basis=num_rand_basis,
        is_disp=False,
        **search,
    )
    before = str(policy.rng.bit_generator.state["state"])
    res = policy.bayes_search(
        max_num_probes=2,
        num_search_each_probe=3,
        simulator=objective,
        optimizer=_GridOptimizer(),
        interval=-1,
        is_disp=False,
        **search,
    )
    drew = str(policy.rng.bit_generator.state["state"]) != before
    return res.action_X[: res.total_num_search], drew


def check_range_multi_point_proposal_vs_serial():
    """The virtual observations of a multi-point proposal of the range
    policies are drawn on rank 0 and broadcast, as in the discrete ones:
    the processes maximize the same function, so that, with an optimizer
    that uses no random numbers, they propose the points of a serial run
    with the generator of rank 0, even if their own generators differ.
    Only rank 0 draws."""
    cases = [("range", "NDS"), ("range_multi", "NDS")]
    # ParEGO draws the weights of the unified objective, on rank 0
    cases += [("range_unified", "NDS"), ("range_unified", "ParEGO")]
    for kind, unify in cases:
        for num_rand_basis in (0, 20):
            label = f"{kind} multi-point proposal (num_rand_basis={num_rand_basis}"
            label += f", {unify})" if kind == "range_unified" else ")"
            rng = 4711 if comm.rank == 0 else 9000 + comm.rank
            parallel, drew = _range_multi_point_search(
                kind, comm, rng, num_rand_basis, unify
            )
            assert_identical_over_ranks(parallel, label)
            if drew != (comm.rank == 0):
                raise AssertionError(
                    f"generator of rank {comm.rank} used: {drew} ({label})"
                )
            serial, _ = _range_multi_point_search(
                kind, None, 4711, num_rand_basis, unify
            )
            if comm.rank == 0 and not np.array_equal(parallel, serial):
                raise AssertionError(f"parallel != serial ({label})")
    log("range multi-point proposals vs serial: OK")


def _unified_search(kind, comm_, rng):
    """A search of a unified policy with ParEGO, two points per step; the
    unified objective at the end and the points searched."""
    X = make_grid()
    discrete = kind.startswith("discrete")
    space = {"test_X": X} if discrete else {"min_X": [-1.0, -1.0], "max_X": [1.0, 1.0]}
    policy = getattr(physbo.search, kind).Policy(
        num_objectives=2, comm=comm_, rng=rng, **space
    )
    sim = (lambda action: f_multi(X[action])) if discrete else f_multi
    search = dict(
        simulator=sim,
        score="EI",
        unify_method=physbo.search.unify.ParEGO(num_objectives=2),
        is_disp=False,
    )
    if not discrete:
        search["optimizer"] = _GridOptimizer()
    policy.random_search(max_num_probes=6, simulator=sim, is_disp=False)
    res = policy.bayes_search(
        max_num_probes=3, num_search_each_probe=2, interval=1, **search
    )
    N = res.total_num_search
    searched = res.chosen_actions[:N] if discrete else res.action_X[:N]
    return policy.training_unified.t.copy(), searched


def check_unified_objective_is_the_same_on_every_rank():
    """The objectives are unified on rank 0 for every process: ParEGO draws
    its weights there, so that the processes agree on the unified objective
    and on the search even if their generators differ, and find what a
    serial run with the generator of rank 0 finds."""
    for kind in ["discrete_unified", "range_unified"]:
        label = f"{kind} with ParEGO"
        rng = 2718 if comm.rank == 0 else 7000 + comm.rank
        unified, searched = _unified_search(kind, comm, rng)
        assert_identical_over_ranks(unified, f"unified objective ({label})")
        assert_identical_over_ranks(searched, f"points searched ({label})")
        serial_unified, serial_searched = _unified_search(kind, None, 2718)
        if comm.rank == 0 and not (
            np.array_equal(unified, serial_unified)
            and np.array_equal(searched, serial_searched)
        ):
            raise AssertionError(f"parallel != serial ({label})")
    log("unified objective is the same on every rank: OK")


def check_unified_posterior_on_one_rank_alone():
    """Outside a search, a method of a unified policy may be called by one
    process alone (post-processing on rank 0), also when observations were
    added since the predictor was updated: it does not wait for the others.
    The search that follows unifies the objectives anew for every process."""
    X = make_grid()
    sim = lambda action: f_multi(X[action])
    for kind in ["discrete_unified", "range_unified"]:
        discrete = kind.startswith("discrete")
        space = {"test_X": X} if discrete else {"min_X": [-1.0, -1.0], "max_X": [1.0, 1.0]}
        simulator = sim if discrete else f_multi
        policy = getattr(physbo.search, kind).Policy(
            num_objectives=2, comm=comm, rng=1618, **space
        )
        search = dict(
            simulator=simulator,
            score="EI",
            unify_method=physbo.search.unify.ParEGO(num_objectives=2),
            is_disp=False,
        )
        if not discrete:
            search["optimizer"] = _GridOptimizer()
        policy.random_search(max_num_probes=5, simulator=simulator, is_disp=False)
        policy.bayes_search(max_num_probes=2, interval=0, **search)
        policy.random_search(max_num_probes=1, simulator=simulator, is_disp=False)
        if comm.rank == 0:
            fmean = policy.get_post_fmean(X[:3])
            if fmean.shape[0] != 3 or not np.all(np.isfinite(fmean)):
                raise AssertionError(f"{kind}: posterior on rank 0: {fmean}")
        comm.barrier()
        res = policy.bayes_search(
            max_num_probes=2, num_search_each_probe=2, interval=-1, **search
        )
        label = f"{kind} after a posterior on rank 0 alone"
        assert_identical_over_ranks(policy.training_unified.t, f"unified ({label})")
        N = res.total_num_search
        searched = res.chosen_actions[:N] if discrete else res.action_X[:N]
        assert_identical_over_ranks(searched, f"points searched ({label})")
    log("unified posterior on one rank alone: OK")


def check_proposal_without_virtual_observations():
    """multi_probe_num_sampling = 0: nothing is drawn; the processes agree
    on the points of a multi-point proposal."""
    X = make_grid()
    for kind in ["discrete", "range", "range_multi", "range_unified"]:
        discrete = kind == "discrete"
        kwargs = {} if kind in ("discrete", "range") else {"num_objectives": 2}
        space = (
            {"test_X": X} if discrete else {"min_X": [-1.0, -1.0], "max_X": [1.0, 1.0]}
        )
        policy = getattr(physbo.search, kind).Policy(
            comm=comm, rng=1234, **space, **kwargs
        )
        policy.config.search.multi_probe_num_sampling = 0
        objective = f if kind in ("discrete", "range") else f_multi
        sim = (lambda action: objective(X[action])) if discrete else objective
        search = {"score": "HVPI"} if kind == "range_multi" else {"score": "EI"}
        if kind == "range_unified":
            search["unify_method"] = physbo.search.unify.NDS(num_objectives=2)
        policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)
        res = policy.bayes_search(
            max_num_probes=2, num_search_each_probe=3, simulator=sim,
            is_disp=False, **search,
        )
        N = res.total_num_search
        if N != 11:
            raise AssertionError(f"{kind} without virtual observations: {N} points")
        searched = res.chosen_actions[:N] if discrete else res.action_X[:N]
        assert_identical_over_ranks(searched, f"{kind} without virtual observations")
    log("proposals without virtual observations: OK")


def _raised(fn):
    """(name of the exception, message) of what fn() raises on this
    process, gathered over the processes; None where it did not raise."""
    try:
        fn()
    except Exception as e:
        got = (type(e).__name__, str(e))
    else:
        got = None
    return comm.allgather(got)


def check_policy_without_communicator_is_refused():
    """A policy unpickled from a parallel run has no communicator until
    set_comm() is called; searching with it is refused on every process
    instead of going on as a serial run on each."""
    X = make_grid()
    sim = lambda action: f(X[action])
    policy = physbo.search.discrete.Policy(test_X=X, comm=comm, rng=4321)
    clone = pickle.loads(pickle.dumps(policy))
    results = _raised(
        lambda: clone.random_search(max_num_probes=1, simulator=sim, is_disp=False)
    )
    for got in results:
        if got is None or got[0] != "RuntimeError" or "set_comm" not in got[1]:
            raise AssertionError(f"search without set_comm(): {results}")
    # nor is it written to a checkpoint, which would hold one process only
    tmpdir = tempfile.mkdtemp() if comm.rank == 0 else None
    tmpdir = comm.bcast(tmpdir, root=0)
    fname = os.path.join(tmpdir, "ckpt.pkl")
    try:
        results = _raised(lambda: clone.save_checkpoint(fname))
        for got in results:
            if got is None or got[0] != "RuntimeError" or "set_comm" not in got[1]:
                raise AssertionError(f"save_checkpoint without set_comm(): {results}")
        if comm.rank == 0 and os.path.exists(fname):
            raise AssertionError("a checkpoint was written without set_comm()")
    finally:
        comm.barrier()
        if comm.rank == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
    clone.set_comm(comm)
    res = clone.random_search(max_num_probes=2, simulator=sim, is_disp=False)
    assert_identical_over_ranks(res.chosen_actions[:2], "search after set_comm()")
    log("policy without its communicator is refused: OK")


def check_checkpoint_is_refused_on_every_rank():
    """A checkpoint that cannot be used or cannot be read is refused on
    every process; none is left waiting for rank 0."""
    X = make_grid()
    cls = physbo.search.discrete.Policy

    tmpdir = tempfile.mkdtemp() if comm.rank == 0 else None
    tmpdir = comm.bcast(tmpdir, root=0)
    fname = os.path.join(tmpdir, "ckpt.pkl")
    try:
        cls(test_X=X, comm=comm, rng=1).save_checkpoint(fname)
        comm.barrier()
        if comm.rank == 0:
            with open(fname, "rb") as fp:
                header = pickle.load(fp)
                state = fp.read()

        def write(name, header_, state_):
            if comm.rank == 0:
                with open(os.path.join(tmpdir, name), "wb") as fp:
                    pickle.dump(header_, fp)
                    fp.write(state_)
            comm.barrier()
            return os.path.join(tmpdir, name)

        def refused(label, name, exception, text):
            # the exception of rank 0, or a RuntimeError that names it where
            # it cannot be copied to the other processes
            results = _raised(lambda: cls.load_checkpoint(name, comm=comm))
            for got in results:
                if got is None or text not in got[1]:
                    raise AssertionError(f"{label}: {results}")
                if got[0] != exception and not (
                    got[0] == "RuntimeError" and exception in got[1]
                ):
                    raise AssertionError(f"{label}: {results}")

        other = write(
            "version.pkl",
            dict(header, format_version=header["format_version"] + 1)
            if comm.rank == 0
            else None,
            state if comm.rank == 0 else None,
        )
        refused("another format version", other, "RuntimeError", "format version")

        missing = os.path.join(tmpdir, "missing.pkl")
        refused("missing file", missing, "FileNotFoundError", "missing.pkl")

        broken = write(
            "broken.pkl", header if comm.rank == 0 else None, b"not a pickle"
        )
        refused("state that cannot be unpickled", broken, "UnpicklingError", "")

        if comm.rank == 0:
            with open(os.path.join(tmpdir, "empty.pkl"), "wb"):
                pass
        comm.barrier()
        refused("empty file", os.path.join(tmpdir, "empty.pkl"), "EOFError", "")
    finally:
        comm.barrier()
        if comm.rank == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
    log("unusable checkpoints are refused on every rank: OK")


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
    check_random_optimizer_constraint()
    for kind in ["range", "range_multi", "range_unified"]:
        check_range_constraint(kind)
    check_no_predictor_warning_on_rank_0_only()
    check_multi_probe_header_obeys_is_disp()
    check_negative_interval_needs_a_learned_predictor()
    check_random_search_until_exhausted()
    check_random_proposals_do_not_repeat()
    check_process_without_candidates_does_not_win()
    check_range_random_points_are_those_of_rank_0()
    check_range_multi_point_proposal_vs_serial()
    check_unified_objective_is_the_same_on_every_rank()
    check_unified_posterior_on_one_rank_alone()
    check_proposal_without_virtual_observations()
    check_policy_without_communicator_is_refused()
    check_checkpoint_is_refused_on_every_rank()
    log("all MPI checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        comm.Abort(1)

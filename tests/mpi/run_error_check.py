# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""MPI checks for exceptions raised on some processes only.

Run under MPI, e.g.:

    mpirun -np 2 python run_error_check.py

An exception raised on some processes only (by the simulator on rank 0, or in
the scores of one process's share of the candidates, ...) must be raised on
every process, instead of leaving the others in a collective operation. Without
that, this script hangs, and tests/mpi/test_mpi.py fails by its timeout.
"""

import inspect
from contextlib import contextmanager
from itertools import product

import numpy as np
from mpi4py import MPI

import physbo
from physbo import _mpi

comm = MPI.COMM_WORLD
last = comm.size - 1


def log(msg):
    if comm.rank == 0:
        print(msg, flush=True)


class Failure(Exception):
    pass


def expect_raised_everywhere(
    label, run, failing_rank, message="injected", name="Failure"
):
    """run() must raise the exception name(message) on every process; on a
    process other than failing_rank it must be caused by the ProcessError of
    failing_rank.

    If the code under test leaves a process in one of its collective
    operations, the allgather here may pair with it, and the check fails with
    a confusing message (or hangs) instead of passing.
    """
    try:
        run()
        error = None
    except Exception as e:
        error = e
    if error is None:
        info = None
    else:
        cause = error.__cause__
        info = (
            type(error).__name__,
            str(error),
            cause.rank if isinstance(cause, _mpi.ProcessError) else None,
        )
    for rank, i in enumerate(comm.allgather(info)):
        if i is None:
            raise AssertionError(f"{label}: no exception on rank {rank}")
        expected_cause = None if rank == failing_rank else failing_rank
        if i != (name, message, expected_cause):
            raise AssertionError(f"{label}: rank {rank} raised {i}")
    log(f"{label}: raised on every process")


def assert_identical_over_ranks(arr, label):
    gathered = comm.allgather(np.asarray(arr))
    for other in gathered[1:]:
        if not np.array_equal(gathered[0], other):
            raise AssertionError(f"{label} differs between ranks")


@contextmanager
def patched(owner, name, on_rank, after=0):
    """Make owner.name raise Failure on rank on_rank, from its call number
    after + 1 on."""
    original = getattr(owner, name)
    calls = [0]

    def wrapper(*args, **kwargs):
        calls[0] += 1
        if comm.rank == on_rank and calls[0] > after:
            raise Failure("injected")
        return original(*args, **kwargs)

    setattr(owner, name, wrapper)
    try:
        yield
    finally:
        setattr(owner, name, original)


@contextmanager
def patched_in(owner, name, caller, on_rank):
    """Make owner.name raise Failure on rank on_rank when called (directly or
    not) from a function named caller."""
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        if comm.rank == on_rank and any(
            frame.function == caller for frame in inspect.stack()
        ):
            raise Failure("injected")
        return original(*args, **kwargs)

    setattr(owner, name, wrapper)
    try:
        yield
    finally:
        setattr(owner, name, original)


def make_grid():
    a = np.linspace(0.0, 1.0, 11)
    return np.array(list(product(a, a)))


X = make_grid()


def f(x):
    return -np.sum((x - 0.5) ** 2, axis=-1)


def f_multi(x):
    x = np.atleast_2d(x)
    return np.stack(
        [-np.sum((x - 0.25) ** 2, axis=-1), -np.sum((x - 0.75) ** 2, axis=-1)],
        axis=-1,
    )


def failing_simulator(x):
    raise Failure("injected")


def discrete(n_random=5, config=None):
    policy = physbo.search.discrete.Policy(test_X=X, config=config, comm=comm)
    policy.set_seed(12345)
    if n_random:
        sim = lambda action: f(X[action])
        policy.random_search(max_num_probes=n_random, simulator=sim, is_disp=False)
    return policy


def range_single(n_random=5):
    policy = physbo.search.range.Policy(
        min_X=np.zeros(2), max_X=np.ones(2), comm=comm
    )
    policy.set_seed(12345)
    if n_random:
        sim = lambda x: f(np.atleast_2d(x))
        policy.random_search(max_num_probes=n_random, simulator=sim, is_disp=False)
    return policy


def discrete_multi(n_random=5):
    policy = physbo.search.discrete_multi.Policy(
        test_X=X, num_objectives=2, comm=comm
    )
    policy.set_seed(12345)
    if n_random:
        sim = lambda action: f_multi(X[action])
        policy.random_search(max_num_probes=n_random, simulator=sim, is_disp=False)
    return policy


def range_multi(n_random=5):
    policy = physbo.search.range_multi.Policy(
        min_X=np.zeros(2), max_X=np.ones(2), num_objectives=2, comm=comm
    )
    policy.set_seed(12345)
    if n_random:
        policy.random_search(max_num_probes=n_random, simulator=f_multi, is_disp=False)
    return policy


def discrete_unified(n_random=5):
    policy = physbo.search.discrete_unified.Policy(
        test_X=X, num_objectives=2, comm=comm
    )
    policy.set_seed(12345)
    if n_random:
        sim = lambda action: f_multi(X[action])
        policy.random_search(max_num_probes=n_random, simulator=sim, is_disp=False)
    return policy


def range_unified(n_random=5):
    policy = physbo.search.range_unified.Policy(
        min_X=np.zeros(2), max_X=np.ones(2), num_objectives=2, comm=comm
    )
    policy.set_seed(12345)
    if n_random:
        policy.random_search(max_num_probes=n_random, simulator=f_multi, is_disp=False)
    return policy


def unify():
    return physbo.search.unify.ParEGO(num_objectives=2)


def check_simulator():
    """A simulator that raises (it runs on rank 0 only)."""
    for name, make in [
        ("discrete", discrete),
        ("range", range_single),
        ("discrete_multi", discrete_multi),
        ("range_multi", range_multi),
        ("discrete_unified", discrete_unified),
        ("range_unified", range_unified),
    ]:
        expect_raised_everywhere(
            f"{name} random_search, simulator",
            lambda: make(n_random=0).random_search(
                max_num_probes=1, simulator=failing_simulator, is_disp=False
            ),
            failing_rank=0,
        )
        kwargs = {"unify_method": unify()} if "unified" in name else {}
        score = "HVPI" if name.endswith("multi") else "EI"
        expect_raised_everywhere(
            f"{name} bayes_search, simulator",
            lambda: make().bayes_search(
                max_num_probes=1,
                simulator=failing_simulator,
                score=score,
                is_disp=False,
                **kwargs,
            ),
            failing_rank=0,
        )


def check_scores():
    """Scores that raise on the last process only."""
    score = physbo.search.score
    score_multi = physbo.search.score_multi
    sim = lambda action: f(X[action])
    sim_multi = lambda action: f_multi(X[action])
    sim_x = lambda x: f(np.atleast_2d(x))
    cases = [
        ("discrete", discrete, score, sim, {"score": "EI"}),
        ("discrete, 2 per step", discrete, score, sim,
         {"score": "EI", "num_search_each_probe": 2}),
        ("range", range_single, score, sim_x, {"score": "EI"}),
        ("discrete_multi", discrete_multi, score_multi, sim_multi, {"score": "HVPI"}),
        ("discrete_multi, 2 per step", discrete_multi, score_multi, sim_multi,
         {"score": "HVPI", "num_search_each_probe": 2}),
        ("range_multi", range_multi, score_multi, f_multi, {"score": "HVPI"}),
        ("discrete_unified", discrete_unified, score, sim_multi,
         {"score": "EI", "unify_method": None}),
        ("discrete_unified, 2 per step", discrete_unified, score, sim_multi,
         {"score": "EI", "unify_method": None, "num_search_each_probe": 2}),
        ("range_unified", range_unified, score, f_multi,
         {"score": "EI", "unify_method": None}),
    ]
    for name, make, module, simulator, kwargs in cases:
        if "unify_method" in kwargs:
            kwargs = dict(kwargs, unify_method=unify())
        policy = make()
        with patched(module, "score", on_rank=last):
            expect_raised_everywhere(
                f"{name} bayes_search, scores",
                lambda: policy.bayes_search(
                    max_num_probes=1, simulator=simulator, is_disp=False, **kwargs
                ),
                failing_rank=last,
            )

    # only the scores of the second candidate of a step (the marginal scores)
    policy = discrete()
    with patched(score, "score", on_rank=last, after=1):
        expect_raised_everywhere(
            "discrete bayes_search, marginal scores",
            lambda: policy.bayes_search(
                max_num_probes=1,
                simulator=sim,
                score="EI",
                num_search_each_probe=2,
                is_disp=False,
            ),
            failing_rank=last,
        )

    # the samples for the marginal scores, drawn on rank 0 only (the discrete
    # policies broadcast them)
    policy = discrete()
    with patched(physbo.gp.Predictor, "get_predict_samples", on_rank=0):
        expect_raised_everywhere(
            "discrete bayes_search, samples",
            lambda: policy.bayes_search(
                max_num_probes=1,
                simulator=sim,
                score="EI",
                num_search_each_probe=2,
                is_disp=False,
            ),
            failing_rank=0,
        )

    # only the marginal scores, and the samples for them, of the other
    # discrete policies
    for name, make, module, simulator, kwargs in [
        ("discrete_multi", discrete_multi, score_multi, sim_multi,
         {"score": "HVPI"}),
        ("discrete_unified", discrete_unified, score, sim_multi,
         {"score": "EI", "unify_method": unify()}),
    ]:
        for label, owner, attr, after, on_rank in [
            ("marginal scores", module, "score", 1, last),
            ("samples", physbo.gp.Predictor, "get_predict_samples", 0, 0),
        ]:
            policy = make()
            with patched(owner, attr, on_rank=on_rank, after=after):
                expect_raised_everywhere(
                    f"{name} bayes_search, {label}",
                    lambda: policy.bayes_search(
                        max_num_probes=1,
                        simulator=simulator,
                        num_search_each_probe=2,
                        is_disp=False,
                        **kwargs,
                    ),
                    failing_rank=on_rank,
                )

    # the posterior sample of Thompson sampling with the BLM predictor, drawn
    # on rank 0 only and broadcast: for the first candidate of a step and for
    # the marginal scores of the second one
    for label, after in [("TS sample", 0), ("TS samples for marginal scores", 1)]:
        policy = discrete()
        with patched(
            physbo.blm.Predictor, "draw_post_sample_params", on_rank=0, after=after
        ):
            expect_raised_everywhere(
                f"discrete bayes_search, {label}",
                lambda: policy.bayes_search(
                    max_num_probes=1,
                    simulator=sim,
                    score="TS",
                    num_rand_basis=20,
                    num_search_each_probe=2,
                    is_disp=False,
                ),
                failing_rank=0,
            )

    # the samples for the second candidate of a step in the range policies,
    # drawn before the optimizer
    for name, make, simulator, kwargs in [
        ("range", range_single, sim_x, {"score": "EI"}),
        ("range_multi", range_multi, f_multi, {"score": "HVPI"}),
        ("range_unified", range_unified, f_multi,
         {"score": "EI", "unify_method": unify()}),
    ]:
        policy = make()
        with patched(physbo.gp.Predictor, "get_predict_samples", on_rank=last):
            expect_raised_everywhere(
                f"{name} bayes_search, samples",
                lambda: policy.bayes_search(
                    max_num_probes=1,
                    simulator=simulator,
                    num_search_each_probe=2,
                    is_disp=False,
                    **kwargs,
                ),
                failing_rank=last,
            )

    # the same without samples (multi_probe_num_sampling = 0), where only the
    # process that failed has no samples to set up predictors for
    for name, make, simulator, kwargs in [
        ("range", range_single, sim_x, {"score": "EI"}),
        ("range_unified", range_unified, f_multi,
         {"score": "EI", "unify_method": unify()}),
    ]:
        policy = make()
        policy.config.search.multi_probe_num_sampling = 0
        with patched(physbo.gp.Predictor, "get_predict_samples", on_rank=last):
            expect_raised_everywhere(
                f"{name} bayes_search, samples, none drawn",
                lambda: policy.bayes_search(
                    max_num_probes=1,
                    simulator=simulator,
                    num_search_each_probe=2,
                    is_disp=False,
                    **kwargs,
                ),
                failing_rank=last,
            )

    # the setup of the predictors for these samples, before the optimizer
    for name, make, simulator, kwargs in [
        ("range", range_single, sim_x, {"score": "EI"}),
        ("range_multi", range_multi, f_multi, {"score": "HVPI"}),
        ("range_unified", range_unified, f_multi,
         {"score": "EI", "unify_method": unify()}),
    ]:
        policy = make()
        with patched_in(
            physbo.gp.Predictor, "update", "marginal_score_function", on_rank=last
        ):
            expect_raised_everywhere(
                f"{name} bayes_search, predictors for the samples",
                lambda: policy.bayes_search(
                    max_num_probes=1,
                    simulator=simulator,
                    num_search_each_probe=2,
                    is_disp=False,
                    **kwargs,
                ),
                failing_rank=last,
            )

    # get_score in parallel, which bayes_search does not use
    for name, make, module, simulator, kwargs in [
        ("discrete", discrete, score, sim, {"score": "EI"}),
        ("discrete_multi", discrete_multi, score_multi, sim_multi,
         {"score": "HVPI"}),
        ("discrete_unified", discrete_unified, score, sim_multi,
         {"score": "EI", "unify_method": unify()}),
        ("range", range_single, score, sim_x, {"score": "EI"}),
        ("range_multi", range_multi, score_multi, f_multi, {"score": "HVPI"}),
        ("range_unified", range_unified, score, f_multi,
         {"score": "EI", "unify_method": unify()}),
    ]:
        policy = make()
        policy.bayes_search(
            max_num_probes=1, simulator=simulator, is_disp=False, **kwargs
        )
        xs = {"xs": X} if name.startswith("range") else {}
        with patched(module, "score", on_rank=last):
            expect_raised_everywhere(
                f"{name} get_score",
                lambda: policy.get_score(kwargs["score"], parallel=True, **xs),
                failing_rank=last,
            )

    # user-given candidates, invalid in the share of the last process only
    actions = list(range(2 * comm.size - 1)) + [10**6]
    for name, make, simulator, kwargs in [
        ("discrete", discrete, sim, {"score": "EI"}),
        ("discrete_multi", discrete_multi, sim_multi, {"score": "HVPI"}),
        ("discrete_unified", discrete_unified, sim_multi,
         {"score": "EI", "unify_method": unify()}),
    ]:
        policy = make()
        policy.bayes_search(
            max_num_probes=1, simulator=simulator, is_disp=False, **kwargs
        )
        expect_raised_everywhere(
            f"{name} get_score, invalid candidate",
            lambda: policy.get_score(kwargs["score"], actions=actions, parallel=True),
            failing_rank=last,
            message=f"index {10**6} is out of bounds for axis 0 with size {len(X)}",
            name="IndexError",
        )


def check_fit():
    """Learning of the hyperparameters, and the random features of the
    Bayesian linear model, raising on one process (on rank 0, whose result is
    used, or on another one) is raised on every process."""
    sim = lambda action: f(X[action])
    learning = physbo.gp.core.learning
    for name, owner, attr, kwargs in [
        ("learning (Adam)", learning.Adam, "run", {"score": "EI"}),
        ("learning (Batch)", learning.Batch, "run", {"score": "EI"}),
        ("random features", physbo.gp.cov.Gauss, "rand_expans",
         {"score": "TS", "num_rand_basis": 50}),
    ]:
        for failing_rank in [0, last]:
            if "Batch" in name:
                learning_config = physbo.misc._set_config.Batch()
            else:
                learning_config = physbo.misc._set_config.Adam()
            config = physbo.misc.SetConfig(learning_config=learning_config)
            policy = discrete(config=config)

            def run():
                return policy.bayes_search(
                    max_num_probes=2,
                    simulator=sim,
                    is_disp=False,
                    interval=1,
                    **kwargs,
                )

            with patched(owner, attr, on_rank=failing_rank):
                expect_raised_everywhere(
                    f"{name} on rank {failing_rank}", run, failing_rank=failing_rank
                )


class Unpicklable(Exception):
    def __init__(self, a, b):
        super().__init__(f"{a} {b}")


class BadMessage(Exception):
    def __str__(self):
        raise AttributeError("no message")


def check_helpers():
    """Exceptions that cannot be copied to the other processes (by pickle) are
    raised there as RuntimeError; an exception without message is sent too."""

    def run(make_error):
        error = make_error() if comm.rank == last else None
        _mpi.allgather(comm, 1, error)

    if comm.rank == last:
        expected = ("Unpicklable", "1 2", None)
    else:
        expected = ("RuntimeError", "Unpicklable: 1 2", last)
    tb = ""
    try:
        run(lambda: Unpicklable(1, 2))
        info = None
    except Exception as e:
        cause = e.__cause__
        info = (
            type(e).__name__,
            str(e),
            cause.rank if isinstance(cause, _mpi.ProcessError) else None,
        )
        if isinstance(cause, _mpi.ProcessError):
            tb = cause.tb
    if info != expected:
        raise AssertionError(f"exception that cannot be copied: {info}")
    if comm.rank != last and "Unpicklable: 1 2" not in tb:
        raise AssertionError("the traceback of the other process is missing")

    # an exception that cannot be pickled
    def unpicklable():
        e = Failure("injected")
        e.f = lambda: None
        return e

    try:
        run(unpicklable)
        info = None
    except Exception as e:
        info = type(e).__name__, str(e)
    if comm.rank == last:
        expected = ("Failure", "injected")
    else:
        expected = ("RuntimeError", "Failure: injected")
    if info != expected:
        raise AssertionError(f"exception that cannot be pickled: {info}")
    # an exception whose message cannot be formatted
    try:
        run(BadMessage)
        info = None
    except Exception as e:
        cause = e.__cause__
        info = (
            type(e).__name__,
            cause.rank if isinstance(cause, _mpi.ProcessError) else None,
        )
    expected = ("BadMessage", None if comm.rank == last else last)
    if info != expected:
        raise AssertionError(f"exception without message: {info}")
    log("helpers: OK")


def main():
    log(f"running on {comm.size} MPI process(es)")
    check_helpers()
    check_simulator()
    check_scores()
    check_fit()
    log("all MPI checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        comm.Abort(1)

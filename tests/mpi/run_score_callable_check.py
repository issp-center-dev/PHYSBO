# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""MPI checks for callable acquisition functions (score).

Run under MPI, e.g.:

    mpirun -np 2 python run_score_callable_check.py

The script exits with a non-zero status if any check fails; it is
invoked by tests/mpi/test_score_callable_mpi.py via subprocess.
"""

from itertools import product

import numpy as np
from mpi4py import MPI

import physbo

comm = MPI.COMM_WORLD

MIN_X = np.array([0.0, 0.0])
MAX_X = np.array([1.0, 1.0])
INITIAL_X_RANGE = np.array([[0.1, 0.1], [0.9, 0.2], [0.3, 0.8], [0.5, 0.5]])
SINGLE_KINDS = ["discrete", "range", "discrete_unified", "range_unified"]
MULTI_KINDS = ["discrete_multi", "range_multi"]
ALL_KINDS = SINGLE_KINDS + MULTI_KINDS
DISCRETE_KINDS = ["discrete", "discrete_unified", "discrete_multi"]
RANGE_KINDS = ["range", "range_unified", "range_multi"]


def log(msg):
    if comm.rank == 0:
        print(msg, flush=True)


def assert_identical_over_ranks(arr, label):
    """All ranks must hold an identical copy of arr."""
    gathered = comm.allgather(np.asarray(arr))
    for other in gathered[1:]:
        if gathered[0].shape != other.shape or not np.array_equal(
            gathered[0], other
        ):
            raise AssertionError(f"{label} differs between ranks")


def grid_X():
    a = np.linspace(0.0, 1.0, 5)
    return np.array(list(product(a, a)))


def f_single(x):
    x = np.atleast_2d(x)
    return -np.sum((x - 0.6) ** 2, axis=-1)


def f_multi(x):
    x = np.atleast_2d(x)
    return np.stack(
        [-np.sum((x - 0.25) ** 2, axis=-1), -np.sum((x - 0.75) ** 2, axis=-1)],
        axis=-1,
    )


def is_multi(kind):
    return kind in MULTI_KINDS


def make_config():
    config = physbo.misc.SetConfig()
    config.search.multi_probe_num_sampling = 3
    config.learning.is_disp = False
    config.learning.num_init_params_search = 5
    config.learning.max_epoch = 50
    config.learning.max_epoch_init_params_search = 10
    return config


def make_policy(kind, initial_actions=None):
    config = make_config()
    X = grid_X()
    if initial_actions is None:
        initial_actions = np.array([0, 7, 13, 24])
    if kind == "discrete":
        return physbo.search.discrete.Policy(
            test_X=X,
            config=config,
            initial_data=(initial_actions, f_single(X[initial_actions])),
            comm=comm,
        )
    if kind == "range":
        return physbo.search.range.Policy(
            min_X=MIN_X,
            max_X=MAX_X,
            config=config,
            initial_data=(INITIAL_X_RANGE, f_single(INITIAL_X_RANGE)),
            comm=comm,
        )
    if kind in ("discrete_unified", "discrete_multi"):
        return getattr(physbo.search, kind).Policy(
            test_X=X,
            num_objectives=2,
            config=config,
            initial_data=(initial_actions, f_multi(X[initial_actions])),
            comm=comm,
        )
    return getattr(physbo.search, kind).Policy(
        num_objectives=2,
        min_X=MIN_X,
        max_X=MAX_X,
        config=config,
        initial_data=(INITIAL_X_RANGE, f_multi(INITIAL_X_RANGE)),
        comm=comm,
    )


def make_simulator(kind):
    X = grid_X()
    if kind == "discrete":
        return lambda action: f_single(X[action])
    if kind == "range":
        return lambda x: f_single(x)
    if kind in ("discrete_unified", "discrete_multi"):
        return lambda action: f_multi(X[action])
    return lambda x: f_multi(x)


def search_kwargs(kind):
    kw = {}
    if kind.endswith("unified"):
        kw["unify_method"] = physbo.search.unify.NDS(num_objectives=2)
    if kind in RANGE_KINDS:
        kw["optimizer"] = physbo.search.optimize.random.Optimizer(
            min_X=MIN_X, max_X=MAX_X, nsamples=20
        )
    return kw


class CountingSingle:
    """Deterministic single-objective score (mean + std) counting its calls."""

    def __init__(self):
        self.test_X = []

    def __call__(self, predictor, training, test):
        self.test_X.append(np.array(test.X, copy=True))
        fmean = predictor.get_post_fmean(training, test)
        fcov = predictor.get_post_fcov(training, test)
        return (fmean + np.sqrt(fcov)).flatten()


class CountingMulti:
    """Deterministic multi-objective score (sum of means) counting its calls."""

    def __init__(self):
        self.test_X = []

    def __call__(self, predictor_list, training, test, pareto):
        self.test_X.append(np.array(test.X, copy=True))
        return np.sum(
            [
                p.get_post_fmean(training, test, objective_index=i)
                for i, p in enumerate(predictor_list)
            ],
            axis=0,
        ).flatten()


def make_score(kind):
    return CountingMulti() if is_multi(kind) else CountingSingle()


def check_get_score_parallel(kind):
    """G1a: get_score(callable, xs, parallel=True) agrees over ranks and with serial."""
    policy = make_policy(kind)
    policy.set_seed(12345)
    policy.bayes_search(max_num_probes=0, is_disp=False, **search_kwargs(kind))
    X = grid_X()
    for npoints in [1, comm.size - 1, comm.size + 2]:
        if npoints < 1:
            continue
        xs = X[2 : 2 + npoints]
        fn = make_score(kind)
        res = policy.get_score(fn, xs=xs, parallel=True)
        res = np.asarray(res)
        if res.shape != (npoints,):
            raise AssertionError(
                f"get_score ({kind}, {npoints} points): shape {res.shape}"
            )
        assert_identical_over_ranks(res, f"get_score ({kind}, {npoints} points)")

        local = np.array_split(np.arange(npoints), comm.size)[comm.rank]
        if len(local) == 0:
            if len(fn.test_X) != 0:
                raise AssertionError(
                    f"get_score ({kind}): callable called on a rank with 0 points"
                )
        else:
            if len(fn.test_X) != 1 or not np.array_equal(fn.test_X[0], xs[local]):
                raise AssertionError(
                    f"get_score ({kind}): test is not this rank's share"
                )

        serial = np.asarray(policy.get_score(make_score(kind), xs=xs, parallel=False))
        if not np.allclose(res, serial, rtol=1e-12, atol=0):
            raise AssertionError(f"get_score ({kind}): parallel != serial")
    log(f"get_score parallel ({kind}): OK")


def check_discrete_bayes_search(kind, N):
    """G1b: discrete bayes_search with fewer unsearched candidates than ranks."""
    n_cand = grid_X().shape[0]
    n_rest = comm.size + 1
    initial_actions = np.arange(n_cand - n_rest)
    policy = make_policy(kind, initial_actions=initial_actions)
    policy.set_seed(12345)
    res = policy.bayes_search(
        max_num_probes=2,
        num_search_each_probe=N,
        simulator=make_simulator(kind),
        score=make_score(kind),
        is_disp=False,
        **search_kwargs(kind),
    )
    total = res.total_num_search
    n_new = total - len(initial_actions)
    if n_new < 1:
        raise AssertionError(f"bayes_search ({kind}, N={N}): nothing proposed")
    chosen = res.chosen_actions[:total]
    if len(set(chosen.tolist())) != total:
        raise AssertionError(f"bayes_search ({kind}, N={N}): duplicated actions")
    assert_identical_over_ranks(chosen, f"chosen_actions ({kind}, N={N})")
    assert_identical_over_ranks(res.fx[:total], f"fx ({kind}, N={N})")
    log(f"discrete bayes_search ({kind}, N={N}): OK")


def check_range_bayes_search(kind, N):
    """G1c: range bayes_search with the MPI-aware random optimizer."""
    policy = make_policy(kind)
    policy.set_seed(12345)
    n0 = policy.history.total_num_search
    res = policy.bayes_search(
        max_num_probes=2,
        num_search_each_probe=N,
        simulator=make_simulator(kind),
        score=make_score(kind),
        is_disp=False,
        **search_kwargs(kind),
    )
    total = res.total_num_search
    if total != n0 + 2 * N:
        raise AssertionError(f"bayes_search ({kind}, N={N}): {total - n0} proposals")
    X = res.action_X[n0:total]
    if np.any(X < MIN_X) or np.any(X > MAX_X):
        raise AssertionError(f"bayes_search ({kind}, N={N}): proposal out of range")
    assert_identical_over_ranks(res.action_X[:total], f"action_X ({kind}, N={N})")
    assert_identical_over_ranks(res.fx[:total], f"fx ({kind}, N={N})")
    log(f"range bayes_search ({kind}, N={N}): OK")


def main():
    log(f"running on {comm.size} MPI process(es)")
    for kind in ALL_KINDS:
        check_get_score_parallel(kind)
    for kind in DISCRETE_KINDS:
        for N in [1, 2]:
            check_discrete_bayes_search(kind, N)
    for kind in RANGE_KINDS:
        for N in [1, 2]:
            check_range_bayes_search(kind, N)
    log("all score-callable MPI checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        comm.Abort(1)

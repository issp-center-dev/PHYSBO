# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""MPI consistency checks for failed observations (non-finite objective values).

Run under MPI, e.g.:

    mpirun -np 2 python run_failed_observation_check.py

The script exits with a non-zero status if any check fails; it is
invoked by tests/mpi/test_mpi.py via subprocess.
"""

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
        if not np.array_equal(gathered[0], other, equal_nan=True):
            raise AssertionError(f"{label} differs between ranks")


def assert_on_all_ranks(flag, label):
    if not all(comm.allgather(bool(flag))):
        raise AssertionError(f"{label} does not hold on all the ranks")


def f(x):
    return -np.sum((x - 0.5) ** 2, axis=-1)


def check_discrete():
    """Failed observations are recorded identically on all the ranks."""
    a = np.linspace(0.0, 1.0, 11)
    X = np.array(list(product(a, a)))

    def sim(action):
        return np.where(X[action, 0] > 0.6, np.nan, f(X[action]))

    policy = physbo.search.discrete.Policy(test_X=X, comm=comm)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=10, simulator=sim, is_disp=False)
    res = policy.bayes_search(
        max_num_probes=5, simulator=sim, score="EI", is_disp=False, interval=0
    )
    N = res.total_num_search
    assert_on_all_ranks(N == 15, "discrete: the number of the observations")
    assert_on_all_ranks(not res.valid_mask.all(), "discrete: failed observations")
    assert_identical_over_ranks(res.chosen_actions[:N], "discrete chosen_actions")
    assert_identical_over_ranks(res.fx[:N], "discrete fx")
    assert_identical_over_ranks(policy.training.X, "discrete training.X")
    assert_on_all_ranks(
        policy.training.X.shape[0] == res.valid_mask.sum(), "discrete: training data"
    )
    log("discrete with failed observations: OK")


def check_range():
    """All the ranks raise the error together, and the policy stays usable."""
    policy = physbo.search.range.Policy(
        min_X=np.array([0.0, 0.0]), max_X=np.array([1.0, 1.0]), comm=comm
    )
    policy.set_seed(12345)
    sim = lambda x: f(np.atleast_2d(x))
    simnan = lambda x: np.full(np.atleast_2d(x).shape[0], np.nan)

    policy.random_search(max_num_probes=5, simulator=sim, is_disp=False)

    for search in ("random_search", "bayes_search"):
        raised = False
        try:
            if search == "random_search":
                policy.random_search(max_num_probes=2, simulator=simnan, is_disp=False)
            else:
                policy.bayes_search(
                    max_num_probes=2,
                    simulator=simnan,
                    score="EI",
                    is_disp=False,
                    interval=0,
                )
        except ValueError:
            raised = True
        assert_on_all_ranks(raised, f"range {search}: ValueError")
        assert_on_all_ranks(
            policy.history.total_num_search == 5, f"range {search}: history unchanged"
        )
        assert_on_all_ranks(
            policy.training.X.shape[0] == 5, f"range {search}: training unchanged"
        )

    res = policy.bayes_search(
        max_num_probes=2, simulator=sim, score="EI", is_disp=False, interval=0
    )
    N = res.total_num_search
    assert_on_all_ranks(N == 7, "range: the number of the observations")
    assert_identical_over_ranks(res.action_X[:N], "range action_X")
    assert_identical_over_ranks(res.fx[:N], "range fx")
    log("range with non-finite values: OK")


def main():
    log(f"running on {comm.size} MPI process(es)")
    check_discrete()
    check_range()
    log("all MPI checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        comm.Abort(1)

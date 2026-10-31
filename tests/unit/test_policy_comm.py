# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The MPI attributes set by the constructors of the six policies.

A communicator is imitated by an object with ``size`` and ``rank``, which
is all the constructors use of it.
"""

from types import SimpleNamespace

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")

KINDS = [
    "discrete",
    "discrete_multi",
    "discrete_unified",
    "range",
    "range_multi",
    "range_unified",
]


def make_policy(kind, comm):
    config = physbo.misc.SetConfig()
    config.learning.is_disp = True
    X = np.linspace(0.0, 1.0, 10)[:, None]
    if kind == "discrete":
        return physbo.search.discrete.Policy(test_X=X, config=config, comm=comm)
    if kind == "discrete_multi":
        return physbo.search.discrete_multi.Policy(
            test_X=X, num_objectives=2, config=config, comm=comm
        )
    if kind == "discrete_unified":
        return physbo.search.discrete_unified.Policy(
            test_X=X, num_objectives=2, config=config, comm=comm
        )
    kwargs = dict(min_X=np.zeros(1), max_X=np.ones(1), config=config, comm=comm)
    if kind == "range":
        return physbo.search.range.Policy(**kwargs)
    if kind == "range_multi":
        return physbo.search.range_multi.Policy(num_objectives=2, **kwargs)
    return physbo.search.range_unified.Policy(num_objectives=2, **kwargs)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rank", [0, 1])
def test_only_rank_0_displays(kind, rank):
    """is_disp is kept on rank 0 and cleared on the other ranks, for every policy."""
    policy = make_policy(kind, SimpleNamespace(size=2, rank=rank))
    assert policy.mpisize == 2 and policy.mpirank == rank
    assert policy.config.learning.is_disp == (rank == 0)


@pytest.mark.parametrize("kind", KINDS)
def test_serial_run_keeps_is_disp(kind):
    policy = make_policy(kind, None)
    assert (policy.mpisize, policy.mpirank, policy.mpicomm) == (1, 0, None)
    assert policy.config.learning.is_disp is True


@pytest.mark.parametrize("kind", ["discrete", "discrete_multi", "discrete_unified"])
def test_discrete_candidates_are_split_over_ranks(kind):
    policies = [make_policy(kind, SimpleNamespace(size=3, rank=r)) for r in range(3)]
    parts = [p.actions for p in policies]
    assert all(len(part) > 0 for part in parts)
    np.testing.assert_array_equal(np.concatenate(parts), np.arange(10))

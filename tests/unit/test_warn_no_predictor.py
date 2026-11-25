# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The warning printed when a policy without a predictor is asked for a
posterior (`_warn_no_predictor`).

It must be printed on rank 0 only, like every other message of the policies,
and it must name the method that was called.  A communicator is imitated by
an object with ``size`` and ``rank``, which is all the constructors use of it.
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

WARNING_HEAD = "Warning: Since policy.predictor is not yet set,"

X = np.linspace(0.0, 1.0, 10)[:, None]


def make_config():
    config = physbo.misc.SetConfig()
    config.learning.is_disp = False
    config.learning.num_init_params_search = 2
    config.learning.max_epoch = 50
    config.learning.max_epoch_init_params_search = 10
    return config


def make_policy(kind, comm=None):
    config = make_config()
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


def make_simulator(kind):
    def single(x):
        return -np.sum((np.atleast_2d(x) - 0.3) ** 2, axis=-1)

    def multi(x):
        x = np.atleast_2d(x)
        return np.c_[-np.sum((x - 0.3) ** 2, axis=-1), -np.sum((x - 0.7) ** 2, axis=-1)]

    if kind == "discrete":
        return lambda action: single(X[action])
    if kind in ("discrete_multi", "discrete_unified"):
        return lambda action: multi(X[action])
    if kind == "range":
        return single
    return multi


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rank", [0, 1])
def test_warning_is_printed_on_rank_0_only(kind, rank, capsys):
    policy = make_policy(kind, SimpleNamespace(size=2, rank=rank))
    policy._warn_no_predictor("get_post_fmean()")
    out = capsys.readouterr().out
    if rank == 0:
        assert WARNING_HEAD in out
        assert "before calling get_post_fmean()." in out
    else:
        assert out == ""


@pytest.mark.parametrize("kind", KINDS)
def test_warning_is_printed_in_a_serial_run(kind, capsys):
    policy = make_policy(kind)
    policy._warn_no_predictor("get_score()")
    out = capsys.readouterr().out
    assert WARNING_HEAD in out
    assert "before calling get_score()." in out


@pytest.mark.parametrize("kind", ["discrete", "discrete_multi", "range_multi"])
def test_permutation_importance_names_itself(kind, capsys):
    """The warning of get_permutation_importance names that method, not
    get_post_fmean as it did in these three policies.

    range.Policy has no get_permutation_importance; the unified policies
    name the method correctly already, and cannot be asked for it before
    bayes_search (#132)."""
    policy = make_policy(kind)
    policy.set_seed(1)
    policy.random_search(
        max_num_probes=4, simulator=make_simulator(kind), is_disp=False
    )
    capsys.readouterr()
    policy.get_permutation_importance(n_perm=2)
    out = capsys.readouterr().out
    assert WARNING_HEAD in out
    assert "before calling get_permutation_importance()." in out
    assert "get_post_fmean()" not in out

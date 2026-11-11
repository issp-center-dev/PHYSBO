# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The header of a multiple-probe step of bayes_search
("NNNN-th multiple probe search (score)") obeys is_disp, as the one of
random_search does, in the six policies."""

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

HEADER = "multiple probe search"

MIN_X = np.zeros(1)
MAX_X = np.ones(1)
X = np.linspace(0.0, 1.0, 10)[:, None]


def make_config():
    config = physbo.misc.SetConfig()
    config.search.multi_probe_num_sampling = 2
    config.learning.is_disp = False
    config.learning.num_init_params_search = 2
    config.learning.max_epoch = 50
    config.learning.max_epoch_init_params_search = 10
    return config


def make_policy(kind):
    config = make_config()
    if kind == "discrete":
        return physbo.search.discrete.Policy(test_X=X, config=config)
    if kind == "discrete_multi":
        return physbo.search.discrete_multi.Policy(
            test_X=X, num_objectives=2, config=config
        )
    if kind == "discrete_unified":
        return physbo.search.discrete_unified.Policy(
            test_X=X, num_objectives=2, config=config
        )
    kwargs = dict(min_X=MIN_X, max_X=MAX_X, config=config)
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


def search_kwargs(kind):
    kwargs = dict(max_num_probes=1, num_search_each_probe=2, interval=0)
    if kind.endswith("unified"):
        kwargs["unify_method"] = physbo.search.unify.ParEGO(num_objectives=2)
    if kind.startswith("range"):
        kwargs["optimizer"] = physbo.search.optimize.random.Optimizer(
            MIN_X, MAX_X, nsamples=20
        )
    return kwargs


def run(kind, is_disp, capsys):
    policy = make_policy(kind)
    policy.set_seed(1)
    sim = make_simulator(kind)
    policy.random_search(max_num_probes=3, simulator=sim, is_disp=False)
    capsys.readouterr()
    policy.bayes_search(simulator=sim, is_disp=is_disp, **search_kwargs(kind))
    return capsys.readouterr().out


@pytest.mark.parametrize("kind", KINDS)
def test_header_is_not_printed_with_is_disp_false(kind, capsys):
    out = run(kind, False, capsys)
    assert HEADER not in out


@pytest.mark.parametrize("kind", KINDS)
def test_header_is_printed_with_is_disp_true(kind, capsys):
    out = run(kind, True, capsys)
    assert out.count(HEADER) == 1

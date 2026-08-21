# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import sys

import numpy as np
import pytest
from packaging.version import Version

physbo = pytest.importorskip("physbo")
odatse = pytest.importorskip("odatse")

from physbo.search.optimize.odatse import default_alg_dict, Optimizer


def fn(x):
    # maximized at x = (0.5, 0.5)
    return -np.sum((np.array(x) - 0.5) ** 2, axis=-1)


@pytest.fixture
def min_X():
    return np.array([0.0, 0.0])


@pytest.fixture
def max_X():
    return np.array([1.0, 1.0])


@pytest.mark.parametrize(
    "algorithm_name", ["exchange", "pamc", "minsearch", "mapper", "bayes"]
)
def test_default_alg_dict(min_X, max_X, algorithm_name):
    d = default_alg_dict(min_X, max_X, algorithm_name)
    assert d["name"] == algorithm_name
    assert "param" in d
    assert "seed" in d


def test_default_alg_dict_unknown(min_X, max_X):
    with pytest.raises(ValueError):
        default_alg_dict(min_X, max_X, "unknown_algorithm")


def test_optimizer_minsearch(min_X, max_X, tmp_path, monkeypatch):
    # ODAT-SE writes its results under the current directory
    monkeypatch.chdir(tmp_path)
    optimizer = Optimizer(default_alg_dict(min_X, max_X, "minsearch"))
    X = optimizer(fn)
    assert X.shape == (1, 2)
    assert np.allclose(X[0], [0.5, 0.5], atol=0.05)


@pytest.mark.skipif(
    sys.platform == "win32" and Version(odatse.__version__) <= Version("3.2.1"),
    reason="ODAT-SE <= 3.2.1 removes ColorMap.txt.tmp while it is still open, "
    "which fails on Windows (fixed in ODAT-SE main, not yet released)",
)
def test_optimizer_mapper(min_X, max_X, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    optimizer = Optimizer(default_alg_dict(min_X, max_X, "mapper"))
    X = optimizer(fn)
    assert X.shape == (1, 2)
    # the default mapper grid has 11 points per dimension, including 0.5
    assert np.allclose(X[0], [0.5, 0.5], atol=0.05)


def test_bayes_search_with_odatse_optimizer(min_X, max_X, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def simulator(X):
        return fn(np.atleast_2d(X))

    policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X)
    policy.set_seed(12345)
    policy.random_search(max_num_probes=5, simulator=simulator)

    optimizer = Optimizer(default_alg_dict(min_X, max_X, "minsearch"))
    res = policy.bayes_search(
        max_num_probes=2, simulator=simulator, score="EI", optimizer=optimizer
    )
    best_fx, best_X = res.export_all_sequence_best_fx()
    assert res.total_num_search == 7
    assert best_fx[-1] >= best_fx[4]  # Bayes steps never lose the best value

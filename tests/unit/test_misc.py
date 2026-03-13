# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import sys

import pytest
import numpy as np

physbo = pytest.importorskip("physbo")
set_config_module = pytest.importorskip("physbo.misc._set_config")


@pytest.fixture
def X():
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )


# @pytest.mark.parametrize("index", [2, [0, 1]])
def test_centering(X):
    dim = X.shape[1]
    centered = physbo.misc.centering(X)
    assert np.array_equal(centered.mean(axis=0), np.zeros(dim))
    assert np.array_equal(centered.std(axis=0), np.ones(dim))


def test_gauss_elim():
    np.random.seed(12345)
    N = 5
    U = np.random.randn(N, N)
    for i in range(N):
        for j in range(i):
            U[i, j] = 0.0
    ref = np.random.randn(N)
    t = np.dot(U, ref)
    t = np.dot(U.transpose(), t)
    res = physbo.misc.gauss_elim(U, t)
    np.testing.assert_array_almost_equal(res, ref)


@pytest.mark.parametrize("NM", [(3, 5), (5, 3), (5, 5)])
def test_diagAB(NM):
    np.random.seed(12345)
    N = NM[0]
    M = NM[1]
    A = np.random.randn(N, M)
    B = np.random.randn(M, N)
    ref = np.dot(A, B).diagonal()
    res = physbo.misc.diagAB(A, B)
    np.testing.assert_array_almost_equal(res, ref)


# Now work only for diagnal matrices
@pytest.mark.parametrize("NM", [(3, 5), (5, 3), (5, 5)])
def test_traceAB2(NM):
    np.random.seed(12345)
    N = NM[0]
    M = NM[1]
    A = np.random.randn(N, M)
    B = np.random.randn(M, N)
    ref = np.dot(A, B).trace()
    res = physbo.misc.traceAB2(A, B)
    assert res == pytest.approx(ref)


# Now work only for diagnal matrices
@pytest.mark.parametrize("NM", [(3, 5), (5, 3), (5, 5)])
def test_traceAB3(NM):
    np.random.seed(12345)
    d = 2
    N = NM[0]
    M = NM[1]
    A = np.random.randn(N, M)
    B = np.random.randn(d, M, N)
    ref = [np.dot(A, B[i, :, :]).trace() for i in range(d)]
    res = physbo.misc.traceAB3(A, B)
    np.testing.assert_array_almost_equal(res, ref)


def test_logsumexp():
    np.random.seed(12345)
    N = 10
    xs = np.random.randn(N)
    ref = np.log(sum(np.exp(xs)))
    res = physbo.misc.logsumexp(xs)
    assert res == pytest.approx(ref)


def _write_config(tmp_path, content):
    config_path = tmp_path / "config.ini"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_setconfig_load_adam(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
[search]
multi_probe_num_sampling = 11
alpha = 0.5

[learning]
method = adam
is_disp = yes
num_disp = 7
num_init_params_search = 3

[online]
max_epoch = 123
max_epoch_init_params_search = 9
batch_size = 32
eval_size = 456

[adam]
alpha = 0.01
beta = 0.8
gamma = 0.95
epsilon = 1e-7
""",
    )

    config = physbo.misc.SetConfig()
    config.load(str(config_path))

    assert config.search.multi_probe_num_sampling == 11
    assert config.search.alpha == pytest.approx(0.5)
    assert config.learning.method == "adam"
    assert config.learning.is_disp is True
    assert config.learning.num_disp == 7
    assert config.learning.num_init_params_search == 3
    assert config.learning.max_epoch == 123
    assert config.learning.batch_size == 32
    assert config.learning.alpha == pytest.approx(0.01)
    assert config.learning.gamma == pytest.approx(0.95)


def test_setconfig_load_batch(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
[search]
multi_probe_num_sampling = 20
alpha = 1.0

[learning]
method = batch
is_disp = false
num_disp = 10
num_init_params_search = 20

[batch]
max_iter = 111
max_iter_init_params_search = 22
batch_size = 333
""",
    )

    config = physbo.misc.SetConfig()
    config.load(str(config_path))

    assert config.learning.method == "batch"
    assert config.learning.is_disp is False
    assert config.learning.max_iter == 111
    assert config.learning.max_iter_init_params_search == 22
    assert config.learning.batch_size == 333


def test_setconfig_load_missing_file():
    config = physbo.misc.SetConfig()
    with pytest.raises(FileNotFoundError):
        config.load("definitely_missing_file.ini")


def test_setconfig_load_missing_required_section(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
[search]
multi_probe_num_sampling = 20
alpha = 1.0
""",
    )

    config = physbo.misc.SetConfig()
    with pytest.raises(ValueError, match=r"\[learning\]"):
        config.load(str(config_path))


def test_setconfig_load_unknown_method(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
[search]
multi_probe_num_sampling = 20
alpha = 1.0

[learning]
method = sgd
""",
    )

    config = physbo.misc.SetConfig()
    with pytest.raises(ValueError, match="Unknown learning method"):
        config.load(str(config_path))


@pytest.mark.parametrize(
    "value, expected",
    [
        ("True", True),
        ("true", True),
        ("YES", True),
        ("1", True),
        ("on", True),
        ("False", False),
        ("false", False),
        ("No", False),
        ("0", False),
        ("off", False),
        (True, True),
        (False, False),
        (1, True),
        (0, False),
    ],
)
def test_boolean_parser(value, expected):
    assert set_config_module.boolean(value) is expected


@pytest.mark.parametrize("value", ["", "maybe", 2, -1, None])
def test_boolean_parser_invalid(value):
    with pytest.raises(ValueError, match="Cannot parse boolean value"):
        set_config_module.boolean(value)

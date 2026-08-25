# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for loading the search/learning configuration from config.ini."""

import os

import pytest

physbo = pytest.importorskip("physbo")

from physbo.misc import _set_config


def test_defaults():
    config = physbo.misc.SetConfig()
    assert config.search.multi_probe_num_sampling == 20
    assert config.search.alpha == 1.0
    # the default learning method is adam
    assert isinstance(config.learning, _set_config.Adam)
    assert config.learning.method == "adam"
    assert config.learning.max_epoch == 500


def test_load_adam(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "\n".join(
            [
                "[search]",
                "multi_probe_num_sampling = 30",
                "alpha = 2.0",
                "[learning]",
                "method = adam",
                "is_disp = False",
                "num_disp = 5",
                "num_init_params_search = 7",
                "[online]",
                "max_epoch = 123",
                "max_epoch_init_params_search = 11",
                "batch_size = 32",
                "eval_size = 100",
                "[adam]",
                "alpha = 0.05",
                "beta = 0.8",
                "gamma = 0.99",
                "epsilon = 1e-4",
            ]
        )
    )

    config = physbo.misc.SetConfig()
    config.load(str(ini))

    assert config.search.multi_probe_num_sampling == 30
    assert config.search.alpha == 2.0

    assert isinstance(config.learning, _set_config.Adam)
    assert config.learning.method == "adam"
    assert config.learning.is_disp is False
    assert config.learning.num_disp == 5
    assert config.learning.num_init_params_search == 7
    assert config.learning.max_epoch == 123
    assert config.learning.max_epoch_init_params_search == 11
    assert config.learning.batch_size == 32
    assert config.learning.eval_size == 100
    assert config.learning.alpha == pytest.approx(0.05)
    assert config.learning.beta == pytest.approx(0.8)
    assert config.learning.gamma == pytest.approx(0.99)
    assert config.learning.epsilon == pytest.approx(1e-4)


def test_load_batch(tmp_path):
    ini = tmp_path / "config.ini"
    ini.write_text(
        "\n".join(
            [
                "[search]",
                "[learning]",
                "method = bfgs",
                "is_disp = True",
                "[batch]",
                "max_iter = 42",
                "max_iter_init_params_search = 5",
                "batch_size = 100",
            ]
        )
    )

    config = physbo.misc.SetConfig()
    config.load(str(ini))

    assert isinstance(config.learning, _set_config.Batch)
    assert config.learning.method == "bfgs"
    assert config.learning.is_disp is True
    assert config.learning.max_iter == 42
    assert config.learning.max_iter_init_params_search == 5
    assert config.learning.batch_size == 100


def test_show(capsys):
    config = physbo.misc.SetConfig()
    config.show()
    out = capsys.readouterr().out
    assert "multi_probe_num_sampling" in out
    assert "method :  adam" in out


def test_boolean_helper():
    assert _set_config.boolean(True) is True
    assert _set_config.boolean("True") is True
    assert _set_config.boolean(False) is False
    assert _set_config.boolean("False") is False
    # common truthy/falsy spellings are accepted
    assert _set_config.boolean("yes") is True
    assert _set_config.boolean("on") is True
    assert _set_config.boolean("1") is True
    assert _set_config.boolean("no") is False
    assert _set_config.boolean("off") is False
    assert _set_config.boolean("0") is False
    # unknown strings are rejected
    with pytest.raises(ValueError):
        _set_config.boolean("maybe")

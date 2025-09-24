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


@pytest.fixture
def X():
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )


@pytest.fixture
def t():
    return np.array([0.0, 1.0, 2.0, 3.0])


@pytest.fixture
def Z():
    return np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])


@pytest.fixture
def variable(X, t, Z):
    return physbo.Variable(X=X, t=t, Z=Z)


@pytest.mark.parametrize("index", [2, [0, 1]])
def test_get_subset(variable, X, t, Z, index):
    var2 = variable.get_subset(index)
    assert np.array_equal(var2.X, X[index, :])
    assert np.array_equal(var2.t, t[index])
    assert np.array_equal(var2.Z, Z[index, :])


@pytest.mark.parametrize("index", [2, [0, 1]])
def test_delete_X(variable, X, index):
    variable.delete_X(index)
    ref = np.delete(X, index, 0)
    assert np.array_equal(variable.X, ref)


@pytest.mark.parametrize("index", [2, [0, 1]])
def test_delete_t(variable, t, index):
    variable.delete_t(index)
    ref = np.delete(t, index)
    assert np.array_equal(variable.t, ref)


@pytest.mark.parametrize("index", [2, [0, 1]])
def test_delete_Z(variable, Z, index):
    variable.delete_Z(index)
    ref = np.delete(Z, index, 0)
    assert np.array_equal(variable.Z, ref)


def test_add(variable, X, t, Z, mocker):
    variable.add(X, t, Z)
    assert np.array_equal(variable.X, np.vstack((X, X)))
    assert np.array_equal(variable.t, np.hstack((t, t)))
    assert np.array_equal(variable.Z, np.vstack((Z, Z)))


def test_add_X(variable, X):
    n = X.shape[0]
    variable.add_X(X)
    assert np.array_equal(variable.X[n : 2 * n, :], X)


def test_add_t(variable, t):
    n = t.shape[0]
    variable.add_t(t)
    assert np.array_equal(variable.t[n : 2 * n], t)


def test_add_Z(variable, Z):
    n = Z.shape[0]
    variable.add_Z(Z)
    assert np.array_equal(variable.Z[n : 2 * n, :], Z)


def test_save_load(variable, tmpdir):
    tmpfile = tmpdir.join("tmpfile.npz")
    filename = str(tmpfile)
    variable.save(filename)
    var2 = physbo.Variable()
    var2.load(filename)
    assert np.array_equal(variable.X, var2.X)
    assert np.array_equal(variable.t, var2.t)
    assert np.array_equal(variable.Z, var2.Z)

    variable.Z = None
    variable.save(filename)
    var2 = physbo.Variable()
    var2.load(filename)
    assert np.array_equal(variable.X, var2.X)
    assert np.array_equal(variable.t, var2.t)
    assert np.array_equal(variable.Z, var2.Z)
    tmpfile.remove()

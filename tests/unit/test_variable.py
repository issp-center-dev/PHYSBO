# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
    return np.array([[0.0], [1.0], [2.0], [3.0]])  # 2D array with shape (4, 1)


@pytest.fixture
def Z():
    return np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])


@pytest.fixture
def variable(X, t, Z):
    return physbo.Variable(X=X, t=t, Z=Z)


@pytest.fixture
def t_2d():
    return np.array([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])


@pytest.fixture
def variable_2d(X, t_2d, Z):
    return physbo.Variable(X=X, t=t_2d, Z=Z)


@pytest.mark.parametrize("index", [2, [0, 1]])
def test_get_subset(variable, X, t, Z, index):
    var2 = variable.get_subset(index)
    if isinstance(index, int):
        index = [index]
    assert np.array_equal(var2.X, X[index, :])
    assert np.array_equal(var2.t, t[index, :])  # t is now 2D
    assert np.array_equal(var2.Z, Z[index, :])


@pytest.mark.parametrize("index", [2, [0, 1]])
def test_delete_X(variable, X, index):
    variable.delete_X(index)
    ref = np.delete(X, index, 0)
    assert np.array_equal(variable.X, ref)


@pytest.mark.parametrize("index", [2, [0, 1]])
def test_delete_t(variable, t, index):
    variable.delete_t(index)
    ref = np.delete(t, index, axis=0)  # Use axis=0 for 2D array
    assert np.array_equal(variable.t, ref)


@pytest.mark.parametrize("index", [2, [0, 1]])
def test_delete_Z(variable, Z, index):
    variable.delete_Z(index)
    ref = np.delete(Z, index, 0)
    assert np.array_equal(variable.Z, ref)


def test_add(variable, X, t, Z, mocker):
    variable.add(X, t, Z)
    assert np.array_equal(variable.X, np.vstack((X, X)))
    assert np.array_equal(variable.t, np.vstack((t, t)))  # Use vstack for 2D array
    assert np.array_equal(variable.Z, np.vstack((Z, Z)))


def test_add_X(variable, X):
    n = X.shape[0]
    variable.add_X(X)
    assert np.array_equal(variable.X[n : 2 * n, :], X)


def test_add_t(variable, t):
    n = t.shape[0]
    variable.add_t(t)
    assert np.array_equal(variable.t[n : 2 * n, :], t)  # t is now 2D


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


# Tests for 2D t array (multi-objective)
@pytest.mark.parametrize("index", [2, [0, 1]])
def test_get_subset_2d(variable_2d, X, t_2d, Z, index):
    var2 = variable_2d.get_subset(index)
    if isinstance(index, int):
        index = [index]
    assert np.array_equal(var2.X, X[index, :])
    # t_2d has shape (4, 2), so subset should also be 2D
    assert np.array_equal(var2.t, t_2d[index, :])
    assert np.array_equal(var2.Z, Z[index, :])
    # Verify that result is 2D
    assert var2.t.ndim == 2, f"t should be 2D, got ndim={var2.t.ndim}"


@pytest.mark.parametrize("index", [2, [0, 1]])
def test_delete_t_2d(variable_2d, t_2d, index):
    variable_2d.delete_t(index)
    ref = np.delete(t_2d, index, axis=0)
    assert np.array_equal(variable_2d.t, ref)


def test_add_t_2d(variable_2d, t_2d):
    n = t_2d.shape[0]
    variable_2d.add_t(t_2d)
    assert np.array_equal(variable_2d.t[n : 2 * n, :], t_2d)




def test_add_2d(variable_2d, X, t_2d, Z):
    variable_2d.add(X, t_2d, Z)
    assert np.array_equal(variable_2d.X, np.vstack((X, X)))
    assert np.array_equal(variable_2d.t, np.vstack((t_2d, t_2d)))
    assert np.array_equal(variable_2d.Z, np.vstack((Z, Z)))


def test_save_load_2d(variable_2d, tmpdir):
    tmpfile = tmpdir.join("tmpfile_2d.npz")
    filename = str(tmpfile)
    variable_2d.save(filename)
    var2 = physbo.Variable()
    var2.load(filename)
    assert np.array_equal(variable_2d.X, var2.X)
    assert np.array_equal(variable_2d.t, var2.t)
    assert np.array_equal(variable_2d.Z, var2.Z)
    tmpfile.remove()


# Tests for requiring 2D arrays
def test_X_must_be_2d():
    """Test that X must be a 2D array"""
    X_1d = np.array([1.0, 2.0, 3.0])
    X_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    t = np.array([[1.0], [2.0], [3.0]])

    # X as 1D should raise AssertionError
    with pytest.raises(AssertionError, match="X must be a 2D array"):
        physbo.Variable(X=X_1d, t=t)

    # X as 3D should raise AssertionError
    with pytest.raises(AssertionError, match="X must be a 2D array"):
        physbo.Variable(X=X_3d, t=np.array([[1.0]]))


def test_t_must_be_2d():
    """Test that t must be a 2D array (or is converted to 2D)"""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    t_1d = np.array([1.0, 2.0, 3.0])
    t_3d = np.array([[[1.0], [2.0], [3.0]]])

    # t as 1D - should either raise error or be converted to 2D
    # This test verifies that t is treated as 2D (either by error or conversion)
    try:
        var = physbo.Variable(X=X, t=t_1d)
        # If creation succeeds, t should be 2D (either converted or already 2D)
        assert var.t.ndim == 2, f"t should be 2D array, got ndim={var.t.ndim}"
        assert var.t.shape == (3, 1), f"t should have shape (3, 1), got {var.t.shape}"
    except AssertionError as e:
        # If creation fails, verify it's because t must be 2D
        assert "2D array" in str(e) or "must be" in str(e), f"Unexpected error: {e}"

    # t as 3D should raise AssertionError
    with pytest.raises(AssertionError):
        physbo.Variable(X=X, t=t_3d)


def test_Z_must_be_2d():
    """Test that Z must be a 2D array"""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    t = np.array([[1.0], [2.0], [3.0]])
    Z_1d = np.array([1.0, 2.0, 3.0])
    Z_3d = np.array([[[1.0], [2.0], [3.0]]])

    # Z as 1D should raise AssertionError
    with pytest.raises(AssertionError, match="Z must be a 2D array"):
        physbo.Variable(X=X, t=t, Z=Z_1d)

    # Z as 3D should raise AssertionError
    with pytest.raises(AssertionError, match="Z must be a 2D array"):
        physbo.Variable(X=X, t=t, Z=Z_3d)


def test_all_2d_arrays_valid():
    """Test that valid 2D arrays work correctly"""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    t = np.array([[1.0], [2.0], [3.0]])
    Z = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])

    var = physbo.Variable(X=X, t=t, Z=Z)
    assert var.X.ndim == 2, "X should be 2D array"
    assert var.t.ndim == 2, "t should be 2D array"
    assert var.Z.ndim == 2, "Z should be 2D array"
    assert var.X.shape == (3, 2), f"X should have shape (3, 2), got {var.X.shape}"
    assert var.t.shape == (3, 1), f"t should have shape (3, 1), got {var.t.shape}"
    assert var.Z.shape == (3, 2), f"Z should have shape (3, 2), got {var.Z.shape}"

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
    return np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    )

@pytest.fixture
def variable(X, t, Z):
    return physbo.variable(X=X, t=t, Z=Z)


@pytest.mark.parametrize("index", [2, [0, 1]])
def test_get_subset(variable, X, t, Z, index):
    var2 = variable.get_subset(index)
    assert np.array_equal(var2.X, X[index, :])
    assert np.array_equal(var2.t, t[index])
    assert np.array_equal(var2.Z, Z[index, :])


def test_delete(variable, mocker):
    delete_X = mocker.patch("physbo.variable.delete_X")
    delete_t = mocker.patch("physbo.variable.delete_t")
    delete_Z = mocker.patch("physbo.variable.delete_Z")
    variable.delete(1)
    delete_X.assert_called_once_with(1)
    delete_t.assert_called_once_with(1)
    delete_Z.assert_called_once_with(1)


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
    add_X = mocker.patch("physbo.variable.add_X")
    add_t = mocker.patch("physbo.variable.add_t")
    add_Z = mocker.patch("physbo.variable.add_Z")
    variable.add(X, t, Z)
    add_X.assert_called_once_with(X)
    add_t.assert_called_once_with(t)
    add_Z.assert_called_once_with(Z)


def test_add_X(variable, X):
    n = X.shape[0]
    variable.add_X(X)
    assert np.array_equal(variable.X[n:2*n, :], X)


def test_add_t(variable, t):
    n = t.shape[0]
    variable.add_t(t)
    assert np.array_equal(variable.t[n:2*n], t)


def test_add_Z(variable, Z):
    n = Z.shape[0]
    variable.add_Z(Z)
    assert np.array_equal(variable.Z[n:2*n, :], Z)


def test_save_load(variable, tmpdir):
    tmpfile = tmpdir.join("tmpfile.npz")
    filename = str(tmpfile)
    variable.save(filename)
    var2 = physbo.variable()
    var2.load(filename)
    assert np.array_equal(variable.X, var2.X)
    assert np.array_equal(variable.t, var2.t)
    assert np.array_equal(variable.Z, var2.Z)

    variable.Z = None
    variable.save(filename)
    var2 = physbo.variable()
    var2.load(filename)
    assert np.array_equal(variable.X, var2.X)
    assert np.array_equal(variable.t, var2.t)
    assert np.array_equal(variable.Z, var2.Z)
    tmpfile.remove()

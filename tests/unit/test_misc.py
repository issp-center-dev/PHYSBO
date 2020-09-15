import sys

import pytest
import numpy as np

physbo = pytest.importorskip("physbo")


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
@pytest.mark.xfail
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
@pytest.mark.xfail
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


def test_logsumexp64():
    np.random.seed(12345)
    N = 10
    xs = np.random.randn(N)
    ref = np.log(sum(np.exp(xs)))
    res = physbo.misc.logsumexp64(xs)
    assert res == pytest.approx(ref)

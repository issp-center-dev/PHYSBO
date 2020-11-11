from __future__ import print_function

import pytest
import numpy as np
import numpy.random as random

physbo = pytest.importorskip("physbo")


def get_post_fmean(training, test):
    print(test)
    N = test.shape[0]
    return random.randn(N, 1)


def get_post_fcov(training, test):
    print(test)
    N = test.shape[0]
    return np.ones((N, 1))


def get_post_samples(training, test, alpha=1.0):
    print(test)
    N = test.shape[0]
    return random.randn(N, 1)


@pytest.fixture
def predictor(mocker):
    p = mocker.MagicMock()
    p.get_post_fmean = mocker.MagicMock(side_effect=get_post_fmean)
    p.get_post_fcov = mocker.MagicMock(side_effect=get_post_fcov)
    p.get_post_samples = mocker.MagicMock(side_effect=get_post_samples)
    return p


@pytest.fixture
def X():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
        ]
    )


@pytest.fixture
def Y():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
    )


@pytest.fixture
def t():
    return np.array([0.0, 1.0, 2.0, 3.0])


def test_gp(predictor, X, t, mocker):
    X = physbo.misc.centering(X)
    np.random.seed(12345)
    N = len(t)
    Ntrain = N - 1
    Ntest = N - Ntrain
    id_all = np.random.choice(len(t), len(t), replace=False)
    id_train = id_all[0:Ntrain]
    id_test = id_all[Ntrain:N]

    X_train = X[id_train]
    X_test = X[id_test]

    t_train = t[id_train]
    t_test = t[id_test]

    cov = physbo.gp.cov.gauss(X_train.shape[1], ard=False)
    mean = physbo.gp.mean.const()
    lik = physbo.gp.lik.gauss()

    adam_run_spy = mocker.spy(physbo.gp.core.learning.adam, "run")
    model_set_params_spy = mocker.spy(physbo.gp.core.model, "set_params")

    gp = physbo.gp.model(lik=lik, mean=mean, cov=cov)
    config = physbo.misc.set_config()
    gp.fit(X_train, t_train, config)

    adam_run_spy.assert_called()
    model_set_params_spy.assert_called()

    inf_prepare_spy = mocker.spy(physbo.gp.inf.exact, "prepare")
    gp.prepare(X_train, t_train)
    fmean = gp.get_post_fmean(X_train, X_test)
    inf_prepare_spy.assert_called()
    _ = gp.get_post_fcov(X_train, X_test)

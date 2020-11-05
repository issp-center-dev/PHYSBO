from __future__ import print_function

from itertools import product

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")


def f(x):
    return -np.sum((x - 0.5) ** 2)


class simulator:
    def __init__(self, noise=1.0e-5):
        np.random.seed(12345)
        self.nslice = 11
        self.dim = 2
        self.N = self.nslice ** self.dim
        self.X = np.zeros((self.N, self.dim))
        self.t = np.zeros(self.N)
        for i, x in enumerate(
            product(np.linspace(0.0, 1.0, self.nslice), repeat=self.dim)
        ):
            lx = list(x)
            self.X[i, :] = lx
            self.t[i] = f(self.X[i, :]) + noise * np.random.randn()


def test_gp():
    sim = simulator()
    X = sim.X
    t = sim.t
    X = physbo.misc.centering(X)

    np.random.seed(12345)
    N = len(t)
    Ntrain = int(N * 0.8)
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

    gp = physbo.gp.model(lik=lik, mean=mean, cov=cov)
    config = physbo.misc.set_config()

    gp.fit(X_train, t_train, config)
    gp.print_params()

    gp.prepare(X_train, t_train)
    fmean = gp.get_post_fmean(X_train, X_test)
    _ = gp.get_post_fcov(X_train, X_test)

    res = np.mean((fmean - t_test) ** 2)
    ref = 1.0891454518475606e-05
    assert res == pytest.approx(ref, rel=1e-3)

    # restart
    cov = physbo.gp.cov.gauss(X_train.shape[1], ard=False)
    mean = physbo.gp.mean.const()
    lik = physbo.gp.lik.gauss()
    gp2 = physbo.gp.model(lik=lik, mean=mean, cov=cov)
    gp_params = np.append(
        np.append(gp.lik.params, gp.prior.mean.params), gp.prior.cov.params
    )
    gp2.set_params(gp_params)
    gp2.prepare(X_train, t_train)
    fmean = gp2.get_post_fmean(X_train, X_test)
    _ = gp2.get_post_fcov(X_train, X_test)

    res2 = np.mean((fmean - t_test) ** 2)
    assert res == res2

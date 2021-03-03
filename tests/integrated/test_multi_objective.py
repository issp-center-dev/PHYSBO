from __future__ import print_function

from itertools import product

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")


def vlmop2_minus(x):
    n = x.shape[1]
    y1 = 1 - np.exp(-1 * np.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
    y2 = 1 - np.exp(-1 * np.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))

    return np.c_[-y1, -y2]


class simulator:
    def __init__(self):
        a = np.linspace(-2, 2, 101)
        self.X = np.array(list(product(a, a)))
        self.t = vlmop2_minus(self.X)

    def __call__(self, action):
        return self.t[action]


def test_bayes_search():
    sim = simulator()
    nrand = 10
    nsearch = 5
    policy = physbo.search.discrete.policy_mo(test_X=sim.X, num_objectives=2)
    policy.set_seed(12345)
    res = policy.random_search(max_num_probes=nrand, simulator=sim)
    res = policy.bayes_search(max_num_probes=nsearch, simulator=sim, score="EHVI")
    vid = res.pareto.volume_in_dominance([-1, -1], [0, 0])
    vid_ref = 0.20374417006226475
    assert vid == pytest.approx(vid_ref, rel=1e-3)

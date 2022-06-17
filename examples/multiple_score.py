import itertools

import numpy as np
import physbo

# Make a set of candidates, test_X
D = 2  # The number of params (the dimension of parameter space)
Nx = 11  # The number of candidates
N = Nx*Nx

# score = "HVPI"
score = "EHVI"

a = np.linspace(-2, 2, Nx)
test_X = np.array(list(itertools.product(a, a)))


def vlmop2_minus(x):
    n = x.shape[1]
    y1 = 1 - np.exp(-1 * np.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
    y2 = 1 - np.exp(-1 * np.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))

    return np.c_[-y1, -y2]


class simulator(object):
    def __init__(self, X):
        self.t = vlmop2_minus(X)

    def __call__(self, action):
        return self.t[action]


sim = simulator(test_X)

policy = physbo.search.discrete_multi.policy(test_X, num_objectives=2)
policy.set_seed(0)
# Random search (10 times)
policy.random_search(max_num_probes=10, simulator=sim)

# Bayesian search (40 times)
#   score function (acquisition function): expectation of improvement (EI)
policy.bayes_search(max_num_probes=40, simulator=sim, score=score, interval=0)

print("Mean values of prediction")
scores = policy.get_post_fmean(xs=test_X)
print(scores)
print()

print("Standard deviations of prediction")
scores = policy.get_post_fcov(xs=test_X)
print(np.sqrt(scores))
print()

print("Acquisition function")
scores = policy.get_score(mode=score, xs=test_X)
print(scores)

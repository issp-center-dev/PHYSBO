# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import itertools

import numpy as np
import physbo

# Make a set of candidates, test_X
D = 2  # The number of params (the dimension of parameter space)
Nx = 11  # The number of candidates
N = Nx * Nx

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

policy = physbo.search.discrete_multi.Policy(test_X, num_objectives=2)
policy.set_seed(0)
# Random search (10 times)
policy.random_search(max_num_probes=10, simulator=sim)

# Bayesian search (40 times)
#   score function (acquition function): expectation of improvement (EI)
policy.bayes_search(max_num_probes=40, simulator=sim, score=score, interval=0)

print("Pareto fronts:")
res = policy.history
front, front_index = res.export_pareto_front()
with open("pareto_front_discrete.txt", "w") as f:
    for fr, ifr in zip(front, front_index):
        action = res.chosen_actions[ifr]
        X = test_X[action, :]
        print("  action: ", action)
        print("  X: ", X)
        print("  f: ", fr)
        print()
        for y in fr:
            f.write(f"{y} ")
        for x in X:
            f.write(f"{x} ")
        f.write("\n")

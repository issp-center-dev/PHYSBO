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
D = 3  # The number of params (the dimension of parameter space)
Nx = 21  # The number of candidates
N = Nx ** D

# score = "HVPI"
score = "EHVI"

a = np.linspace(-2, 2, Nx)
test_X = np.array(list(itertools.product(a, repeat=D)))


def vlmop2_minus(x):
    n = x.shape[1]
    y1 = 1 - np.exp(-1 * np.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
    y2 = 1 - np.exp(-1 * np.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))

    return np.c_[-y1, -y2]


def simulator(actions: np.ndarray) -> np.ndarray:
    X = test_X[actions, :2]
    return vlmop2_minus(X)


policy = physbo.search.discrete_multi.policy(test_X, num_objectives=2)
policy.set_seed(0)
# Random search (10 times)
policy.random_search(max_num_probes=30, simulator=simulator)

# Bayesian search (40 times)
#   score function (acquition function): expectation of improvement (EI)
policy.bayes_search(max_num_probes=20, simulator=simulator, score=score, interval=0)

print(policy.get_permutation_importance(n_perm=20))

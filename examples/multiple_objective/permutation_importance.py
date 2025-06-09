# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import sys
import itertools

import numpy as np
import matplotlib.pyplot as plt
import physbo

num_rand_basis = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print("num_rand_basis = ", num_rand_basis)

# Make a set of candidates, test_X
D = 3  # The number of params (the dimension of parameter space)
Nx = 11  # The number of candidates
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
policy.bayes_search(max_num_probes=20, simulator=simulator, score=score, interval=0, num_rand_basis=num_rand_basis)

importance_mean, importance_std = policy.get_permutation_importance(n_perm=20)


num_objectives = importance_mean.shape[1]

fig, ax = plt.subplots(num_objectives, 1, figsize=(8, 5))
for i in range(num_objectives):
    ax[i].barh(
        range(D),
        importance_mean[:, i],
        xerr=importance_std[:, i],
    )
    ax[i].invert_yaxis()
    ax[i].set_ylabel("Parameters")
    ax[i].text(0.8, 0.1, f"{i}-th objective", transform=ax[i].transAxes)
ax[-1].set_xlabel("Permutation Importance")

# to share the same x-axis
xmin = np.inf
xmax = -np.inf
for i in range(2):
    xmin_ax, xmax_ax = ax[i].get_xlim()
    xmin = min(xmin, xmin_ax)
    xmax = max(xmax, xmax_ax)
for i in range(num_objectives):
    ax[i].set_xlim(xmin, xmax)

print("save permutation_importance.pdf")
fig.savefig("permutation_importance.pdf")

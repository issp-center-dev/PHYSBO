# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import physbo

np.random.seed(137)

# Make a set of candidates, test_X
D = 6  # The number of params (the dimension of parameter space)
N = 1000  # The number of candidates
test_X = np.random.randn(N, D)  # Generated from Gaussian
test_X[0, :] = 0.0  # true solution
weights = np.linspace(1.0, 0.0, D)
weights = weights**2
weights = weights.reshape(1, D)

def simulator(actions: np.ndarray) -> np.ndarray:
    """Objective function

    Quadratic function, -Σ_i x_i^2
    Receives an array of actions (indices of candidates) and returns the corresponding results as an array
    """

    return -np.sum(weights * test_X[actions, :] ** 2, axis=1)


policy = physbo.search.discrete.policy(test_X)
policy.set_seed(12345)

# Random search (10 times)
policy.random_search(max_num_probes=30, simulator=simulator)

# Bayesian search (40 times)
#   score function (acquition function): expectation of improvement (EI)
policy.bayes_search(max_num_probes=10, simulator=simulator, score="EI")

print("weights = ", weights)
print("permutation importance = ", policy.get_permutation_importance(n_perm=10))

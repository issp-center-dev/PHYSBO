# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import numpy as np
import matplotlib.pyplot as plt
import physbo


D = 3  # The number of params (the dimension of parameter space)
N = 10000  # The number of candidates
nrss = [0, 10, 100, 300]  # The number of random features

# Make a set of candidates, test_X
test_X = np.random.randn(N, D)  # Generated from Gaussian
test_X[0, :] = 0.0  # true solution


def simulator(actions: np.ndarray) -> np.ndarray:
    """Objective function

    Quadratic function, -Σ_i x_i^2
    Receives an array of actions (indices of candidates) and returns the corresponding results as an array
    """
    return -np.sum(test_X[actions, :] ** 2, axis=1)


def run(num_rand_basis):
    policy = physbo.search.discrete.policy(test_X)
    policy.set_seed(12345)

    # Random search (10 times)
    policy.random_search(max_num_probes=10, simulator=simulator)

    # Bayesian search (40 times)
    #   score function (acquition function): expectation of improvement (EI)
    policy.bayes_search(
        max_num_probes=190,
        simulator=simulator,
        score="EI",
        num_rand_basis=num_rand_basis,
    )

    # Print the best result
    # best_actions[i] and best_fx[i] stores the best action and value up to the i-th search (random + bayes)
    best_fx, best_actions = policy.history.export_sequence_best_fx()

    result = {}
    result["best_fx"] = best_fx
    result["time_get_action"] = policy.history.time_get_action

    return result


results = []

for nrs in nrss:
    local_result = run(nrs)
    results.append(local_result)

for name in results[0].keys():
    fig, ax = plt.subplots()
    for nrs, result in zip(nrss, results):
        ax.plot(result[name], label=f"nrs={nrs}")
    ax.legend()
    fig.savefig(f"{name}.pdf")
    ax.clear()

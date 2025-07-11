# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import physbo
from physbo.search.optimize.random import Optimizer as RandomOptimizer

# Make a set of candidates, test_X
D = 2  # The number of params (the dimension of parameter space)

min_X = [-2, -2]
max_X = [2, 2]

# score = "HVPI"
score = "EHVI"
# score = "TS"

def vlmop2_minus(x):
    n = x.shape[1]
    y1 = 1 - np.exp(-1 * np.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
    y2 = 1 - np.exp(-1 * np.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))

    return np.c_[-y1, -y2]


optimizer = RandomOptimizer(min_X=min_X, max_X=max_X, nsamples=121)
policy = physbo.search.range_multi.Policy(min_X=min_X, max_X=max_X, num_objectives=2)
policy.set_seed(0)
# Random search (10 times)
policy.random_search(max_num_probes=10, simulator=vlmop2_minus)

# Bayesian search (40 times)
#   score function (acquition function): expectation of improvement (EI)
policy.bayes_search(max_num_probes=40, simulator=vlmop2_minus, score=score, interval=0, optimizer=optimizer)

print("Pareto fronts:")
res = policy.history
front, front_index = res.export_pareto_front()

with open("pareto_front_range.txt", "w") as f:
    for fr, ifr in zip(front, front_index):
        X = res.action_X[ifr, :]
        print("  action: ", ifr)
        print("  X: ", X)
        print("  f: ", fr)
        print()
        for y in fr:
            f.write(f"{y} ")
        for x in X:
            f.write(f"{x} ")
        f.write("\n")

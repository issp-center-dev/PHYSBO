# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import physbo

from mpi4py import MPI

min_X = np.array([-2, -2])
max_X = np.array([2, 2])
alg_name = "exchange"

def simulator(x):
    return -np.sum(x**2, axis=1)

policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X, comm=MPI.COMM_WORLD)
policy.set_seed(12345)
policy.random_search(max_num_probes=10, simulator=simulator)
policy.bayes_search(max_num_probes=10, num_search_each_probe=1, simulator=simulator, score="EI", alg_name=alg_name, num_rand_basis=100)

print(policy.history.export_sequence_best_fx())

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from mpi4py import MPI

import physbo
from physbo.search.optimize.odatse import Optimizer, default_alg_dict

nsamples = 10000

min_X = np.array([-2, -2])
max_X = np.array([2, 2])

alg_name = "exchange"
alg_dict = default_alg_dict(min_X, max_X, alg_name)
optimizer = Optimizer(alg_dict)

# optimizer = Optimizer(nsamples)


def simulator(x):
    return -np.sum(x**2, axis=1)


policy = physbo.search.range.Policy(min_X=min_X, max_X=max_X, comm=MPI.COMM_WORLD)
policy.set_seed(12345)
policy.random_search(max_num_probes=10, simulator=simulator)
# policy.bayes_search(max_num_probes=10, num_search_each_probe=1, simulator=simulator, score="EI", optimizer=optimizer, num_rand_basis=0)
policy.bayes_search(
    max_num_probes=10,
    num_search_each_probe=1,
    simulator=simulator,
    score="EI",
    num_rand_basis=0,
)

if MPI.COMM_WORLD.Get_rank() == 0:
    print(policy.history.export_sequence_best_fx())

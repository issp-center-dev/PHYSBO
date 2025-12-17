# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import time
import os

import numpy as np

import matplotlib.pyplot as plt

import physbo


def benchmark(
    fn: physbo.test_functions.multi_objective.MultiTestFunction,
    score: str,
    num_random_search: int,
    num_bayes_search: int,
    num_bayes_search_set: int,
    nrand_basis: int,
    N: int,
    seed: int,
    output_dir: str,
    pdffilename_prefix: str,
    optimizer: physbo.search.optimize.random.Optimizer | None = None,
):
    min_X = fn.min_X
    max_X = fn.max_X

    if fn.dim <= 3:
        X = physbo.search.utility.make_grid(min_X, max_X, N)
    else:
        np.random.seed(seed)
        X = np.random.uniform(min_X, max_X, size=(N * N * N, fn.dim))
        X[0, :] = min_X
        X[-1, :] = max_X

    use_range = optimizer is not None

    if use_range:
        policy = physbo.search.range_multi.Policy(min_X=min_X, max_X=max_X, num_objectives=fn.nobj)
        sim = fn
    else:
        policy = physbo.search.discrete_multi.Policy(test_X=X, num_objectives=fn.nobj)
        sim = physbo.search.utility.Simulator(test_X=X, test_function=fn)
    policy.set_seed(seed)
    policy.random_search(max_num_probes=num_random_search, simulator=sim, is_disp=False)

    res = []
    elapsed_time = []

    start_time = time.time()
    for i in range(num_bayes_search_set):
        r = policy.bayes_search(
            max_num_probes=num_bayes_search,
            simulator=sim,
            score=score,
            is_disp=False,
            num_rand_basis=nrand_basis,
            optimizer=optimizer,
        )
        end_time = time.time()
        elapsed_time.append(end_time - start_time)
        res.append(r)

    nobj = fn.nobj
    fig, ax = plt.subplots(nobj - 1, nobj - 1, figsize=(5 * (nobj - 1), 5 * (nobj - 1)), sharex="col", sharey="row")

    vid = []
    num_bayes = []
    for i in range(num_bayes_search_set):
        num_bayes.append((i + 1) * num_bayes_search)
        v = res[i].pareto.volume_in_dominance(fn.reference_min, fn.reference_max)
        vid.append(v)
        physbo.search.utility.plot_pareto_front_all(
            res[i], steps_end=num_random_search, marker="+", ax=ax
        )
        physbo.search.utility.plot_pareto_front_all(
            res[i], steps_begin=num_random_search, marker="o", ax=ax
        )
        filename = os.path.join(
            output_dir,
            f"{pdffilename_prefix}_{score}_{(i + 1) * num_bayes_search}.pdf",
        )
        fig.savefig(filename)
    plt.close(fig)
    return vid, elapsed_time, num_bayes

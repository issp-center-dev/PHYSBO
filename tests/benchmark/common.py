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
    pdffilename_prefix: str | None = None,
    optimizer: physbo.search.optimize.random.Optimizer | None = None,
    unify_method=None,
    skip_volume_calculation=False,
):
    min_X = fn.min_X
    max_X = fn.max_X

    if fn.dim <= 3:
        X = physbo.search.utility.make_grid(min_X, max_X, N, constraint=fn.constraint)
    else:
        np.random.seed(seed)
        X = np.random.uniform(min_X, max_X, size=(N * N * N, fn.dim))
        X[0, :] = min_X
        X[-1, :] = max_X
        X = X[fn.constraint(X).reshape(-1)]

    use_range = optimizer is not None
    use_unify = unify_method is not None

    if use_range:
        sim = fn
        if use_unify:
            policy = physbo.search.range_unified.Policy(min_X=min_X, max_X=max_X, num_objectives=fn.nobj)
        else:
            policy = physbo.search.range_multi.Policy(min_X=min_X, max_X=max_X, num_objectives=fn.nobj)
    else:
        sim = physbo.search.utility.Simulator(test_X=X, test_function=fn)
        if use_unify:
            policy = physbo.search.discrete_unified.Policy(test_X=X, num_objectives=fn.nobj)
        else:
            policy = physbo.search.discrete_multi.Policy(test_X=X, num_objectives=fn.nobj)
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
            unify_method=unify_method,
        )
        end_time = time.time()
        elapsed_time.append(end_time - start_time)
        res.append(r)

    nobj = fn.nobj
    if pdffilename_prefix is not None:
        fig, ax = plt.subplots(nobj - 1, nobj - 1, figsize=(5 * (nobj - 1), 5 * (nobj - 1)), sharex="col", sharey="row")
    else:
        fig = None
        ax = None

    vid = []
    vid_times = []
    num_bayes = []
    time_start = time.time()
    for i in range(num_bayes_search_set):
        num_bayes.append((i + 1) * num_bayes_search)
        if not skip_volume_calculation:
            v = res[i].pareto.volume_in_dominance(fn.reference_min, fn.reference_max)
        else:
            v = -1.0
        etime = time.time() - time_start
        vid_times.append(etime)
        vid.append(v)
        if fig is not None:
            physbo.search.utility.plot_pareto_front_all(
                res[i], steps_end=num_random_search, marker="+", ax=ax
            )
            physbo.search.utility.plot_pareto_front_all(
                res[i], steps_begin=num_random_search, marker="o", ax=ax
            )
            filename = os.path.join(
                output_dir,
                f"{pdffilename_prefix}_{(i + 1) * num_bayes_search}.pdf",
            )
            fig.savefig(filename)
    if fig is not None:
        plt.close(fig)
    return vid, elapsed_time, num_bayes


def gen_centers(nobj: int, dim: int) -> np.ndarray:
    centers = np.zeros((nobj, dim))
    if dim == 1:
        centers[0, 0] = 1.0
        for i in range(1, nobj):
            centers[i, 0] = -0.75 * centers[i-1, 0]
    elif dim == 2:
        R = np.zeros((2, 2))
        R[0, 0] = np.cos(2 * np.pi / nobj)
        R[0, 1] = np.sin(2 * np.pi / nobj)
        R[1, 0] = -np.sin(2 * np.pi / nobj)
        R[1, 1] = np.cos(2 * np.pi / nobj)
        centers[0, 0] = 1.0
        centers[0, 1] = 0.0
        for i in range(1, nobj):
            centers[i, :] = R @ centers[i-1, :]
    else:
        raise ValueError(f"Dimension {dim} is not supported")
    return centers


def set_parameters(
    use_range: bool,
    fn: physbo.test_functions.multi_objective.MultiTestFunction,
    N: int,
    score: str,
    nrand_basis_TS: int,
    unify_method_name: str,
):
    if use_range:
        optimizer = physbo.search.optimize.random.Optimizer(min_X=fn.min_X, max_X=fn.max_X, nsamples=N*N)
    else:
        optimizer = None

    if score == "TS":
        nrand_basis = nrand_basis_TS
    else:
        nrand_basis = 0

    if unify_method_name == "ParEGO":
        unify_method = physbo.search.unify.ParEGO(num_objectives=fn.nobj)
    elif unify_method_name == "NDS":
        unify_method = physbo.search.unify.NDS(num_objectives=fn.nobj)
    else:
        unify_method = None

    if unify_method is None:
        if score == "EI":
            score = "EHVI"
        elif score == "PI":
            score = "HVPI"
        score_name = score
    else:
        score_name = f"{unify_method_name}-{score}"

    return {
        "optimizer": optimizer,
        "nrand_basis": nrand_basis,
        "unify_method": unify_method,
        "score": score,
        "score_name": score_name,
    }

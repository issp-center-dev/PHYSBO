# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import argparse
import sys
import os

import numpy as np

import physbo

import common


def gen_center(nobj: int, dim: int) -> np.ndarray:
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


parser = argparse.ArgumentParser()
parser.add_argument("--nobj_list", type=str, default="2")
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--score_list", type=str, default="EHVI,HVPI,TS")
parser.add_argument("--num_random_search", type=int, default=10)
parser.add_argument("--num_bayes_search", type=int, default=10)
parser.add_argument("--num_bayes_search_set", type=int, default=4)
parser.add_argument("--nrand_basis_TS", type=int, default=300)
parser.add_argument("--N", type=int, default=51)
parser.add_argument("--use_range", action="store_true")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--output_dir", type=str, default="output_nobjs")
args = parser.parse_args()

nobj_list = [int(nobj) for nobj in args.nobj_list.split(",")]
dim = args.dim
score_list = args.score_list.split(",")
num_random_search = args.num_random_search
num_bayes_search = args.num_bayes_search
num_bayes_search_set = args.num_bayes_search_set
nrand_basis_TS = args.nrand_basis_TS
N = args.N
use_range = args.use_range
seed = args.seed
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

for nobj in nobj_list:
    pdffilename_prefix = f"solutions_nobj_{nobj}_dim_{dim}"
    centers = gen_center(nobj, dim)
    fn = physbo.test_functions.multi_objective.Gaussian(centers=centers)
    print(f"Benchmarking {nobj=}...")
    filename = os.path.join(output_dir, f"nobj_{nobj}_dim_{dim}.txt")
    if use_range:
        optimizer = physbo.search.optimize.random.Optimizer(min_X=fn.min_X, max_X=fn.max_X, nsamples=N*N)
    else:
        optimizer = None
    with open(filename, "w") as f:
        for score in score_list:
            if score == "TS":
                nrand_basis = nrand_basis_TS
            else:
                nrand_basis = 0
            vid, elapsed_time, num_bayes = common.benchmark(
                fn=fn,
                score=score,
                num_random_search=num_random_search,
                num_bayes_search=num_bayes_search,
                num_bayes_search_set=num_bayes_search_set,
                nrand_basis=nrand_basis,
                N=N,
                optimizer=optimizer,
                seed=seed,
                output_dir=output_dir,
                pdffilename_prefix=pdffilename_prefix,
            )
            for v, t, n in zip(vid, elapsed_time, num_bayes):
                f.write(f"{score} {n} {v} {t}\n")
                sys.stdout.write(f"{score} {n} {v} {t}\n")

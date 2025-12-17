# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import argparse
import os
import sys

import physbo

import common


parser = argparse.ArgumentParser()
parser.add_argument("--test_function", type=str, default="VLMOP2", help="Test function name")
parser.add_argument("--Gaussian_dim", type=int, default=2, help="Dimension for Gaussian test function")
parser.add_argument("--Gaussian_nobj", type=int, default=2, help="Number of objectives for Gaussian test function")
parser.add_argument("--score", type=str, default="TS", help="Score name")
parser.add_argument("--unify_method", type=str, default="None", help="Unify method name")
parser.add_argument("--num_random_search", type=int, default=10, help="Number of random search")
parser.add_argument("--num_bayes_search", type=int, default=10, help="Number of Bayes search")
parser.add_argument("--num_bayes_search_set", type=int, default=4, help="Number of Bayes search set")
parser.add_argument("--nrand_basis_TS", type=int, default=300, help="Number of random basis for TS")
parser.add_argument("--N", type=int, default=51, help="Number of points in each dimension")
parser.add_argument("--use_range", action="store_true", help="Use range-based search")
parser.add_argument("--seed", type=int, default=12345, help="Random seed")
parser.add_argument("--output_dir", type=str, default="output_functions", help="Output directory")
parser.add_argument("--header", action="store_true", help="Write header to output file and exit")
args = parser.parse_args()

test_function = args.test_function
if test_function == "Gaussian":
    nobj = args.Gaussian_nobj
    dim = args.Gaussian_dim
    centers = common.gen_centers(nobj, dim)
    fn = physbo.test_functions.multi_objective.Gaussian(centers=centers)
    test_name = f"Gaussian_{nobj}objs_{dim}dim"
else:
    fn = getattr(physbo.test_functions.multi_objective, test_function)()
    test_name = test_function

header = args.header
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, f"{test_name}.txt")
if header:
    with open(filename, "w") as f:
        f.write("score num_bayes volume elapsed_time\n")
    sys.stdout.write("score num_bayes volume elapsed_time\n")
    sys.exit(0)

score = args.score
unify_method_name = args.unify_method
num_random_search = args.num_random_search
num_bayes_search = args.num_bayes_search
num_bayes_search_set = args.num_bayes_search_set
nrand_basis_TS = args.nrand_basis_TS
N = args.N
use_range = args.use_range
seed = args.seed

parameters = common.set_parameters(
    use_range=use_range,
    fn=fn,
    N=N,
    score=score,
    nrand_basis_TS=nrand_basis_TS,
    unify_method_name=unify_method_name,
)
optimizer = parameters["optimizer"]
nrand_basis = parameters["nrand_basis"]
unify_method = parameters["unify_method"]
score_name = parameters["score_name"]

pdffilename_prefix = f"solutions_{test_name}_{score_name}"
with open(filename, "a") as f:
    if header:
        f.write("score num_bayes volume elapsed_time\n")

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
        unify_method=unify_method,
    )
    for v, t, n in zip(vid, elapsed_time, num_bayes):
        f.write(f"{score_name} {n} {v} {t}\n")
        sys.stdout.write(f"{score_name} {n} {v} {t}\n")

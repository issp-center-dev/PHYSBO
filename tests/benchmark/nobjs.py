# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import argparse
import sys
import os

import physbo

import common


parser = argparse.ArgumentParser()
parser.add_argument("--nobj", type=int, default=2)
parser.add_argument("--dim", type=int, default=2)
parser.add_argument("--score", type=str, default="TS")
parser.add_argument("--unify_method", type=str, default="None")
parser.add_argument("--num_random_search", type=int, default=10)
parser.add_argument("--num_bayes_search", type=int, default=40)
parser.add_argument("--nrand_basis_TS", type=int, default=300)
parser.add_argument("--N", type=int, default=51)
parser.add_argument("--use_range", action="store_true")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--output_dir", type=str, default="output_nobjs")
parser.add_argument("--header", action="store_true")
parser.add_argument("--savefig", action="store_true")
args = parser.parse_args()

dim = args.dim

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, f"dim_{dim}.txt")
if args.header:
    with open(filename, "w") as f:
        f.write("score num_bayes volume elapsed_time\n")
    sys.stdout.write("score num_bayes volume elapsed_time\n")
    sys.exit(0)

nobj = args.nobj
centers = common.gen_centers(nobj, dim)
fn = physbo.test_functions.multi_objective.Gaussian(centers=centers)

num_random_search = args.num_random_search
num_bayes_search = args.num_bayes_search
num_bayes_search_set = 1
N = args.N
use_range = args.use_range
seed = args.seed

parameters = common.set_parameters(
    use_range=use_range,
    fn=fn,
    N=N,
    score=args.score,
    nrand_basis_TS=args.nrand_basis_TS,
    unify_method_name=args.unify_method,
)
optimizer = parameters["optimizer"]
nrand_basis = parameters["nrand_basis"]
unify_method = parameters["unify_method"]
score = parameters["score"]
score_name = parameters["score_name"]

if args.savefig:
    pdffilename_prefix = f"solutions_nobj_{nobj}_dim_{dim}_{score_name}"
else:
    pdffilename_prefix = None

with open(filename, "a") as f:
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
    for v, t in zip(vid, elapsed_time):
        f.write(f"{score_name} {nobj} {v} {t}\n")
        sys.stdout.write(f"{score_name} {nobj} {v} {t}\n")

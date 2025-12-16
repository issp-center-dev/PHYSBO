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
parser.add_argument("--class_list", type=str, default="VLMOP1,VLMOP2,BinhKorn,ChankongHaimes,ConstrEX")
parser.add_argument("--score_list", type=str, default="EHVI,HVPI,TS")
parser.add_argument("--num_random_search", type=int, default=10)
parser.add_argument("--num_bayes_search", type=int, default=10)
parser.add_argument("--num_bayes_search_set", type=int, default=4)
parser.add_argument("--nrand_basis_TS", type=int, default=300)
parser.add_argument("--N", type=int, default=51)
parser.add_argument("--use_range", action="store_true")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--output_dir", type=str, default="output_functions")
args = parser.parse_args()

class_list = args.class_list.split(",")
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

for class_name in class_list:
    print(f"Benchmarking {class_name}...")
    fn_class = getattr(physbo.test_functions.multi_objective, class_name)
    fn = fn_class()
    if use_range:
        optimizer = physbo.search.optimize.random.Optimizer(min_X=fn.min_X, max_X=fn.max_X, nsamples=N*N)
    else:
        optimizer = None
    filename = os.path.join(output_dir, f"{class_name}.txt")
    pdffilename_prefix = f"solutions_{class_name}"
    with open(filename, "w") as f:
        f.write("score num_bayes volume elapsed_time\n")
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

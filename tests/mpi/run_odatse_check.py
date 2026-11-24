# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Check serial and parallel ODAT-SE searches under MPI."""

import os
import tempfile
import traceback

import numpy as np
from mpi4py import MPI

import odatse.mpi
from physbo.search.optimize.odatse import Optimizer, default_alg_dict


def check_serial_requires_one_process():
    optimizer = Optimizer(default_alg_dict([0.0, 0.0], [1.0, 1.0], "mapper"))
    try:
        optimizer(lambda x: -np.sum((x - 0.5) ** 2), mpicomm=None)
    except ValueError:
        pass
    else:
        raise AssertionError("mpicomm=None accepted a multi-process MPI context")


def main():
    comm = MPI.COMM_WORLD
    if hasattr(odatse.mpi, "setup") and comm.size > 1:
        check_serial_requires_one_process()

    previous_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as local_dir:
        output_dir = comm.bcast(local_dir if comm.rank == 0 else None, root=0)
        os.chdir(output_dir)
        try:
            if hasattr(odatse.mpi, "setup") and comm.size == 1:
                optimizer = Optimizer(default_alg_dict([0.0, 0.0], [1.0, 1.0], "minsearch"))
                for target in [0.25, 0.75]:
                    X = optimizer(lambda x: -np.sum((x - target) ** 2))
                    assert np.allclose(X, [[target, target]], atol=0.05)
                    assert odatse.mpi.algsize() == 1
                    assert odatse.mpi.solsize() == 1
            names = ["minsearch"] if comm.size == 1 else [
                "mapper", "minsearch", "exchange", "pamc", "bayes"
            ]
            for name in names:
                optimizer = Optimizer(default_alg_dict([0.0, 0.0], [1.0, 1.0], name))
                for target in [0.3, 0.7]:
                    X = optimizer(lambda x: -np.sum((x - target) ** 2), mpicomm=comm)
                    assert X.shape == (1, 2)
                    assert np.allclose(X[0], [target, target], atol=0.05)
                    for other in comm.allgather(X):
                        assert np.array_equal(X, other)
            if hasattr(odatse.mpi, "setup") and comm.size > 1:
                check_serial_requires_one_process()
            comm.Barrier()
        finally:
            os.chdir(previous_dir)
    if comm.rank == 0:
        print("all MPI checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)

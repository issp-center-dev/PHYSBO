# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Launcher for the MPI consistency checks.

The actual checks live in run_mpi_check.py and are executed under
``mpirun -np 2``. This test is skipped when mpi4py or an MPI launcher
is not available, so the regular (serial) test runs are unaffected.
"""

import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("physbo")

MPIEXEC = shutil.which("mpirun") or shutil.which("mpiexec")


@pytest.mark.skipif(MPIEXEC is None, reason="no MPI launcher (mpirun/mpiexec)")
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
@pytest.mark.parametrize(
    "script_name, nprocs",
    [("run_mpi_check.py", 2), ("run_odatse_check.py", 1), ("run_odatse_check.py", 2)],
)
def test_mpi_consistency(script_name, nprocs):
    script = os.path.join(os.path.dirname(__file__), script_name)
    res = subprocess.run(
        [MPIEXEC, "-np", str(nprocs), sys.executable, script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    assert res.returncode == 0
    assert "all MPI checks passed" in res.stdout

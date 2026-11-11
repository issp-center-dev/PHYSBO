# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Code shared by the six policies.

The policies of the discrete and of the range search spaces, and those of
one, several and unified objectives, repeat a good part of their code.
What is literally the same is kept here: a function, and classes that only
hold methods (no state and no constructor of their own) and are listed
among the bases of the policies that share them.
"""

from .. import _mpi


def run_simulator(simulator, action, comm=None):
    """
    Run the simulator once for the proposed points of a step.

    In a parallel run the simulator is called on rank 0 only, and its
    result is broadcast; an exception it raises is raised on every process.

    Parameters
    ----------
    simulator: callable
        Function that takes the proposed points (indices of candidates in
        the discrete policies, coordinates in the range policies) and
        returns the value(s) of the objective function(s).
    action: numpy.ndarray
        The proposed points.
    comm: MPI.Comm, optional
        MPI communicator

    Returns
    -------
    The return value of the simulator, unchanged.
    """
    if comm is None:
        return simulator(action)
    if comm.rank == 0:
        t, error = _mpi.call(comm, simulator, action)
    else:
        t, error = 0.0, None
    return _mpi.bcast(comm, t, error, root=0)

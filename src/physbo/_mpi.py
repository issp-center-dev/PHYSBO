# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Collective operations that raise on every process when one failed.

A computation that runs on some processes only, or on data that differ
between the processes (the simulator on rank 0, the scores of each process's
share of the candidates, random samples, ...), can raise on some processes
while the others wait in the next collective operation. The functions here
send, along with each process's result, whether its computation raised, and
raise on every process when one did: a process that raised re-raises its own
exception, the others a copy of the exception of the first process that
raised (a ``RuntimeError`` with its type and message if it cannot be copied
with pickle), caused by a :class:`ProcessError` that holds its traceback.

Computations that run on identical data on every process are assumed to fail
on every process alike, and are not wrapped. Exceptions that are not an
``Exception`` (e.g. ``KeyboardInterrupt``) are not caught.

Usage::

    f, error = _mpi.call(comm, compute, ...)
    fs = _mpi.allgather(comm, f, error)
"""

import pickle
import traceback


class ProcessError(Exception):
    """The exception of another process, with its formatted traceback."""

    def __init__(self, rank, tb):
        super().__init__(rank, tb)
        self.rank = rank
        self.tb = tb

    def __str__(self):
        return f"raised on process {self.rank}:\n{self.tb}"


def is_parallel(comm):
    return comm is not None and comm.Get_size() > 1


def call(comm, fn, *args, **kwargs):
    """Call ``fn(*args, **kwargs)`` and return ``(result, error)``.

    In a parallel run an ``Exception`` is returned as ``error`` (with
    ``result`` None) to be passed to the next collective operation of this
    module; otherwise it is raised.
    """
    if not is_parallel(comm):
        return fn(*args, **kwargs), None
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, e


def _pack(error):
    """What is sent of ``error``; must not raise, or the collective operation
    would be left on this process."""
    if error is None:
        return None
    name = type(error).__name__
    try:
        summary = f"{name}: {error}"
    except Exception:
        summary = f"{name}: (message unavailable)"
    try:
        tb = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    except Exception:
        tb = f"{summary}\n(traceback unavailable)\n"
    try:
        data = pickle.dumps(error)
    except Exception:
        data = None
    return data, summary, tb


def _raise(packed_errors, error):
    """Raise ``error``, or else the first of ``packed_errors``, pairs of a rank
    and its packed error."""
    if error is not None:
        raise error
    for rank, packed in packed_errors:
        if packed is not None:
            break
    else:
        return
    data, summary, tb = packed
    exc = None
    if data is not None:
        try:
            exc = pickle.loads(data)
        except Exception:
            exc = None
    if not isinstance(exc, Exception):
        exc = RuntimeError(summary)
    raise exc from ProcessError(rank, tb)


def allgather(comm, value, error=None):
    """``comm.allgather(value)``, raising on every process if one passes ``error``."""
    gathered = comm.allgather((None if error is not None else value, _pack(error)))
    _raise([(rank, packed) for rank, (_, packed) in enumerate(gathered)], error)
    return [value for value, _ in gathered]


def bcast(comm, value, error=None, root=0):
    """The ``value`` of ``root`` on every process, as ``comm.bcast(value, root)``,
    raising on every process if one passes ``error``.

    Implemented with ``allgather``, in which the other processes send only
    whether they failed.
    """
    if comm.Get_rank() == root and error is None:
        message = (value, None)
    else:
        message = (None, _pack(error))
    gathered = comm.allgather(message)
    _raise([(rank, packed) for rank, (_, packed) in enumerate(gathered)], error)
    return gathered[root][0]

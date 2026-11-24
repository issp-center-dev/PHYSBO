# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The helpers of physbo._mpi, with a communicator simulated in one process.

The checks under MPI are in tests/mpi/run_error_check.py.
"""

import pytest

pytest.importorskip("physbo")

from physbo import _mpi


class FakeComm:
    """Process ``rank`` of ``size``; the other processes send ``others``,
    a dict from rank to their message."""

    def __init__(self, rank, size, others=None):
        self.rank = rank
        self.size = size
        self.others = others or {}

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def allgather(self, message):
        return [
            message if r == self.rank else self.others[r] for r in range(self.size)
        ]

    def bcast(self, message, root=0):
        return message if self.rank == root else self.others[root]


class Failure(Exception):
    pass


def fail():
    raise Failure("injected")


def test_call_raises_without_parallel_run():
    with pytest.raises(Failure):
        _mpi.call(None, fail)
    with pytest.raises(Failure):
        _mpi.call(FakeComm(0, 1), fail)


def test_call_returns_error_in_parallel_run():
    result, error = _mpi.call(FakeComm(0, 2), fail)
    assert result is None
    assert isinstance(error, Failure)
    assert _mpi.call(FakeComm(0, 2), lambda x: x + 1, 1) == (2, None)


def test_call_does_not_catch_base_exceptions():
    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _mpi.call(FakeComm(0, 2), interrupt)


def packed_error_of_other_process():
    """What another process sends after an error."""
    _, error = _mpi.call(FakeComm(1, 2), fail)
    return _mpi._pack(error)


def test_allgather_without_errors():
    comm = FakeComm(0, 3, {1: (11, None), 2: (12, None)})
    assert _mpi.allgather(comm, 10) == [10, 11, 12]


def test_allgather_raises_own_error():
    comm = FakeComm(0, 2, {1: (11, None)})
    _, error = _mpi.call(comm, fail)
    with pytest.raises(Failure) as info:
        _mpi.allgather(comm, None, error)
    assert info.value is error


def test_allgather_raises_copy_of_first_error_of_others():
    packed = packed_error_of_other_process()
    other = _mpi._pack(ValueError("later"))
    comm = FakeComm(0, 3, {1: (None, packed), 2: (None, other)})
    with pytest.raises(Failure, match="^injected$") as info:
        _mpi.allgather(comm, 10)
    cause = info.value.__cause__
    assert isinstance(cause, _mpi.ProcessError)
    assert cause.rank == 1
    assert "Failure: injected" in cause.tb and "in fail" in cause.tb
    assert "raised on process 1" in str(cause)


def test_bcast_without_errors():
    # bcast is an allgather in which the other processes send no value
    comm = FakeComm(1, 3, {0: (5, None), 2: (None, None)})
    assert _mpi.bcast(comm, None) == 5


def test_bcast_raises_error_of_any_process():
    packed = packed_error_of_other_process()
    # of the root
    with pytest.raises(Failure) as info:
        _mpi.bcast(FakeComm(1, 2, {0: (None, packed)}), None)
    assert info.value.__cause__.rank == 0

    # of another process, also on the root
    with pytest.raises(Failure) as info:
        _mpi.bcast(FakeComm(0, 3, {1: (None, packed), 2: (None, None)}), 5)
    assert info.value.__cause__.rank == 1

    # the process that raised raises its own
    _, error = _mpi.call(FakeComm(1, 2), fail)
    with pytest.raises(Failure) as info:
        _mpi.bcast(FakeComm(1, 2, {0: (5, None)}), None, error)
    assert info.value is error


class NeedsTwoArguments(Exception):
    def __init__(self, a, b):
        super().__init__(f"{a} {b}")


def test_exception_that_cannot_be_copied_becomes_runtime_error():
    # pickled, but not unpickled
    packed = _mpi._pack(NeedsTwoArguments(1, 2))
    comm = FakeComm(0, 2, {1: (None, packed)})
    with pytest.raises(RuntimeError, match="^NeedsTwoArguments: 1 2$"):
        _mpi.allgather(comm, 10)

    # not pickled
    error = Failure("injected")
    error.f = lambda: None
    packed = _mpi._pack(error)
    assert packed[0] is None
    comm = FakeComm(0, 2, {1: (None, packed)})
    with pytest.raises(RuntimeError, match="^Failure: injected$"):
        _mpi.allgather(comm, 10)


class BadMessage(Exception):
    def __str__(self):
        raise AttributeError("no message")


def fail_without_message():
    raise BadMessage()


def test_pack_does_not_raise_when_the_message_fails():
    _, error = _mpi.call(FakeComm(1, 2), fail_without_message)
    packed = _mpi._pack(error)
    assert packed[1] == "BadMessage: (message unavailable)"
    # the process that raised still raises its own exception
    with pytest.raises(BadMessage):
        _mpi.allgather(FakeComm(1, 2, {0: (1, None)}), None, error)
    # the others raise a copy
    with pytest.raises(BadMessage) as info:
        _mpi.allgather(FakeComm(0, 2, {1: (None, packed)}), 1)
    assert info.value.__cause__.rank == 1

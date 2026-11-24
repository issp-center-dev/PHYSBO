# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Checkpoint support for the search policies.

This module provides :class:`CheckpointMixin`, which makes a policy

- safely picklable (the MPI communicator is excluded and can be
  re-attached with :meth:`CheckpointMixin.set_comm`), and
- resumable through :meth:`CheckpointMixin.save_checkpoint` /
  :meth:`CheckpointMixin.load_checkpoint`, a single-file checkpoint
  consolidated on rank 0.

Unlike ``Policy.save()``/``Policy.load()`` (which persist the search
*results* in a portable, version-tolerant form), a checkpoint captures
the complete execution state -- including the random number generator --
so that a resumed run continues bit-exactly as if it had never stopped.

A checkpoint file holds two pickles, one after the other: a header (a dict
of the format version, the PHYSBO version, the policy class and the number
of MPI processes) and the state (the state shared by the processes and the
part of each process). ``load_checkpoint`` reads and checks the header
before it unpickles the state, so that a file it cannot use is refused with
a clear error whatever the state contains. Under MPI, the file is read on
rank 0 and a failure to read it is raised on every rank.

The pickled state of a policy carries the format version too, so that a
policy pickled on its own (inside the checkpoint of another application) is
refused in the same way when the layout of the state has changed.
"""

import os
import pickle
import warnings

import numpy as np

from .. import _mpi
from .._rng import LegacyRNG, get_rng

FORMAT_VERSION = 2

# the entry of the pickled state of a policy that holds FORMAT_VERSION
_STATE_VERSION_KEY = "_format_version"


def _class_name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def _export_rng_state(rng):
    """Serialize the RNG state.

    In the legacy mode the state lives in the global ``numpy.random``
    module, so it is captured explicitly; a Generator carries its own
    state and is pickled as is.
    """
    if isinstance(rng, LegacyRNG):
        return ("legacy", np.random.get_state())
    return ("generator", rng)


class CheckpointMixin:
    """Pickling and checkpoint support shared by the search policies."""

    def __getstate__(self):
        # The MPI communicator cannot be pickled (except, incidentally,
        # predefined ones such as COMM_WORLD); it is dropped here and must
        # be re-attached with set_comm() after unpickling.
        state = self.__dict__.copy()
        state["mpicomm"] = None
        state[_STATE_VERSION_KEY] = FORMAT_VERSION
        return state

    def __setstate__(self, state):
        state = dict(state)
        fv = state.pop(_STATE_VERSION_KEY, None)
        if fv != FORMAT_VERSION:
            raise RuntimeError(
                "this policy was pickled with the state in format version "
                f"{1 if fv is None else fv}, but this version of PHYSBO "
                f"reads format version {FORMAT_VERSION}. A pickled policy "
                "is readable by the minor version of PHYSBO that wrote it; "
                "use save() and load() to keep results across versions."
            )
        self.__dict__.update(state)
        # normalize the legacy adapter back to the module-wide singleton
        if isinstance(self.rng, LegacyRNG):
            self.rng = get_rng()

    def set_comm(self, comm):
        """Re-attach an MPI communicator to a restored policy.

        Parameters
        ----------
        comm: MPI.Comm or None
            The communicator to attach. Its size and rank must match the
            ``mpisize`` and ``mpirank`` stored in the policy state; a
            policy restored from a serial run accepts only ``None``.

        Raises
        ------
        RuntimeError
            If the communicator size or rank does not match the stored
            state.
        """
        if comm is None:
            if self.mpisize != 1:
                raise RuntimeError(
                    f"this policy state was taken with mpisize={self.mpisize}; "
                    "it cannot be used without an MPI communicator"
                )
            self.mpicomm = None
            return
        if comm.size != self.mpisize:
            raise RuntimeError(
                f"this policy state was taken with mpisize={self.mpisize}, "
                f"but the communicator has size={comm.size}"
            )
        if comm.rank != self.mpirank:
            raise RuntimeError(
                f"this policy state belongs to rank {self.mpirank}, "
                f"but the communicator rank is {comm.rank}"
            )
        self.mpicomm = comm
        self.config.learning.is_disp = (
            self.config.learning.is_disp and self.mpirank == 0
        )

    def _require_comm(self):
        """Refuse to go on with a communicator that is not the one of the
        state of the policy.

        The state says how many processes share the candidates and the
        random numbers (``mpisize``), and the collective operations ask the
        communicator. They differ in a policy restored by pickle or copied
        from a parallel run, which has no communicator until set_comm() is
        called: a search would go on as a serial run on each process, and a
        checkpoint would hold the part of one process only.
        """
        size = 1 if self.mpicomm is None else self.mpicomm.size
        if size != self.mpisize:
            raise RuntimeError(
                f"ERROR: this policy is in the state of a run with "
                f"mpisize={self.mpisize}, but its communicator has {size} "
                "process(es). Attach the communicator of the run with "
                "set_comm() after unpickling or copying a policy."
            )

    def _local_checkpoint_state(self):
        """Rank-local part of the checkpoint (candidate partition and RNG)."""
        return {
            "rng": _export_rng_state(self.rng),
            "space": self.space.local_state(),
        }

    def _restore_local_checkpoint_state(self, state, rank):
        kind, payload = state["rng"]
        if kind == "legacy":
            # NOTE: restoring a legacy-mode checkpoint sets the *global*
            # numpy.random state (that is where the stream lives).
            np.random.set_state(payload)
            self.rng = get_rng()
        else:
            self.rng = payload
        self.space.restore_local_state(state["space"])
        self.mpirank = rank

    def save_checkpoint(self, filename):
        """Save the complete execution state into a single file.

        Under MPI this is a collective operation: every rank must call
        it. The rank-local parts (remaining candidates and RNG state) are
        gathered, and rank 0 writes one file atomically (via a temporary
        file and ``os.replace``).

        In the legacy RNG mode the global ``numpy.random`` state of each
        rank is captured; in the Generator mode (``rng=`` given to the
        policy) the policy-owned generators are stored.

        Parameters
        ----------
        filename: str
            The name of the checkpoint file.

        See Also
        --------
        load_checkpoint: restores the state and resumes bit-exactly.
        save: persists only the search results (portable across runs
            with different numbers of MPI processes).
        """
        self._require_comm()
        local = self._local_checkpoint_state()
        if self.mpicomm is None:
            per_rank = [local]
        else:
            per_rank = self.mpicomm.gather(local, root=0)
        if self.mpirank == 0:
            import physbo

            header = {
                "format_version": FORMAT_VERSION,
                "physbo_version": physbo.__version__,
                "policy_class": _class_name(type(self)),
                "mpisize": self.mpisize,
            }
            state = {"shared": self.__getstate__(), "per_rank": per_rank}
            tmpname = f"{filename}.tmp"
            with open(tmpname, "wb") as f:
                pickle.dump(header, f)
                pickle.dump(state, f)
            os.replace(tmpname, filename)

    @classmethod
    def load_checkpoint(cls, filename, comm=None):
        """Restore a policy from a checkpoint file.

        Under MPI this is a collective operation: every rank must call
        it with the same ``filename``; rank 0 reads the file and
        broadcasts it. The run must use the same number of MPI processes
        as the run that saved the checkpoint.

        In the legacy RNG mode this restores the global ``numpy.random``
        state of each rank as a side effect (the stream lives there); in
        the Generator mode the restored policy carries its own generator
        and the global state is not touched.

        Parameters
        ----------
        filename: str
            The name of the checkpoint file.
        comm: MPI.Comm, optional
            MPI communicator of the resuming run (default: None).

        Returns
        -------
        policy
            The restored policy, with ``comm`` attached.

        Raises
        ------
        RuntimeError
            If the checkpoint was taken with a different policy class,
            a different number of MPI processes, or an incompatible
            checkpoint format.
        """
        if comm is None:
            rank, size = 0, 1
        else:
            rank, size = comm.rank, comm.size

        parallel = _mpi.is_parallel(comm)
        f = None

        def read_header():
            nonlocal f
            f = open(filename, "rb")
            return pickle.load(f)

        try:
            # The header is read and checked, on every rank alike, before
            # the state is unpickled: the state of a file this version
            # cannot use may not even be loadable (classes that moved or do
            # not exist). Rank 0 reads the file, in one pass; an exception
            # while reading is raised on every rank.
            header, error = None, None
            if rank == 0:
                header, error = _mpi.call(comm, read_header)
            if parallel:
                header = _mpi.bcast(comm, header, error, root=0)
            cls._check_checkpoint_header(header, size)

            state, error = None, None
            if rank == 0:
                state, error = _mpi.call(comm, pickle.load, f)
            if parallel:
                state = _mpi.bcast(comm, state, error, root=0)
        finally:
            if f is not None:
                f.close()

        policy = cls.__new__(cls)
        policy.__setstate__(state["shared"])
        policy._restore_local_checkpoint_state(state["per_rank"][rank], rank)
        policy.set_comm(comm)
        return policy

    @classmethod
    def _check_checkpoint_header(cls, header, size):
        """Refuse a checkpoint that this class, this version of PHYSBO or a
        run of ``size`` processes cannot use."""
        fv = header.get("format_version") if isinstance(header, dict) else None
        if fv != FORMAT_VERSION:
            raise RuntimeError(
                f"unsupported checkpoint format version {fv} "
                f"(this version of PHYSBO supports {FORMAT_VERSION})"
            )
        if header["policy_class"] != _class_name(cls):
            raise RuntimeError(
                f"checkpoint was saved by {header['policy_class']}, "
                f"but is being loaded as {_class_name(cls)}"
            )
        if header["mpisize"] != size:
            raise RuntimeError(
                f"checkpoint was taken with mpisize={header['mpisize']}, "
                f"but the current run has mpisize={size}"
            )
        import physbo

        if header["physbo_version"] != physbo.__version__:
            warnings.warn(
                f"checkpoint was saved with PHYSBO {header['physbo_version']}, "
                f"but is being loaded with {physbo.__version__}; "
                "bit-exact resumption is not guaranteed",
                RuntimeWarning,
            )

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
"""

import os
import pickle
import warnings

import numpy as np

from .._rng import LegacyRNG, get_rng

FORMAT_VERSION = 1


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
        return state

    def __setstate__(self, state):
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

    def _local_checkpoint_state(self):
        """Rank-local part of the checkpoint (candidate partition and RNG)."""
        state = {"rng": _export_rng_state(self.rng)}
        if hasattr(self, "actions"):
            state["actions"] = self.actions
        return state

    def _restore_local_checkpoint_state(self, state, rank):
        kind, payload = state["rng"]
        if kind == "legacy":
            # NOTE: restoring a legacy-mode checkpoint sets the *global*
            # numpy.random state (that is where the stream lives).
            np.random.set_state(payload)
            self.rng = get_rng()
        else:
            self.rng = payload
        if "actions" in state:
            self.actions = state["actions"]
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
        local = self._local_checkpoint_state()
        if self.mpicomm is None:
            per_rank = [local]
        else:
            per_rank = self.mpicomm.gather(local, root=0)
        if self.mpirank == 0:
            import physbo

            ckpt = {
                "format_version": FORMAT_VERSION,
                "physbo_version": physbo.__version__,
                "policy_class": _class_name(type(self)),
                "mpisize": self.mpisize,
                "shared": self.__getstate__(),
                "per_rank": per_rank,
            }
            tmpname = f"{filename}.tmp"
            with open(tmpname, "wb") as f:
                pickle.dump(ckpt, f)
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

        ckpt = None
        if rank == 0:
            with open(filename, "rb") as f:
                ckpt = pickle.load(f)
        if comm is not None:
            ckpt = comm.bcast(ckpt, root=0)

        fv = ckpt.get("format_version")
        if fv != FORMAT_VERSION:
            raise RuntimeError(
                f"unsupported checkpoint format version {fv} "
                f"(this version of PHYSBO supports {FORMAT_VERSION})"
            )
        if ckpt["policy_class"] != _class_name(cls):
            raise RuntimeError(
                f"checkpoint was saved by {ckpt['policy_class']}, "
                f"but is being loaded as {_class_name(cls)}"
            )
        if ckpt["mpisize"] != size:
            raise RuntimeError(
                f"checkpoint was taken with mpisize={ckpt['mpisize']}, "
                f"but the current run has mpisize={size}"
            )
        import physbo

        if ckpt["physbo_version"] != physbo.__version__:
            warnings.warn(
                f"checkpoint was saved with PHYSBO {ckpt['physbo_version']}, "
                f"but is being loaded with {physbo.__version__}; "
                "bit-exact resumption is not guaranteed",
                RuntimeWarning,
            )

        policy = cls.__new__(cls)
        policy.__setstate__(ckpt["shared"])
        policy._restore_local_checkpoint_state(ckpt["per_rank"][rank], rank)
        policy.set_comm(comm)
        return policy

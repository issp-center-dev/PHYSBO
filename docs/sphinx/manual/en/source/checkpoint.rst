Checkpoint and restart
==================================

PHYSBO provides two ways to persist a search:

- ``policy.save()`` / ``policy.load()`` save the search *results*
  (history, training data, and predictor) into portable files.
  They are suitable for analyzing results later or for warm-starting a
  new search, and the files can be loaded by runs with a different
  number of MPI processes. They do **not** save the random number
  generator state, so a restarted run does not reproduce the
  uninterrupted one.
- The checkpoint API described here saves the *complete execution
  state*, including the random number generator, so that a suspended
  run can be resumed **bit-exactly**: the continued search selects
  exactly the same candidates as if it had never stopped.

Random number generator modes
----------------------------------

The checkpoint mechanism interacts with the ``rng`` argument of the
policies:

.. code-block:: python

    # legacy mode (default): uses the global numpy.random state.
    # set_seed() seeds the global state, as in previous versions.
    policy = physbo.search.discrete.Policy(test_X=X)

    # Generator mode: the policy owns a numpy.random.Generator.
    # The state is stored on the policy itself.
    policy = physbo.search.discrete.Policy(test_X=X, rng=12345)

In the Generator mode the RNG state is part of the policy object, so
pickling the policy captures everything needed for exact resumption.
In the legacy mode the RNG state lives in the global ``numpy.random``
module; the checkpoint API captures and restores it explicitly (note
that loading a legacy-mode checkpoint therefore *sets the global
numpy.random state* as a side effect).

Saving and restoring a checkpoint
----------------------------------

.. code-block:: python

    import physbo

    policy = physbo.search.discrete.Policy(test_X=X, rng=12345)
    policy.random_search(max_num_probes=10, simulator=simulator)
    policy.bayes_search(max_num_probes=20, simulator=simulator, score="TS",
                        num_rand_basis=500)

    # save the complete execution state into a single file
    policy.save_checkpoint("search.ckpt")

    # ... later, possibly in a new process ...

    policy = physbo.search.discrete.Policy.load_checkpoint("search.ckpt")
    # continues exactly as if the run had never stopped
    policy.bayes_search(max_num_probes=20, simulator=simulator, score="TS",
                        num_rand_basis=500)

``load_checkpoint`` is a class method of the policy class that saved
the checkpoint; loading with a different policy class raises an error.
The checkpoint file records the PHYSBO version (a mismatch emits a
warning: bit-exact resumption is guaranteed only within the same
version) and the checkpoint format version.

Usage under MPI
----------------------------------

``save_checkpoint`` and ``load_checkpoint`` are *collective*
operations: every rank must call them. The rank-local state (the
remaining candidates of each rank and its RNG state) is gathered, and
rank 0 writes a single file; on load, rank 0 reads the file and
broadcasts it.

.. code-block:: python

    policy = physbo.search.discrete.Policy(test_X=X, comm=comm, rng=12345)
    policy.random_search(max_num_probes=10, simulator=simulator)
    policy.save_checkpoint("search.ckpt")     # all ranks call this

    # ... restart with the SAME number of MPI processes ...

    policy = physbo.search.discrete.Policy.load_checkpoint(
        "search.ckpt", comm=comm)             # all ranks call this

The resuming run must use the same number of MPI processes as the run
that saved the checkpoint; ``load_checkpoint`` raises an error
otherwise.

.. note::

   With the BLM predictor (``num_rand_basis > 0``), Thompson sampling
   draws its posterior sample on rank 0 and broadcasts it, so the
   search result is independent of the number of ranks. With the GP
   predictor (``num_rand_basis == 0``), Thompson sampling under MPI is
   a rank-local approximation and the result depends on the number of
   ranks; the BLM predictor is recommended for TS under MPI.

Embedding PHYSBO in another application
----------------------------------------

Applications that embed PHYSBO (such as ODAT-SE) and have their own
checkpoint mechanism can simply pickle the policy as part of their own
state. The MPI communicator is excluded from the pickled state
automatically and must be re-attached with ``set_comm()`` after
restoring:

.. code-block:: python

    # saving (inside the host application's own checkpoint routine)
    state = {
        "step": step,
        "policy": self.policy,     # picklable; the communicator is excluded
        # ... other host state ...
    }
    with open(filename, "wb") as f:
        pickle.dump(state, f)

    # restoring
    with open(filename, "rb") as f:
        state = pickle.load(f)
    self.policy = state["policy"]
    self.policy.set_comm(self.mpicomm)   # re-attach the communicator

``set_comm()`` validates that the size and rank of the communicator
match the stored state. If the policy uses the Generator mode
(``rng=`` given), its RNG state is included in the pickle
automatically; in the legacy mode the host must additionally save and
restore ``numpy.random.get_state()`` itself (as ODAT-SE already does).
